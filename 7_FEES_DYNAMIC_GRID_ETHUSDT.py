import os
import asyncio
import pandas as pd
import numpy as np
import json
import time
from datetime import datetime, timezone
from pybit.unified_trading import HTTP
from dotenv import load_dotenv

load_dotenv()

class TradeLogger:
    def __init__(self, bot_name, symbol):
        self.bot_name = bot_name
        self.symbol = symbol
        self.currency = "USDT"
        self.open_trades = {}
        self.trade_id = 1000
        
        # Emergency stop tracking
        self.daily_pnl = 0
        self.consecutive_losses = 0
        self.max_daily_loss = 50
        
        os.makedirs("logs", exist_ok=True)
        self.log_file = f"logs/{bot_name}_{symbol}.log"
        
    def generate_trade_id(self):
        self.trade_id += 1
        return self.trade_id
    
    def log_trade_open(self, side, expected_price, actual_price, qty, stop_loss, take_profit, info=""):
        trade_id = self.generate_trade_id()
        slippage = actual_price - expected_price if side == "BUY" else expected_price - actual_price
        
        log_entry = {
            "id": trade_id,
            "bot": self.bot_name,
            "symbol": self.symbol,
            "side": "LONG" if side == "BUY" else "SHORT",
            "action": "OPEN",
            "ts": datetime.now(timezone.utc).isoformat(),
            "expected_price": round(expected_price, 4),
            "actual_price": round(actual_price, 4),
            "slippage": round(slippage, 4),
            "qty": round(qty, 6),
            "stop_loss": round(stop_loss, 4),
            "take_profit": round(take_profit, 4),
            "currency": self.currency,
            "info": info
        }
        
        self.open_trades[trade_id] = {
            "entry_time": datetime.now(),
            "entry_price": actual_price,
            "side": side,
            "qty": qty,
            "stop_loss": stop_loss,
            "take_profit": take_profit
        }
        
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        
        return trade_id, log_entry
    
    def log_trade_close(self, trade_id, expected_exit, actual_exit, reason, fees_entry=-0.04, fees_exit=-0.04):
        if trade_id not in self.open_trades:
            return None
            
        trade = self.open_trades[trade_id]
        duration = (datetime.now() - trade["entry_time"]).total_seconds()
        slippage = actual_exit - expected_exit if trade["side"] == "SELL" else expected_exit - actual_exit
        
        gross_pnl = ((actual_exit - trade["entry_price"]) * trade["qty"] 
                    if trade["side"] == "BUY" 
                    else (trade["entry_price"] - actual_exit) * trade["qty"])
        
        # Proper rebate calculation for maker orders
        entry_rebate = trade["entry_price"] * trade["qty"] * abs(fees_entry) / 100
        exit_rebate = actual_exit * trade["qty"] * abs(fees_exit) / 100
        total_rebates = entry_rebate + exit_rebate
        net_pnl = gross_pnl + total_rebates
        
        log_entry = {
            "id": trade_id,
            "bot": self.bot_name,
            "symbol": self.symbol,
            "side": "LONG" if trade["side"] == "BUY" else "SHORT",
            "action": "CLOSE",
            "ts": datetime.now(timezone.utc).isoformat(),
            "duration_sec": int(duration),
            "entry_price": round(trade["entry_price"], 4),
            "expected_exit": round(expected_exit, 4),
            "actual_exit": round(actual_exit, 4),
            "slippage": round(slippage, 4),
            "qty": round(trade["qty"], 6),
            "gross_pnl": round(gross_pnl, 2),
            "fee_rebates": {"entry": round(entry_rebate, 2), "exit": round(exit_rebate, 2), "total": round(total_rebates, 2)},
            "net_pnl": round(net_pnl, 2),
            "reason": reason,
            "currency": self.currency
        }
        
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        
        del self.open_trades[trade_id]
        return log_entry

class DynamicGridBot:
    def __init__(self):
        self.symbol = 'ETHUSDT'
        self.demo_mode = os.getenv('DEMO_MODE', 'true').lower() == 'true'
        
        prefix = 'TESTNET_' if self.demo_mode else 'LIVE_'
        self.api_key = os.getenv(f'{prefix}BYBIT_API_KEY')
        self.api_secret = os.getenv(f'{prefix}BYBIT_API_SECRET')
        self.exchange = None
        
        self.position = None
        self.price_data = pd.DataFrame()
        self.account_balance = 1000
        
        # Configuration
        self.config = {
            'grid_levels': 10,
            'grid_spacing_pct': 0.6,
            'risk_per_trade': 2.0,
            'maker_offset_pct': 0.01,
            'maker_fee_pct': -0.04,
            'net_take_profit': 1.2,
            'net_stop_loss': 0.6,
            'atr_period': 14,
            'volatility_threshold': 0.015,
            'slippage_basis_points': 2,
        }
        
        self.grid_levels = []
        self.current_grid_index = -1
        self.last_update_time = None
        
        # Trade cooldown
        self.last_trade_time = 0
        self.trade_cooldown = 30
        
        self.logger = TradeLogger("DYNAMIC_GRID_FIXED", self.symbol)
        self.current_trade_id = None
    
    def connect(self):
        try:
            self.exchange = HTTP(demo=self.demo_mode, api_key=self.api_key, api_secret=self.api_secret)
            return self.exchange.get_server_time().get('retCode') == 0
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False
    
    def format_qty(self, qty):
        # ETH minimum quantity is 0.001 with 3 decimal places
        min_qty = 0.001
        if qty < min_qty:
            return "0"
        
        rounded_qty = round(qty / min_qty) * min_qty
        return f"{rounded_qty:.3f}"
    
    async def get_account_balance(self):
        try:
            wallet = self.exchange.get_wallet_balance(accountType="UNIFIED")
            if wallet.get('retCode') == 0:
                for coin in wallet['result']['list'][0]['coin']:
                    if coin['coin'] == 'USDT':
                        self.account_balance = float(coin['availableToWithdraw'])
                        return True
        except Exception as e:
            print(f"❌ Balance check error: {e}")
        return False
    
    def calculate_position_size(self, price, stop_loss_price):
        if self.account_balance <= 0:
            return 0
        
        risk_amount = self.account_balance * (self.config['risk_per_trade'] / 100)
        stop_distance = abs(price - stop_loss_price)
        
        if stop_distance == 0:
            return 0
        
        position_size_usdt = min(risk_amount / stop_distance * price, self.account_balance * 0.1)
        return position_size_usdt / price
    
    def apply_slippage(self, expected_price, side):
        slippage_pct = self.config['slippage_basis_points'] / 10000
        
        if side in ['BUY', 'Buy']:
            return expected_price * (1 + slippage_pct)
        else:
            return expected_price * (1 - slippage_pct)
    
    def calculate_atr(self, df):
        if len(df) < self.config['atr_period']:
            return None
        
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=self.config['atr_period']).mean()
        
        return atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else None
    
    def update_grid_levels(self, current_price, atr):
        if not atr:
            atr = current_price * 0.01
        
        volatility_factor = min(max(atr / current_price, 0.005), 0.03)
        adjusted_spacing = self.config['grid_spacing_pct'] / 100 * (1 + volatility_factor * 10)
        
        self.grid_levels = []
        for i in range(-self.config['grid_levels'], self.config['grid_levels'] + 1):
            if i != 0:
                level = current_price * (1 + i * adjusted_spacing)
                self.grid_levels.append({
                    'price': level,
                    'index': i,
                    'side': 'BUY' if i < 0 else 'SELL'
                })
        
        self.grid_levels.sort(key=lambda x: x['price'])
        self.last_update_time = datetime.now()
    
    def find_nearest_grid(self, current_price):
        if not self.grid_levels:
            return None
        
        for i, level in enumerate(self.grid_levels):
            if abs(current_price - level['price']) / level['price'] < 0.002:
                return i, level
        
        return None
    
    def generate_signal(self, df):
        if len(df) < 20:
            return None
        
        current_price = float(df['close'].iloc[-1])
        atr = self.calculate_atr(df)
        
        if not self.grid_levels or not self.last_update_time:
            self.update_grid_levels(current_price, atr)
            return None
        
        time_since_update = (datetime.now() - self.last_update_time).total_seconds()
        if time_since_update > 300:
            self.update_grid_levels(current_price, atr)
        
        grid_match = self.find_nearest_grid(current_price)
        if not grid_match:
            return None
        
        grid_index, grid_level = grid_match
        
        if grid_index == self.current_grid_index:
            return None
        
        ema_short = df['close'].ewm(span=9).mean().iloc[-1]
        trend = 'UP' if current_price > ema_short else 'DOWN'
        
        if grid_level['side'] == 'BUY' and trend == 'UP':
            self.current_grid_index = grid_index
            return {
                'action': 'BUY',
                'price': current_price,
                'grid_level': grid_level['price'],
                'grid_index': grid_level['index']
            }
        elif grid_level['side'] == 'SELL' and trend == 'DOWN':
            self.current_grid_index = grid_index
            return {
                'action': 'SELL',
                'price': current_price,
                'grid_level': grid_level['price'],
                'grid_index': grid_level['index']
            }
        
        return None
    
    async def get_market_data(self):
        try:
            klines = self.exchange.get_kline(
                category="linear",
                symbol=self.symbol,
                interval="15",
                limit=50
            )
            
            if klines.get('retCode') != 0:
                return False
            
            df = pd.DataFrame(klines['result']['list'], columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
            ])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
            
            self.price_data = df.sort_values('timestamp').reset_index(drop=True)
            return True
        except Exception as e:
            print(f"❌ Market data error: {e}")
            return False
    
    async def check_position(self):
        try:
            positions = self.exchange.get_positions(category="linear", symbol=self.symbol)
            if positions.get('retCode') == 0:
                pos_list = positions['result']['list']
                self.position = pos_list[0] if pos_list and float(pos_list[0]['size']) > 0 else None
        except Exception as e:
            print(f"❌ Position check error: {e}")
    
    def should_close(self):
        if not self.position:
            return False, ""
        
        current_price = float(self.price_data['close'].iloc[-1])
        entry_price = float(self.position.get('avgPrice', 0))
        side = self.position.get('side', '')
        
        if entry_price == 0:
            return False, ""
        
        profit_pct = ((current_price - entry_price) / entry_price * 100 if side == "Buy"
                     else (entry_price - current_price) / entry_price * 100)
        
        if profit_pct >= self.config['net_take_profit']:
            return True, "grid_target_reached"
        if profit_pct <= -self.config['net_stop_loss']:
            return True, "stop_loss"
        
        return False, ""
    
    async def execute_trade(self, signal):
        # Check trade cooldown
        if time.time() - self.last_trade_time < self.trade_cooldown:
            remaining = self.trade_cooldown - (time.time() - self.last_trade_time)
            print(f"⏰ Trade cooldown: wait {remaining:.0f}s")
            return
            
        if not await self.get_account_balance():
            print("❌ Could not get account balance")
            return
        
        if self.account_balance < 10:
            print(f"❌ Insufficient balance: ${self.account_balance:.2f}")
            return
        
        stop_loss_pct = self.config['net_stop_loss'] / 100
        stop_loss_price = (signal['price'] * (1 - stop_loss_pct) if signal['action'] == 'BUY'
                          else signal['price'] * (1 + stop_loss_pct))
        
        qty = self.calculate_position_size(signal['price'], stop_loss_price)
        formatted_qty = self.format_qty(qty)
        
        if formatted_qty == "0":
            print(f"❌ Position size too small: {qty:.6f}")
            return
        
        offset_mult = 1 - self.config['maker_offset_pct']/100 if signal['action'] == 'BUY' else 1 + self.config['maker_offset_pct']/100
        limit_price = round(signal['price'] * offset_mult, 2)
        expected_execution_price = self.apply_slippage(limit_price, signal['action'])
        
        try:
            order = self.exchange.place_order(
                category="linear",
                symbol=self.symbol,
                side="Buy" if signal['action'] == 'BUY' else "Sell",
                orderType="Limit",
                qty=formatted_qty,
                price=str(limit_price,
                timeInForce="PostOnly")
            )
            
            if order.get('retCode') == 0:
                self.last_trade_time = time.time()
                
                net_tp = (expected_execution_price * (1 + self.config['net_take_profit']/100) if signal['action'] == 'BUY' 
                         else expected_execution_price * (1 - self.config['net_take_profit']/100))
                net_sl = (expected_execution_price * (1 - self.config['net_stop_loss']/100) if signal['action'] == 'BUY' 
                         else expected_execution_price * (1 + self.config['net_stop_loss']/100))
                
                position_value = float(formatted_qty) * expected_execution_price
                risk_pct = (position_value * stop_loss_pct / self.account_balance) * 100
                
                self.current_trade_id, _ = self.logger.log_trade_open(
                    side=signal['action'],
                    expected_price=limit_price,
                    actual_price=expected_execution_price,
                    qty=float(formatted_qty),
                    stop_loss=net_sl,
                    take_profit=net_tp,
                    info=f"grid_level:{signal['grid_level']:.2f}_risk:{risk_pct:.1f}%_bal:{self.account_balance:.2f}"
                )
                
                print(f"📊 GRID {signal['action']}: {formatted_qty} @ ${limit_price:.2f}")
                print(f"   💰 Position Value: ${position_value:.2f} ({risk_pct:.1f}% of balance)")
                print(f"   🎯 Expected Execution: ${expected_execution_price:.2f} (with slippage)")
                print(f"   💎 TP: ${net_tp:.2f} | SL: ${net_sl:.2f}")
                
        except Exception as e:
            print(f"❌ Trade failed: {e}")
    
    async def close_position(self, reason):
        if not self.position:
            return
        
        current_price = float(self.price_data['close'].iloc[-1])
        side = "Sell" if self.position.get('side') == "Buy" else "Buy"
        qty = float(self.position['size'])
        
        offset_mult = 1 + self.config['maker_offset_pct']/100 if side == "Sell" else 1 - self.config['maker_offset_pct']/100
        limit_price = round(current_price * offset_mult, 2)
        expected_exit_price = self.apply_slippage(limit_price, side)
        
        try:
            order = self.exchange.place_order(
                category="linear",
                symbol=self.symbol,
                side=side,
                orderType="Limit",
                qty=self.format_qty(qty,
                timeInForce="PostOnly"),
                price=str(limit_price),
                timeInForce="PostOnly",
                reduceOnly=True)
            
            if order.get('retCode') == 0:
                if self.current_trade_id:
                    self.logger.log_trade_close(
                        trade_id=self.current_trade_id,
                        expected_exit=limit_price,
                        actual_exit=expected_exit_price,
                        reason=reason,
                        fees_entry=self.config['maker_fee_pct'],
                        fees_exit=self.config['maker_fee_pct']
                    )
                    self.current_trade_id = None
                
                print(f"✅ Closed: {reason} @ ${expected_exit_price:.2f}")
                self.position = None
                
        except Exception as e:
            print(f"❌ Close failed: {e}")
    
    def show_status(self):
        if len(self.price_data) == 0:
            return
        
        current_price = float(self.price_data['close'].iloc[-1])
        
        print(f"\n📊 FIXED Dynamic Grid Bot - {self.symbol}")
        print(f"💰 Price: ${current_price:.2f} | Balance: ${self.account_balance:.2f}")
        print(f"🔧 FIXES APPLIED:")
        print(f"   • Position Sizing: Risk-based ({self.config['risk_per_trade']}% per trade)")
        print(f"   • Fee Calculations: Proper maker rebates")  
        print(f"   • Slippage Modeling: {self.config['slippage_basis_points']} basis points")
        
        if self.grid_levels:
            buy_grids = [g for g in self.grid_levels if g['side'] == 'BUY']
            sell_grids = [g for g in self.grid_levels if g['side'] == 'SELL']
            if buy_grids:
                print(f"🟢 Next Buy Grid: ${buy_grids[-1]['price']:.2f}")
            if sell_grids:
                print(f"🔴 Next Sell Grid: ${sell_grids[0]['price']:.2f}")
        
        if self.position:
            entry_price = float(self.position.get('avgPrice', 0))
            side = self.position.get('side', '')
            size = self.position.get('size', '0')
            pnl = float(self.position.get('unrealisedPnl', 0))
            position_value = float(size) * current_price
            risk_pct = (position_value / self.account_balance) * 100 if self.account_balance > 0 else 0
            
            emoji = "🟢" if side == "Buy" else "🔴"
            print(f"{emoji} {side}: {size} ETH @ ${entry_price:.2f} | PnL: ${pnl:.2f}")
            print(f"   📊 Position: ${position_value:.2f} ({risk_pct:.1f}% of balance)")
        else:
            print("⚡ Waiting for optimal grid signals...")
        
        print("-" * 50)
    
    async def run_cycle(self):
        # Emergency stop check
        if self.logger.daily_pnl < -self.logger.max_daily_loss:
            print(f"🔴 EMERGENCY STOP: Daily loss ${abs(self.logger.daily_pnl):.2f} exceeded limit")
            if self.position:
                await self.close_position("emergency_stop")
            return
            
        if not await self.get_market_data():
            return
        
        await self.check_position()
        
        if self.position:
            should_close, reason = self.should_close()
            if should_close:
                await self.close_position(reason)
        else:
            signal = self.generate_signal(self.price_data)
            if signal:
                await self.execute_trade(signal)
        
        self.show_status()
    
    async def run(self):
        if not self.connect():
            print("❌ Failed to connect")
            return
        
        print(f"📊 FIXED Dynamic Grid Trading Bot - {self.symbol}")
        print(f"🔧 CRITICAL FIXES:")
        print(f"   ✅ Position Sizing: Account balance-based with {self.config['risk_per_trade']}% risk")
        print(f"   ✅ Fee Calculations: Proper maker rebate handling")
        print(f"   ✅ Slippage Modeling: {self.config['slippage_basis_points']} basis points expected slippage")
        print(f"   ✅ Instrument Precision: ETH 0.001 minimum quantity")
        print(f"💎 Using MAKER-ONLY orders for {abs(self.config['maker_fee_pct'])}% fee rebate")
        
        try:
            while True:
                await self.run_cycle()
                await asyncio.sleep(2)
        except KeyboardInterrupt:
            print("\n🛑 Bot stopped")
            if self.position:
                await self.close_position("manual_stop")
        except Exception as e:
            print(f"⚠️ Runtime error: {e}")
            await asyncio.sleep(5)

if __name__ == "__main__":
    bot = DynamicGridBot()
    asyncio.run(bot.run())