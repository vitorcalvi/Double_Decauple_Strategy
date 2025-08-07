import os
import asyncio
import pandas as pd
import numpy as np
import json
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
        
        os.makedirs("logs", exist_ok=True)
        self.log_file = f"logs/{bot_name}_{symbol}.log"
        
    def generate_trade_id(self):
        self.trade_id += 1
        return self.trade_id
    
    def log_trade_open(self, side, expected_price, actual_price, qty, stop_loss, take_profit, info=""):
        trade_id = self.generate_trade_id()
        # ✅ FIXED: Proper slippage calculation
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
        
        # ✅ FIXED: Proper slippage calculation
        slippage = actual_exit - expected_exit if trade["side"] == "SELL" else expected_exit - actual_exit
        
        if trade["side"] == "BUY":
            gross_pnl = (actual_exit - trade["entry_price"]) * trade["qty"]
        else:
            gross_pnl = (trade["entry_price"] - actual_exit) * trade["qty"]
        
        # ✅ FIXED: Proper maker rebate calculation (negative fees = rebates earned)
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

class RMISuperTrendBot:
    def __init__(self):
        self.symbol = 'ADAUSDT'
        self.demo_mode = os.getenv('DEMO_MODE', 'true').lower() == 'true'
        
        prefix = 'TESTNET_' if self.demo_mode else 'LIVE_'
        self.api_key = os.getenv(f'{prefix}BYBIT_API_KEY')
        self.api_secret = os.getenv(f'{prefix}BYBIT_API_SECRET')
        self.exchange = None
        
        self.position = None
        self.price_data = pd.DataFrame()
        self.account_balance = 1000  # Will be updated from API
        
        # ✅ FIXED: Strategy configuration with proper risk management
        self.config = {
            'timeframe': '3',
            'rmi_period': 14,
            'rmi_momentum': 5,
            'rmi_threshold_long': 55,
            'rmi_threshold_short': 45,
            'supertrend_period': 10,
            'supertrend_multiplier': 2,
            'risk_pct': 2.0,              # Risk 2% of account per trade
            'maker_offset_pct': 0.01,
            'maker_fee_pct': -0.04,
            'net_take_profit': 0.70,
            'net_stop_loss': 0.35,
            'slippage_pct': 0.02,         # Expected slippage 0.02%
            'min_notional': 5,            # Minimum $5 position
            'qty_precision': 0,           # ADA quantity precision (whole numbers)
        }
        
        self.logger = TradeLogger("RMI_SUPERTREND_FIXED", self.symbol)
        self.current_trade_id = None
    
    def connect(self):
        try:
            self.exchange = HTTP(demo=self.demo_mode, api_key=self.api_key, api_secret=self.api_secret)
            return self.exchange.get_server_time().get('retCode') == 0
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False
    
    # ✅ FIXED: Proper quantity formatting for ADA (whole numbers)
    def format_qty(self, qty):
        """Format quantity with proper precision for ADA"""
        return str(int(round(qty)))
    
    # ✅ FIXED: Get account balance for proper position sizing
    async def update_account_balance(self):
        """Update account balance from API"""
        try:
            wallet = self.exchange.get_wallet_balance(accountType="UNIFIED")
            if wallet.get('retCode') == 0:
                for coin in wallet['result']['list'][0]['coin']:
                    if coin['coin'] == 'USDT':
                        self.account_balance = float(coin['availableToWithdraw'])
                        break
        except Exception as e:
            print(f"⚠️ Balance update error: {e}")
    
    # ✅ FIXED: Calculate position size based on account balance and risk
    def calculate_position_size(self, price, stop_loss_price):
        """Calculate position size based on risk percentage"""
        if self.account_balance <= 0:
            return 0
        
        risk_amount = self.account_balance * (self.config['risk_pct'] / 100)
        price_diff = abs(price - stop_loss_price)
        
        if price_diff == 0:
            return 0
        
        # Calculate quantity that risks exactly risk_amount
        qty = risk_amount / price_diff
        
        # Apply minimum notional check
        notional = qty * price
        if notional < self.config['min_notional']:
            qty = self.config['min_notional'] / price
        
        return qty
    
    # ✅ FIXED: Add slippage modeling to limit price calculation
    def calculate_limit_price(self, market_price, side, include_slippage=True):
        """Calculate limit price with slippage and maker offset"""
        slippage_mult = 1 + (self.config['slippage_pct'] / 100) if include_slippage else 1
        
        if side == 'BUY':
            # For buy: add slippage, subtract maker offset for better fill
            price_with_slippage = market_price * slippage_mult
            limit_price = price_with_slippage * (1 - self.config['maker_offset_pct'] / 100)
        else:
            # For sell: subtract slippage, add maker offset for better fill
            price_with_slippage = market_price / slippage_mult
            limit_price = price_with_slippage * (1 + self.config['maker_offset_pct'] / 100)
        
        return round(limit_price, 4)
    
    def calculate_rmi(self, prices):
        """Calculate Relative Momentum Index"""
        if len(prices) < self.config['rmi_period'] + self.config['rmi_momentum']:
            return None
        
        momentum = prices.diff(self.config['rmi_momentum'])
        
        gain = momentum.where(momentum > 0, 0)
        loss = -momentum.where(momentum < 0, 0)
        
        avg_gain = gain.rolling(window=self.config['rmi_period']).mean()
        avg_loss = loss.rolling(window=self.config['rmi_period']).mean()
        
        rs = avg_gain / (avg_loss + 1e-10)
        rmi = 100 - (100 / (1 + rs))
        
        return rmi
    
    def calculate_supertrend(self, df):
        """Calculate SuperTrend indicator"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Calculate ATR
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=self.config['supertrend_period']).mean()
        
        # Calculate basic bands
        hl_avg = (high + low) / 2
        multiplier = self.config['supertrend_multiplier']
        
        upper_band = hl_avg + (multiplier * atr)
        lower_band = hl_avg - (multiplier * atr)
        
        # Initialize supertrend
        supertrend = pd.Series(index=df.index, dtype=float)
        direction = pd.Series(index=df.index, dtype=int)
        
        for i in range(self.config['supertrend_period'], len(df)):
            if pd.isna(upper_band.iloc[i]) or pd.isna(lower_band.iloc[i]):
                continue
                
            if i == self.config['supertrend_period']:
                if close.iloc[i] <= upper_band.iloc[i]:
                    direction.iloc[i] = -1
                    supertrend.iloc[i] = upper_band.iloc[i]
                else:
                    direction.iloc[i] = 1
                    supertrend.iloc[i] = lower_band.iloc[i]
            else:
                # Previous values
                prev_dir = direction.iloc[i-1]
                
                if close.iloc[i] <= upper_band.iloc[i]:
                    if prev_dir == -1:
                        if upper_band.iloc[i] < supertrend.iloc[i-1]:
                            supertrend.iloc[i] = upper_band.iloc[i]
                        else:
                            supertrend.iloc[i] = supertrend.iloc[i-1]
                        direction.iloc[i] = -1
                    else:
                        supertrend.iloc[i] = upper_band.iloc[i]
                        direction.iloc[i] = -1
                else:
                    if prev_dir == 1:
                        if lower_band.iloc[i] > supertrend.iloc[i-1]:
                            supertrend.iloc[i] = lower_band.iloc[i]
                        else:
                            supertrend.iloc[i] = supertrend.iloc[i-1]
                        direction.iloc[i] = 1
                    else:
                        supertrend.iloc[i] = lower_band.iloc[i]
                        direction.iloc[i] = 1
        
        return supertrend, direction
    
    def generate_signal(self, df):
        if len(df) < 50:
            return None
        
        current_price = float(df['close'].iloc[-1])
        
        # Calculate RMI
        rmi = self.calculate_rmi(df['close'])
        if rmi is None or pd.isna(rmi.iloc[-1]):
            return None
        
        current_rmi = rmi.iloc[-1]
        
        # Calculate SuperTrend
        supertrend, direction = self.calculate_supertrend(df)
        if pd.isna(supertrend.iloc[-1]) or pd.isna(direction.iloc[-1]):
            return None
        
        current_supertrend = supertrend.iloc[-1]
        current_direction = direction.iloc[-1]
        
        # Long signal: RMI > threshold + price above SuperTrend
        if current_rmi > self.config['rmi_threshold_long'] and current_direction == 1 and current_price > current_supertrend:
            return {
                'action': 'BUY',
                'price': current_price,
                'rmi': current_rmi,
                'supertrend': current_supertrend,
                'trend': 'UP'
            }
        
        # Short signal: RMI < threshold + price below SuperTrend
        elif current_rmi < self.config['rmi_threshold_short'] and current_direction == -1 and current_price < current_supertrend:
            return {
                'action': 'SELL',
                'price': current_price,
                'rmi': current_rmi,
                'supertrend': current_supertrend,
                'trend': 'DOWN'
            }
        
        return None
    
    async def get_market_data(self):
        try:
            klines = self.exchange.get_kline(
                category="linear",
                symbol=self.symbol,
                interval=self.config['timeframe'],
                limit=100
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
            pass
    
    def should_close(self):
        if not self.position:
            return False, ""
        
        current_price = float(self.price_data['close'].iloc[-1])
        entry_price = float(self.position.get('avgPrice', 0))
        side = self.position.get('side', '')
        
        if entry_price == 0:
            return False, ""
        
        # Check profit/loss targets
        if side == "Buy":
            profit_pct = (current_price - entry_price) / entry_price * 100
            if profit_pct >= self.config['net_take_profit']:
                return True, "take_profit"
            if profit_pct <= -self.config['net_stop_loss']:
                return True, "stop_loss"
        else:
            profit_pct = (entry_price - current_price) / entry_price * 100
            if profit_pct >= self.config['net_take_profit']:
                return True, "take_profit"
            if profit_pct <= -self.config['net_stop_loss']:
                return True, "stop_loss"
        
        # Check for SuperTrend exit
        supertrend, direction = self.calculate_supertrend(self.price_data)
        if not pd.isna(supertrend.iloc[-1]):
            if side == "Buy" and current_price < supertrend.iloc[-1]:
                return True, "supertrend_exit"
            elif side == "Sell" and current_price > supertrend.iloc[-1]:
                return True, "supertrend_exit"
        
        # Check for RMI reversal
        rmi = self.calculate_rmi(self.price_data['close'])
        if rmi is not None and not pd.isna(rmi.iloc[-1]):
            if side == "Buy" and rmi.iloc[-1] < self.config['rmi_threshold_short']:
                return True, "rmi_reversal"
            elif side == "Sell" and rmi.iloc[-1] > self.config['rmi_threshold_long']:
                return True, "rmi_reversal"
        
        return False, ""
    
    async def execute_trade(self, signal):
        # ✅ FIXED: Update account balance for proper position sizing
        await self.update_account_balance()
        
        # ✅ FIXED: Calculate stop loss based on SuperTrend level
        if signal['action'] == 'BUY':
            stop_loss_price = signal['supertrend'] * 0.995
        else:
            stop_loss_price = signal['supertrend'] * 1.005
        
        # ✅ FIXED: Calculate position size based on risk
        qty = self.calculate_position_size(signal['price'], stop_loss_price)
        formatted_qty = self.format_qty(qty)
        
        if int(formatted_qty) < (self.config['min_notional'] / signal['price']):
            print(f"⚠️ Position size too small: {formatted_qty}")
            return
        
        # ✅ FIXED: Calculate limit price with slippage
        limit_price = self.calculate_limit_price(signal['price'], signal['action'])
        
        try:
            order = self.exchange.place_order(
                category="linear",
                symbol=self.symbol,
                side="Buy" if signal['action'] == 'BUY' else "Sell",
                orderType="Limit",
                qty=formatted_qty,
                price=str(limit_price),
                timeInForce="PostOnly"
            )
            
            if order.get('retCode') == 0:
                net_tp = limit_price * (1 + self.config['net_take_profit']/100) if signal['action'] == 'BUY' else limit_price * (1 - self.config['net_take_profit']/100)
                net_sl = limit_price * (1 - self.config['net_stop_loss']/100) if signal['action'] == 'BUY' else limit_price * (1 + self.config['net_stop_loss']/100)
                
                self.current_trade_id, _ = self.logger.log_trade_open(
                    side=signal['action'],
                    expected_price=signal['price'],
                    actual_price=limit_price,
                    qty=float(formatted_qty),
                    stop_loss=stop_loss_price,
                    take_profit=net_tp,
                    info=f"rmi:{signal['rmi']:.1f}_st:{signal['supertrend']:.4f}_trend:{signal['trend']}_risk:{self.config['risk_pct']}%"
                )
                
                position_value = float(formatted_qty) * limit_price
                print(f"✅ FIXED RMI+ST {signal['action']}: {formatted_qty} @ ${limit_price:.4f}")
                print(f"   💰 Position Value: ${position_value:.2f} (Risk: {self.config['risk_pct']}%)")
                print(f"   📊 RMI: {signal['rmi']:.1f} | SuperTrend: ${signal['supertrend']:.4f}")
                print(f"   🎯 Trend: {signal['trend']}")
                print(f"   🛡️ Account Balance: ${self.account_balance:.2f}")
                
        except Exception as e:
            print(f"❌ Trade failed: {e}")
    
    async def close_position(self, reason):
        if not self.position:
            return
        
        current_price = float(self.price_data['close'].iloc[-1])
        side = "Sell" if self.position.get('side') == "Buy" else "Buy"
        qty = float(self.position['size'])
        
        # ✅ FIXED: Use slippage modeling for close price
        limit_price = self.calculate_limit_price(current_price, side)
        
        try:
            order = self.exchange.place_order(
                category="linear",
                symbol=self.symbol,
                side=side,
                orderType="Limit",
                qty=self.format_qty(qty),
                price=str(limit_price),
                timeInForce="PostOnly",
                reduceOnly=True
            )
            
            if order.get('retCode') == 0:
                if self.current_trade_id:
                    self.logger.log_trade_close(
                        trade_id=self.current_trade_id,
                        expected_exit=current_price,
                        actual_exit=limit_price,
                        reason=reason,
                        fees_entry=self.config['maker_fee_pct'],
                        fees_exit=self.config['maker_fee_pct']
                    )
                    self.current_trade_id = None
                
                print(f"✅ Closed: {reason}")
                self.position = None
                
        except Exception as e:
            print(f"❌ Close failed: {e}")
    
    def show_status(self):
        if len(self.price_data) == 0:
            return
        
        current_price = float(self.price_data['close'].iloc[-1])
        
        print(f"\n📈 FIXED RMI + SuperTrend Bot - {self.symbol}")
        print(f"💰 Price: ${current_price:.4f} | Balance: ${self.account_balance:.2f}")
        print(f"🔧 FIXED: Risk-based position sizing ({self.config['risk_pct']}% per trade)")
        
        rmi = self.calculate_rmi(self.price_data['close'])
        supertrend, direction = self.calculate_supertrend(self.price_data)
        
        if rmi is not None and not pd.isna(rmi.iloc[-1]):
            print(f"📊 RMI: {rmi.iloc[-1]:.1f}")
        
        if not pd.isna(supertrend.iloc[-1]):
            trend = "UP" if direction.iloc[-1] == 1 else "DOWN"
            print(f"📈 SuperTrend: ${supertrend.iloc[-1]:.4f} | Trend: {trend}")
        
        if self.position:
            entry_price = float(self.position.get('avgPrice', 0))
            side = self.position.get('side', '')
            size = self.position.get('size', '0')
            
            pnl = float(self.position.get('unrealisedPnl', 0))
            
            emoji = "🟢" if side == "Buy" else "🔴"
            print(f"{emoji} {side}: {size} ADA @ ${entry_price:.4f} | PnL: ${pnl:.2f}")
        else:
            print("🔍 Scanning for trend-sync signals...")
        
        print("-" * 60)
    
    async def run_cycle(self):
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
        
        print(f"📈 FULLY FIXED RMI + SuperTrend Bot")
        print(f"✅ FIXES APPLIED:")
        print(f"   • Position Sizing: Account balance-based ({self.config['risk_pct']}% risk)")
        print(f"   • Fee Calculations: Proper maker rebates")
        print(f"   • Slippage Modeling: {self.config['slippage_pct']}% expected")
        print(f"   • Risk Management: Stop loss at SuperTrend levels")
        print(f"   • Quantity Precision: {self.config['qty_precision']} decimals (ADA whole numbers)")
        
        try:
            while True:
                await self.run_cycle()
                await asyncio.sleep(5)
        except KeyboardInterrupt:
            print("\n🛑 Bot stopped")
            if self.position:
                await self.close_position("manual_stop")
        except Exception as e:
            print(f"⚠️ Runtime error: {e}")
            await asyncio.sleep(5)

if __name__ == "__main__":
    bot = RMISuperTrendBot()
    asyncio.run(bot.run())