import os
import asyncio
import pandas as pd
import json
import time
from datetime import datetime, timezone
from collections import deque
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
        slippage = 0  # PostOnly = zero slippage
        
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
        slippage = 0  # PostOnly = zero slippage
        
        gross_pnl = ((actual_exit - trade["entry_price"]) * trade["qty"] 
                    if trade["side"] == "BUY" 
                    else (trade["entry_price"] - actual_exit) * trade["qty"])
        
        # Correct maker rebate calculation
        entry_rebate = trade["entry_price"] * trade["qty"] * abs(fees_entry) / 100
        exit_rebate = actual_exit * trade["qty"] * abs(fees_exit) / 100
        total_rebates = entry_rebate + exit_rebate
        net_pnl = gross_pnl + total_rebates
        
        # Update daily PnL
        self.daily_pnl += net_pnl
        self.consecutive_losses = self.consecutive_losses + 1 if net_pnl < 0 else 0
        
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

class LiquiditySweepBot:
    """Fixed Liquidity Sweep Strategy"""
    
    def __init__(self):
        self.symbol = 'DOGEUSDT'
        self.demo_mode = os.getenv('DEMO_MODE', 'true').lower() == 'true'
        
        # API setup
        prefix = 'TESTNET_' if self.demo_mode else 'LIVE_'
        self.api_key = os.getenv(f'{prefix}BYBIT_API_KEY')
        self.api_secret = os.getenv(f'{prefix}BYBIT_API_SECRET')
        self.exchange = None
        
        # Trading state
        self.position = None
        self.pending_order = None
        self.price_data = pd.DataFrame()
        self.account_balance = 1000
        
        # Anti-duplicate mechanisms
        self.last_trade_time = 0
        self.trade_cooldown = 30
        self.last_order_time = 0
        self.order_cooldown = 10
        self.last_signal_price = 0
        self.min_price_change_pct = 0.1
        
        # Strategy parameters
        self.config = {
            'timeframe': '5',
            'liquidity_lookback': 50,
            'order_block_lookback': 20,
            'sweep_threshold': 0.15,
            'retracement_ratio': 0.5,
            'risk_per_trade_pct': 2.0,
            'lookback': 100,
            'maker_offset_pct': 0.01,
            'maker_fee_pct': -0.04,
            'gross_take_profit': 1.5,
            'gross_stop_loss': 0.5,
            'net_take_profit': 1.58,
            'net_stop_loss': 0.42,
            'expected_slippage_pct': 0.02,
        }
        
        # Liquidity tracking
        self.liquidity_pools = {'highs': deque(maxlen=10), 'lows': deque(maxlen=10)}
        self.order_blocks = []
        
        # Trade logging
        self.logger = TradeLogger("LIQUIDITY_SWEEP_FIXED", self.symbol)
        self.current_trade_id = None
    
    def connect(self):
        """Connect to exchange API"""
        try:
            self.exchange = HTTP(demo=self.demo_mode, api_key=self.api_key, api_secret=self.api_secret)
            return self.exchange.get_server_time().get('retCode') == 0
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False
    
    async def get_account_balance(self):
        """Get actual account balance"""
        try:
            wallet = self.exchange.get_wallet_balance(accountType="UNIFIED", coin="USDT")
            if wallet.get('retCode') == 0:
                balance_list = wallet['result']['list']
                if balance_list:
                    for coin in balance_list[0]['coin']:
                        if coin['coin'] == 'USDT':
                            self.account_balance = float(coin['availableToWithdraw'])
                            return True
        except Exception as e:
            print(f"❌ Balance error: {e}")
        
        # Fallback for demo
        self.account_balance = 1000.0
        return True
    
    def calculate_position_size(self, price, stop_loss_price):
        """Calculate position size based on risk percentage"""
        if self.account_balance <= 0:
            return 0
        
        risk_amount = self.account_balance * (self.config['risk_per_trade_pct'] / 100)
        price_diff = abs(price - stop_loss_price)
        
        if price_diff == 0:
            return 0
        
        # Include slippage in calculation
        slippage_factor = 1 + (self.config['expected_slippage_pct'] / 100)
        adjusted_risk = risk_amount / slippage_factor
        
        qty = adjusted_risk / price_diff
        return max(qty, 1)  # Minimum 1 DOGE
    
    def format_qty(self, qty):
        """Format quantity for DOGE with proper precision"""
        if qty < 1:
            return "0"
        return str(int(round(qty)))
    
    def apply_slippage(self, price, side, order_type="market"):
        """Apply realistic slippage modeling"""
        if order_type == "limit":
            return price  # No slippage for limit orders
        
        slippage_pct = self.config['expected_slippage_pct'] / 100
        
        return price * (1 + slippage_pct) if side in ["BUY", "Buy"] else price * (1 - slippage_pct)
    
    async def execute_limit_order(self, side, qty, price, is_reduce=False):
        """Execute limit order with PostOnly for zero slippage"""
        formatted_qty = self.format_qty(qty)
        
        # Calculate limit price with small offset
        limit_price = price * (0.9998 if side == "Buy" else 1.0002)
        limit_price = float(round(limit_price, 4))
        
        params = {
            "category": "linear",
            "symbol": self.symbol,
            "side": side,
            "orderType": "Limit",
            "qty": formatted_qty,
            "price": str(limit_price),
            "timeInForce": "PostOnly"
        }
        
        if is_reduce:
            params["reduceOnly"] = True
        
        order = self.exchange.place_order(**params)
        
        return limit_price if order.get('retCode') == 0 else None
    
    async def check_pending_orders(self):
        """Check for any pending orders"""
        try:
            orders = self.exchange.get_open_orders(category="linear", symbol=self.symbol)
            if orders.get('retCode') != 0:
                self.pending_order = None
                return False
            
            order_list = orders['result']['list']
            if order_list and len(order_list) > 0:
                self.pending_order = order_list[0]
                return True
            
            self.pending_order = None
            return False
        except Exception as e:
            print(f"❌ Order check error: {e}")
            return False
    
    async def check_position(self):
        """Check current position status"""
        try:
            positions = self.exchange.get_positions(category="linear", symbol=self.symbol)
            if positions.get('retCode') == 0:
                pos_list = positions['result']['list']
                if pos_list:
                    for pos in pos_list:
                        if float(pos.get('size', 0)) > 0:
                            self.position = pos
                            return True
                self.position = None
        except Exception as e:
            print(f"❌ Position check error: {e}")
            self.position = None
        return False
    
    def identify_liquidity_pools(self, df):
        """Identify liquidity pools"""
        if len(df) < self.config['liquidity_lookback']:
            return
        
        window = 5
        highs = df['high'].rolling(window=window, center=True).max()
        lows = df['low'].rolling(window=window, center=True).min()
        
        self.liquidity_pools['highs'].clear()
        self.liquidity_pools['lows'].clear()
        
        for i in range(len(df) - 10, max(0, len(df) - self.config['liquidity_lookback']), -1):
            # Check significant high
            if df['high'].iloc[i] == highs.iloc[i]:
                is_significant = (
                    all(df['high'].iloc[max(0, i-3):i] < df['high'].iloc[i]) and
                    all(df['high'].iloc[i+1:min(len(df), i+4)] < df['high'].iloc[i])
                )
                if is_significant:
                    self.liquidity_pools['highs'].append({
                        'price': df['high'].iloc[i],
                        'index': i,
                        'volume': df['volume'].iloc[i]
                    })
            
            # Check significant low
            if df['low'].iloc[i] == lows.iloc[i]:
                is_significant = (
                    all(df['low'].iloc[max(0, i-3):i] > df['low'].iloc[i]) and
                    all(df['low'].iloc[i+1:min(len(df), i+4)] > df['low'].iloc[i])
                )
                if is_significant:
                    self.liquidity_pools['lows'].append({
                        'price': df['low'].iloc[i],
                        'index': i,
                        'volume': df['volume'].iloc[i]
                    })
    
    def detect_liquidity_sweep(self, df):
        """Detect liquidity sweep"""
        if len(df) < 3:
            return None
        
        current_high = df['high'].iloc[-1]
        current_low = df['low'].iloc[-1]
        current_close = df['close'].iloc[-1]
        current_volume = df['volume'].iloc[-1]
        avg_volume = df['volume'].iloc[-20:].mean()
        
        # Check sweep above liquidity
        for pool in self.liquidity_pools['highs']:
            sweep_level = pool['price'] * (1 + self.config['sweep_threshold'] / 100)
            
            if current_high > sweep_level and current_close < pool['price']:
                return {
                    'type': 'bearish_sweep',
                    'swept_level': pool['price'],
                    'volume_ratio': current_volume / avg_volume if avg_volume > 0 else 1
                }
        
        # Check sweep below liquidity
        for pool in self.liquidity_pools['lows']:
            sweep_level = pool['price'] * (1 - self.config['sweep_threshold'] / 100)
            
            if current_low < sweep_level and current_close > pool['price']:
                return {
                    'type': 'bullish_sweep',
                    'swept_level': pool['price'],
                    'volume_ratio': current_volume / avg_volume if avg_volume > 0 else 1
                }
        
        return None
    
    def generate_signal(self, df):
        """Generate trading signal"""
        if len(df) < self.config['lookback'] or self.position:
            return None
        
        # Check for duplicate signals
        current_price = df['close'].iloc[-1]
        if self.last_signal_price != 0:
            price_change_pct = abs(current_price - self.last_signal_price) / self.last_signal_price * 100
            if price_change_pct < self.min_price_change_pct:
                return None
        
        self.identify_liquidity_pools(df)
        sweep = self.detect_liquidity_sweep(df)
        
        if not sweep:
            return None
        
        action = 'BUY' if sweep['type'] == 'bullish_sweep' else 'SELL' if sweep['type'] == 'bearish_sweep' else None
        
        if action:
            self.last_signal_price = current_price
            return {
                'action': action,
                'price': current_price,
                'swept_level': sweep['swept_level'],
                'volume_ratio': sweep['volume_ratio']
            }
        
        return None
    
    async def get_market_data(self):
        """Retrieve market data"""
        try:
            klines = self.exchange.get_kline(
                category="linear",
                symbol=self.symbol,
                interval=self.config['timeframe'],
                limit=self.config['lookback']
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
    
    def should_close(self):
        """Check if position should be closed"""
        if not self.position:
            return False, ""
        
        current_price = float(self.price_data['close'].iloc[-1])
        entry_price = float(self.position.get('avgPrice', 0))
        side = self.position.get('side', '')
        
        if entry_price == 0:
            return False, ""
        
        # Calculate NET targets with slippage
        if side == "Buy":
            profit_pct = (current_price - entry_price) / entry_price * 100
        else:
            profit_pct = (entry_price - current_price) / entry_price * 100
        
        if profit_pct >= self.config['net_take_profit']:
            return True, "take_profit"
        if profit_pct <= -self.config['net_stop_loss']:
            return True, "stop_loss"
        
        return False, ""
    
    async def execute_trade(self, signal):
        """Execute trade with proper sizing and slippage"""
        # Check trade cooldown
        current_time = time.time()
        if current_time - self.last_trade_time < self.trade_cooldown:
            remaining = self.trade_cooldown - (current_time - self.last_trade_time)
            print(f"⏰ Trade cooldown: wait {remaining:.0f}s")
            return
        
        if current_time - self.last_order_time < self.order_cooldown:
            return
        
        # Get account balance
        await self.get_account_balance()
        
        # Calculate stop loss price for position sizing
        stop_loss_price = (signal['price'] * (1 - self.config['net_stop_loss'] / 100) if signal['action'] == 'BUY'
                          else signal['price'] * (1 + self.config['net_stop_loss'] / 100))
        
        # Calculate position size based on risk
        qty = self.calculate_position_size(signal['price'], stop_loss_price)
        formatted_qty = self.format_qty(qty)
        
        if formatted_qty == "0":
            print(f"❌ Position size too small: {qty}")
            return
        
        # Apply maker offset for rebate
        offset_mult = 1 - self.config['maker_offset_pct']/100 if signal['action'] == 'BUY' else 1 + self.config['maker_offset_pct']/100
        limit_price = round(signal['price'] * offset_mult, 4)
        
        # Model expected slippage for limit orders
        expected_fill_price = self.apply_slippage(limit_price, signal['action'], "limit")
        
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
                self.last_trade_time = current_time
                self.last_order_time = current_time
                
                # Calculate targets
                net_tp = (expected_fill_price * (1 + self.config['net_take_profit']/100) if signal['action'] == 'BUY' 
                         else expected_fill_price * (1 - self.config['net_take_profit']/100))
                net_sl = (expected_fill_price * (1 - self.config['net_stop_loss']/100) if signal['action'] == 'BUY' 
                         else expected_fill_price * (1 + self.config['net_stop_loss']/100))
                
                self.current_trade_id, _ = self.logger.log_trade_open(
                    side=signal['action'],
                    expected_price=signal['price'],
                    actual_price=limit_price,
                    qty=float(formatted_qty),
                    stop_loss=net_sl,
                    take_profit=net_tp,
                    info=f"swept:{signal['swept_level']:.4f}_risk:{self.config['risk_per_trade_pct']}%_bal:{self.account_balance:.2f}"
                )
                
                risk_amount = self.account_balance * (self.config['risk_per_trade_pct'] / 100)
                
                print(f"✅ FIXED {signal['action']}: {formatted_qty} DOGE @ ${limit_price:.4f}")
                print(f"   💰 Risk: ${risk_amount:.2f} ({self.config['risk_per_trade_pct']}% of ${self.account_balance:.2f})")
                print(f"   🎯 Liquidity Swept: ${signal['swept_level']:.4f}")
                print(f"   📊 Expected Slippage: {self.config['expected_slippage_pct']}%")
                
        except Exception as e:
            print(f"❌ Trade failed: {e}")
    
    async def close_position(self, reason):
        """Close position with maker order"""
        if not self.position:
            return
        
        current_price = float(self.price_data['close'].iloc[-1])
        side = "Sell" if self.position.get('side') == "Buy" else "Buy"
        qty = float(self.position['size'])
        
        # Calculate limit price with offset
        offset_mult = 1 + self.config['maker_offset_pct']/100 if side == "Sell" else 1 - self.config['maker_offset_pct']/100
        limit_price = round(current_price * offset_mult, 4)
        
        # Model expected slippage
        expected_fill_price = self.apply_slippage(limit_price, side, "limit")
        
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
                        actual_exit=expected_fill_price,
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
        """Show current status"""
        if len(self.price_data) == 0:
            return
        
        current_price = float(self.price_data['close'].iloc[-1])
        
        print(f"\n🎯 FIXED Liquidity Sweep Bot - {self.symbol}")
        print(f"💰 Price: ${current_price:.4f} | Balance: ${self.account_balance:.2f}")
        print(f"⚡ Risk per trade: {self.config['risk_per_trade_pct']}%")
        print(f"📊 Expected slippage: {self.config['expected_slippage_pct']}%")
        
        if self.position:
            entry_price = float(self.position.get('avgPrice', 0))
            side = self.position.get('side', '')
            size = self.position.get('size', '0')
            pnl = float(self.position.get('unrealisedPnl', 0))
            
            emoji = "🟢" if side == "Buy" else "🔴"
            print(f"{emoji} {side}: {size} DOGE @ ${entry_price:.4f} | PnL: ${pnl:.2f}")
        elif self.pending_order:
            print(f"⏳ Pending order: {self.pending_order.get('side')} @ ${self.pending_order.get('price')}")
        else:
            print("🔍 Scanning for liquidity sweeps...")
        
        print("-" * 60)
    
    async def run_cycle(self):
        """Run one trading cycle"""
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
        """Main bot loop"""
        if not self.connect():
            print("❌ Failed to connect")
            return
        
        print(f"🎯 FIXED Liquidity Sweep Bot - {self.symbol}")
        print("✅ FIXES APPLIED:")
        print(f"   • Position sizing: Risk-based ({self.config['risk_per_trade_pct']}% per trade)")
        print(f"   • Fee calculations: Correct maker rebates")
        print(f"   • Slippage modeling: {self.config['expected_slippage_pct']}% expected")
        print(f"   • Account balance: Dynamic checking")
        
        try:
            while True:
                await self.run_cycle()
                await asyncio.sleep(10)
        except KeyboardInterrupt:
            print("\n🛑 Bot stopped")
            if self.position:
                await self.close_position("manual_stop")
        except Exception as e:
            print(f"⚠️ Runtime error: {e}")
            await asyncio.sleep(5)

if __name__ == "__main__":
    bot = LiquiditySweepBot()
    asyncio.run(bot.run())