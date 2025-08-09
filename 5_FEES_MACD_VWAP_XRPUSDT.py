import os
import asyncio
import pandas as pd
import numpy as np
import json
import time
import math
from datetime import datetime, timezone
from pybit.unified_trading import HTTP
from dotenv import load_dotenv

load_dotenv()

class UnifiedLogger:
    def __init__(self, bot_name, symbol):
        self.bot_name = bot_name
        self.symbol = symbol
        self.currency = "USDT"
        self.open_trades = {}
        self.trade_id = 1000
        
        os.makedirs("logs", exist_ok=True)
        self.log_file = f"logs/5_FEES_MACD_VWAP_XRPUSDT.log"
        
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
            "expected_price": round(expected_price, 6),
            "actual_price": round(actual_price, 6),
            "slippage": round(slippage, 6),
            "qty": round(qty, 6),
            "stop_loss": round(stop_loss, 6),
            "take_profit": round(take_profit, 6),
            "currency": self.currency,
            "info": info
        }
        
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        self.open_trades[trade_id] = log_entry
        return trade_id, log_entry

class EmaRsiTrader:
    def __init__(self):
        self.demo_mode = True  # Set to False for live trading
        self.symbol = "XRPUSDT"
        self.exchange = None
        self.position = None
        self.account_balance = 1000.0
        self.position_data = {}
        self.price_data = pd.DataFrame()
        
        prefix = "DEMO_" if self.demo_mode else "LIVE_"
        self.api_key = os.getenv(f'{prefix}BYBIT_API_KEY')
        self.api_secret = os.getenv(f'{prefix}BYBIT_API_SECRET')
        
        self.config = {
            'ema_fast': 10,
            'ema_slow': 21,
            'rsi_period': 14,
            'rsi_long_threshold': 30,
            'rsi_short_threshold': 70,
            'stop_loss_pct': 0.5,
            'take_profit_pct': 1.5,
            'trailing_stop_pct': 0.7,
            'risk_per_trade_pct': 2.0,
            'expected_slippage_pct': 0.02,
            'maker_offset_pct': 0.02,
            'limit_order_timeout': 30,
            'limit_order_retries': 3
        }
        
        self.pending_order = False
        self.last_trade_time = 0
        self.trade_cooldown = 30
        
        os.makedirs("logs", exist_ok=True)
        self.log_file = f"logs/5_FEES_MACD_VWAP_XRPUSDT.log"
        self.unified_logger = UnifiedLogger("STRATEGY5_EMA_RSI_FIXED", self.symbol)
        self.current_trade_id = None
        self.entry_price = None
        self.trailing_stop = None
        self.last_ema_state = None
    
    def connect(self):
        try:
            self.exchange = HTTP(demo=self.demo_mode, api_key=self.api_key, api_secret=self.api_secret)
            return self.exchange.get_server_time().get('retCode') == 0
        except:
            return False
    
    async def get_account_balance(self):
        try:
            wallet = self.exchange.get_wallet_balance(accountType="UNIFIED", coin="USDT")
            if wallet.get('retCode') == 0:
                balance_list = wallet['result']['list']
                if balance_list:
                    for coin in balance_list[0]['coin']:
                        if coin['coin'] == 'USDT':
                            balance_str = coin['availableToWithdraw']
                            if balance_str and balance_str.strip():
                                self.account_balance = float(balance_str)
                                return True
        except:
            pass
        
        self.account_balance = 1000.0
        return True
    
    def calculate_position_size(self, price, stop_loss_price):
        if self.account_balance <= 0:
            return 0
        
        risk_amount = self.account_balance * (self.config['risk_per_trade_pct'] / 100)
        price_diff = abs(price - stop_loss_price)
        
        if price_diff == 0:
            return 0
        
        slippage_factor = 1 + (self.config['expected_slippage_pct'] / 100)
        adjusted_risk = risk_amount / slippage_factor
        
        raw_qty = adjusted_risk / price_diff
        max_position_value = self.account_balance * 0.1
        max_qty = max_position_value / price
        
        final_qty = min(raw_qty, max_qty)
        return max(final_qty, 0.1)
    
    def format_qty(self, qty):
        """FIXED: Return a properly formatted quantity rounded up to meet exchange minimums."""
        min_qty = 0.1  # XRP minimum
        qty_step = 0.1
        min_notional = 5.0
        
        # Get current price
        try:
            current_price = float(self.price_data['close'].iloc[-1]) if len(self.price_data) > 0 else 1.0
        except:
            current_price = 1.0
        
        # Round to step
        qty = math.floor(qty / qty_step) * qty_step
        
        # Ensure minimum
        if qty < min_qty:
            qty = min_qty
            
        # Check notional
        if qty * current_price < min_notional:
            qty = math.ceil(min_notional / current_price / qty_step) * qty_step
            
        return str(qty)
    
    def apply_slippage(self, price, side, order_type="market"):
        if order_type == "limit":
            return price
        
        slippage_pct = self.config['expected_slippage_pct'] / 100
        return price * (1 + slippage_pct) if side in ["BUY", "Buy"] else price * (1 - slippage_pct)
    
    def calculate_indicators(self, df):
        if len(df) < max(self.config['ema_slow'], self.config['rsi_period']):
            return None
        
        # Calculate EMAs
        df['ema_fast'] = df['close'].ewm(span=self.config['ema_fast'], adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=self.config['ema_slow'], adjust=False).mean()
        
        # Calculate RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.config['rsi_period']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.config['rsi_period']).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        current_ema_fast = df['ema_fast'].iloc[-1]
        current_ema_slow = df['ema_slow'].iloc[-1]
        prev_ema_fast = df['ema_fast'].iloc[-2]
        prev_ema_slow = df['ema_slow'].iloc[-2]
        
        crossover_up = prev_ema_fast <= prev_ema_slow and current_ema_fast > current_ema_slow
        crossover_down = prev_ema_fast >= prev_ema_slow and current_ema_fast < current_ema_slow
        
        return {
            'ema_fast': current_ema_fast,
            'ema_slow': current_ema_slow,
            'crossover_up': crossover_up,
            'crossover_down': crossover_down,
            'rsi': rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
        }
    
    def generate_signal(self, df):
        if self.position:
            return None
            
        indicators = self.calculate_indicators(df)
        if not indicators:
            return None
        
        price = float(df['close'].iloc[-1])
        
        # Strong signal requirements
        # BUY: EMA crossover up + RSI < 30 (oversold)
        if indicators['crossover_up'] and indicators['rsi'] < self.config['rsi_long_threshold']:
            return {'action': 'BUY', 'price': price, 'rsi': indicators['rsi']}
        
        # SELL: EMA crossover down + RSI > 70 (overbought)
        elif indicators['crossover_down'] and indicators['rsi'] > self.config['rsi_short_threshold']:
            return {'action': 'SELL', 'price': price, 'rsi': indicators['rsi']}
        
        return None
    
    async def execute_limit_order(self, side, qty, base_price, is_reduce=False):
        """Execute limit order with PostOnly ladder strategy for zero slippage"""
        formatted_qty = self.format_qty(qty)
        
        if formatted_qty == "0":
            print(f"❌ Quantity too small: {qty}")
            return None
        
        offsets = [0.01, 0.02, 0.05, 0.10]  # Progressive offsets
        
        for retry, offset_pct in enumerate(offsets[:self.config['limit_order_retries']]):
            # Calculate limit price with offset
            if side == "Buy":
                limit_price = base_price * (1 - offset_pct / 100)
            else:
                limit_price = base_price * (1 + offset_pct / 100)
            
            limit_price = round(limit_price, 4)
            
            try:
                # Place PostOnly order
                order = self.exchange.place_order(
                    category="linear",
                    symbol=self.symbol,
                    side=side,
                    orderType="Limit",
                    qty=formatted_qty,
                    price=str(limit_price),
                    timeInForce="PostOnly",
                    reduceOnly=is_reduce
                )
                
                if order.get('retCode') != 0:
                    continue
                
                order_id = order['result']['orderId']
                
                # Wait for fill or timeout
                start_time = time.time()
                while time.time() - start_time < self.config['limit_order_timeout']:
                    check = self.exchange.get_open_orders(
                        category="linear",
                        symbol=self.symbol,
                        orderId=order_id
                    )
                    
                    if check.get('retCode') == 0:
                        if not check['result']['list']:
                            # Order filled
                            return limit_price
                    
                    await asyncio.sleep(1)
                
                # Cancel unfilled order
                self.exchange.cancel_order(
                    category="linear",
                    symbol=self.symbol,
                    orderId=order_id
                )
                
            except Exception as e:
                print(f"❌ Order attempt {retry+1} failed: {e}")
        
        # Fallback to IOC if all PostOnly attempts fail
        try:
            order = self.exchange.place_order(
                category="linear",
                symbol=self.symbol,
                side=side,
                orderType="Limit",
                qty=formatted_qty,
                price=str(base_price),
                timeInForce="IOC",
                reduceOnly=is_reduce
            )
            
            if order.get('retCode') == 0:
                return base_price
        except:
            pass
        
        return None
    
    async def get_market_data(self):
        try:
            klines = self.exchange.get_kline(category="linear", symbol=self.symbol, interval="1", limit=50)
            if klines.get('retCode') != 0:
                return False
            
            df = pd.DataFrame(klines['result']['list'], 
                            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
            df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
            
            self.price_data = df.sort_values('timestamp').reset_index(drop=True)
            return True
        except:
            return False
    
    async def check_position(self):
        try:
            positions = self.exchange.get_positions(category="linear", symbol=self.symbol)
            if positions.get('retCode') == 0:
                pos_list = positions['result']['list']
                if pos_list and float(pos_list[0]['size']) > 0:
                    self.position = pos_list[0]
                    if not self.entry_price:
                        self.entry_price = float(self.position.get('avgPrice', 0))
                else:
                    self.position = None
                    self.entry_price = None
                    self.trailing_stop = None
        except:
            pass
    
    def should_close(self):
        if not self.position or not self.entry_price:
            return False, ""
        
        current_price = float(self.price_data['close'].iloc[-1])
        is_long = self.position.get('side') == "Buy"
        
        if is_long:
            profit_pct = (current_price - self.entry_price) / self.entry_price * 100
            
            # Update trailing stop
            if profit_pct > self.config['trailing_stop_pct']:
                new_stop = current_price * (1 - self.config['trailing_stop_pct']/100)
                if not self.trailing_stop or new_stop > self.trailing_stop:
                    self.trailing_stop = new_stop
            
            # Check exit conditions
            if self.trailing_stop and current_price <= self.trailing_stop:
                return True, "trailing_stop"
            if profit_pct >= self.config['take_profit_pct']:
                return True, "take_profit"
            if profit_pct <= -self.config['stop_loss_pct']:
                return True, "stop_loss"
        else:
            profit_pct = (self.entry_price - current_price) / self.entry_price * 100
            
            # Update trailing stop
            if profit_pct > self.config['trailing_stop_pct']:
                new_stop = current_price * (1 + self.config['trailing_stop_pct']/100)
                if not self.trailing_stop or new_stop < self.trailing_stop:
                    self.trailing_stop = new_stop
            
            # Check exit conditions
            if self.trailing_stop and current_price >= self.trailing_stop:
                return True, "trailing_stop"
            if profit_pct >= self.config['take_profit_pct']:
                return True, "take_profit"
            if profit_pct <= -self.config['stop_loss_pct']:
                return True, "stop_loss"
        
        return False, ""
    
    async def execute_trade(self, signal):
        if self.pending_order:
            return
        
        if time.time() - self.last_trade_time < self.trade_cooldown:
            return
        
        self.pending_order = True
        
        try:
            price = signal['price']
            side = "Buy" if signal['action'] == 'BUY' else "Sell"
            
            # Calculate stop loss
            if side == "Buy":
                stop_loss_price = price * (1 - self.config['stop_loss_pct']/100)
                take_profit = price * (1 + self.config['take_profit_pct']/100)
            else:
                stop_loss_price = price * (1 + self.config['stop_loss_pct']/100)
                take_profit = price * (1 - self.config['take_profit_pct']/100)
            
            qty = self.calculate_position_size(price, stop_loss_price)
            
            if qty == 0:
                self.pending_order = False
                return
            
            # Execute with PostOnly ladder
            actual_price = await self.execute_limit_order(side, qty, price)
            
            if actual_price:
                self.entry_price = actual_price
                self.last_trade_time = time.time()
                
                self.current_trade_id, _ = self.unified_logger.log_trade_open(
                    side=signal['action'],
                    expected_price=price,
                    actual_price=actual_price,
                    qty=qty,
                    stop_loss=stop_loss_price,
                    take_profit=take_profit,
                    info=f"rsi:{signal['rsi']:.1f}"
                )
                
                print(f"✅ {signal['action']} {qty:.1f} @ ${actual_price:.4f} | ZERO SLIPPAGE")
                print(f"   RSI: {signal['rsi']:.1f}")
        
        except Exception as e:
            print(f"❌ Trade failed: {e}")
        
        finally:
            self.pending_order = False
    
    async def close_position(self, reason):
        if not self.position:
            return
        
        try:
            current_price = float(self.price_data['close'].iloc[-1])
            side = "Sell" if self.position.get('side') == "Buy" else "Buy"
            qty = float(self.position['size'])
            
            actual_price = await self.execute_limit_order(side, qty, current_price, is_reduce=True)
            
            if actual_price:
                print(f"✅ Closed: {reason} @ ${actual_price:.4f} | ZERO SLIPPAGE")
                self.position = None
                self.entry_price = None
                self.trailing_stop = None
        except Exception as e:
            print(f"❌ Close failed: {e}")
    
    def show_status(self):
        if len(self.price_data) == 0:
            return
        
        current_price = float(self.price_data['close'].iloc[-1])
        indicators = self.calculate_indicators(self.price_data)
        
        if indicators:
            print(f"\n📊 EMA-RSI - {self.symbol}")
            print(f"💰 Price: ${current_price:.4f} | Balance: ${self.account_balance:.2f}")
            print(f"📈 EMA: {indicators['ema_fast']:.4f}/{indicators['ema_slow']:.4f} | RSI: {indicators['rsi']:.1f}")
        
        if self.position:
            entry = self.entry_price or 0
            is_long = self.position.get('side') == "Buy"
            size = float(self.position.get('size', 0))
            
            if is_long:
                pnl_pct = ((current_price - entry) / entry * 100) if entry else 0
            else:
                pnl_pct = ((entry - current_price) / entry * 100) if entry else 0
            
            print(f"📈 Position: {'LONG' if is_long else 'SHORT'} {size:.1f} @ ${entry:.4f}")
            print(f"   P&L: {pnl_pct:+.2f}%")
            
            if self.trailing_stop:
                print(f"   Trailing Stop: ${self.trailing_stop:.4f}")
    
    async def run(self):
        print(f"🚀 Starting EMA-RSI Trader - {self.symbol}")
        print(f"✅ ZERO SLIPPAGE MODE: PostOnly Ladder Strategy")
        
        if not self.connect():
            print("❌ Failed to connect to exchange")
            return
        
        print("✅ Connected to Bybit")
        
        iteration = 0
        while True:
            try:
                await self.get_account_balance()
                await self.get_market_data()
                await self.check_position()
                
                # Check exit conditions
                should_exit, reason = self.should_close()
                if should_exit:
                    await self.close_position(reason)
                
                # Check for new signals
                if not self.position and len(self.price_data) > 0:
                    signal = self.generate_signal(self.price_data)
                    if signal:
                        await self.execute_trade(signal)
                
                # Show status every 5 iterations
                if iteration % 5 == 0:
                    self.show_status()
                
                iteration += 1
                await asyncio.sleep(60)
                
            except Exception as e:
                print(f"❌ Error: {e}")
                await asyncio.sleep(30)

if __name__ == "__main__":
    trader = EmaRsiTrader()
    asyncio.run(trader.run())