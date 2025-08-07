import os
import asyncio
import pandas as pd
import json
from datetime import datetime
from pybit.unified_trading import HTTP
from dotenv import load_dotenv

load_dotenv()

class UnifiedLogger:
    def __init__(self, bot_name, symbol):
        self.bot_name = bot_name
        self.symbol = symbol
        self.currency = "USDT"
        self.open_trades = {}
        self.trade_counter = 1000
        
    def generate_trade_id(self):
        self.trade_counter += 1
        return self.trade_counter
    
    def log_trade_open(self, side, expected_price, actual_price, qty, stop_loss, take_profit, info=""):
        trade_id = self.generate_trade_id()
        slippage = actual_price - expected_price if side == "BUY" else expected_price - actual_price
        
        log_entry = {
            "id": trade_id,
            "bot": self.bot_name,
            "symbol": self.symbol,
            "side": "LONG" if side == "BUY" else "SHORT",
            "action": "OPEN",
            "ts": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
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
        
        return trade_id, log_entry

    def log_trade_close(self, trade_id, expected_exit, actual_exit, reason, fees_entry=0.1, fees_exit=0.25):
        if trade_id not in self.open_trades:
            return None
            
        trade = self.open_trades[trade_id]
        duration = (datetime.now() - trade["entry_time"]).total_seconds()
        
        slippage = actual_exit - expected_exit if trade["side"] == "SELL" else expected_exit - actual_exit
        
        if trade["side"] == "BUY":
            gross_pnl = (actual_exit - trade["entry_price"]) * trade["qty"]
        else:
            gross_pnl = (trade["entry_price"] - actual_exit) * trade["qty"]
        
        fee_rate = 0.001
        fees_entry = trade["entry_price"] * trade["qty"] * fee_rate
        fees_exit = actual_exit * trade["qty"] * fee_rate
        total_fees = fees_entry + fees_exit
        net_pnl = gross_pnl - total_fees
        
        log_entry = {
            "id": trade_id,
            "bot": self.bot_name,
            "symbol": self.symbol,
            "side": "LONG" if trade["side"] == "BUY" else "SHORT",
            "action": "CLOSE",
            "ts": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "duration_sec": int(duration),
            "entry_price": round(trade["entry_price"], 4),
            "expected_exit": round(expected_exit, 4),
            "actual_exit": round(actual_exit, 4),
            "slippage": round(slippage, 4),
            "qty": round(trade["qty"], 6),
            "gross_pnl": round(gross_pnl, 2),
            "fees": {"entry": round(fees_entry, 4), "exit": round(fees_exit, 4), "total": round(total_fees, 4)},
            "net_pnl": round(net_pnl, 2),
            "reason": reason,
            "currency": self.currency
        }
        
        del self.open_trades[trade_id]
        return log_entry
    
    def write_log(self, log_entry, log_file):
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

class Strategy5_EMARSIBot:
    def __init__(self):
        self.symbol = 'XRPUSDT'
        self.demo_mode = os.getenv('DEMO_MODE', 'true').lower() == 'true'
        
        prefix = 'TESTNET_' if self.demo_mode else 'LIVE_'
        self.api_key = os.getenv(f'{prefix}BYBIT_API_KEY')
        self.api_secret = os.getenv(f'{prefix}BYBIT_API_SECRET')
        
        self.exchange = None
        self.position = None
        self.pending_order = None
        self.price_data = pd.DataFrame()
        self.ema_divergence = 0
        
        # UPDATED CONFIG TO MATCH STRATEGY 5
        self.config = {
            'ema_fast': 9,           # Changed from 5
            'ema_slow': 21,          # Changed from 13
            'rsi_period': 7,         # Changed from 5
            'rsi_long_filter': 40,   # Long only if RSI > 40
            'rsi_short_filter': 60,  # Short only if RSI < 60
            'position_size': 100,
            'maker_offset_pct': 0.01,
            'stop_loss': 0.35,       # Hard stop 0.35%
            'trail_divergence': 0.15,# Trail when EMAs diverge > 0.15%
            'order_timeout': 180,
            'qty_step': 0.01,
        }
        
        os.makedirs("logs", exist_ok=True)
        self.log_file = "logs/5_EMA_RSI_BNBUSDT.log"
        self.unified_logger = UnifiedLogger("STRATEGY5_EMA_RSI", self.symbol)
        self.current_trade_id = None
        self.entry_price = None
        self.trailing_stop = None
    
    def connect(self):
        try:
            self.exchange = HTTP(demo=self.demo_mode, api_key=self.api_key, api_secret=self.api_secret)
            return self.exchange.get_server_time().get('retCode') == 0
        except:
            return False
    
    async def check_pending_orders(self):
        try:
            orders = self.exchange.get_open_orders(category="linear", symbol=self.symbol)
            if orders.get('retCode') != 0:
                return False
            
            order_list = orders['result']['list']
            if not order_list:
                self.pending_order = None
                return False
            
            order = order_list[0]
            age = datetime.now().timestamp() - int(order['createdTime']) / 1000
            
            if age > self.config['order_timeout']:
                self.exchange.cancel_order(category="linear", symbol=self.symbol, orderId=order['orderId'])
                self.pending_order = None
                return False
            
            self.pending_order = order
            return True
        except:
            return False
    
    def calculate_indicators(self, df):
        if len(df) < max(self.config['ema_slow'], self.config['rsi_period']) + 1:
            return None
        
        close = df['close']
        
        # EMAs
        ema_fast = close.ewm(span=self.config['ema_fast']).mean()
        ema_slow = close.ewm(span=self.config['ema_slow']).mean()
        
        # EMA divergence for trailing
        self.ema_divergence = abs((ema_fast.iloc[-1] - ema_slow.iloc[-1]) / ema_slow.iloc[-1] * 100)
        
        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(window=self.config['rsi_period']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.config['rsi_period']).mean()
        rsi = 100 - (100 / (1 + gain / (loss + 1e-10))).iloc[-1]
        
        # Crossover detection
        crossover_up = ema_fast.iloc[-2] <= ema_slow.iloc[-2] and ema_fast.iloc[-1] > ema_slow.iloc[-1]
        crossover_down = ema_fast.iloc[-2] >= ema_slow.iloc[-2] and ema_fast.iloc[-1] < ema_slow.iloc[-1]
        
        return {
            'ema_fast': ema_fast.iloc[-1],
            'ema_slow': ema_slow.iloc[-1],
            'crossover_up': crossover_up,
            'crossover_down': crossover_down,
            'rsi': rsi if pd.notna(rsi) else 50
        }
    
    def generate_signal(self, df):
        indicators = self.calculate_indicators(df)
        if not indicators:
            return None
        
        price = float(df['close'].iloc[-1])
        
        # BUY: EMA crossover up + RSI > 40
        if indicators['crossover_up'] and indicators['rsi'] > self.config['rsi_long_filter']:
            return {'action': 'BUY', 'price': price, 'rsi': indicators['rsi']}
        
        # SELL: EMA crossover down + RSI < 60
        elif indicators['crossover_down'] and indicators['rsi'] < self.config['rsi_short_filter']:
            return {'action': 'SELL', 'price': price, 'rsi': indicators['rsi']}
        
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
        
        # Calculate profit percentage
        profit_pct = ((current_price - self.entry_price) / self.entry_price * 100) if is_long else ((self.entry_price - current_price) / self.entry_price * 100)
        
        # Hard stop loss
        if profit_pct <= -self.config['stop_loss']:
            return True, "stop_loss"
        
        # Trailing stop when EMAs diverge > 0.15%
        if self.ema_divergence > self.config['trail_divergence']:
            # Initialize trailing stop
            if not self.trailing_stop:
                if is_long:
                    self.trailing_stop = current_price * (1 - self.config['stop_loss'] / 100)
                else:
                    self.trailing_stop = current_price * (1 + self.config['stop_loss'] / 100)
            else:
                # Update trailing stop
                if is_long:
                    new_stop = current_price * (1 - self.config['stop_loss'] / 100)
                    if new_stop > self.trailing_stop:
                        self.trailing_stop = new_stop
                    if current_price <= self.trailing_stop:
                        return True, "trailing_stop"
                else:
                    new_stop = current_price * (1 + self.config['stop_loss'] / 100)
                    if new_stop < self.trailing_stop:
                        self.trailing_stop = new_stop
                    if current_price >= self.trailing_stop:
                        return True, "trailing_stop"
        
        return False, ""
    
    def format_qty(self, qty):
        step = self.config['qty_step']
        return f"{round(qty / step) * step:.2f}"
    
    def log_trade(self, action, price, info=""):
        if action in ["BUY", "SELL"] and not self.current_trade_id:
            qty = self.config['position_size'] / price
            stop_loss = price * (1 - self.config['stop_loss']/100) if action == "BUY" else price * (1 + self.config['stop_loss']/100)
            take_profit = price * 1.01 if action == "BUY" else price * 0.99  # Placeholder since we use trailing
            
            self.current_trade_id, log_entry = self.unified_logger.log_trade_open(
                side=action,
                expected_price=price,
                actual_price=price,
                qty=qty,
                stop_loss=stop_loss,
                take_profit=take_profit,
                info=info
            )
            
            self.unified_logger.write_log(log_entry, self.log_file)
            
        elif action == "CLOSE" and self.current_trade_id:
            log_entry = self.unified_logger.log_trade_close(
                trade_id=self.current_trade_id,
                expected_exit=price,
                actual_exit=price,
                reason=info
            )
            
            if log_entry:
                self.unified_logger.write_log(log_entry, self.log_file)
            
            self.current_trade_id = None
    
    async def execute_trade(self, signal):
        if await self.check_pending_orders() or self.position:
            return
        
        qty = self.config['position_size'] / signal['price']
        formatted_qty = self.format_qty(qty)
        
        if float(formatted_qty) < self.config['qty_step']:
            return
        
        is_buy = signal['action'] == 'BUY'
        offset = 1 - self.config['maker_offset_pct']/100 if is_buy else 1 + self.config['maker_offset_pct']/100
        limit_price = round(signal['price'] * offset, 2)
        
        try:
            order = self.exchange.place_order(
                category="linear",
                symbol=self.symbol,
                side="Buy" if is_buy else "Sell",
                orderType="Limit",
                qty=formatted_qty,
                price=str(limit_price),
                timeInForce="PostOnly"
            )
            
            if order.get('retCode') == 0:
                self.pending_order = order['result']
                print(f"✅ {signal['action']}: {formatted_qty} @ ${limit_price:.2f} | RSI:{signal['rsi']:.1f}")
                self.log_trade(signal['action'], limit_price, f"RSI:{signal['rsi']:.1f}_EMA_cross")
        except Exception as e:
            print(f"❌ Trade failed: {e}")

    async def close_position(self, reason):
        if not self.position:
            return
        
        current_price = float(self.price_data['close'].iloc[-1])
        side = "Sell" if self.position.get('side') == "Buy" else "Buy"
        qty = float(self.position['size'])
        
        # Calculate limit price with maker offset to ensure rebate
        offset_mult = 1 + self.config['maker_offset_pct']/100 if side == "Sell" else 1 - self.config['maker_offset_pct']/100
        limit_price = round(current_price * offset_mult, 2)
        
        try:
            order = self.exchange.place_order(
                category="linear",
                symbol=self.symbol,
                side=side,
                orderType="Limit",           # ✅ FIXED: Limit instead of Market
                qty=self.format_qty(qty),
                price=str(limit_price),      # ✅ ADDED: Limit price for maker rebate
                timeInForce="PostOnly",      # ✅ ADDED: Ensures maker execution
                reduceOnly=True
            )
            
            if order.get('retCode') == 0:
                current = float(self.price_data['close'].iloc[-1])
                print(f"✅ Closed: {reason} @ ${limit_price:.2f}")
                self.log_trade("CLOSE", current, reason)
                self.entry_price = None
                self.trailing_stop = None
        except Exception as e:
            print(f"❌ Close failed: {e}")
    
    def show_status(self):
        if len(self.price_data) == 0:
            return
        
        current_price = float(self.price_data['close'].iloc[-1])
        status_parts = [f"📊 BNB: ${current_price:.2f}"]
        
        if self.position:
            side = self.position.get('side', '')
            status_parts.append(f"📍 {side} @ ${self.entry_price:.2f}")
            if self.trailing_stop:
                status_parts.append(f"Trail: ${self.trailing_stop:.2f}")
            status_parts.append(f"EMA Div: {self.ema_divergence:.2f}%")
        elif self.pending_order:
            order_price = float(self.pending_order.get('price', 0))
            order_side = self.pending_order.get('side', '')
            age = int(datetime.now().timestamp() - int(self.pending_order.get('createdTime', 0)) / 1000)
            status_parts.append(f"⏳ {order_side} @ ${order_price:.2f} ({age}s)")
        else:
            indicators = self.calculate_indicators(self.price_data)
            if indicators:
                status_parts.append(f"RSI: {indicators['rsi']:.1f}")
                if indicators['ema_fast'] > indicators['ema_slow']:
                    status_parts.append("EMA: ↑")
                else:
                    status_parts.append("EMA: ↓")
        
        print(" | ".join(status_parts), end='\r')
    
    async def run_cycle(self):
        if not await self.get_market_data():
            return
        
        await self.check_position()
        await self.check_pending_orders()
        
        if self.position:
            should_close, reason = self.should_close()
            if should_close:
                await self.close_position(reason)
        elif not self.pending_order:
            signal = self.generate_signal(self.price_data)
            if signal:
                await self.execute_trade(signal)
        
        self.show_status()
    
    async def run(self):
        if not self.connect():
            print("❌ Failed to connect")
            return
        
        print(f"✅ Strategy 5: EMA Crossover + RSI Filter")
        print(f"📊 {self.symbol} | EMA 9/21 | RSI 7")
        print(f"🎯 Hard Stop: {self.config['stop_loss']}% | Trail at EMA div > {self.config['trail_divergence']}%")
        
        while True:
            try:
                await self.run_cycle()
                await asyncio.sleep(1)
            except KeyboardInterrupt:
                print("\n🛑 Bot stopped")
                try:
                    self.exchange.cancel_all_orders(category="linear", symbol=self.symbol)
                except:
                    pass
                if self.position:
                    await self.close_position("manual_stop")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                await asyncio.sleep(5)

if __name__ == "__main__":
    bot = Strategy5_EMARSIBot()
    asyncio.run(bot.run())