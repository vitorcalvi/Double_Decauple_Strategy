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
        """Log position closing with slippage and PnL calculation"""
        if trade_id not in self.open_trades:
            return None
            
        trade = self.open_trades[trade_id]
        duration = (datetime.now() - trade["entry_time"]).total_seconds()
        
        slippage = actual_exit - expected_exit if trade["side"] == "SELL" else expected_exit - actual_exit
        
        if trade["side"] == "BUY":
            gross_pnl = (actual_exit - trade["entry_price"]) * trade["qty"]
        else:
            gross_pnl = (trade["entry_price"] - actual_exit) * trade["qty"]
        
        # FIX: Calculate fees as percentage of trade value
        fee_rate = 0.001  # 0.1% fee rate
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

class ETHScalpingBot:
    def __init__(self):
        self.symbol = 'ETHUSDT'
        self.demo_mode = os.getenv('DEMO_MODE', 'true').lower() == 'true'
        
        prefix = 'TESTNET_' if self.demo_mode else 'LIVE_'
        self.api_key = os.getenv(f'{prefix}BYBIT_API_KEY')
        self.api_secret = os.getenv(f'{prefix}BYBIT_API_SECRET')
        self.exchange = None
        
        self.position = None
        self.price_data = pd.DataFrame()
        self.pending_order = None
        self.last_signal = None
        self.last_order_time = None
        self.order_timeout = 60
        self.order_cooldown = 30
        
        self.config = {
            'ema_fast': 5,
            'ema_slow': 13,
            'bb_period': 20,
            'bb_std': 2.0,
            'position_size': 500,
            'maker_offset_pct': 0.02,
            'net_take_profit': 1.05,
            'net_stop_loss': 0.45,
        }
        
        os.makedirs("logs", exist_ok=True)
        self.log_file = "logs/1_FEES_EMA_BB_ETHUSDT.log"
        self.unified_logger = UnifiedLogger("1_FEES_EMA_BB", self.symbol)
        self.current_trade_id = None
    
    def connect(self):
        try:
            self.exchange = HTTP(demo=self.demo_mode, api_key=self.api_key, api_secret=self.api_secret)
            return self.exchange.get_server_time().get('retCode') == 0
        except:
            return False
    
    def format_qty(self, qty):
        return f"{round(qty / 0.001) * 0.001:.3f}"
    
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
            
            if age > self.order_timeout:
                self.exchange.cancel_order(category="linear", symbol=self.symbol, orderId=order['orderId'])
                self.pending_order = None
                self.last_signal = None
                return False
            
            self.pending_order = order
            return True
        except:
            return False
    
    def is_valid_signal(self, signal):
        if not signal:
            return False
            
        if self.last_order_time:
            elapsed = (datetime.now() - self.last_order_time).total_seconds()
            if elapsed < self.order_cooldown:
                return False
        
        if self.last_signal:
            price_change = abs(signal['price'] - self.last_signal['price']) / self.last_signal['price']
            if price_change < 0.001:
                return False
        
        return True
    
    def calculate_indicators(self, df):
        if len(df) < 20:
            return None
        
        close = df['close']
        ema_fast = close.ewm(span=self.config['ema_fast']).mean().iloc[-1]
        ema_slow = close.ewm(span=self.config['ema_slow']).mean().iloc[-1]
        
        sma = close.rolling(window=self.config['bb_period']).mean()
        std = close.rolling(window=self.config['bb_period']).std()
        upper_band = (sma + std * self.config['bb_std']).iloc[-1]
        lower_band = (sma - std * self.config['bb_std']).iloc[-1]
        
        bb_range = upper_band - lower_band
        bb_position = (close.iloc[-1] - lower_band) / bb_range if bb_range != 0 else 0.5
        
        return {
            'price': close.iloc[-1],
            'ema_fast': ema_fast,
            'ema_slow': ema_slow,
            'bb_position': bb_position,
            'trend': 'UP' if ema_fast > ema_slow else 'DOWN'
        }
    
    def generate_signal(self, df):
        indicators = self.calculate_indicators(df)
        if not indicators:
            return None
        
        if indicators['trend'] == 'UP' and indicators['bb_position'] <= 0.3:
            signal = {'action': 'BUY', 'price': indicators['price'], 'bb_pos': indicators['bb_position']}
            if self.is_valid_signal(signal):
                return signal
        
        if indicators['trend'] == 'DOWN' and indicators['bb_position'] >= 0.7:
            signal = {'action': 'SELL', 'price': indicators['price'], 'bb_pos': indicators['bb_position']}
            if self.is_valid_signal(signal):
                return signal
        
        return None
    
    async def get_market_data(self):
        try:
            klines = self.exchange.get_kline(category="linear", symbol=self.symbol, interval="1", limit=50)
            if klines.get('retCode') != 0:
                return False
            
            df = pd.DataFrame(klines['result']['list'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
            
            self.price_data = df.sort_values('timestamp').reset_index(drop=True)
            return True
        except:
            return False
    
    async def check_position(self):
        try:
            positions = self.exchange.get_positions(category="linear", symbol=self.symbol)
            if positions.get('retCode') == 0:
                pos_list = positions['result']['list']
                self.position = pos_list[0] if pos_list and float(pos_list[0]['size']) > 0 else None
        except:
            pass
    
    def should_close(self):
        if not self.position:
            return False, ""
        
        current_price = float(self.price_data['close'].iloc[-1])
        entry_price = float(self.position.get('avgPrice', 0))
        side = self.position.get('side', '')
        
        if entry_price == 0:
            return False, ""
        
        profit_pct = ((current_price - entry_price) / entry_price * 100) if side == "Buy" else ((entry_price - current_price) / entry_price * 100)
        
        if profit_pct >= self.config['net_take_profit']:
            return True, "take_profit"
        if profit_pct <= -self.config['net_stop_loss']:
            return True, "stop_loss"
        
        return False, ""
    
    def log_trade(self, action, price, info=""):
        if action in ["BUY", "SELL"] and not self.current_trade_id:
            qty = self.config['position_size'] / price
            stop_loss = price * (1 - self.config['net_stop_loss']/100) if action == "BUY" else price * (1 + self.config['net_stop_loss']/100)
            take_profit = price * (1 + self.config['net_take_profit']/100) if action == "BUY" else price * (1 - self.config['net_take_profit']/100)
            
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
                reason=info.split("_")[0] if "_" in info else info
            )
            
            if log_entry:
                self.unified_logger.write_log(log_entry, self.log_file)
            
            self.current_trade_id = None
    
    async def execute_trade(self, signal):
        if await self.check_pending_orders() or self.position or not self.is_valid_signal(signal):
            return
        
        qty = self.config['position_size'] / signal['price']
        formatted_qty = self.format_qty(qty)
        
        if float(formatted_qty) < 0.001:
            return
        
        offset = 1 - self.config['maker_offset_pct']/100 if signal['action'] == 'BUY' else 1 + self.config['maker_offset_pct']/100
        limit_price = round(signal['price'] * offset, 2)
        
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
                self.last_signal = signal
                self.last_order_time = datetime.now()
                self.pending_order = order['result']
                
                print(f"✅ {signal['action']}: {formatted_qty} ETH @ ${limit_price:.2f} | BB: {signal['bb_pos']:.2f}")
                self.log_trade(signal['action'], limit_price, f"BB:{signal['bb_pos']:.2f}")
        except Exception as e:
            print(f"❌ Trade failed: {e}")
    
    async def close_position(self, reason):
        if not self.position:
            return
        
        side = "Sell" if self.position.get('side') == "Buy" else "Buy"
        qty = float(self.position['size'])
        formatted_qty = self.format_qty(qty)
        
        try:
            order = self.exchange.place_order(
                category="linear",
                symbol=self.symbol,
                side=side,
                orderType="Market",
                qty=formatted_qty,
                reduceOnly=True
            )
            
            if order.get('retCode') == 0:
                current = float(self.price_data['close'].iloc[-1])
                print(f"💰 Position Closed: {reason}")
                self.log_trade("CLOSE", current, reason)
                self.position = None
                self.last_signal = None
        except Exception as e:
            print(f"❌ Close failed: {e}")
    
    def show_status(self):
        if len(self.price_data) == 0:
            return
        
        current_price = float(self.price_data['close'].iloc[-1])
        indicators = self.calculate_indicators(self.price_data)
        
        status = f"📊 ETH: ${current_price:,.2f}"
        
        if self.position:
            entry = float(self.position.get('avgPrice', 0))
            side = self.position.get('side', '')
            pct = ((current_price - entry) / entry * 100) if side == "Buy" else ((entry - current_price) / entry * 100)
            status += f" | 📍 {side} @ ${entry:.2f} | {pct:+.2f}%"
        elif self.pending_order:
            order_price = float(self.pending_order.get('price', 0))
            order_side = self.pending_order.get('side', '')
            age = int(datetime.now().timestamp() - int(self.pending_order.get('createdTime', 0)) / 1000)
            status += f" | ⏳ Pending {order_side} @ ${order_price:.2f} ({age}s)"
        elif indicators:
            status += f" | BB: {indicators['bb_position']:.2f} | Trend: {indicators['trend']}"
        
        print(status, end='\r')
    
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
        
        print(f"🚀 EMA + BB Bot for {self.symbol}")
        print(f"🎯 TP: {self.config['net_take_profit']}% | SL: {self.config['net_stop_loss']}%")
        print("✅ Connected! Starting bot...")
        
        while True:
            try:
                await self.run_cycle()
                await asyncio.sleep(10 if not self.position and not self.pending_order else 3)
                    
            except KeyboardInterrupt:
                print("\n🛑 Shutting down...")
                try:
                    self.exchange.cancel_all_orders(category="linear", symbol=self.symbol)
                except:
                    pass
                if self.position:
                    await self.close_position("manual_stop")
                print("✅ Bot stopped")
                break
            except Exception as e:
                print(f"⚠️ Error: {e}")
                await asyncio.sleep(5)

if __name__ == "__main__":
    bot = ETHScalpingBot()
    asyncio.run(bot.run())