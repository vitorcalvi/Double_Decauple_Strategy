# 3_FEES_EMAMACDRSI_SOLUSDT.py - Streamlined

import os
import asyncio
import pandas as pd
from datetime import datetime
from pybit.unified_trading import HTTP
from dotenv import load_dotenv
from unified_logger import UnifiedLogger

load_dotenv()

class EMAMACDRSIBot:
    def __init__(self):
        self.symbol = 'SOLUSDT'
        self.demo_mode = os.getenv('DEMO_MODE', 'true').lower() == 'true'
        
        prefix = 'TESTNET_' if self.demo_mode else 'LIVE_'
        self.api_key = os.getenv(f'{prefix}BYBIT_API_KEY')
        self.api_secret = os.getenv(f'{prefix}BYBIT_API_SECRET')
        
        self.exchange = None
        self.position = None
        self.price_data = pd.DataFrame()
        self.pending_order = None
        self.last_signal = None
        self.order_timeout = 180
        
        self.config = {
            'timeframe': '5',
            'ema_short': 12,
            'ema_long': 26,
            'macd_signal': 9,
            'rsi_period': 14,
            'position_size': 100,
            'lookback': 100,
            'maker_offset_pct': 0.01,
            'net_take_profit': 1.08,
            'net_stop_loss': 0.42,
        }
        
        os.makedirs("logs", exist_ok=True)
        self.log_file = "logs/3_FEES_EMAMACDRSI_SOLUSDT.log"
        self.unified_logger = UnifiedLogger("3_FEES_EMAMACDRSI", self.symbol)
        self.current_trade_id = None
    
    def connect(self):
        try:
            self.exchange = HTTP(demo=self.demo_mode, api_key=self.api_key, api_secret=self.api_secret)
            return self.exchange.get_server_time().get('retCode') == 0
        except:
            return False
    
    def format_qty(self, qty):
        return str(int(round(qty)))
    
    async def check_pending_orders(self):
        try:
            orders = self.exchange.get_open_orders(category="linear", symbol=self.symbol)
            if orders.get('retCode') == 0:
                order_list = orders['result']['list']
                if order_list:
                    order = order_list[0]
                    age = datetime.now().timestamp() - int(order['createdTime']) / 1000
                    if age > self.order_timeout:
                        self.exchange.cancel_order(category="linear", symbol=self.symbol, orderId=order['orderId'])
                        self.pending_order = None
                    else:
                        self.pending_order = order
                        return True
                else:
                    self.pending_order = None
            return False
        except:
            return False
    
    def identify_swings(self):
        if len(self.price_data) < 5:
            return []
        
        df = self.price_data
        swings = []
        
        for i in range(1, len(df) - 1):
            if df['high'].iloc[i] > df['high'].iloc[i-1] and df['high'].iloc[i] > df['high'].iloc[i+1]:
                swings.append({'index': i, 'type': 'HIGH', 'price': df['high'].iloc[i]})
            elif df['low'].iloc[i] < df['low'].iloc[i-1] and df['low'].iloc[i] < df['low'].iloc[i+1]:
                swings.append({'index': i, 'type': 'LOW', 'price': df['low'].iloc[i]})
        
        return swings
    
    def generate_signal(self):
        if len(self.price_data) < 10:
            return None
        
        swings = self.identify_swings()
        if len(swings) < 2:
            return None
        
        current_price = float(self.price_data['close'].iloc[-1])
        last_swing = swings[-1]
        
        if last_swing['type'] == 'LOW' and current_price > last_swing['price']:
            return {'action': 'BUY', 'price': current_price, 'reason': 'swing_low'}
        elif last_swing['type'] == 'HIGH' and current_price < last_swing['price']:
            return {'action': 'SELL', 'price': current_price, 'reason': 'swing_high'}
        
        return None
    
    async def get_market_data(self):
        try:
            klines = self.exchange.get_kline(
                category="linear", symbol=self.symbol,
                interval=self.config['timeframe'], limit=self.config['lookback']
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
        
        if side == "Buy":
            if current_price >= entry_price * (1 + self.config['net_take_profit'] / 100):
                return True, "take_profit"
            if current_price <= entry_price * (1 - self.config['net_stop_loss'] / 100):
                return True, "stop_loss"
        else:
            if current_price <= entry_price * (1 - self.config['net_take_profit'] / 100):
                return True, "take_profit"
            if current_price >= entry_price * (1 + self.config['net_stop_loss'] / 100):
                return True, "stop_loss"
        
        return False, ""
    
    def log_trade(self, action, price, info=""):
        if action in ["BUY", "SELL"] and not self.current_trade_id:
            qty = self.config['position_size'] / price
            stop_loss = price * (1 - self.config['net_stop_loss']/100) if action == "BUY" else price * (1 + self.config['net_stop_loss']/100)
            take_profit = price * (1 + self.config['net_take_profit']/100) if action == "BUY" else price * (1 - self.config['net_take_profit']/100)
            
            self.current_trade_id, log_entry = self.unified_logger.log_trade_open(
                side=action, expected_price=price, actual_price=price, qty=qty,
                stop_loss=stop_loss, take_profit=take_profit, info=info
            )
            self.unified_logger.write_log(log_entry, self.log_file)
            
        elif action == "CLOSE" and self.current_trade_id:
            log_entry = self.unified_logger.log_trade_close(
                trade_id=self.current_trade_id, expected_exit=price, actual_exit=price,
                reason=info.split("_")[0] if "_" in info else info
            )
            if log_entry:
                self.unified_logger.write_log(log_entry, self.log_file)
            self.current_trade_id = None
    
    async def execute_trade(self, signal):
        if await self.check_pending_orders() or self.position:
            return
        
        qty = self.config['position_size'] / signal['price']
        formatted_qty = self.format_qty(qty)
        
        if float(formatted_qty) < 1:
            return
        
        limit_price = round(signal['price'] * (1 - self.config['maker_offset_pct']/100 if signal['action'] == 'BUY' else 1 + self.config['maker_offset_pct']/100), 4)
        
        try:
            order = self.exchange.place_order(
                category="linear", symbol=self.symbol,
                side="Buy" if signal['action'] == 'BUY' else "Sell",
                orderType="Limit", qty=formatted_qty, price=str(limit_price),
                timeInForce="PostOnly"
            )
            
            if order.get('retCode') == 0:
                self.pending_order = order['result']
                print(f"✅ {signal['action']}: {formatted_qty} @ ${limit_price:.4f} | {signal['reason']}")
                self.log_trade(signal['action'], limit_price, signal['reason'])
        except Exception as e:
            print(f"❌ Trade error: {e}")
    
    async def close_position(self, reason):
        if not self.position:
            return
        
        side = "Sell" if self.position.get('side') == "Buy" else "Buy"
        qty = float(self.position['size'])
        
        try:
            order = self.exchange.place_order(
                category="linear", symbol=self.symbol, side=side,
                orderType="Market", qty=self.format_qty(qty), reduceOnly=True
            )
            
            if order.get('retCode') == 0:
                current = float(self.price_data['close'].iloc[-1])
                print(f"✅ Closed: {reason}")
                self.log_trade("CLOSE", current, reason)
        except Exception as e:
            print(f"❌ Close failed: {e}")
    
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
            signal = self.generate_signal()
            if signal:
                await self.execute_trade(signal)
    
    async def run(self):
        if not self.connect():
            print("❌ Failed to connect")
            return
        
        print(f"✅ ZigZag Trading Bot - {self.symbol}")
        
        try:
            while True:
                await self.run_cycle()
                await asyncio.sleep(2)
        except KeyboardInterrupt:
            print("🛑 Bot stopped")
            try:
                self.exchange.cancel_all_orders(category="linear", symbol=self.symbol)
            except:
                pass
            if self.position:
                await self.close_position("manual_stop")

if __name__ == "__main__":
    bot = ZigZagTradingBot()
    asyncio.run(bot.run())

#==============================================================================

# 5_FEES_MACD_VWAP_BTCUSDT.py - Streamlined

import os
import asyncio
import pandas as pd
from datetime import datetime
from pybit.unified_trading import HTTP
from dotenv import load_dotenv
from unified_logger import UnifiedLogger

load_dotenv()

class PivotReversalBot:
    def __init__(self):
        self.symbol = 'BTCUSDT'
        self.demo_mode = os.getenv('DEMO_MODE', 'true').lower() == 'true'
        
        prefix = 'TESTNET_' if self.demo_mode else 'LIVE_'
        self.api_key = os.getenv(f'{prefix}BYBIT_API_KEY')
        self.api_secret = os.getenv(f'{prefix}BYBIT_API_SECRET')
        self.exchange = None
        
        self.position = None
        self.price_data = pd.DataFrame()
        self.pending_order = None
        self.last_order_time = None
        self.order_timeout = 180
        self.last_signal = None
        
        self.config = {
            'rsi_period': 7,
            'rsi_oversold': 40,
            'rsi_overbought': 60,
            'position_size': 100,
            'maker_offset_pct': 0.01,
            'net_take_profit': 0.68,
            'net_stop_loss': 0.07,
        }
        
        self.last_status_time = None
        self.status_interval = 5
        
        os.makedirs("logs", exist_ok=True)
        self.log_file = "logs/5_FEES_MACD_VWAP_BTCUSDT.log"
        self.unified_logger = UnifiedLogger("5_FEES_MACD_VWAP", self.symbol)
        self.current_trade_id = None
    
    def connect(self):
        try:
            self.exchange = HTTP(demo=self.demo_mode, api_key=self.api_key, api_secret=self.api_secret)
            return self.exchange.get_server_time().get('retCode') == 0
        except:
            return False
    
    async def check_pending_orders(self):
        try:
            orders = self.exchange.get_open_orders(category="linear", symbol=self.symbol)
            if orders.get('retCode') == 0:
                order_list = orders['result']['list']
                if order_list:
                    order = order_list[0]
                    age = datetime.now().timestamp() - int(order['createdTime']) / 1000
                    if age > self.order_timeout:
                        self.exchange.cancel_order(category="linear", symbol=self.symbol, orderId=order['orderId'])
                        self.pending_order = None
                        self.last_signal = None
                    else:
                        self.pending_order = order
                        return True
                else:
                    self.pending_order = None
            return False
        except:
            return False
    
    def format_qty(self, qty):
        return f"{round(qty / 0.001) * 0.001:.3f}"
    
    def calculate_indicators(self, df):
        if len(df) < self.config['rsi_period'] + 1:
            return None
        
        close = df['close']
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(window=self.config['rsi_period']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.config['rsi_period']).mean()
        
        epsilon = 1e-10
        rs = gain / (loss + epsilon)
        rsi = 100 - (100 / (1 + rs))
        
        return {
            'rsi': rsi.iloc[-1] if not rsi.empty else 50,
            'rsi_prev': rsi.iloc[-2] if len(rsi) > 1 else 50
        }
    
    def generate_signal(self, df):
        if len(df) < 20:
            return None
        
        current_price = float(df['close'].iloc[-1])
        indicators = self.calculate_indicators(df)
        if not indicators:
            return None
        
        rsi = indicators['rsi']
        rsi_prev = indicators['rsi_prev']
        
        if self.last_signal:
            price_change = abs(current_price - self.last_signal['price']) / self.last_signal['price']
            if price_change < 0.001:
                return None
        
        if rsi <= self.config['rsi_oversold'] and rsi > rsi_prev:
            return {'action': 'BUY', 'price': current_price, 'rsi': rsi}
        
        if rsi >= self.config['rsi_overbought'] and rsi < rsi_prev:
            return {'action': 'SELL', 'price': current_price, 'rsi': rsi}
        
        return None
    
    async def get_market_data(self):
        try:
            klines = self.exchange.get_kline(
                category="linear", symbol=self.symbol, interval="1", limit=50
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
        
        if side == "Buy":
            profit_pct = ((current_price - entry_price) / entry_price) * 100
            if profit_pct >= self.config['net_take_profit']:
                return True, "take_profit"
            if profit_pct <= -self.config['net_stop_loss']:
                return True, "stop_loss"
        else:
            profit_pct = ((entry_price - current_price) / entry_price) * 100
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
                side=action, expected_price=price, actual_price=price, qty=qty,
                stop_loss=stop_loss, take_profit=take_profit, info=info
            )
            self.unified_logger.write_log(log_entry, self.log_file)
            
        elif action == "CLOSE" and self.current_trade_id:
            log_entry = self.unified_logger.log_trade_close(
                trade_id=self.current_trade_id, expected_exit=price, actual_exit=price,
                reason=info.split("_")[0] if "_" in info else info
            )
            if log_entry:
                self.unified_logger.write_log(log_entry, self.log_file)
            self.current_trade_id = None
    
    async def execute_trade(self, signal):
        if await self.check_pending_orders() or self.position:
            return
        
        qty = self.config['position_size'] / signal['price']
        formatted_qty = self.format_qty(qty)
        
        if float(formatted_qty) < 0.001:
            return
        
        offset_mult = 1 - self.config['maker_offset_pct']/100 if signal['action'] == 'BUY' else 1 + self.config['maker_offset_pct']/100
        limit_price = round(signal['price'] * offset_mult, 2)
        
        try:
            order = self.exchange.place_order(
                category="linear", symbol=self.symbol,
                side="Buy" if signal['action'] == 'BUY' else "Sell",
                orderType="Limit", qty=formatted_qty, price=str(limit_price),
                timeInForce="PostOnly"
            )
            
            if order.get('retCode') == 0:
                self.last_order_time = datetime.now()
                self.last_signal = signal
                self.pending_order = order['result']
                print(f"✅ {signal['action']}: {formatted_qty} BTC @ ${limit_price:,.2f} | RSI: {signal['rsi']:.1f}")
                self.log_trade(signal['action'], limit_price, f"RSI:{signal['rsi']:.1f}")
        except Exception as e:
            print(f"❌ Trade failed: {e}")
    
    async def close_position(self, reason):
        if not self.position:
            return
        
        side = "Sell" if self.position.get('side') == "Buy" else "Buy"
        qty = float(self.position['size'])
        
        try:
            order = self.exchange.place_order(
                category="linear", symbol=self.symbol, side=side,
                orderType="Market", qty=self.format_qty(qty), reduceOnly=True
            )
            
            if order.get('retCode') == 0:
                current_price = float(self.price_data['close'].iloc[-1])
                print(f"💰 Position Closed: {reason}")
                self.log_trade("CLOSE", current_price, reason)
                self.position = None
                self.last_signal = None
        except Exception as e:
            print(f"❌ Close failed: {e}")
    
    def show_status(self):
        now = datetime.now()
        if self.last_status_time and (now - self.last_status_time).total_seconds() < self.status_interval:
            return
        
        self.last_status_time = now
        
        if len(self.price_data) == 0:
            return
        
        current_price = float(self.price_data['close'].iloc[-1])
        status_parts = [f"📊 BTC: ${current_price:,.2f}"]
        
        if self.position:
            entry = float(self.position.get('avgPrice', 0))
            side = self.position.get('side', '')
            status_parts.append(f"📍 {side} @ ${entry:,.2f}")
        elif self.pending_order:
            order_price = float(self.pending_order.get('price', 0))
            order_side = self.pending_order.get('side', '')
            status_parts.append(f"⏳ Pending {order_side} @ ${order_price:,.2f}")
        else:
            indicators = self.calculate_indicators(self.price_data)
            if indicators:
                status_parts.append(f"RSI: {indicators['rsi']:.1f}")
        
        print(" ".join(status_parts), end='\r')
    
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
        
        print(f"🚀 Pivot Reversal Bot for {self.symbol}")
        
        while True:
            try:
                await self.run_cycle()
                await asyncio.sleep(2)
            except KeyboardInterrupt:
                print("🛑 Bot stopped")
                try:
                    self.exchange.cancel_all_orders(category="linear", symbol=self.symbol)
                except:
                    pass
                if self.position:
                    await self.close_position("manual_stop")
                break
            except Exception as e:
                print(f"⚠️ Error: {e}")
                await asyncio.sleep(5)

if __name__ == "__main__":
    bot = PivotReversalBot()
    asyncio.run(bot.run())secret)
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
            
            if age > self.order_timeout:
                self.exchange.cancel_order(category="linear", symbol=self.symbol, orderId=order['orderId'])
                self.pending_order = None
                self.last_signal = None
                return False
            
            self.pending_order = order
            return True
        except:
            return False
    
    def calculate_indicators(self, df):
        if len(df) < self.config['lookback']:
            return None
        
        close = df['close']
        
        ema_short = close.ewm(span=self.config['ema_short']).mean()
        ema_long = close.ewm(span=self.config['ema_long']).mean()
        macd_line = ema_short - ema_long
        signal_line = macd_line.ewm(span=self.config['macd_signal']).mean()
        histogram = macd_line - signal_line
        
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(window=self.config['rsi_period']).mean()
        loss = -delta.clip(upper=0).rolling(window=self.config['rsi_period']).mean()
        epsilon = 1e-10
        rsi = 100 - (100 / (1 + gain / (loss + epsilon)))
        
        return {
            'price': close.iloc[-1],
            'ema_aligned': ema_short.iloc[-1] > ema_long.iloc[-1],
            'histogram': histogram.iloc[-1],
            'histogram_prev': histogram.iloc[-2] if len(histogram) > 1 else 0,
            'rsi': rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
        }
    
    def generate_signal(self, df):
        analysis = self.calculate_indicators(df)
        if not analysis:
            return None
        
        if self.last_signal:
            price_change = abs(analysis['price'] - self.last_signal['price']) / self.last_signal['price']
            if price_change < 0.002:
                return None
        
        histogram_positive = analysis['histogram'] > 0
        histogram_turning_positive = analysis['histogram_prev'] <= 0 and analysis['histogram'] > 0
        histogram_turning_negative = analysis['histogram_prev'] >= 0 and analysis['histogram'] < 0
        
        if (histogram_positive and analysis['rsi'] < 60) or (histogram_turning_positive and analysis['rsi'] < 70):
            return {'action': 'BUY', 'price': analysis['price'], 'rsi': analysis['rsi']}
        
        if (not histogram_positive and analysis['rsi'] > 40) or (histogram_turning_negative and analysis['rsi'] > 30):
            return {'action': 'SELL', 'price': analysis['price'], 'rsi': analysis['rsi']}
        
        return None
    
    async def get_market_data(self):
        try:
            klines = self.exchange.get_kline(
                category="linear",
                symbol=self.symbol,
                interval=self.config['timeframe'],
                limit=self.config['lookback']
            )
            
            if klines.get('retCode') != 0:
                return False
            
            df = pd.DataFrame(klines['result']['list'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
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
        
        analysis = self.calculate_indicators(self.price_data)
        if analysis:
            histogram_turning_negative = analysis['histogram_prev'] >= 0 and analysis['histogram'] < 0
            if side == "Buy" and histogram_turning_negative and analysis['rsi'] > 60:
                return True, "macd_reversal"
            
            histogram_turning_positive = analysis['histogram_prev'] <= 0 and analysis['histogram'] > 0
            if side == "Sell" and histogram_turning_positive and analysis['rsi'] < 40:
                return True, "macd_reversal"
        
        return False, ""
    
    def log_trade(self, action, price, info=""):
        if action in ["BUY", "SELL"] and not self.current_trade_id:
            qty = self.config['position_size'] / price
            stop_loss = price * (1 - self.config['net_stop_loss']/100) if action == "BUY" else price * (1 + self.config['net_stop_loss']/100)
            take_profit = price * (1 + self.config['net_take_profit']/100) if action == "BUY" else price * (1 - self.config['net_take_profit']/100)
            
            self.current_trade_id, log_entry = self.unified_logger.log_trade_open(
                side=action, expected_price=price, actual_price=price, qty=qty,
                stop_loss=stop_loss, take_profit=take_profit, info=info
            )
            self.unified_logger.write_log(log_entry, self.log_file)
            
        elif action == "CLOSE" and self.current_trade_id:
            log_entry = self.unified_logger.log_trade_close(
                trade_id=self.current_trade_id, expected_exit=price, actual_exit=price,
                reason=info.split("_")[0] if "_" in info else info
            )
            if log_entry:
                self.unified_logger.write_log(log_entry, self.log_file)
            self.current_trade_id = None
    
    async def execute_trade(self, signal):
        if await self.check_pending_orders() or self.position:
            return
        
        qty = self.config['position_size'] / signal['price']
        formatted_qty = str(int(round(qty)))
        
        if int(formatted_qty) == 0:
            return
        
        offset = 1 - self.config['maker_offset_pct']/100 if signal['action'] == 'BUY' else 1 + self.config['maker_offset_pct']/100
        limit_price = round(signal['price'] * offset, 4)
        
        try:
            order = self.exchange.place_order(
                category="linear", symbol=self.symbol,
                side="Buy" if signal['action'] == 'BUY' else "Sell",
                orderType="Limit", qty=formatted_qty, price=str(limit_price),
                timeInForce="PostOnly"
            )
            
            if order.get('retCode') == 0:
                self.last_signal = signal
                self.pending_order = order['result']
                print(f"✅ {signal['action']}: {formatted_qty} SOL @ ${limit_price:.4f} | RSI: {signal['rsi']:.1f}")
                self.log_trade(signal['action'], limit_price, f"RSI:{signal['rsi']:.1f}")
        except Exception as e:
            print(f"❌ Trade failed: {e}")
    
    async def close_position(self, reason):
        if not self.position:
            return
        
        side = "Sell" if self.position.get('side') == "Buy" else "Buy"
        qty = float(self.position['size'])
        
        try:
            order = self.exchange.place_order(
                category="linear", symbol=self.symbol, side=side,
                orderType="Market", qty=str(int(round(qty))), reduceOnly=True
            )
            
            if order.get('retCode') == 0:
                current = float(self.price_data['close'].iloc[-1])
                print(f"💰 Closed: {reason}")
                self.log_trade("CLOSE", current, reason)
                self.position = None
                self.last_signal = None
        except Exception as e:
            print(f"❌ Close failed: {e}")
    
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
    
    async def run(self):
        if not self.connect():
            print("❌ Failed to connect")
            return
        
        print(f"🚀 EMA+MACD+RSI Bot for {self.symbol}")
        
        while True:
            try:
                await self.run_cycle()
                await asyncio.sleep(5)
            except KeyboardInterrupt:
                print("🛑 Bot stopped")
                try:
                    self.exchange.cancel_all_orders(category="linear", symbol=self.symbol)
                except:
                    pass
                if self.position:
                    await self.close_position("manual_stop")
                break
            except Exception as e:
                print(f"⚠️ Error: {e}")
                await asyncio.sleep(5)

if __name__ == "__main__":
    bot = EMAMACDRSIBot()
    asyncio.run(bot.run())

#==============================================================================

# 4_FEES_LIQUIDITYSWEEPBOT_DOGEUSDT.py - Streamlined

import os
import asyncio
import pandas as pd
from datetime import datetime
from pybit.unified_trading import HTTP
from dotenv import load_dotenv
from unified_logger import UnifiedLogger

load_dotenv()

class ZigZagTradingBot:
    def __init__(self):
        self.symbol = 'DOGEUSDT'
        self.demo_mode = os.getenv('DEMO_MODE', 'true').lower() == 'true'
        
        prefix = 'TESTNET_' if self.demo_mode else 'LIVE_'
        self.api_key = os.getenv(f'{prefix}BYBIT_API_KEY')
        self.api_secret = os.getenv(f'{prefix}BYBIT_API_SECRET')
        self.exchange = None
        
        self.position = None
        self.price_data = pd.DataFrame()
        self.pending_order = None
        self.order_timeout = 180
        
        self.config = {
            'timeframe': '1',
            'lookback': 50,
            'maker_offset_pct': 0.01,
            'net_take_profit': 1.08,
            'net_stop_loss': 0.42,
            'position_size': 100,
        }
        
        os.makedirs("logs", exist_ok=True)
        self.log_file = "logs/4_FEES_LIQUIDITYSWEEPBOT_DOGEUSDT.log"
        self.unified_logger = UnifiedLogger("4_FEES_LIQUIDITYSWEEP", self.symbol)
        self.current_trade_id = None
    
    def connect(self):
        try:
            self.exchange = HTTP(demo=self.demo_mode, api_key=self.api_key, api_secret=self.api_