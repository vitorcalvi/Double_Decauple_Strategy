import os
import asyncio
import pandas as pd
import json
from datetime import datetime
from pybit.unified_trading import HTTP
from dotenv import load_dotenv

load_dotenv()

class MACDVWAPBot:
    def __init__(self):
        self.symbol = 'BNBUSDT'
        self.demo_mode = os.getenv('DEMO_MODE', 'true').lower() == 'true'
        
        prefix = 'TESTNET_' if self.demo_mode else 'LIVE_'
        self.exchange = HTTP(
            demo=self.demo_mode,
            api_key=os.getenv(f'{prefix}BYBIT_API_KEY'),
            api_secret=os.getenv(f'{prefix}BYBIT_API_SECRET')
        )
        
        self.position = None
        self.pending_order = None
        self.last_signal = None
        self.last_order_time = None
        self.trade_id = 0
        
        self.config = {
            'macd_fast': 8,
            'macd_slow': 21,
            'macd_signal': 5,
            'rsi_period': 9,
            'ema_period': 13,
            'rsi_oversold': 30,  # CHANGED from 35 (more room for signals)
            'rsi_overbought': 70,  # CHANGED from 65 (more room for signals)
            'position_size': 100,
            'maker_offset_pct': 0.01,
            'net_take_profit': 0.43,
            'net_stop_loss': 0.17,
            'order_cooldown': 30,
            'order_timeout': 60
        }
        
        os.makedirs("logs", exist_ok=True)
        self.log_file = "logs/7_FEES_PIVOTREVERSAL_ADAUSDT.log"
    
    def calculate_indicators(self, df):
        if len(df) < 50:
            return None
        
        close = df['close']
        
        # MACD
        ema_fast = close.ewm(span=self.config['macd_fast']).mean()
        ema_slow = close.ewm(span=self.config['macd_slow']).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.config['macd_signal']).mean()
        histogram = macd_line - signal_line
        
        # VWAP - fixed for zero volume edge case
        recent = df.tail(min(1440, len(df)))
        typical_price = (recent['high'] + recent['low'] + recent['close']) / 3
        volume = recent['volume']
        
        # Avoid division by zero
        cum_volume = volume.cumsum()
        vwap = ((typical_price * volume).cumsum() / cum_volume).iloc[-1] if cum_volume.iloc[-1] > 0 else close.iloc[-1]
        
        # RSI - fixed for division by zero
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(window=self.config['rsi_period']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.config['rsi_period']).mean()
        
        # Proper epsilon to avoid division by zero and infinity
        epsilon = 1e-10
        rs = gain / (loss + epsilon)
        rsi = 100 - (100 / (1 + rs))
        
        # EMA
        ema = close.ewm(span=self.config['ema_period']).mean()
        
        return {
            'histogram': histogram.iloc[-1],
            'histogram_prev': histogram.iloc[-2],
            'vwap': vwap,
            'rsi': rsi.iloc[-1],
            'rsi_prev': rsi.iloc[-2],
            'ema': ema.iloc[-1],
            'price': close.iloc[-1]
        }
    
    def generate_signal(self, df):
        ind = self.calculate_indicators(df)
        if not ind:
            return None
        
        # Detect crossovers (including exact threshold touches)
        hist_bullish = ind['histogram_prev'] <= 0 < ind['histogram']
        hist_bearish = ind['histogram_prev'] >= 0 > ind['histogram']
        
        rsi_bullish = ind['rsi_prev'] < self.config['rsi_oversold'] <= ind['rsi']
        rsi_bearish = ind['rsi_prev'] > self.config['rsi_overbought'] >= ind['rsi']
        
        vwap_bullish = ind['price'] > ind['vwap'] and ind['ema'] > ind['vwap']
        vwap_bearish = ind['price'] < ind['vwap'] and ind['ema'] < ind['vwap']
        
        # All three conditions must align
        if hist_bullish and rsi_bullish and vwap_bullish:
            return self._create_signal('BUY', ind)
        
        if hist_bearish and rsi_bearish and vwap_bearish:
            return self._create_signal('SELL', ind)
        
        return None
    
    def _create_signal(self, action, ind):
        return {
            'action': action,
            'price': ind['price'],
            'macd_hist': ind['histogram'],
            'rsi': ind['rsi'],
            'vwap': ind['vwap']
        }
    
    async def get_market_data(self):
        try:
            response = self.exchange.get_kline(
                category="linear",
                symbol=self.symbol,
                interval="1",
                limit=100
            )
            
            if response.get('retCode') != 0:
                return None
            
            df = pd.DataFrame(response['result']['list'], columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
            ])
            
            df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].apply(pd.to_numeric)
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
            
            return df.sort_values('timestamp').reset_index(drop=True)
        except:
            return None
    
    async def update_state(self):
        """Update position and order states."""
        try:
            # Check position
            pos_resp = self.exchange.get_positions(category="linear", symbol=self.symbol)
            if pos_resp.get('retCode') == 0:
                pos_list = pos_resp['result']['list']
                self.position = pos_list[0] if pos_list and float(pos_list[0]['size']) > 0 else None
            
            # Check orders
            ord_resp = self.exchange.get_open_orders(category="linear", symbol=self.symbol)
            if ord_resp.get('retCode') == 0:
                orders = ord_resp['result']['list']
                
                if orders:
                    order = orders[0]
                    order_age = (datetime.now().timestamp() - int(order['createdTime']) / 1000)
                    
                    if order_age > self.config['order_timeout']:
                        self.exchange.cancel_order(
                            category="linear",
                            symbol=self.symbol,
                            orderId=order['orderId']
                        )
                        self.pending_order = None
                        self.last_signal = None
                    else:
                        self.pending_order = order
                else:
                    self.pending_order = None
        except:
            pass
    
    def should_close(self):
        if not self.position:
            return False, ""
        
        price = float(self.price_data['close'].iloc[-1])
        entry = float(self.position['avgPrice'])
        is_long = self.position['side'] == "Buy"
        
        # Calculate exact trigger prices to avoid floating point issues
        if is_long:
            tp_price = entry * (1 + self.config['net_take_profit'] / 100)
            sl_price = entry * (1 - self.config['net_stop_loss'] / 100)
            
            if price >= tp_price:
                return True, "take_profit"
            if price <= sl_price:
                return True, "stop_loss"
        else:
            tp_price = entry * (1 - self.config['net_take_profit'] / 100)
            sl_price = entry * (1 + self.config['net_stop_loss'] / 100)
            
            if price <= tp_price:
                return True, "take_profit"
            if price >= sl_price:
                return True, "stop_loss"
        
        return False, ""
    
    def can_trade(self, signal):
        """Check if trade can be executed."""
        # Cooldown check
        if self.last_order_time:
            if (datetime.now() - self.last_order_time).total_seconds() < self.config['order_cooldown']:
                return False
        
        # Duplicate signal check
        if self.last_signal and signal:
            if (self.last_signal['action'] == signal['action'] and 
                abs(self.last_signal['price'] - signal['price']) / signal['price'] < 0.001):
                return False
        
        return not (self.pending_order or self.position)
    
    async def execute_trade(self, signal):
        if not self.can_trade(signal):
            return
        
        qty = int(round(self.config['position_size'] / signal['price']))
        if qty == 0:
            return
        
        offset = self.config['maker_offset_pct'] / 100
        price = round(signal['price'] * (1 - offset if signal['action'] == 'BUY' else 1 + offset), 4)
        
        try:
            order = self.exchange.place_order(
                category="linear",
                symbol=self.symbol,
                side="Buy" if signal['action'] == 'BUY' else "Sell",
                orderType="Limit",
                qty=str(qty),
                price=str(price),
                timeInForce="PostOnly"
            )
            
            if order.get('retCode') == 0:
                self.trade_id += 1
                self.last_order_time = datetime.now()
                self.last_signal = signal
                self.pending_order = order['result']
                
                print(f"✅ {signal['action']}: {qty} @ ${price:.4f}")
                print(f"   📈 MACD:{signal['macd_hist']:.6f} | RSI:{signal['rsi']:.1f} | VWAP:${signal['vwap']:.4f}")
                self.log_trade(signal['action'], price, f"RSI:{signal['rsi']:.1f}")
                
                await asyncio.sleep(5)
        except Exception as e:
            print(f"❌ Trade failed: {e}")
    
    async def close_position(self, reason):
        if not self.position:
            return
        
        side = "Sell" if self.position['side'] == "Buy" else "Buy"
        qty = int(round(float(self.position['size'])))
        
        try:
            order = self.exchange.place_order(
                category="linear",
                symbol=self.symbol,
                side=side,
                orderType="Market",
                qty=str(qty),
                reduceOnly=True
            )
            
            if order.get('retCode') == 0:
                pnl = float(self.position.get('unrealisedPnl', 0))
                print(f"✅ Closed: {reason} | PnL: ${pnl:.2f}")
                self.log_trade("CLOSE", 0, f"{reason}_PnL:${pnl:.2f}")
                self.last_signal = None
        except Exception as e:
            print(f"❌ Close failed: {e}")
    
    def log_trade(self, action, price, info):
        with open(self.log_file, "a") as f:
            f.write(json.dumps({
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'action': action,
                'price': round(price, 6),
                'info': info
            }) + "\n")
    
    def show_status(self):
        print(f"📊 MACD+VWAP Bot - {datetime.now().strftime('%H:%M:%S')}")
        
        if self.position:
            print(f"{'🟢' if self.position['side'] == 'Buy' else '🔴'} Position: {self.position['side']} {float(self.position['size'])} @ ${float(self.position['avgPrice']):.4f}")
            print(f"💰 Unrealized PnL: ${float(self.position.get('unrealisedPnl', 0)):.2f}")
        elif self.pending_order:
            print(f"⏳ Pending: {self.pending_order.get('side')} @ ${self.pending_order.get('price')}")
        else:
            print("🔍 Scanning...")
        
        if self.last_order_time:
            cooldown_left = self.config['order_cooldown'] - (datetime.now() - self.last_order_time).total_seconds()
            if cooldown_left > 0:
                print(f"⏰ Cooldown: {cooldown_left:.0f}s")
        
    
    async def run_cycle(self):
        self.price_data = await self.get_market_data()
        if self.price_data is None:
            return
        
        await self.update_state()
        
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
        try:
            # Test connection
            if self.exchange.get_server_time().get('retCode') != 0:
                print("❌ Failed to connect")
                return
        except:
            print("❌ Connection error")
            return
        
        print(f"✅ Starting MACD + VWAP bot for {self.symbol}")
        print("📊 Strategy: MACD histogram + RSI cross + VWAP alignment")
        print(f"🎯 TP: {self.config['net_take_profit']}% | SL: {self.config['net_stop_loss']}%")
        print(f"⏰ Order cooldown: {self.config['order_cooldown']}s")
        
        while True:
            try:
                await self.run_cycle()
                await asyncio.sleep(5)
            except KeyboardInterrupt:
                print("\n🛑 Stopping...")
                
                if self.pending_order:
                    try:
                        self.exchange.cancel_all_orders(category="linear", symbol=self.symbol)
                    except:
                        pass
                
                if self.position:
                    await self.close_position("manual_stop")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                await asyncio.sleep(10)

if __name__ == "__main__":
    bot = MACDVWAPBot()
    asyncio.run(bot.run())