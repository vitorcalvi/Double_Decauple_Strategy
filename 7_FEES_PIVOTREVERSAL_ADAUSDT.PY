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
        self.api_key = os.getenv(f'{prefix}BYBIT_API_KEY')
        self.api_secret = os.getenv(f'{prefix}BYBIT_API_SECRET')
        self.exchange = None
        
        self.position = None
        self.price_data = pd.DataFrame()
        self.trade_id = 0
        
        self.config = {
            'macd_fast': 8,
            'macd_slow': 21,
            'macd_signal': 5,
            'rsi_period': 9,
            'ema_period': 13,
            'rsi_oversold': 35,
            'rsi_overbought': 65,
            'position_size': 100,
            'maker_offset_pct': 0.01,
            'net_take_profit': 0.43,
            'net_stop_loss': 0.17,
        }
        
        os.makedirs("logs", exist_ok=True)
        self.log_file = "logs/macd_vwap_trades.log"
    
    def connect(self):
        try:
            self.exchange = HTTP(demo=self.demo_mode, api_key=self.api_key, api_secret=self.api_secret)
            return self.exchange.get_server_time().get('retCode') == 0
        except:
            return False
    
    def calculate_vwap(self, df):
        if len(df) < 20:
            return None
        
        recent_data = df.tail(min(1440, len(df)))
        typical_price = (recent_data['high'] + recent_data['low'] + recent_data['close']) / 3
        volume = recent_data['volume']
        
        vwap = (typical_price * volume).cumsum() / volume.cumsum()
        return vwap.iloc[-1] if not vwap.empty else None
    
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
        
        # VWAP
        vwap = self.calculate_vwap(df)
        if not vwap:
            return None
        
        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(window=self.config['rsi_period']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.config['rsi_period']).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # EMA
        ema = close.ewm(span=self.config['ema_period']).mean()
        
        return {
            'histogram': histogram.iloc[-1],
            'histogram_prev': histogram.iloc[-2] if len(histogram) > 1 else histogram.iloc[-1],
            'vwap': vwap,
            'rsi': rsi.iloc[-1],
            'rsi_prev': rsi.iloc[-2] if len(rsi) > 1 else rsi.iloc[-1],
            'ema': ema.iloc[-1],
            'price': close.iloc[-1]
        }
    
    def generate_signal(self, df):
        indicators = self.calculate_indicators(df)
        if not indicators:
            return None
        
        # Detect signals
        hist_bullish = indicators['histogram_prev'] <= 0 and indicators['histogram'] > 0
        hist_bearish = indicators['histogram_prev'] >= 0 and indicators['histogram'] < 0
        
        rsi_bullish = indicators['rsi_prev'] <= self.config['rsi_oversold'] and indicators['rsi'] > self.config['rsi_oversold']
        rsi_bearish = indicators['rsi_prev'] >= self.config['rsi_overbought'] and indicators['rsi'] < self.config['rsi_overbought']
        
        vwap_bullish = indicators['price'] > indicators['vwap'] and indicators['ema'] > indicators['vwap']
        vwap_bearish = indicators['price'] < indicators['vwap'] and indicators['ema'] < indicators['vwap']
        
        # Bullish signal: All three align
        if hist_bullish and rsi_bullish and vwap_bullish:
            return {
                'action': 'BUY',
                'price': indicators['price'],
                'macd_hist': indicators['histogram'],
                'rsi': indicators['rsi'],
                'vwap': indicators['vwap']
            }
        
        # Bearish signal: All three align
        if hist_bearish and rsi_bearish and vwap_bearish:
            return {
                'action': 'SELL',
                'price': indicators['price'],
                'macd_hist': indicators['histogram'],
                'rsi': indicators['rsi'],
                'vwap': indicators['vwap']
            }
        
        return None
    
    async def get_market_data(self):
        try:
            klines = self.exchange.get_kline(
                category="linear",
                symbol=self.symbol,
                interval="1",
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
    
    async def execute_trade(self, signal):
        qty = self.config['position_size'] / signal['price']
        formatted_qty = str(int(round(qty))) if qty >= 1 else "0"
        
        if formatted_qty == "0":
            return
        
        # LIMIT order for entry
        limit_price = round(signal['price'] * (1 - self.config['maker_offset_pct']/100 if signal['action'] == 'BUY' else 1 + self.config['maker_offset_pct']/100), 4)
        
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
                self.trade_id += 1
                print(f"✅ {signal['action']}: {formatted_qty} @ ${limit_price:.4f}")
                print(f"   📈 MACD Hist:{signal['macd_hist']:.6f} | RSI:{signal['rsi']:.1f} | VWAP:${signal['vwap']:.4f}")
                self.log_trade(signal['action'], limit_price, f"RSI:{signal['rsi']:.1f}")
        except Exception as e:
            print(f"❌ Trade failed: {e}")
    
    async def close_position(self, reason):
        if not self.position:
            return
        
        side = "Sell" if self.position.get('side') == "Buy" else "Buy"
        qty = float(self.position['size'])
        
        # MARKET order for exit
        try:
            order = self.exchange.place_order(
                category="linear",
                symbol=self.symbol,
                side=side,
                orderType="Market",
                qty=str(int(round(qty))),
                reduceOnly=True
            )
            
            if order.get('retCode') == 0:
                pnl = float(self.position.get('unrealisedPnl', 0))
                print(f"✅ Closed: {reason} | PnL: ${pnl:.2f}")
                self.log_trade("CLOSE", 0, f"{reason}_PnL:${pnl:.2f}")
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
    
    async def run_cycle(self):
        if not await self.get_market_data():
            return
        
        await self.check_position()
        
        if self.position:
            should_close, reason = self.should_close()
            if should_close:
                await self.close_position(reason)
        elif signal := self.generate_signal(self.price_data):
            await self.execute_trade(signal)
    
    async def run(self):
        if not self.connect():
            print("❌ Failed to connect")
            return
        
        print(f"✅ Starting MACD + VWAP bot for {self.symbol}")
        print("📊 Strategy: MACD histogram flip + RSI cross + VWAP alignment")
        print(f"🎯 TP: {self.config['net_take_profit']}% | SL: {self.config['net_stop_loss']}%")
        
        while True:
            try:
                await self.run_cycle()
                await asyncio.sleep(1)
            except KeyboardInterrupt:
                print("\n🛑 Bot stopped")
                if self.position:
                    await self.close_position("manual_stop")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                await asyncio.sleep(5)

if __name__ == "__main__":
    bot = MACDVWAPBot()
    asyncio.run(bot.run())