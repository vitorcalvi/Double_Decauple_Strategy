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
        self.symbol = 'BTCUSDT'
        self.demo_mode = os.getenv('DEMO_MODE', 'true').lower() == 'true'
        prefix = 'TESTNET_' if self.demo_mode else 'LIVE_'
        self.api_key = os.getenv(f'{prefix}BYBIT_API_KEY')
        self.api_secret = os.getenv(f'{prefix}BYBIT_API_SECRET')
        self.exchange = None
        self.position = None
        self.price_data = pd.DataFrame()
        self.trade_id = 0
        
        self.config = {
            'macd_fast': 8, 'macd_slow': 21, 'macd_signal': 5,
            'rsi_period': 9, 'ema_period': 13, 'rsi_oversold': 35,
            'rsi_overbought': 65, 'take_profit': 0.35, 'stop_loss': 0.25,
            'position_size': 100, 'maker_offset_pct': 0.01
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
        
        recent = df.tail(min(1440, len(df)))
        typical_price = (recent['high'] + recent['low'] + recent['close']) / 3
        vwap = (typical_price * recent['volume']).cumsum() / recent['volume'].cumsum()
        return vwap.iloc[-1] if not vwap.empty else None
    
    def calculate_indicators(self, df):
        if len(df) < 50:
            return None
        
        close = df['close']
        
        ema_fast = close.ewm(span=self.config['macd_fast']).mean()
        ema_slow = close.ewm(span=self.config['macd_slow']).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.config['macd_signal']).mean()
        histogram = macd_line - signal_line
        
        vwap = self.calculate_vwap(df)
        if not vwap:
            return None
        
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(self.config['rsi_period']).mean()
        loss = -delta.clip(upper=0).rolling(self.config['rsi_period']).mean()
        rsi = 100 - (100 / (1 + gain / loss))
        
        ema = close.ewm(span=self.config['ema_period']).mean()
        
        return {
            'histogram': histogram.iloc[-1], 'histogram_prev': histogram.iloc[-2],
            'vwap': vwap, 'rsi': rsi.iloc[-1], 'rsi_prev': rsi.iloc[-2],
            'ema': ema.iloc[-1], 'price': close.iloc[-1]
        }
    
    def generate_signal(self, df):
        ind = self.calculate_indicators(df)
        if not ind:
            return None
        
        hist_bullish = ind['histogram_prev'] <= 0 and ind['histogram'] > 0
        hist_bearish = ind['histogram_prev'] >= 0 and ind['histogram'] < 0
        
        rsi_bullish = ind['rsi_prev'] <= self.config['rsi_oversold'] and ind['rsi'] > self.config['rsi_oversold']
        rsi_bearish = ind['rsi_prev'] >= self.config['rsi_overbought'] and ind['rsi'] < self.config['rsi_overbought']
        
        vwap_bullish = ind['price'] > ind['vwap'] and ind['ema'] > ind['vwap']
        vwap_bearish = ind['price'] < ind['vwap'] and ind['ema'] < ind['vwap']
        
        if hist_bullish and rsi_bullish and vwap_bullish:
            return {'action': 'BUY', 'price': ind['price'], 'macd_hist': ind['histogram'],
                   'rsi': ind['rsi'], 'vwap': ind['vwap']}
        
        if hist_bearish and rsi_bearish and vwap_bearish:
            return {'action': 'SELL', 'price': ind['price'], 'macd_hist': ind['histogram'],
                   'rsi': ind['rsi'], 'vwap': ind['vwap']}
        
        return None
    
    async def get_market_data(self):
        try:
            klines = self.exchange.get_kline(category="linear", symbol=self.symbol, interval="1", limit=100)
            if klines.get('retCode') != 0:
                return False
            
            df = pd.DataFrame(klines['result']['list'],
                            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
            df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
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
        if entry_price == 0:
            return False, ""
        
        side = self.position.get('side', '')
        pnl_pct = ((current_price - entry_price) / entry_price * 100) * (1 if side == "Buy" else -1)
        
        if pnl_pct >= self.config['take_profit']:
            return True, "take_profit"
        if pnl_pct <= -self.config['stop_loss']:
            return True, "stop_loss"
        
        return False, ""
    
    async def execute_order(self, side, qty, price, reduce_only=False):
        params = {
            'category': "linear", 'symbol': self.symbol, 'side': side,
            'orderType': "Limit", 'qty': str(int(qty)),
            'price': str(price), 'timeInForce': "PostOnly"
        }
        if reduce_only:
            params['reduceOnly'] = True
        
        try:
            return self.exchange.place_order(**params).get('retCode') == 0
        except:
            return False
    
    async def execute_trade(self, signal):
        qty = self.config['position_size'] / signal['price']
        if qty < 1:
            return
        
        offset = 1 - self.config['maker_offset_pct']/100 if signal['action'] == 'BUY' else 1 + self.config['maker_offset_pct']/100
        limit_price = round(signal['price'] * offset, 4)
        
        if await self.execute_order("Buy" if signal['action'] == 'BUY' else "Sell", qty, limit_price):
            self.trade_id += 1
            print(f"✅ {signal['action']}: {int(qty)} DOGE @ ${limit_price:.4f}")
            print(f"   MACD: {signal['macd_hist']:.6f} | RSI: {signal['rsi']:.1f} | VWAP: ${signal['vwap']:.4f}")
    
    async def close_position(self, reason):
        if not self.position:
            return
        
        current_price = float(self.price_data['close'].iloc[-1])
        side = "Sell" if self.position.get('side') == "Buy" else "Buy"
        qty = float(self.position['size'])
        
        offset = 1 + self.config['maker_offset_pct']/100 if side == "Sell" else 1 - self.config['maker_offset_pct']/100
        limit_price = round(current_price * offset, 4)
        
        if await self.execute_order(side, qty, limit_price, True):
            pnl = float(self.position.get('unrealisedPnl', 0))
            print(f"✅ Closed: {reason} | PnL: ${pnl:.2f}")
    
    def show_status(self):
        if self.price_data.empty:
            return
        
        ind = self.calculate_indicators(self.price_data)
        if ind:
            print(f"\n⚡ {self.symbol}: ${ind['price']:.4f}")
            print(f"📊 MACD: {ind['histogram']:.6f} | RSI: {ind['rsi']:.1f} | VWAP: ${ind['vwap']:.4f}")
        
        if self.position:
            pnl = float(self.position.get('unrealisedPnl', 0))
            print(f"{self.position.get('side')}: ${pnl:.2f}")
        print("-" * 40)
    
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
        
        self.show_status()
    
    async def run(self):
        if not self.connect():
            print("Failed to connect")
            return
        
        print(f"✅ MACD+VWAP Bot | {self.symbol} | TP: {self.config['take_profit']}% | SL: {self.config['stop_loss']}%")
        
        while True:
            try:
                await self.run_cycle()
                await asyncio.sleep(1)
            except KeyboardInterrupt:
                if self.position:
                    await self.close_position("manual_stop")
                break
            except:
                await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(MACDVWAPBot().run())