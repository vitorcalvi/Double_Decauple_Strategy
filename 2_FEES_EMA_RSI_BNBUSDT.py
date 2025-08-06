import os
import asyncio
import pandas as pd
import json
from datetime import datetime
from pybit.unified_trading import HTTP
from dotenv import load_dotenv

load_dotenv()

class EMARSIBot:
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
            'ema_fast': 5, 'ema_slow': 13, 'rsi_period': 5,
            'rsi_oversold': 25, 'rsi_overbought': 75, 'volume_threshold': 1.5,
            'take_profit': 0.35, 'stop_loss': 0.20, 'position_size': 100,
            'min_rsi_diff': 2.0, 'maker_offset_pct': 0.01
        }
        
        os.makedirs("logs", exist_ok=True)
        self.log_file = "logs/ema_rsi_trades.log"
    
    def connect(self):
        try:
            self.exchange = HTTP(demo=self.demo_mode, api_key=self.api_key, api_secret=self.api_secret)
            return self.exchange.get_server_time().get('retCode') == 0
        except:
            return False
    
    def calculate_indicators(self, df):
        if len(df) < max(self.config['ema_slow'], self.config['rsi_period']) + 1:
            return None
        
        close = df['close']
        ema_fast = close.ewm(span=self.config['ema_fast']).mean().iloc[-1]
        ema_slow = close.ewm(span=self.config['ema_slow']).mean().iloc[-1]
        
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(self.config['rsi_period']).mean()
        loss = -delta.clip(upper=0).rolling(self.config['rsi_period']).mean()
        rsi = 100 - (100 / (1 + gain / loss))
        
        vol_ratio = df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]
        
        return {
            'ema_fast': ema_fast, 'ema_slow': ema_slow,
            'rsi': rsi.iloc[-1], 'rsi_prev': rsi.iloc[-2],
            'volume_ratio': vol_ratio, 'trend': 'UP' if ema_fast > ema_slow else 'DOWN'
        }
    
    def generate_signal(self, df):
        ind = self.calculate_indicators(df)
        if not ind or ind['volume_ratio'] < self.config['volume_threshold']:
            return None
        
        if abs(ind['rsi'] - ind['rsi_prev']) < self.config['min_rsi_diff']:
            return None
        
        current_price = float(df['close'].iloc[-1])
        
        if (ind['trend'] == 'UP' and ind['rsi_prev'] <= self.config['rsi_oversold'] and 
            ind['rsi'] > ind['rsi_prev'] and current_price > ind['ema_fast']):
            return {'action': 'BUY', 'price': current_price, 'rsi': ind['rsi'], 
                   'trend': ind['trend'], 'volume_ratio': ind['volume_ratio']}
        
        if (ind['trend'] == 'DOWN' and ind['rsi_prev'] >= self.config['rsi_overbought'] and 
            ind['rsi'] < ind['rsi_prev'] and current_price < ind['ema_fast']):
            return {'action': 'SELL', 'price': current_price, 'rsi': ind['rsi'],
                   'trend': ind['trend'], 'volume_ratio': ind['volume_ratio']}
        
        return None
    
    async def get_market_data(self):
        try:
            klines = self.exchange.get_kline(category="linear", symbol=self.symbol, interval="1", limit=50)
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
    
    async def place_order(self, side, qty, price, reduce_only=False):
        params = {
            'category': "linear", 'symbol': self.symbol, 'side': side,
            'orderType': "Limit", 'qty': f"{round(qty / 0.001) * 0.001:.3f}",
            'price': str(price), 'timeInForce': "PostOnly"
        }
        if reduce_only:
            params['reduceOnly'] = True
        try:
            return self.exchange.place_order(**params)
        except:
            return {'retCode': -1}
    
    async def execute_trade(self, signal):
        qty = self.config['position_size'] / signal['price']
        if qty < 0.001:
            return
        
        offset = 1 - self.config['maker_offset_pct']/100 if signal['action'] == 'BUY' else 1 + self.config['maker_offset_pct']/100
        limit_price = round(signal['price'] * offset, 2)
        
        order = await self.place_order("Buy" if signal['action'] == 'BUY' else "Sell", qty, limit_price)
        if order.get('retCode') == 0:
            self.trade_id += 1
            print(f"✅ {signal['action']}: {qty:.3f} BTC @ ${limit_price:,.0f} | RSI:{signal['rsi']:.1f}")
    
    async def close_position(self, reason):
        if not self.position:
            return
        
        current_price = float(self.price_data['close'].iloc[-1])
        side = "Sell" if self.position.get('side') == "Buy" else "Buy"
        qty = float(self.position['size'])
        
        offset = 1 + self.config['maker_offset_pct']/100 if side == "Sell" else 1 - self.config['maker_offset_pct']/100
        limit_price = round(current_price * offset, 2)
        
        order = await self.place_order(side, qty, limit_price, True)
        if order.get('retCode') == 0:
            pnl = float(self.position.get('unrealisedPnl', 0))
            print(f"✅ Closed: {reason} | PnL: ${pnl:.2f}")
    
    def show_status(self):
        if self.price_data.empty:
            return
        
        ind = self.calculate_indicators(self.price_data)
        if ind:
            print(f"\n📊 {self.symbol}: ${float(self.price_data['close'].iloc[-1]):,.0f} | RSI: {ind['rsi']:.1f}")
        
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
        
        print(f"🚀 EMA+RSI Bot | {self.symbol} | TP: {self.config['take_profit']}% | SL: {self.config['stop_loss']}%")
        
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
    asyncio.run(EMARSIBot().run())