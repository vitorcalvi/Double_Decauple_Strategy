import os
import asyncio
import pandas as pd
import json
from datetime import datetime, timezone
from pybit.unified_trading import HTTP
from dotenv import load_dotenv

load_dotenv()

class ZigZagTradingBot:
    def __init__(self):
        self.symbol = 'XRPUSDT'
        self.demo_mode = os.getenv('DEMO_MODE', 'true').lower() == 'true'
        prefix = 'TESTNET_' if self.demo_mode else 'LIVE_'
        self.api_key = os.getenv(f'{prefix}BYBIT_API_KEY')
        self.api_secret = os.getenv(f'{prefix}BYBIT_API_SECRET')
        self.exchange = None
        self.position = None
        self.price_data = pd.DataFrame()
        self.trade_id = 0
        self.last_trade_bar = 0
        
        self.config = {
            'timeframe': '3', 'lookback': 100, 'zigzag_pct': 0.5,
            'min_swing_bars': 3, 'stop_loss': 0.5, 'take_profit': 1.0,
            'trailing_stop': 0.4, 'risk_per_trade': 0.05, 'position_size': 100,
            'cooldown_bars': 3, 'maker_offset_pct': 0.01
        }
        
        self.current_capital = 1000
        os.makedirs("logs", exist_ok=True)
        self.log_file = f"logs/zigzag_{datetime.now().strftime('%Y%m%d')}.log"
    
    def connect(self):
        try:
            self.exchange = HTTP(demo=self.demo_mode, api_key=self.api_key, api_secret=self.api_secret)
            return self.exchange.get_server_time().get('retCode') == 0
        except:
            return False
    
    def identify_swings(self):
        if len(self.price_data) < 10:
            return []
        
        df = self.price_data
        swings = []
        
        for i in range(2, len(df) - 2):
            is_high = (df['high'].iloc[i] > max(df['high'].iloc[i-2:i].max(), df['high'].iloc[i+1:i+3].max()))
            is_low = (df['low'].iloc[i] < min(df['low'].iloc[i-2:i].min(), df['low'].iloc[i+1:i+3].min()))
            
            if is_high:
                swings.append({'index': i, 'type': 'HIGH', 'price': df['high'].iloc[i]})
            elif is_low:
                swings.append({'index': i, 'type': 'LOW', 'price': df['low'].iloc[i]})
        
        filtered = []
        for swing in swings:
            if not filtered or (
                abs(swing['price'] - filtered[-1]['price']) / filtered[-1]['price'] * 100 >= self.config['zigzag_pct'] and
                swing['type'] != filtered[-1]['type']
            ):
                filtered.append(swing)
        
        return filtered
    
    def generate_signal(self):
        if len(self.price_data) < 20:
            return None
        
        current_bar = len(self.price_data) - 1
        if current_bar - self.last_trade_bar < self.config['cooldown_bars']:
            return None
        
        swings = self.identify_swings()
        if len(swings) < 3:
            return None
        
        current_price = float(self.price_data['close'].iloc[-1])
        last_swing = swings[-1]
        bars_since = current_bar - last_swing['index']
        
        vol_ratio = self.price_data['volume'].iloc[-3:].mean() / self.price_data['volume'].iloc[-20:].mean()
        if vol_ratio <= 0.8:
            return None
        
        if (last_swing['type'] == 'LOW' and bars_since <= self.config['min_swing_bars'] and
            current_price > last_swing['price'] * 1.001):
            return {'action': 'BUY', 'price': current_price, 'reason': 'swing_low_reversal'}
        
        if (last_swing['type'] == 'HIGH' and bars_since <= self.config['min_swing_bars'] and
            current_price < last_swing['price'] * 0.999):
            return {'action': 'SELL', 'price': current_price, 'reason': 'swing_high_reversal'}
        
        if last_swing['type'] == 'HIGH' and current_price > last_swing['price'] * 1.002:
            return {'action': 'BUY', 'price': current_price, 'reason': 'breakout_high'}
        
        if last_swing['type'] == 'LOW' and current_price < last_swing['price'] * 0.998:
            return {'action': 'SELL', 'price': current_price, 'reason': 'breakout_low'}
        
        return None
    
    async def get_market_data(self):
        try:
            klines = self.exchange.get_kline(
                category="linear", symbol=self.symbol,
                interval=self.config['timeframe'], limit=self.config['lookback']
            )
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
                
                if not pos_list or float(pos_list[0]['size']) == 0:
                    if self.position:
                        pnl = float(self.position.get('realisedPnl', 0))
                        self.current_capital += pnl
                    self.position = None
                else:
                    self.position = pos_list[0]
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
        
        swings = self.identify_swings()
        if swings:
            last_swing = swings[-1]
            if ((side == "Buy" and last_swing['type'] == 'HIGH' and current_price >= entry_price * 1.002) or
                (side == "Sell" and last_swing['type'] == 'LOW' and current_price <= entry_price * 0.998)):
                return True, "swing_exit"
        
        if pnl_pct >= self.config['take_profit']:
            return True, "take_profit"
        if pnl_pct <= -self.config['stop_loss']:
            return True, "stop_loss"
        
        if pnl_pct > self.config['trailing_stop'] and pnl_pct < self.config['trailing_stop'] * 0.5:
            return True, "trailing_stop"
        
        return False, ""
    
    async def execute_order(self, side, qty, price, sl=None, tp=None, reduce_only=False):
        qty_str = f"{round(qty / 0.1) * 0.1:.1f}"
        params = {
            'category': "linear", 'symbol': self.symbol, 'side': side,
            'orderType': "Limit", 'qty': qty_str,
            'price': str(price), 'timeInForce': "PostOnly"
        }
        if sl:
            params['stopLoss'] = str(sl)
        if tp:
            params['takeProfit'] = str(tp)
        if reduce_only:
            params['reduceOnly'] = True
        
        try:
            return self.exchange.place_order(**params).get('retCode') == 0
        except:
            return False
    
    async def execute_trade(self, signal):
        qty = self.config['position_size'] / signal['price']
        if qty < 0.1:
            return
        
        if signal['action'] == 'BUY':
            limit_price = round(signal['price'] * (1 - self.config['maker_offset_pct']/100), 2)
            sl_price = round(signal['price'] * (1 - self.config['stop_loss']/100), 2)
            tp_price = round(signal['price'] * (1 + self.config['take_profit']/100), 2)
        else:
            limit_price = round(signal['price'] * (1 + self.config['maker_offset_pct']/100), 2)
            sl_price = round(signal['price'] * (1 + self.config['stop_loss']/100), 2)
            tp_price = round(signal['price'] * (1 - self.config['take_profit']/100), 2)
        
        if await self.execute_order("Buy" if signal['action'] == 'BUY' else "Sell", qty, limit_price, sl_price, tp_price):
            self.trade_id += 1
            self.last_trade_bar = len(self.price_data) - 1
            print(f"✅ {signal['action']}: {qty:.1f} @ ${limit_price:.2f} | {signal['reason']}")
    
    async def close_position(self, reason):
        if not self.position:
            return
        
        current_price = float(self.price_data['close'].iloc[-1])
        side = "Sell" if self.position.get('side') == "Buy" else "Buy"
        qty = float(self.position['size'])
        
        offset = 1 + self.config['maker_offset_pct']/100 if side == "Sell" else 1 - self.config['maker_offset_pct']/100
        limit_price = round(current_price * offset, 2)
        
        if await self.execute_order(side, qty, limit_price, reduce_only=True):
            pnl = float(self.position.get('unrealisedPnl', 0))
            print(f"✅ Closed: {reason} | PnL: ${pnl:.2f}")
    
    def show_status(self):
        if self.price_data.empty:
            return
        
        price = float(self.price_data['close'].iloc[-1])
        swings = self.identify_swings()
        
        print(f"\n📈 {self.symbol}: ${price:.2f} | Capital: ${self.current_capital:.2f}")
        
        if swings:
            last = swings[-1]
            print(f"🔄 Last Swing: {last['type']} @ ${last['price']:.2f}")
        
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
        elif signal := self.generate_signal():
            await self.execute_trade(signal)
        
        self.show_status()
    
    async def run(self):
        if not self.connect():
            print("Failed to connect")
            return
        
        print(f"🚀 ZigZag Bot | {self.symbol} | TP: {self.config['take_profit']}% | SL: {self.config['stop_loss']}%")
        
        while True:
            try:
                await self.run_cycle()
                await asyncio.sleep(10)
            except KeyboardInterrupt:
                if self.position:
                    await self.close_position("manual_stop")
                break
            except:
                await asyncio.sleep(10)

if __name__ == "__main__":
    asyncio.run(ZigZagTradingBot().run())