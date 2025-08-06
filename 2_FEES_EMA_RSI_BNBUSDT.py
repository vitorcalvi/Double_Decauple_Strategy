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
            'ema_fast': 5,
            'ema_slow': 13,
            'rsi_period': 5,
            'rsi_oversold': 25,
            'rsi_overbought': 75,
            'volume_threshold': 1.5,
            'position_size': 100,
            'min_rsi_diff': 2.0,
            'maker_offset_pct': 0.01,
            'net_take_profit': 0.43,
            'net_stop_loss': 0.12,
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
        required_len = max(self.config['ema_slow'], self.config['rsi_period']) + 1
        if len(df) < required_len:
            return None
        
        close = df['close']
        volume = df['volume']
        
        ema_fast = close.ewm(span=self.config['ema_fast']).mean()
        ema_slow = close.ewm(span=self.config['ema_slow']).mean()
        
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(window=self.config['rsi_period']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.config['rsi_period']).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        vol_avg = volume.rolling(window=20).mean()
        volume_ratio = volume.iloc[-1] / vol_avg.iloc[-1] if vol_avg.iloc[-1] > 0 else 0
        
        return {
            'ema_fast': ema_fast.iloc[-1],
            'ema_slow': ema_slow.iloc[-1],
            'rsi': rsi.iloc[-1],
            'rsi_prev': rsi.iloc[-2] if len(rsi) > 1 else rsi.iloc[-1],
            'volume_ratio': volume_ratio,
            'trend': 'UP' if ema_fast.iloc[-1] > ema_slow.iloc[-1] else 'DOWN'
        }
    
    def generate_signal(self, df):
        indicators = self.calculate_indicators(df)
        if not indicators:
            return None
        
        if indicators['volume_ratio'] < self.config['volume_threshold']:
            return None
        
        rsi_diff = abs(indicators['rsi'] - indicators['rsi_prev'])
        if rsi_diff < self.config['min_rsi_diff']:
            return None
        
        current_price = float(df['close'].iloc[-1])
        
        # Bullish signal
        if (indicators['trend'] == 'UP' and
            indicators['rsi_prev'] <= self.config['rsi_oversold'] and
            indicators['rsi'] > indicators['rsi_prev'] and
            current_price > indicators['ema_fast']):
            return {'action': 'BUY', 'price': current_price, 'rsi': indicators['rsi']}
        
        # Bearish signal
        if (indicators['trend'] == 'DOWN' and
            indicators['rsi_prev'] >= self.config['rsi_overbought'] and
            indicators['rsi'] < indicators['rsi_prev'] and
            current_price < indicators['ema_fast']):
            return {'action': 'SELL', 'price': current_price, 'rsi': indicators['rsi']}
        
        return None
    
    async def get_market_data(self):
        try:
            klines = self.exchange.get_kline(
                category="linear",
                symbol=self.symbol,
                interval="1",
                limit=50
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
        formatted_qty = f"{round(qty / 0.001) * 0.001:.3f}" if qty >= 0.001 else "0"
        
        if formatted_qty == "0":
            return
        
        # LIMIT order for entry
        offset_mult = 1 - self.config['maker_offset_pct']/100 if signal['action'] == 'BUY' else 1 + self.config['maker_offset_pct']/100
        limit_price = round(signal['price'] * offset_mult, 2)
        
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
                print(f"✅ {signal['action']}: {formatted_qty} @ ${limit_price:,.0f}")
                self.log_trade(signal['action'], limit_price, f"RSI:{signal['rsi']:.1f}")
        except Exception as e:
            print(f"❌ Trade failed: {e}")
    
    async def close_position(self, reason):
        if not self.position:
            return
        
        side = "Sell" if self.position.get('side') == "Buy" else "Buy"
        qty = float(self.position['size'])
        formatted_qty = f"{round(qty / 0.001) * 0.001:.3f}"
        
        # MARKET order for exit
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
                'price': round(price, 2),
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
        
        print(f"✅ Starting EMA + RSI bot for {self.symbol}")
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
    bot = EMARSIBot()
    asyncio.run(bot.run())