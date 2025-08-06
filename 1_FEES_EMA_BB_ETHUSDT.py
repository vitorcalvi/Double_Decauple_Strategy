import os
import asyncio
import pandas as pd
import json
from datetime import datetime
from pybit.unified_trading import HTTP
from dotenv import load_dotenv

load_dotenv()

class DOGEScalpingBot:
    def __init__(self):
        self.symbol = 'ETHUSDT'
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
            'bb_period': 20,
            'bb_std': 2.0,
            'min_spread': 0.05,
            'volume_spike': 0.5,
            'position_size': 100,
            'maker_offset_pct': 0.01,
            'maker_fee_pct': -0.04,
            'net_take_profit': 0.28,
            'net_stop_loss': 0.07,
        }
        
        os.makedirs("logs", exist_ok=True)
        self.log_file = "logs/1_FEES_EMA_BB_ETHUSDT.log"
    
    def connect(self):
        try:
            self.exchange = HTTP(demo=self.demo_mode, api_key=self.api_key, api_secret=self.api_secret)
            return self.exchange.get_server_time().get('retCode') == 0
        except:
            return False
    
    def calculate_indicators(self, df):
        if len(df) < 20:
            return None
        
        close = df['close']
        volume = df['volume']
        
        ema_fast = close.ewm(span=self.config['ema_fast']).mean()
        ema_slow = close.ewm(span=self.config['ema_slow']).mean()
        
        sma = close.rolling(window=self.config['bb_period']).mean()
        std = close.rolling(window=self.config['bb_period']).std()
        upper_band = sma + (std * self.config['bb_std'])
        lower_band = sma - (std * self.config['bb_std'])
        
        vol_ma = volume.rolling(window=10).mean()
        volume_ratio = volume.iloc[-1] / vol_ma.iloc[-1] if vol_ma.iloc[-1] > 0 else 0
        
        current_price = close.iloc[-1]
        price_momentum = (ema_fast.iloc[-1] - ema_slow.iloc[-1]) / ema_slow.iloc[-1] * 100
        bb_position = (current_price - lower_band.iloc[-1]) / (upper_band.iloc[-1] - lower_band.iloc[-1]) if upper_band.iloc[-1] != lower_band.iloc[-1] else 0.5
        spread = (df['high'].iloc[-1] - df['low'].iloc[-1]) / df['low'].iloc[-1] * 100
        
        return {
            'price': current_price,
            'price_momentum': price_momentum,
            'volume_ratio': volume_ratio,
            'bb_position': bb_position,
            'spread': spread
        }
    
    def generate_signal(self, df):
        indicators = self.calculate_indicators(df)
        if not indicators:
            return None
        
        # Long signal
        if (indicators['price_momentum'] > 0.1 and
            indicators['volume_ratio'] > self.config['volume_spike'] and
            indicators['bb_position'] < 0.3 and
            indicators['spread'] > self.config['min_spread']):
            return {'action': 'BUY', 'price': indicators['price'], 'momentum': indicators['price_momentum']}
        
        # Short signal
        if (indicators['price_momentum'] < -0.1 and
            indicators['volume_ratio'] > self.config['volume_spike'] and
            indicators['bb_position'] > 0.7 and
            indicators['spread'] > self.config['min_spread']):
            return {'action': 'SELL', 'price': indicators['price'], 'momentum': indicators['price_momentum']}
        
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
            net_tp = entry_price * (1 + self.config['net_take_profit'] / 100)
            net_sl = entry_price * (1 - self.config['net_stop_loss'] / 100)
            if current_price >= net_tp:
                return True, "take_profit"
            if current_price <= net_sl:
                return True, "stop_loss"
        else:
            net_tp = entry_price * (1 - self.config['net_take_profit'] / 100)
            net_sl = entry_price * (1 + self.config['net_stop_loss'] / 100)
            if current_price <= net_tp:
                return True, "take_profit"
            if current_price >= net_sl:
                return True, "stop_loss"
        
        return False, ""
    
    async def execute_trade(self, signal):
        qty = self.config['position_size'] / signal['price']
        
        # ETHUSDT uses 0.001 minimum
        if 'ETH' in self.symbol:
            formatted_qty = f"{round(qty / 0.001) * 0.001:.3f}"
            if float(formatted_qty) < 0.001:
                return
        else:
            formatted_qty = str(int(round(qty)))
            if int(formatted_qty) == 0:
                return
        
        # LIMIT order for entry (maker)
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
                self.log_trade(signal['action'], limit_price, f"momentum:{signal['momentum']:.2f}")
        except Exception as e:
            print(f"❌ Trade failed: {e}")
    
    async def close_position(self, reason):
        if not self.position:
            return
        
        side = "Sell" if self.position.get('side') == "Buy" else "Buy"
        qty = float(self.position['size'])
        
        # MARKET order for exit (immediate execution)
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
                gross_pnl = float(self.position.get('unrealisedPnl', 0))
                print(f"✅ Closed: {reason} | PnL: ${gross_pnl:.2f}")
                self.log_trade("CLOSE", 0, f"{reason}_PnL:${gross_pnl:.2f}")
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
        
        print(f"✅ Starting bot for {self.symbol}")
        print(f"📊 Strategy: EMA + Bollinger Bands")
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
    bot = DOGEScalpingBot()
    asyncio.run(bot.run())