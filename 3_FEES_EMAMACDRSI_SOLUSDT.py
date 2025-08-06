import os
import asyncio
import pandas as pd
import json
from datetime import datetime
from pybit.unified_trading import HTTP
from dotenv import load_dotenv

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
        self.trade_id = 0
        
        self.config = {
            'timeframe': '5',
            'ema_short': 12,
            'ema_long': 26,
            'macd_signal': 9,
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_neutral_low': 50,
            'position_size': 100,
            'lookback': 100,
            'maker_offset_pct': 0.01,
            'net_take_profit': 1.08,
            'net_stop_loss': 0.42,
        }
        
        os.makedirs("logs", exist_ok=True)
        self.log_file = "logs/ema_macd_rsi_trades.log"
    
    def connect(self):
        try:
            self.exchange = HTTP(demo=self.demo_mode, api_key=self.api_key, api_secret=self.api_secret)
            return self.exchange.get_server_time().get('retCode') == 0
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
        rsi = 100 - (100 / (1 + gain / loss))
        
        return {
            'price': close.iloc[-1],
            'ema_aligned': ema_short.iloc[-1] > ema_long.iloc[-1],
            'histogram_flip': histogram.iloc[-2] < 0 and histogram.iloc[-1] > 0,
            'histogram_reversal': histogram.iloc[-2] > 0 and histogram.iloc[-1] < 0,
            'rsi': rsi.iloc[-1],
            'rsi_above_50': rsi.iloc[-1] > self.config['rsi_neutral_low']
        }
    
    def generate_signal(self, df):
        analysis = self.calculate_indicators(df)
        if not analysis:
            return None
        
        if analysis['histogram_flip'] and analysis['rsi_above_50']:
            return {'action': 'BUY', 'price': analysis['price'], 'rsi': analysis['rsi']}
        
        if analysis['histogram_reversal']:
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
            
            df = pd.DataFrame(klines['result']['list'], columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
            ])
            
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
        
        analysis = self.calculate_indicators(self.price_data)
        if analysis and analysis['histogram_reversal']:
            return True, "macd_reversal"
        
        return False, ""
    
    async def execute_trade(self, signal):
        qty = self.config['position_size'] / signal['price']
        
        if qty < 1.0:
            return
        
        # LIMIT order for entry
        offset_mult = 1 - self.config['maker_offset_pct']/100 if signal['action'] == 'BUY' else 1 + self.config['maker_offset_pct']/100
        limit_price = round(signal['price'] * offset_mult, 4)
        
        try:
            order = self.exchange.place_order(
                category="linear",
                symbol=self.symbol,
                side="Buy" if signal['action'] == 'BUY' else "Sell",
                orderType="Limit",
                qty=str(int(round(qty))),
                price=str(limit_price),
                timeInForce="PostOnly"
            )
            
            if order.get('retCode') == 0:
                self.trade_id += 1
                print(f"✅ {signal['action']}: {int(qty)} @ ${limit_price:.4f}")
                self.log_trade(signal['action'], limit_price, f"RSI:{signal['rsi']:.1f}")
        except:
            pass
    
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
        except:
            pass
    
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
            print("Failed to connect")
            return
        
        print(f"✅ Starting EMA+MACD+RSI bot")
        print(f"⏰ Timeframe: 5 minutes")
        print(f"🎯 TP: {self.config['net_take_profit']}% | SL: {self.config['net_stop_loss']}%")
        
        while True:
            try:
                await self.run_cycle()
                await asyncio.sleep(5)
            except KeyboardInterrupt:
                print("\nBot stopped")
                if self.position:
                    await self.close_position("manual_stop")
                break
            except Exception as e:
                print(f"Error: {e}")
                await asyncio.sleep(5)

if __name__ == "__main__":
    bot = EMAMACDRSIBot()
    asyncio.run(bot.run())