import os
import asyncio
import pandas as pd
import json
from datetime import datetime
from pybit.unified_trading import HTTP
from dotenv import load_dotenv

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
        self.trade_id = 0
        
        self.config = {
            'pivot_period': 5,
            'rsi_period': 7,
            'mfi_period': 7,
            'rsi_oversold': 35,
            'rsi_overbought': 65,
            'mfi_oversold': 25,
            'mfi_overbought': 75,
            'position_size': 100,
            'maker_offset_pct': 0.01,
            'net_take_profit': 0.68,
            'net_stop_loss': 0.07,
        }
        
        self.support_levels = []
        self.resistance_levels = []
        
        os.makedirs("logs", exist_ok=True)
        self.log_file = "logs/5_FEES_MACD_VWAP_BTCUSDT.log"
    
    def connect(self):
        try:
            self.exchange = HTTP(demo=self.demo_mode, api_key=self.api_key, api_secret=self.api_secret)
            return self.exchange.get_server_time().get('retCode') == 0
        except:
            return False
    
    def format_qty(self, qty):
        # ADAUSDT uses integer quantities (no decimals)
        return str(int(round(qty)))
    
    def calculate_indicators(self, df):
        min_len = max(self.config['rsi_period'], self.config['mfi_period']) + 1
        if len(df) < min_len:
            return None
        
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(window=self.config['rsi_period']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.config['rsi_period']).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # MFI
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        
        positive_mf = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window=self.config['mfi_period']).sum()
        negative_mf = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window=self.config['mfi_period']).sum()
        
        mfi = 100 - (100 / (1 + positive_mf / negative_mf))
        
        return {
            'rsi': rsi.iloc[-1] if not rsi.empty else 50,
            'mfi': mfi.iloc[-1] if not mfi.empty else 50,
            'rsi_prev': rsi.iloc[-2] if len(rsi) > 1 else 50
        }
    
    def generate_signal(self, df):
        if len(df) < 30:
            return None
        
        current_price = float(df['close'].iloc[-1])
        
        indicators = self.calculate_indicators(df)
        if not indicators:
            return None
        
        rsi = indicators['rsi']
        mfi = indicators['mfi']
        rsi_prev = indicators['rsi_prev']
        
        # BUY Signal
        if mfi <= self.config['mfi_oversold'] or (rsi <= self.config['rsi_oversold'] and rsi > rsi_prev):
            return {'action': 'BUY', 'price': current_price, 'rsi': rsi, 'mfi': mfi}
        
        # SELL Signal
        if mfi >= self.config['mfi_overbought'] or (rsi >= self.config['rsi_overbought'] and rsi < rsi_prev):
            return {'action': 'SELL', 'price': current_price, 'rsi': rsi, 'mfi': mfi}
        
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
    
    def should_close(self, signal=None):
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
        
        # Exit on opposite signal
        if signal and ((side == "Buy" and signal['action'] == 'SELL') or 
                      (side == "Sell" and signal['action'] == 'BUY')):
            return True, "signal_reversal"
        
        return False, ""
    
    async def execute_trade(self, signal):
        qty = self.config['position_size'] / signal['price']
        formatted_qty = self.format_qty(qty)
        
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
                print(f"✅ {signal['action']}: {formatted_qty} @ ${limit_price:.2f}")
                print(f"   📈 RSI:{signal['rsi']:.1f} MFI:{signal['mfi']:.1f}")
                self.log_trade(signal['action'], limit_price, f"RSI:{signal['rsi']:.1f}_MFI:{signal['mfi']:.1f}")
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
                qty=self.format_qty(qty),
                reduceOnly=True
            )
            
            if order.get('retCode') == 0:
                pnl = float(self.position.get('unrealisedPnl', 0))
                print(f"✅ Closed: {reason} | PnL: ${pnl:.2f}")
                self.log_trade("CLOSE", 0, f"{reason}_PnL:${pnl:.2f}")
                self.position = None
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
        
        signal = self.generate_signal(self.price_data)
        
        if self.position:
            should_close, reason = self.should_close(signal)
            if should_close:
                await self.close_position(reason)
        elif signal:
            await self.execute_trade(signal)
    
    async def run(self):
        if not self.connect():
            print("❌ Failed to connect")
            return
        
        print(f"✅ Starting Pivot Reversal bot for {self.symbol}")
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
    bot = PivotReversalBot()
    asyncio.run(bot.run())