import os
import asyncio
import pandas as pd
import json
from datetime import datetime
from collections import deque
from pybit.unified_trading import HTTP
from dotenv import load_dotenv

load_dotenv()

class LiquiditySweepBot:
    def __init__(self):
        self.symbol = 'DOGEUSDT'
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
            'liquidity_lookback': 50,
            'order_block_lookback': 20,
            'sweep_threshold': 0.15,
            'position_size': 100,
            'lookback': 100,
            'maker_offset_pct': 0.01,
            'net_take_profit': 1.58,
            'net_stop_loss': 0.42,
        }
        
        self.liquidity_pools = {'highs': deque(maxlen=10), 'lows': deque(maxlen=10)}
        self.order_blocks = []
        
        os.makedirs("logs", exist_ok=True)
        self.log_file = "logs/liquidity_sweep_trades.log"
    
    def connect(self):
        try:
            self.exchange = HTTP(demo=self.demo_mode, api_key=self.api_key, api_secret=self.api_secret)
            return self.exchange.get_server_time().get('retCode') == 0
        except:
            return False
    
    def identify_liquidity_pools(self, df):
        if len(df) < self.config['liquidity_lookback']:
            return
        
        window = 5
        highs = df['high'].rolling(window=window, center=True).max()
        lows = df['low'].rolling(window=window, center=True).min()
        
        self.liquidity_pools['highs'].clear()
        self.liquidity_pools['lows'].clear()
        
        for i in range(len(df) - 10, max(0, len(df) - self.config['liquidity_lookback']), -1):
            # Check significant high
            if df['high'].iloc[i] == highs.iloc[i]:
                is_significant = (
                    all(df['high'].iloc[max(0, i-3):i] < df['high'].iloc[i]) and
                    all(df['high'].iloc[i+1:min(len(df), i+4)] < df['high'].iloc[i])
                )
                if is_significant:
                    self.liquidity_pools['highs'].append({
                        'price': df['high'].iloc[i],
                        'index': i,
                        'volume': df['volume'].iloc[i]
                    })
            
            # Check significant low
            if df['low'].iloc[i] == lows.iloc[i]:
                is_significant = (
                    all(df['low'].iloc[max(0, i-3):i] > df['low'].iloc[i]) and
                    all(df['low'].iloc[i+1:min(len(df), i+4)] > df['low'].iloc[i])
                )
                if is_significant:
                    self.liquidity_pools['lows'].append({
                        'price': df['low'].iloc[i],
                        'index': i,
                        'volume': df['volume'].iloc[i]
                    })
    
    def identify_order_blocks(self, df):
        if len(df) < self.config['order_block_lookback']:
            return []
        
        blocks = []
        
        for i in range(len(df) - 3, max(0, len(df) - self.config['order_block_lookback']), -1):
            # Bullish order block
            if (df['close'].iloc[i] < df['open'].iloc[i] and
                df['close'].iloc[i+1] > df['open'].iloc[i+1] and
                (df['close'].iloc[i+1] - df['open'].iloc[i+1]) > 2 * abs(df['close'].iloc[i] - df['open'].iloc[i])):
                
                blocks.append({
                    'type': 'bullish',
                    'high': df['high'].iloc[i],
                    'low': df['low'].iloc[i],
                    'index': i
                })
            
            # Bearish order block
            elif (df['close'].iloc[i] > df['open'].iloc[i] and
                  df['close'].iloc[i+1] < df['open'].iloc[i+1] and
                  abs(df['close'].iloc[i+1] - df['open'].iloc[i+1]) > 2 * (df['close'].iloc[i] - df['open'].iloc[i])):
                
                blocks.append({
                    'type': 'bearish',
                    'high': df['high'].iloc[i],
                    'low': df['low'].iloc[i],
                    'index': i
                })
        
        self.order_blocks = blocks[-5:] if blocks else []
    
    def detect_liquidity_sweep(self, df):
        if len(df) < 3:
            return None
        
        current_high = df['high'].iloc[-1]
        current_low = df['low'].iloc[-1]
        current_close = df['close'].iloc[-1]
        current_volume = df['volume'].iloc[-1]
        avg_volume = df['volume'].iloc[-20:].mean()
        
        # Check sweep above liquidity
        for pool in self.liquidity_pools['highs']:
            sweep_level = pool['price'] * (1 + self.config['sweep_threshold'] / 100)
            
            if current_high > sweep_level and current_close < pool['price']:
                return {
                    'type': 'bearish_sweep',
                    'swept_level': pool['price'],
                    'volume_ratio': current_volume / avg_volume if avg_volume > 0 else 1
                }
        
        # Check sweep below liquidity
        for pool in self.liquidity_pools['lows']:
            sweep_level = pool['price'] * (1 - self.config['sweep_threshold'] / 100)
            
            if current_low < sweep_level and current_close > pool['price']:
                return {
                    'type': 'bullish_sweep',
                    'swept_level': pool['price'],
                    'volume_ratio': current_volume / avg_volume if avg_volume > 0 else 1
                }
        
        return None
    
    def check_order_block_confluence(self, sweep_type, current_price):
        if not self.order_blocks:
            return False
        
        for block in self.order_blocks:
            if ((sweep_type == 'bullish_sweep' and block['type'] == 'bullish') or
                (sweep_type == 'bearish_sweep' and block['type'] == 'bearish')):
                if block['low'] <= current_price <= block['high']:
                    return True
        
        return False
    
    def generate_signal(self, df):
        if len(df) < self.config['lookback']:
            return None
        
        self.identify_liquidity_pools(df)
        self.identify_order_blocks(df)
        
        sweep = self.detect_liquidity_sweep(df)
        if not sweep:
            return None
        
        current_price = df['close'].iloc[-1]
        has_confluence = self.check_order_block_confluence(sweep['type'], current_price)
        
        action = 'BUY' if sweep['type'] == 'bullish_sweep' else 'SELL' if sweep['type'] == 'bearish_sweep' else None
        
        if action:
            return {
                'action': action,
                'price': current_price,
                'swept_level': sweep['swept_level'],
                'volume_ratio': sweep['volume_ratio'],
                'confluence': has_confluence
            }
        
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
            
            # Check for next liquidity pool
            for pool in self.liquidity_pools['highs']:
                if current_price >= pool['price'] * 0.995:
                    return True, "liquidity_pool"
        else:
            if current_price <= entry_price * (1 - self.config['net_take_profit'] / 100):
                return True, "take_profit"
            if current_price >= entry_price * (1 + self.config['net_stop_loss'] / 100):
                return True, "stop_loss"
            
            # Check for next liquidity pool
            for pool in self.liquidity_pools['lows']:
                if current_price <= pool['price'] * 1.005:
                    return True, "liquidity_pool"
        
        return False, ""
    
    async def execute_trade(self, signal):
        qty = self.config['position_size'] / signal['price']
        formatted_qty = str(int(round(qty))) if qty >= 1 else "0"
        
        if formatted_qty == "0":
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
                qty=formatted_qty,
                price=str(limit_price),
                timeInForce="PostOnly"
            )
            
            if order.get('retCode') == 0:
                self.trade_id += 1
                print(f"🎯 {signal['action']}: {formatted_qty} @ ${limit_price:.4f}")
                print(f"   💎 Liquidity Swept: ${signal['swept_level']:.4f} | Volume: {signal['volume_ratio']:.1f}x")
                print(f"   📦 Order Block Confluence: {'✅' if signal['confluence'] else '❌'}")
                
                self.log_trade(signal['action'], limit_price, f"swept:{signal['swept_level']:.4f}")
        except Exception as e:
            print(f"Trade failed: {e}")
    
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
                pnl = float(self.position.get('unrealisedPnl', 0))
                print(f"💰 Closed: {reason} | PnL: ${pnl:.2f}")
                self.log_trade("CLOSE", 0, f"{reason}_PnL:${pnl:.2f}")
                self.position = None
        except Exception as e:
            print(f"Close failed: {e}")
    
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
        else:
            signal = self.generate_signal(self.price_data)
            if signal:
                await self.execute_trade(signal)
    
    async def run(self):
        if not self.connect():
            print("Failed to connect")
            return
        
        print(f"💎 Liquidity Sweep Bot")
        print(f"⏰ Timeframe: 5+ minutes")
        print(f"🎯 TP: {self.config['net_take_profit']}% | SL: {self.config['net_stop_loss']}%")
        
        while True:
            try:
                await self.run_cycle()
                await asyncio.sleep(10)
            except KeyboardInterrupt:
                print("\nBot stopped")
                if self.position:
                    await self.close_position("manual_stop")
                break
            except Exception as e:
                print(f"Error: {e}")
                await asyncio.sleep(5)

if __name__ == "__main__":
    bot = LiquiditySweepBot()
    asyncio.run(bot.run())