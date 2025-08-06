# File 4: Liquidity Sweep Bot - Fixed Version (Removed volume_threshold requirement)

import os
import asyncio
import pandas as pd
import numpy as np
import json
from datetime import datetime
from collections import deque
from pybit.unified_trading import HTTP
from dotenv import load_dotenv

load_dotenv()

class LiquiditySweepBot:
    """Strategy 3: Smart-Money Liquidity Sweep (70% Win Rate)"""
    
    def __init__(self):
        self.symbol = 'DOGEUSDT'
        self.demo_mode = os.getenv('DEMO_MODE', 'true').lower() == 'true'
        
        # API setup
        prefix = 'TESTNET_' if self.demo_mode else 'LIVE_'
        self.api_key = os.getenv(f'{prefix}BYBIT_API_KEY')
        self.api_secret = os.getenv(f'{prefix}BYBIT_API_SECRET')
        self.exchange = None
        
        # Trading state
        self.position = None
        self.price_data = pd.DataFrame()
        self.trade_id = 0
        
        # Strategy parameters
        self.config = {
            'timeframe': '5',
            'liquidity_lookback': 50,
            'order_block_lookback': 20,
            'sweep_threshold': 0.15,
            'retracement_ratio': 0.5,
            'take_profit_pct': 1.5,
            'stop_loss_pct': 0.5,
            'position_size': 100,
            'lookback': 100,
            'maker_offset_pct': 0.01,
        }
        
        # Liquidity tracking
        self.liquidity_pools = {'highs': deque(maxlen=10), 'lows': deque(maxlen=10)}
        self.order_blocks = []
        
        os.makedirs("logs", exist_ok=True)
        self.log_file = "logs/liquidity_sweep_trades.log"
    
    def connect(self):
        """Connect to exchange."""
        try:
            self.exchange = HTTP(demo=self.demo_mode, api_key=self.api_key, api_secret=self.api_secret)
            return self.exchange.get_server_time().get('retCode') == 0
        except Exception as e:
            print(f"Connection error: {e}")
            return False
    
    def format_qty(self, qty):
        """Format quantity for DOGE."""
        return str(int(round(qty))) if qty >= 1.0 else "0"
    
    def identify_liquidity_pools(self, df):
        """Identify liquidity pools."""
        if len(df) < self.config['liquidity_lookback']:
            return
        
        window = 5
        highs = df['high'].rolling(window=window, center=True).max()
        lows = df['low'].rolling(window=window, center=True).min()
        
        self.liquidity_pools['highs'].clear()
        self.liquidity_pools['lows'].clear()
        
        # Find significant highs and lows
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
        """Identify order blocks."""
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
        """Detect liquidity sweep."""
        if len(df) < 3:
            return None
        
        current_high = df['high'].iloc[-1]
        current_low = df['low'].iloc[-1]
        current_close = df['close'].iloc[-1]
        current_volume = df['volume'].iloc[-1]
        avg_volume = df['volume'].iloc[-20:].mean()
        
        # Check sweep above liquidity - REMOVED volume threshold requirement
        for pool in self.liquidity_pools['highs']:
            sweep_level = pool['price'] * (1 + self.config['sweep_threshold'] / 100)
            
            if (current_high > sweep_level and 
                current_close < pool['price']):
                
                return {
                    'type': 'bearish_sweep',
                    'swept_level': pool['price'],
                    'volume_ratio': current_volume / avg_volume if avg_volume > 0 else 1
                }
        
        # Check sweep below liquidity - REMOVED volume threshold requirement
        for pool in self.liquidity_pools['lows']:
            sweep_level = pool['price'] * (1 - self.config['sweep_threshold'] / 100)
            
            if (current_low < sweep_level and 
                current_close > pool['price']):
                
                return {
                    'type': 'bullish_sweep',
                    'swept_level': pool['price'],
                    'volume_ratio': current_volume / avg_volume if avg_volume > 0 else 1
                }
        
        return None
    
    def check_order_block_confluence(self, sweep_type, current_price):
        """Check order block confluence."""
        if not self.order_blocks:
            return False
        
        for block in self.order_blocks:
            if ((sweep_type == 'bullish_sweep' and block['type'] == 'bullish') or
                (sweep_type == 'bearish_sweep' and block['type'] == 'bearish')):
                if block['low'] <= current_price <= block['high']:
                    return True
        
        return False
    
    def generate_signal(self, df):
        """Generate trading signal."""
        if len(df) < self.config['lookback']:
            return None
        
        self.identify_liquidity_pools(df)
        self.identify_order_blocks(df)
        
        sweep = self.detect_liquidity_sweep(df)
        if not sweep:
            return None
        
        current_price = df['close'].iloc[-1]
        has_confluence = self.check_order_block_confluence(sweep['type'], current_price)
        
        if sweep['type'] == 'bullish_sweep':
            return {
                'action': 'BUY',
                'price': current_price,
                'swept_level': sweep['swept_level'],
                'volume_ratio': sweep['volume_ratio'],
                'confluence': has_confluence
            }
        elif sweep['type'] == 'bearish_sweep':
            return {
                'action': 'SELL',
                'price': current_price,
                'swept_level': sweep['swept_level'],
                'volume_ratio': sweep['volume_ratio'],
                'confluence': has_confluence
            }
        
        return None
    
    async def get_market_data(self):
        """Get market data."""
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
            
        except Exception as e:
            print(f"Data error: {e}")
            return False
    
    async def check_position(self):
        """Check current position."""
        try:
            positions = self.exchange.get_positions(category="linear", symbol=self.symbol)
            if positions.get('retCode') != 0:
                return
            
            pos_list = positions['result']['list']
            self.position = pos_list[0] if pos_list and float(pos_list[0]['size']) > 0 else None
                
        except Exception as e:
            print(f"Position check error: {e}")
    
    def should_close(self):
        """Check if should close position."""
        if not self.position:
            return False, ""
        
        current_price = float(self.price_data['close'].iloc[-1])
        entry_price = float(self.position.get('avgPrice', 0))
        side = self.position.get('side', '')
        
        if entry_price == 0:
            return False, ""
        
        pnl_pct = ((current_price - entry_price) / entry_price * 100) if side == "Buy" else ((entry_price - current_price) / entry_price * 100)
        
        if pnl_pct >= self.config['take_profit_pct']:
            return True, "take_profit_1.5RR"
        if pnl_pct <= -self.config['stop_loss_pct']:
            return True, "stop_loss"
        
        # Check for next liquidity pool
        if side == "Buy":
            for pool in self.liquidity_pools['highs']:
                if current_price >= pool['price'] * 0.995:
                    return True, "next_liquidity_pool"
        else:
            for pool in self.liquidity_pools['lows']:
                if current_price <= pool['price'] * 1.005:
                    return True, "next_liquidity_pool"
        
        return False, ""
    
    async def execute_trade(self, signal):
        """Execute maker-only trade."""
        qty = self.config['position_size'] / signal['price']
        formatted_qty = self.format_qty(qty)
        
        if formatted_qty == "0":
            return
        
        # Calculate limit price
        if signal['action'] == 'BUY':
            limit_price = round(signal['price'] * (1 - self.config['maker_offset_pct']/100), 4)
        else:
            limit_price = round(signal['price'] * (1 + self.config['maker_offset_pct']/100), 4)
        
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
                confluence_str = "WITH_OB" if signal['confluence'] else "NO_OB"
                self.log_trade(signal['action'], limit_price, 
                             f"swept:{signal['swept_level']:.4f}_vol:{signal['volume_ratio']:.1f}_{confluence_str}")
                
                print(f"🎯 MAKER {signal['action']}: {formatted_qty} DOGE @ ${limit_price:.4f}")
                print(f"   Liquidity Swept: ${signal['swept_level']:.4f} | Volume: {signal['volume_ratio']:.1f}x")
                print(f"   Order Block Confluence: {'✅' if signal['confluence'] else '❌'}")
                
        except Exception as e:
            print(f"Trade failed: {e}")
    
    async def close_position(self, reason):
        """Close position with maker order."""
        if not self.position:
            return
        
        current_price = float(self.price_data['close'].iloc[-1])
        side = "Sell" if self.position.get('side') == "Buy" else "Buy"
        qty = float(self.position['size'])
        
        # Calculate limit price
        if side == "Sell":
            limit_price = round(current_price * (1 + self.config['maker_offset_pct']/100), 4)
        else:
            limit_price = round(current_price * (1 - self.config['maker_offset_pct']/100), 4)
        
        try:
            order = self.exchange.place_order(
                category="linear",
                symbol=self.symbol,
                side=side,
                orderType="Limit",
                qty=self.format_qty(qty),
                price=str(limit_price),
                timeInForce="PostOnly",
                reduceOnly=True
            )
            
            if order.get('retCode') == 0:
                pnl = float(self.position.get('unrealisedPnl', 0))
                self.log_trade("CLOSE", limit_price, f"{reason}_PnL:${pnl:.2f}")
                print(f"💰 Closed: {reason} | PnL: ${pnl:.2f}")
                
        except Exception as e:
            print(f"Close failed: {e}")
    
    def log_trade(self, action, price, info):
        """Log trade."""
        log_data = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'id': self.trade_id,
            'action': action,
            'price': round(price, 6),
            'info': info
        }
        
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_data) + "\n")
    
    def show_status(self):
        """Show current status."""
        if len(self.price_data) == 0:
            return
        
        current_price = float(self.price_data['close'].iloc[-1])
        
        print(f"\n💎 Smart-Money Liquidity Sweep - {self.symbol}")
        print(f"💰 Price: ${current_price:.4f}")
        
        if self.liquidity_pools['highs']:
            top_resistance = max(self.liquidity_pools['highs'], key=lambda x: x['price'])
            print(f"🔴 Next Resistance: ${top_resistance['price']:.4f}")
        
        if self.liquidity_pools['lows']:
            bottom_support = min(self.liquidity_pools['lows'], key=lambda x: x['price'])
            print(f"🟢 Next Support: ${bottom_support['price']:.4f}")
        
        if self.order_blocks:
            bullish = sum(1 for b in self.order_blocks if b['type'] == 'bullish')
            bearish = sum(1 for b in self.order_blocks if b['type'] == 'bearish')
            print(f"📦 Order Blocks: {bullish} Bullish | {bearish} Bearish")
        
        if self.position:
            pnl = float(self.position.get('unrealisedPnl', 0))
            entry = float(self.position.get('avgPrice', 0))
            side = self.position.get('side', '')
            size = self.position.get('size', '0')
            
            emoji = "🟢" if side == "Buy" else "🔴"
            print(f"{emoji} {side}: {size} DOGE @ ${entry:.4f} | PnL: ${pnl:.2f}")
        else:
            print("🔍 Scanning for liquidity sweeps...")
        
        print("-" * 60)
    
    async def run_cycle(self):
        """Main trading cycle."""
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
        """Main bot loop."""
        if not self.connect():
            print("Failed to connect")
            return
        
        print(f"💎 Strategy 3: Smart-Money Liquidity Sweep Bot")
        print(f"⏰ Timeframe: 5+ minutes | Win Rate: 70%")
        print(f"🎯 TP: 1.5 RR ({self.config['take_profit_pct']}%) | SL: {self.config['stop_loss_pct']}%")
        print("💎 Using MAKER-ONLY orders for -0.04% fees")
        
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