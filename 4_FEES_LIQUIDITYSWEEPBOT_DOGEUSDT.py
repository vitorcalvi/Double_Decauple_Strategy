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
    """Strategy 3: Smart-Money Liquidity Sweep - FIXED VERSION"""
    
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
        self.last_sweep_time = None
        self.sweep_cooldown = 60  # Minimum seconds between trades
        
        # OPTIMIZED PARAMETERS
        self.config = {
            'timeframe': '5',
            'liquidity_lookback': 30,  # Reduced from 50
            'order_block_lookback': 15,  # Reduced from 20
            'sweep_threshold': 0.3,  # Increased from 0.15% to reduce false signals
            'sweep_penetration': 0.05,  # How deep price must go beyond level
            'retracement_ratio': 0.618,  # Fibonacci retracement
            'take_profit_pct': 2.0,  # Increased from 1.5%
            'stop_loss_pct': 0.8,  # Increased from 0.5%
            'position_size': 100,
            'lookback': 100,
            'maker_offset_pct': 0.01,
            'min_distance_pct': 0.4,  # Minimum distance from current price to liquidity
            'volume_spike_threshold': 1.5,  # Volume must be 1.5x average
            'confirmation_candles': 2,  # Wait for confirmation after sweep
        }
        
        # Liquidity tracking
        self.liquidity_pools = {'highs': deque(maxlen=5), 'lows': deque(maxlen=5)}
        self.order_blocks = []
        self.recent_sweeps = deque(maxlen=3)  # Track recent sweeps to avoid repetition
        
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
        """Identify significant liquidity pools with better filtering."""
        if len(df) < self.config['liquidity_lookback']:
            return
        
        window = 10  # Increased from 5 for better significance
        current_price = df['close'].iloc[-1]
        
        # Clear old pools
        self.liquidity_pools['highs'].clear()
        self.liquidity_pools['lows'].clear()
        
        # Find significant highs
        for i in range(len(df) - 5, max(0, len(df) - self.config['liquidity_lookback']), -1):
            high_price = df['high'].iloc[i]
            
            # Check if it's a significant high
            is_significant = (
                all(df['high'].iloc[max(0, i-5):i] <= high_price) and
                all(df['high'].iloc[i+1:min(len(df), i+6)] <= high_price)
            )
            
            # Check minimum distance from current price
            distance_pct = abs((high_price - current_price) / current_price * 100)
            
            if is_significant and distance_pct > self.config['min_distance_pct']:
                # Check if not duplicate
                is_duplicate = any(
                    abs(pool['price'] - high_price) / high_price < 0.001 
                    for pool in self.liquidity_pools['highs']
                )
                
                if not is_duplicate:
                    self.liquidity_pools['highs'].append({
                        'price': high_price,
                        'index': i,
                        'volume': df['volume'].iloc[i],
                        'strength': sum(df['volume'].iloc[max(0, i-2):min(len(df), i+3)])
                    })
        
        # Find significant lows
        for i in range(len(df) - 5, max(0, len(df) - self.config['liquidity_lookback']), -1):
            low_price = df['low'].iloc[i]
            
            # Check if it's a significant low
            is_significant = (
                all(df['low'].iloc[max(0, i-5):i] >= low_price) and
                all(df['low'].iloc[i+1:min(len(df), i+6)] >= low_price)
            )
            
            # Check minimum distance from current price
            distance_pct = abs((current_price - low_price) / low_price * 100)
            
            if is_significant and distance_pct > self.config['min_distance_pct']:
                # Check if not duplicate
                is_duplicate = any(
                    abs(pool['price'] - low_price) / low_price < 0.001 
                    for pool in self.liquidity_pools['lows']
                )
                
                if not is_duplicate:
                    self.liquidity_pools['lows'].append({
                        'price': low_price,
                        'index': i,
                        'volume': df['volume'].iloc[i],
                        'strength': sum(df['volume'].iloc[max(0, i-2):min(len(df), i+3)])
                    })
    
    def identify_order_blocks(self, df):
        """Identify order blocks with improved logic."""
        if len(df) < self.config['order_block_lookback']:
            return []
        
        blocks = []
        
        for i in range(len(df) - 4, max(0, len(df) - self.config['order_block_lookback']), -1):
            # Bullish order block (down candle followed by strong up move)
            if (df['close'].iloc[i] < df['open'].iloc[i] and  # Red candle
                df['close'].iloc[i+1] > df['open'].iloc[i+1] and  # Green candle
                (df['close'].iloc[i+1] - df['open'].iloc[i+1]) > 1.5 * abs(df['close'].iloc[i] - df['open'].iloc[i])):
                
                blocks.append({
                    'type': 'bullish',
                    'high': df['high'].iloc[i],
                    'low': df['low'].iloc[i],
                    'index': i,
                    'strength': df['volume'].iloc[i]
                })
            
            # Bearish order block (up candle followed by strong down move)
            elif (df['close'].iloc[i] > df['open'].iloc[i] and  # Green candle
                  df['close'].iloc[i+1] < df['open'].iloc[i+1] and  # Red candle
                  abs(df['close'].iloc[i+1] - df['open'].iloc[i+1]) > 1.5 * (df['close'].iloc[i] - df['open'].iloc[i])):
                
                blocks.append({
                    'type': 'bearish',
                    'high': df['high'].iloc[i],
                    'low': df['low'].iloc[i],
                    'index': i,
                    'strength': df['volume'].iloc[i]
                })
        
        self.order_blocks = blocks[-3:] if blocks else []
    
    def detect_liquidity_sweep(self, df):
        """Detect liquidity sweep with confirmation."""
        if len(df) < 5:
            return None
        
        current_high = df['high'].iloc[-1]
        current_low = df['low'].iloc[-1]
        current_close = df['close'].iloc[-1]
        current_volume = df['volume'].iloc[-1]
        prev_close = df['close'].iloc[-2]
        avg_volume = df['volume'].iloc[-20:].mean()
        
        # Check volume spike
        if current_volume < avg_volume * self.config['volume_spike_threshold']:
            return None
        
        # Check sweep above liquidity (bearish sweep)
        for pool in self.liquidity_pools['highs']:
            sweep_level = pool['price'] * (1 + self.config['sweep_penetration'] / 100)
            
            # Price must sweep above and close back below
            if (current_high > sweep_level and 
                current_close < pool['price'] and
                prev_close < pool['price']):  # Confirmation
                
                # Check if not recently swept
                is_recent = any(
                    abs(sweep['level'] - pool['price']) / pool['price'] < 0.002 
                    for sweep in self.recent_sweeps
                )
                
                if not is_recent:
                    self.recent_sweeps.append({'level': pool['price'], 'time': datetime.now()})
                    return {
                        'type': 'bearish_sweep',
                        'swept_level': pool['price'],
                        'volume_ratio': current_volume / avg_volume if avg_volume > 0 else 1,
                        'strength': pool.get('strength', 0)
                    }
        
        # Check sweep below liquidity (bullish sweep)
        for pool in self.liquidity_pools['lows']:
            sweep_level = pool['price'] * (1 - self.config['sweep_penetration'] / 100)
            
            # Price must sweep below and close back above
            if (current_low < sweep_level and 
                current_close > pool['price'] and
                prev_close > pool['price']):  # Confirmation
                
                # Check if not recently swept
                is_recent = any(
                    abs(sweep['level'] - pool['price']) / pool['price'] < 0.002 
                    for sweep in self.recent_sweeps
                )
                
                if not is_recent:
                    self.recent_sweeps.append({'level': pool['price'], 'time': datetime.now()})
                    return {
                        'type': 'bullish_sweep',
                        'swept_level': pool['price'],
                        'volume_ratio': current_volume / avg_volume if avg_volume > 0 else 1,
                        'strength': pool.get('strength', 0)
                    }
        
        return None
    
    def check_order_block_confluence(self, sweep_type, current_price):
        """Check order block confluence with improved logic."""
        if not self.order_blocks:
            return False
        
        for block in self.order_blocks:
            # Match sweep type with order block type
            if ((sweep_type == 'bullish_sweep' and block['type'] == 'bullish') or
                (sweep_type == 'bearish_sweep' and block['type'] == 'bearish')):
                
                # Check if price is within order block zone
                buffer = 0.001  # 0.1% buffer
                if (block['low'] * (1 - buffer) <= current_price <= block['high'] * (1 + buffer)):
                    return True
        
        return False
    
    def generate_signal(self, df):
        """Generate trading signal with cooldown."""
        if len(df) < self.config['lookback']:
            return None
        
        # Check cooldown
        if self.last_sweep_time:
            time_diff = (datetime.now() - self.last_sweep_time).total_seconds()
            if time_diff < self.sweep_cooldown:
                return None
        
        self.identify_liquidity_pools(df)
        self.identify_order_blocks(df)
        
        sweep = self.detect_liquidity_sweep(df)
        if not sweep:
            return None
        
        current_price = df['close'].iloc[-1]
        has_confluence = self.check_order_block_confluence(sweep['type'], current_price)
        
        # Require confluence for better win rate
        if not has_confluence and sweep['volume_ratio'] < 2.0:
            return None
        
        self.last_sweep_time = datetime.now()
        
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
            return True, "take_profit_2RR"
        if pnl_pct <= -self.config['stop_loss_pct']:
            return True, "stop_loss"
        
        # Don't close at next liquidity pool too quickly
        position_age = (datetime.now() - self.last_sweep_time).total_seconds() if self.last_sweep_time else 0
        if position_age < 30:  # Hold position for at least 30 seconds
            return False, ""
        
        # Check for next liquidity pool (with better distance check)
        if side == "Buy":
            for pool in self.liquidity_pools['highs']:
                distance_pct = (pool['price'] - current_price) / current_price * 100
                if 0 < distance_pct < 0.1:  # Very close to resistance
                    return True, "approaching_resistance"
        else:
            for pool in self.liquidity_pools['lows']:
                distance_pct = (current_price - pool['price']) / pool['price'] * 100
                if 0 < distance_pct < 0.1:  # Very close to support
                    return True, "approaching_support"
        
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
                confluence_str = "WITH_OB" if signal['confluence'] else "HIGH_VOL"
                self.log_trade(signal['action'], limit_price, 
                             f"swept:{signal['swept_level']:.4f}_vol:{signal['volume_ratio']:.1f}_{confluence_str}")
                
                print(f"🎯 MAKER {signal['action']}: {formatted_qty} DOGE @ ${limit_price:.4f}")
                print(f"   Liquidity Swept: ${signal['swept_level']:.4f} | Volume: {signal['volume_ratio']:.1f}x")
                print(f"   Order Block Confluence: {'✅' if signal['confluence'] else '❌ (High Volume Override)'}")
                
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
            top_resistance = min(self.liquidity_pools['highs'], key=lambda x: x['price'])
            print(f"🔴 Next Resistance: ${top_resistance['price']:.4f} ({abs(top_resistance['price'] - current_price) / current_price * 100:.2f}% away)")
        
        if self.liquidity_pools['lows']:
            bottom_support = max(self.liquidity_pools['lows'], key=lambda x: x['price'])
            print(f"🟢 Next Support: ${bottom_support['price']:.4f} ({abs(current_price - bottom_support['price']) / bottom_support['price'] * 100:.2f}% away)")
        
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
            if self.last_sweep_time:
                cooldown_left = max(0, self.sweep_cooldown - (datetime.now() - self.last_sweep_time).total_seconds())
                if cooldown_left > 0:
                    print(f"⏳ Cooldown: {int(cooldown_left)}s remaining")
                else:
                    print("🔍 Scanning for liquidity sweeps...")
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
        
        print(f"💎 Strategy 3: Smart-Money Liquidity Sweep Bot - FIXED")
        print(f"⏰ Timeframe: 5+ minutes | Target Win Rate: 70%")
        print(f"🎯 TP: 2.0 RR ({self.config['take_profit_pct']}%) | SL: {self.config['stop_loss_pct']}%")
        print(f"⏳ Trade Cooldown: {self.sweep_cooldown}s between trades")
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