# File 1: DOGE Scalping Bot - Fixed Version (Removed cooldown)

import os
import time
import asyncio
import pandas as pd
import numpy as np
import json
from datetime import datetime
from pybit.unified_trading import HTTP
from dotenv import load_dotenv

load_dotenv()

class DOGEScalpingBot:
    def __init__(self):
        self.symbol = 'ETHUSDT'
        self.demo_mode = os.getenv('DEMO_MODE', 'true').lower() == 'true'
        
        # API connection
        prefix = 'TESTNET_' if self.demo_mode else 'LIVE_'
        self.api_key = os.getenv(f'{prefix}BYBIT_API_KEY')
        self.api_secret = os.getenv(f'{prefix}BYBIT_API_SECRET')
        self.exchange = None
        
        # Trading state
        self.position = None
        self.price_data = pd.DataFrame()
        self.trade_id = 0
        self.support_resistance = []
        
        # SCALPING Strategy Config
        self.config = {
            'ema_fast': 5,
            'ema_slow': 13,
            'bb_period': 20,
            'bb_std': 2.0,
            'min_spread': 0.05,
            'volume_spike': 0.5,
            'take_profit': 0.2,
            'stop_loss': 0.15,
            'position_size': 100,
            'max_positions': 1,
            'maker_offset_pct': 0.01,  # Offset for limit orders
        }
        
        # DOGE quantity rules
        self.qty_step, self.min_qty = '1', 1.0
        
        os.makedirs("logs", exist_ok=True)
        self.log_file = "logs/scalping_doge_trades.log"
    
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
        return str(int(round(qty))) if qty >= self.min_qty else "0"
    
    def calculate_indicators(self, df):
        """Calculate all technical indicators."""
        if len(df) < 20:
            return None
        
        close = df['close']
        volume = df['volume']
        
        # EMAs
        ema_fast = close.ewm(span=self.config['ema_fast']).mean()
        ema_slow = close.ewm(span=self.config['ema_slow']).mean()
        
        # Bollinger Bands
        sma = close.rolling(window=self.config['bb_period']).mean()
        std = close.rolling(window=self.config['bb_period']).std()
        upper_band = sma + (std * self.config['bb_std'])
        lower_band = sma - (std * self.config['bb_std'])
        
        # Volume analysis
        vol_ma = volume.rolling(window=10).mean()
        vol_ratio = volume.iloc[-1] / vol_ma.iloc[-1] if vol_ma.iloc[-1] > 0 else 0
        
        # Price metrics
        current_price = close.iloc[-1]
        price_momentum = (ema_fast.iloc[-1] - ema_slow.iloc[-1]) / ema_slow.iloc[-1] * 100
        bb_position = (current_price - lower_band.iloc[-1]) / (upper_band.iloc[-1] - lower_band.iloc[-1]) if upper_band.iloc[-1] != lower_band.iloc[-1] else 0.5
        spread = (df['high'].iloc[-1] - df['low'].iloc[-1]) / df['low'].iloc[-1] * 100
        
        return {
            'price': current_price,
            'price_momentum': price_momentum,
            'volume_ratio': vol_ratio,
            'bb_position': bb_position,
            'spread': spread,
            'upper_band': upper_band.iloc[-1],
            'lower_band': lower_band.iloc[-1]
        }
    
    def detect_support_resistance(self, df):
        """Simplified S/R detection."""
        if len(df) < 20:
            return []
        
        levels = []
        window = 5
        
        # Find recent highs and lows
        for i in range(len(df) - 20, len(df) - window):
            if df['high'].iloc[i] == df['high'].iloc[i-window:i+window+1].max():
                levels.append(df['high'].iloc[i])
            if df['low'].iloc[i] == df['low'].iloc[i-window:i+window+1].min():
                levels.append(df['low'].iloc[i])
        
        # Remove duplicates and sort
        return sorted(list(set(levels)))[-5:] if levels else []
    
    def generate_signal(self, df):
        """Generate trading signal."""
        # REMOVED COOLDOWN CHECK - Now can trade immediately
        
        indicators = self.calculate_indicators(df)
        if not indicators:
            return None
        
        self.support_resistance = self.detect_support_resistance(df)
        
        # Signal conditions
        long_signal = (
            indicators['price_momentum'] > 0.1 and
            indicators['volume_ratio'] > self.config['volume_spike'] and
            indicators['bb_position'] < 0.3 and
            indicators['spread'] > self.config['min_spread']
        )
        
        short_signal = (
            indicators['price_momentum'] < -0.1 and
            indicators['volume_ratio'] > self.config['volume_spike'] and
            indicators['bb_position'] > 0.7 and
            indicators['spread'] > self.config['min_spread']
        )
        
        if long_signal:
            return {
                'action': 'BUY',
                'price': indicators['price'],
                'momentum': indicators['price_momentum'],
                'volume': indicators['volume_ratio'],
                'bb_pos': indicators['bb_position']
            }
        elif short_signal:
            return {
                'action': 'SELL',
                'price': indicators['price'],
                'momentum': indicators['price_momentum'],
                'volume': indicators['volume_ratio'],
                'bb_pos': indicators['bb_position']
            }
        
        return None
    
    async def get_market_data(self):
        """Get market data."""
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
            
            # Convert data types
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
        
        if pnl_pct >= self.config['take_profit']:
            return True, "take_profit"
        if pnl_pct <= -self.config['stop_loss']:
            return True, "stop_loss"
        
        return False, ""
    
    async def execute_trade(self, signal):
        """Execute maker-only trade with PostOnly flag."""
        current_price = signal['price']
        qty = self.config['position_size'] / current_price
        formatted_qty = self.format_qty(qty)
        
        if formatted_qty == "0":
            return
        
        # Calculate limit price with offset for maker order
        if signal['action'] == 'BUY':
            limit_price = round(current_price * (1 - self.config['maker_offset_pct']/100), 4)
        else:
            limit_price = round(current_price * (1 + self.config['maker_offset_pct']/100), 4)
        
        try:
            order = self.exchange.place_order(
                category="linear",
                symbol=self.symbol,
                side="Buy" if signal['action'] == 'BUY' else "Sell",
                orderType="Limit",
                qty=formatted_qty,
                price=str(limit_price),
                timeInForce="PostOnly"  # Maker-only order
            )
            
            if order.get('retCode') == 0:
                self.trade_id += 1
                self.log_trade(signal['action'], limit_price, f"MOM:{signal['momentum']:.2f}_VOL:{signal['volume']:.1f}")
                print(f"⚡ MAKER {signal['action']}: {formatted_qty} DOGE @ ${limit_price:.4f}")
                print(f"   📊 Momentum:{signal['momentum']:.2f}% | Volume:{signal['volume']:.1f}x | BB:{signal['bb_pos']:.2f}")
                
        except Exception as e:
            print(f"❌ Trade failed: {e}")
    
    async def close_position(self, reason):
        """Close position with maker order."""
        if not self.position:
            return
        
        current_price = float(self.price_data['close'].iloc[-1])
        side = "Sell" if self.position.get('side') == "Buy" else "Buy"
        qty = float(self.position['size'])
        
        # Calculate limit price for maker close
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
            print(f"❌ Close failed: {e}")
    
    def log_trade(self, action, price, info):
        """Log trade."""
        log_data = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'id': self.trade_id,
            'action': action,
            'price': round(price, 6),
            'info': info
        }
        
        try:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(log_data) + "\n")
        except:
            pass
    
    def show_status(self):
        """Show current status."""
        if len(self.price_data) == 0:
            return
        
        indicators = self.calculate_indicators(self.price_data)
        if not indicators:
            return
        
        print(f"\n⚡ SCALPING BOT - {self.symbol}")
        print(f"💰 Price: ${indicators['price']:.4f}")
        print(f"📊 Momentum: {indicators['price_momentum']:.2f}% | Volume: {indicators['volume_ratio']:.1f}x")
        print(f"📈 BB Position: {indicators['bb_position']:.2f} | Spread: {indicators['spread']:.2f}%")
        
        if self.support_resistance:
            print(f"📍 S/R Levels: {[f'${x:.4f}' for x in self.support_resistance[-3:]]}")
        
        if self.position:
            pnl = float(self.position.get('unrealisedPnl', 0))
            entry = float(self.position.get('avgPrice', 0))
            side = self.position.get('side', '')
            size = self.position.get('size', '0')
            
            emoji = "🟢" if side == "Buy" else "🔴"
            print(f"{emoji} {side}: {size} DOGE @ ${entry:.4f} | PnL: ${pnl:.2f}")
        else:
            print("⚡ Scanning for scalping opportunities...")
        
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
            print("❌ Failed to connect")
            return
        
        print(f"⚡ Connected! Starting SCALPING bot for {self.symbol}")
        print("📊 Strategy: Momentum + Volume + Bollinger Bands")
        print(f"🎯 TP: {self.config['take_profit']}% | SL: {self.config['stop_loss']}%")
        print("💎 Using MAKER-ONLY orders for -0.04% fees")
        
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