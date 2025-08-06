# File 2: EMA RSI Bot (BTC) - Fixed Version (Removed is_trending requirement)

import os
import asyncio
import pandas as pd
import numpy as np
import json
from datetime import datetime
from pybit.unified_trading import HTTP
from dotenv import load_dotenv

load_dotenv()

class EMARSIBot:
    def __init__(self):
        self.symbol = 'BNBUSDT'
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
        
        # EMA + RSI Strategy Config - Tuned for Scalping
        self.config = {
            'ema_fast': 5,
            'ema_slow': 13,
            'rsi_period': 5,
            'rsi_oversold': 25,
            'rsi_overbought': 75,
            'volume_threshold': 1.5,
            'take_profit': 0.35,
            'stop_loss': 0.20,
            'position_size': 100,
            'min_rsi_diff': 2.0,
            'maker_offset_pct': 0.01,  # Maker order offset
        }
        
        # BTC quantity rules
        self.qty_step, self.min_qty = '0.001', 0.001
        
        os.makedirs("logs", exist_ok=True)
        self.log_file = "logs/ema_rsi_btc_trades.log"
    
    def connect(self):
        """Connect to exchange."""
        try:
            self.exchange = HTTP(demo=self.demo_mode, api_key=self.api_key, api_secret=self.api_secret)
            return self.exchange.get_server_time().get('retCode') == 0
        except Exception as e:
            print(f"Connection error: {e}")
            return False
    
    def format_qty(self, qty):
        """Format quantity for BTC."""
        if qty < self.min_qty:
            return "0"
        return f"{round(qty / 0.001) * 0.001:.3f}"
    
    def calculate_indicators(self, df):
        """Calculate all indicators."""
        required_len = max(self.config['ema_slow'], self.config['rsi_period']) + 1
        if len(df) < required_len:
            return None
        
        close = df['close']
        volume = df['volume']
        
        # EMAs
        ema_fast = close.ewm(span=self.config['ema_fast']).mean()
        ema_slow = close.ewm(span=self.config['ema_slow']).mean()
        
        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(window=self.config['rsi_period']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.config['rsi_period']).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Volume ratio
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
        """Generate trading signal."""
        indicators = self.calculate_indicators(df)
        # REMOVED is_trending check - now accepts signals regardless of EMA separation
        if not indicators:
            return None
        
        # Volume confirmation
        if indicators['volume_ratio'] < self.config['volume_threshold']:
            return None
        
        # RSI difference check
        rsi_diff = abs(indicators['rsi'] - indicators['rsi_prev'])
        if rsi_diff < self.config['min_rsi_diff']:
            return None
        
        current_price = float(df['close'].iloc[-1])
        
        # Bullish signal
        if (indicators['trend'] == 'UP' and
            indicators['rsi_prev'] <= self.config['rsi_oversold'] and
            indicators['rsi'] > indicators['rsi_prev'] and
            current_price > indicators['ema_fast']):
            
            return {
                'action': 'BUY',
                'price': current_price,
                'rsi': indicators['rsi'],
                'trend': indicators['trend'],
                'volume_ratio': indicators['volume_ratio']
            }
        
        # Bearish signal
        if (indicators['trend'] == 'DOWN' and
            indicators['rsi_prev'] >= self.config['rsi_overbought'] and
            indicators['rsi'] < indicators['rsi_prev'] and
            current_price < indicators['ema_fast']):
            
            return {
                'action': 'SELL',
                'price': current_price,
                'rsi': indicators['rsi'],
                'trend': indicators['trend'],
                'volume_ratio': indicators['volume_ratio']
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
        """Execute maker-only trade."""
        current_price = signal['price']
        qty = self.config['position_size'] / current_price
        formatted_qty = self.format_qty(qty)
        
        if formatted_qty == "0":
            return
        
        # Calculate limit price for maker order
        if signal['action'] == 'BUY':
            limit_price = round(current_price * (1 - self.config['maker_offset_pct']/100), 2)
        else:
            limit_price = round(current_price * (1 + self.config['maker_offset_pct']/100), 2)
        
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
                self.log_trade(signal['action'], limit_price, f"RSI:{signal['rsi']:.1f}_Vol:{signal['volume_ratio']:.1f}")
                print(f"✅ MAKER {signal['action']}: {formatted_qty} BTC @ ${limit_price:,.0f}")
                print(f"   📊 RSI:{signal['rsi']:.1f} | Trend:{signal['trend']} | Vol:{signal['volume_ratio']:.1f}x")
                
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
            limit_price = round(current_price * (1 + self.config['maker_offset_pct']/100), 2)
        else:
            limit_price = round(current_price * (1 - self.config['maker_offset_pct']/100), 2)
        
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
                print(f"✅ Closed: {reason} | PnL: ${pnl:.2f}")
                
        except Exception as e:
            print(f"❌ Close failed: {e}")
    
    def log_trade(self, action, price, info):
        """Log trade."""
        log_data = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'id': self.trade_id,
            'action': action,
            'price': round(price, 2),
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
        
        price = float(self.price_data['close'].iloc[-1])
        indicators = self.calculate_indicators(self.price_data)
        
        print(f"\n⚡ EMA + RSI BOT - {self.symbol}")
        print(f"💰 Price: ${price:,.0f}")
        
        if indicators:
            print(f"📊 EMA{self.config['ema_fast']}: ${indicators['ema_fast']:,.0f} | EMA{self.config['ema_slow']}: ${indicators['ema_slow']:,.0f}")
            print(f"📈 RSI: {indicators['rsi']:.1f} | Trend: {indicators['trend']}")
        
        if self.position:
            pnl = float(self.position.get('unrealisedPnl', 0))
            entry = float(self.position.get('avgPrice', 0))
            side = self.position.get('side', '')
            size = self.position.get('size', '0')
            
            emoji = "🟢" if side == "Buy" else "🔴"
            print(f"{emoji} {side}: {size} BTC @ ${entry:,.0f} | PnL: ${pnl:.2f}")
        else:
            print("⚡ Scanning for RSI bounce signals...")
        
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
        
        print(f"✅ Connected! Starting EMA + RSI bot for {self.symbol}")
        print(f"📊 Strategy: EMA {self.config['ema_fast']}/{self.config['ema_slow']} + RSI({self.config['rsi_period']})")
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
    bot = EMARSIBot()
    asyncio.run(bot.run())