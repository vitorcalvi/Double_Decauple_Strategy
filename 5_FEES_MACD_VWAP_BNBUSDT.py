import os
import asyncio
import pandas as pd
import numpy as np
import json
from datetime import datetime
from pybit.unified_trading import HTTP
from dotenv import load_dotenv

load_dotenv()

class MACDVWAPBot:
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
        
        # MACD + VWAP Strategy Config - Tuned for Scalping
        self.config = {
            'macd_fast': 8,
            'macd_slow': 21,
            'macd_signal': 5,
            'rsi_period': 9,
            'ema_period': 13,
            'rsi_oversold': 35,
            'rsi_overbought': 65,
            'take_profit': 0.35,
            'stop_loss': 0.25,
            'position_size': 100,
            'maker_offset_pct': 0.01,
        }
        
        # DOGE quantity rules
        self.qty_step, self.min_qty = '1', 1.0
        
        os.makedirs("logs", exist_ok=True)
        self.log_file = "logs/macd_vwap_doge_trades.log"
    
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
    
    def calculate_vwap(self, df):
        """Calculate VWAP."""
        if len(df) < 20:
            return None
        
        recent_data = df.tail(min(1440, len(df)))  # Last 24 hours
        typical_price = (recent_data['high'] + recent_data['low'] + recent_data['close']) / 3
        volume = recent_data['volume']
        
        vwap = (typical_price * volume).cumsum() / volume.cumsum()
        return vwap.iloc[-1] if not vwap.empty else None
    
    def calculate_indicators(self, df):
        """Calculate all indicators."""
        if len(df) < 50:
            return None
        
        close = df['close']
        
        # MACD
        ema_fast = close.ewm(span=self.config['macd_fast']).mean()
        ema_slow = close.ewm(span=self.config['macd_slow']).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.config['macd_signal']).mean()
        histogram = macd_line - signal_line
        
        # VWAP
        vwap = self.calculate_vwap(df)
        if not vwap:
            return None
        
        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(window=self.config['rsi_period']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.config['rsi_period']).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # EMA trend filter
        ema = close.ewm(span=self.config['ema_period']).mean()
        
        return {
            'histogram': histogram.iloc[-1],
            'histogram_prev': histogram.iloc[-2] if len(histogram) > 1 else histogram.iloc[-1],
            'vwap': vwap,
            'rsi': rsi.iloc[-1],
            'rsi_prev': rsi.iloc[-2] if len(rsi) > 1 else rsi.iloc[-1],
            'ema': ema.iloc[-1],
            'price': close.iloc[-1]
        }
    
    def detect_signals(self, indicators):
        """Detect histogram flip, RSI cross, and VWAP alignment."""
        # Histogram flip
        hist_bullish = indicators['histogram_prev'] <= 0 and indicators['histogram'] > 0
        hist_bearish = indicators['histogram_prev'] >= 0 and indicators['histogram'] < 0
        
        # RSI cross
        rsi_bullish = indicators['rsi_prev'] <= self.config['rsi_oversold'] and indicators['rsi'] > self.config['rsi_oversold']
        rsi_bearish = indicators['rsi_prev'] >= self.config['rsi_overbought'] and indicators['rsi'] < self.config['rsi_overbought']
        
        # VWAP alignment
        vwap_bullish = indicators['price'] > indicators['vwap'] and indicators['ema'] > indicators['vwap']
        vwap_bearish = indicators['price'] < indicators['vwap'] and indicators['ema'] < indicators['vwap']
        
        return {
            'hist_bullish': hist_bullish,
            'hist_bearish': hist_bearish,
            'rsi_bullish': rsi_bullish,
            'rsi_bearish': rsi_bearish,
            'vwap_bullish': vwap_bullish,
            'vwap_bearish': vwap_bearish
        }
    
    def generate_signal(self, df):
        """Generate MACD + VWAP signals."""
        indicators = self.calculate_indicators(df)
        if not indicators:
            return None
        
        signals = self.detect_signals(indicators)
        
        # Bullish signal: All three align
        if signals['hist_bullish'] and signals['rsi_bullish'] and signals['vwap_bullish']:
            return {
                'action': 'BUY',
                'price': indicators['price'],
                'macd_hist': indicators['histogram'],
                'rsi': indicators['rsi'],
                'vwap': indicators['vwap']
            }
        
        # Bearish signal: All three align
        if signals['hist_bearish'] and signals['rsi_bearish'] and signals['vwap_bearish']:
            return {
                'action': 'SELL',
                'price': indicators['price'],
                'macd_hist': indicators['histogram'],
                'rsi': indicators['rsi'],
                'vwap': indicators['vwap']
            }
        
        return None
    
    async def get_market_data(self):
        """Get market data."""
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
        
        # Calculate limit price
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
                timeInForce="PostOnly"
            )
            
            if order.get('retCode') == 0:
                self.trade_id += 1
                info = f"HIST:{signal['macd_hist']:.6f}_RSI:{signal['rsi']:.1f}"
                self.log_trade(signal['action'], limit_price, info)
                print(f"✅ MAKER {signal['action']}: {formatted_qty} DOGE @ ${limit_price:.4f}")
                print(f"   📊 MACD Hist:{signal['macd_hist']:.6f} | RSI:{signal['rsi']:.1f}")
                
        except Exception as e:
            print(f"❌ Trade failed: {e}")
    
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
                print(f"✅ Closed: {reason} | PnL: ${pnl:.2f}")
                
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
        
        price = float(self.price_data['close'].iloc[-1])
        indicators = self.calculate_indicators(self.price_data)
        
        print(f"\n⚡ MACD + VWAP BOT - {self.symbol}")
        print(f"💰 Price: ${price:.4f}")
        
        if indicators:
            print(f"📊 MACD Hist: {indicators['histogram']:.6f} | RSI: {indicators['rsi']:.1f}")
            print(f"📈 VWAP: ${indicators['vwap']:.4f} | EMA: ${indicators['ema']:.4f}")
        
        if self.position:
            pnl = float(self.position.get('unrealisedPnl', 0))
            entry = float(self.position.get('avgPrice', 0))
            side = self.position.get('side', '')
            size = self.position.get('size', '0')
            
            emoji = "🟢" if side == "Buy" else "🔴"
            print(f"{emoji} {side}: {size} DOGE @ ${entry:.4f} | PnL: ${pnl:.2f}")
        else:
            print("⚡ Scanning for MACD + VWAP signals...")
        
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
        
        print(f"✅ Connected! Starting MACD + VWAP bot for {self.symbol}")
        print("📊 Strategy: MACD histogram flip + RSI cross + VWAP alignment")
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
    bot = MACDVWAPBot()
    asyncio.run(bot.run())