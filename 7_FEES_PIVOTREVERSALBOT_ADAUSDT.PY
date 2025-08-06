import os
import asyncio
import pandas as pd
import numpy as np
import json
from datetime import datetime
from pybit.unified_trading import HTTP
from dotenv import load_dotenv

load_dotenv()

class PivotReversalBot:
    def __init__(self):
        self.symbol = 'ADAUSDT'
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
        
        # Optimized config for positive edge
        self.config = {
            'pivot_period': 5,
            'rsi_period': 7,
            'mfi_period': 7,
            'rsi_oversold': 35,
            'rsi_overbought': 65,
            'mfi_oversold': 25,
            'mfi_overbought': 75,
            'position_size': 100,
            'tp_percent': 0.6,      # Looser TP
            'sl_percent': 0.15,     # Tighter SL
            'pivot_tolerance': 0.002,
            'range_threshold': 0.02,
            'min_volume_threshold': 50000,  # High volume filter
            'entry_threshold': 0.4,  # Min profit threshold
        }
        
        # Pivot tracking
        self.support_levels = []
        self.resistance_levels = []
        
        # BNB rules
        self.qty_step = '0.01'
        self.min_qty = 0.01
        
        # Logs
        os.makedirs("logs", exist_ok=True)
        self.log_file = "logs/pivot_bnb_trades.log"
    
    def connect(self):
        """Connect to exchange."""
        try:
            self.exchange = HTTP(demo=self.demo_mode, api_key=self.api_key, api_secret=self.api_secret)
            return self.exchange.get_server_time().get('retCode') == 0
        except Exception as e:
            print(f"Connection error: {e}")
            return False
    
    def format_qty(self, qty):
        """Format quantity for exchange."""
        if qty < self.min_qty:
            return "0"
        
        decimals = len(self.qty_step.split('.')[1]) if '.' in self.qty_step else 0
        qty_step_float = float(self.qty_step)
        rounded_qty = round(qty / qty_step_float) * qty_step_float
        
        return f"{rounded_qty:.{decimals}f}" if decimals > 0 else str(int(rounded_qty))
    
    def calculate_indicators(self, df):
        """Calculate RSI and MFI indicators."""
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
            'rsi_prev': rsi.iloc[-2] if len(rsi) > 1 else 50,
            'mfi_prev': mfi.iloc[-2] if len(mfi) > 1 else 50
        }
    
    def detect_pivots(self, df):
        """Detect pivot points."""
        min_len = self.config['pivot_period'] * 2 + 1
        if len(df) < min_len:
            return [], []
        
        period = self.config['pivot_period']
        pivots_high = []
        pivots_low = []
        
        for i in range(period, len(df) - period):
            high_val = df['high'].iloc[i]
            low_val = df['low'].iloc[i]
            
            # Check pivot high
            if all(df['high'].iloc[j] < high_val for j in range(i - period, i + period + 1) if j != i):
                pivots_high.append((high_val, i))
            
            # Check pivot low
            if all(df['low'].iloc[j] > low_val for j in range(i - period, i + period + 1) if j != i):
                pivots_low.append((low_val, i))
        
        return pivots_high, pivots_low
    
    def update_pivot_levels(self, pivot_highs, pivot_lows):
        """Update support and resistance levels."""
        tolerance = 0.003
        
        # Add new resistance levels
        for high, _ in pivot_highs:
            if not any(abs(high - level) < high * tolerance for level in self.resistance_levels):
                self.resistance_levels.append(high)
        
        # Add new support levels
        for low, _ in pivot_lows:
            if not any(abs(low - level) < low * tolerance for level in self.support_levels):
                self.support_levels.append(low)
        
        # Keep only recent levels
        self.resistance_levels = sorted(self.resistance_levels)[-10:]
        self.support_levels = sorted(self.support_levels)[-10:]
    
    def find_next_pivot_target(self, current_price, direction):
        """Find next pivot level for TP."""
        if direction == 'BUY':
            targets = [r for r in self.resistance_levels if r > current_price]
            return min(targets) if targets else current_price * (1 + self.config['tp_percent']/100)
        else:
            targets = [s for s in self.support_levels if s < current_price]
            return max(targets) if targets else current_price * (1 - self.config['tp_percent']/100)
    
    def is_near_level(self, price, levels):
        """Check if price is near any level."""
        tolerance = self.config['pivot_tolerance']
        return any(abs(price - level) / level <= tolerance for level in levels)
    
    def generate_pivot_signal(self, df):
        """Generate signals based on pivot reversal strategy."""
        if len(df) < 30:
            return None
        
        current_price = float(df['close'].iloc[-1])
        current_volume = float(df['volume'].iloc[-1])
        
        # Volume filter
        if current_volume < self.config['min_volume_threshold']:
            return None
        
        indicators = self.calculate_indicators(df)
        if not indicators:
            return None
        
        rsi = indicators['rsi']
        mfi = indicators['mfi']
        rsi_prev = indicators['rsi_prev']
        
        # BUY Signal
        if mfi <= self.config['mfi_oversold'] or (rsi <= self.config['rsi_oversold'] and rsi > rsi_prev):
            target = self.find_next_pivot_target(current_price, 'BUY')
            target = min(target, current_price * (1 + self.config['tp_percent']/100))
            
            # Entry threshold check
            expected_profit = ((target - current_price) / current_price) * 100
            if expected_profit < self.config['entry_threshold']:
                return None
            
            return {
                'action': 'BUY',
                'price': current_price,
                'reason': 'scalp_oversold',
                'rsi': rsi,
                'mfi': mfi,
                'target': target,
                'confidence': 1.2 if self.is_near_level(current_price, self.support_levels) else 1.0
            }
        
        # SELL Signal
        if mfi >= self.config['mfi_overbought'] or (rsi >= self.config['rsi_overbought'] and rsi < rsi_prev):
            target = self.find_next_pivot_target(current_price, 'SELL')
            target = max(target, current_price * (1 - self.config['tp_percent']/100))
            
            # Entry threshold check
            expected_profit = ((current_price - target) / current_price) * 100
            if expected_profit < self.config['entry_threshold']:
                return None
            
            return {
                'action': 'SELL',
                'price': current_price,
                'reason': 'scalp_overbought',
                'rsi': rsi,
                'mfi': mfi,
                'target': target,
                'confidence': 1.2 if self.is_near_level(current_price, self.resistance_levels) else 1.0
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
            
            if not pos_list or float(pos_list[0]['size']) == 0:
                if self.position:
                    self.log_trade("CLOSE", float(self.price_data['close'].iloc[-1]), "position_closed")
                self.position = None
            else:
                self.position = pos_list[0]
                
        except Exception as e:
            print(f"Position check error: {e}")
    
    def should_close_position(self, signal=None):
        """Check if position should be closed."""
        if not self.position:
            return False, ""
        
        current_price = float(self.price_data['close'].iloc[-1])
        entry_price = float(self.position.get('avgPrice', 0))
        side = self.position.get('side', '')
        
        if entry_price == 0:
            return False, ""
        
        # Calculate PnL %
        pnl_pct = ((current_price - entry_price) / entry_price) * 100
        if side == "Sell":
            pnl_pct = -pnl_pct
        
        # Check TP/SL
        if pnl_pct >= self.config['tp_percent']:
            return True, "scalp_tp"
        
        if pnl_pct <= -self.config['sl_percent']:
            return True, "scalp_sl"
        
        # Exit on opposite signal
        if signal and ((side == "Buy" and signal['action'] == 'SELL') or 
                      (side == "Sell" and signal['action'] == 'BUY')):
            return True, "signal_reversal"
        
        return False, ""
    
    async def execute_trade(self, signal):
        """Execute trade with maker orders."""
        current_price = signal['price']
        qty = self.config['position_size'] / current_price
        formatted_qty = self.format_qty(qty)
        
        if formatted_qty == "0":
            return
        
        try:
            order = self.exchange.place_order(
                category="linear",
                symbol=self.symbol,
                side="Buy" if signal['action'] == 'BUY' else "Sell",
                orderType="Limit",
                qty=formatted_qty,
                price=str(current_price),
                timeInForce="PostOnly"
            )
            
            if order.get('retCode') == 0:
                self.trade_id += 1
                info = f"{signal['reason']}_RSI:{signal['rsi']:.1f}_MFI:{signal['mfi']:.1f}"
                self.log_trade(signal['action'], current_price, info)
                print(f"✅ {signal['action']}: {formatted_qty} @ ${current_price:.2f}")
                print(f"   📊 RSI:{signal['rsi']:.1f} MFI:{signal['mfi']:.1f} | Target:${signal['target']:.2f}")
                
        except Exception as e:
            print(f"❌ Trade failed: {e}")
    
    async def close_position(self, reason):
        """Close position with maker order."""
        if not self.position:
            return
        
        current_price = float(self.price_data['close'].iloc[-1])
        pnl = float(self.position.get('unrealisedPnl', 0))
        side = "Sell" if self.position.get('side') == "Buy" else "Buy"
        qty = float(self.position['size'])
        
        try:
            order = self.exchange.place_order(
                category="linear",
                symbol=self.symbol,
                side=side,
                orderType="Limit",
                qty=self.format_qty(qty),
                price=str(current_price),
                timeInForce="PostOnly",
                reduceOnly=True
            )
            
            if order.get('retCode') == 0:
                self.log_trade("CLOSE", current_price, f"{reason} PnL:${pnl:.2f}")
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
        
        print(f"\n⚡ SCALPING BOT - {self.symbol}")
        print(f"💰 Price: ${price:,.2f}")
        
        if indicators:
            print(f"📊 RSI: {indicators['rsi']:.1f} | MFI: {indicators['mfi']:.1f}")
        
        # Show nearest levels
        if self.support_levels:
            near_support = min((s for s in self.support_levels if abs(price - s) / price <= 0.01), 
                             default=None)
            if near_support:
                print(f"🟢 Near Support: ${near_support:.2f}")
        
        if self.resistance_levels:
            near_resistance = min((r for r in self.resistance_levels if abs(price - r) / price <= 0.01), 
                                default=None)
            if near_resistance:
                print(f"🔴 Near Resistance: ${near_resistance:.2f}")
        
        if self.position:
            pnl = float(self.position.get('unrealisedPnl', 0))
            entry = float(self.position.get('avgPrice', 0))
            side = self.position.get('side', '')
            size = self.position.get('size', '0')
            
            emoji = "🟢" if side == "Buy" else "🔴"
            print(f"{emoji} {side}: {size} @ ${entry:.2f} | PnL: ${pnl:.2f}")
        else:
            print("⚡ Scanning for scalping signals...")
        
        print("-" * 50)
    
    async def run_cycle(self):
        """Main trading cycle."""
        # Get market data
        if not await self.get_market_data():
            return
        
        # Detect and update pivot levels
        pivot_highs, pivot_lows = self.detect_pivots(self.price_data)
        self.update_pivot_levels(pivot_highs, pivot_lows)
        
        # Check position
        await self.check_position()
        
        # Generate signal
        signal = self.generate_pivot_signal(self.price_data)
        
        # Position management
        if self.position:
            should_close, reason = self.should_close_position(signal)
            if should_close:
                await self.close_position(reason)
        elif signal:
            await self.execute_trade(signal)
        
        self.show_status()
    
    async def run(self):
        """Main bot loop."""
        if not self.connect():
            print("❌ Failed to connect")
            return
        
        print(f"✅ Connected! Starting Scalping bot for {self.symbol}")
        print(f"📊 Strategy: Maker orders | TP:{self.config['tp_percent']}% SL:{self.config['sl_percent']}%")
        print(f"📈 Entry threshold: {self.config['entry_threshold']}% | Min volume: {self.config['min_volume_threshold']}")
        
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