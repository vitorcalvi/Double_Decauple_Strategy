import os
import asyncio
import pandas as pd
import numpy as np
import json
from datetime import datetime
from pymexc import spot
from dotenv import load_dotenv

load_dotenv()

class ZigZagTradingBot:
    def __init__(self):
        self.symbol = 'SOLUSDT'
        
        # API connection
        self.api_key = os.getenv('MEXC_API_KEY')
        self.api_secret = os.getenv('MEXC_API_SECRET')
        self.exchange = None
        
        # Trading state
        self.position = None
        self.price_data = pd.DataFrame()
        self.trade_id = 0
        
        # ZigZag config
        self.config = {
            'zigzag_period': 10,    # Period for swing detection
            'take_profit': 1.0,     # 1% profit target
            'stop_loss': 0.5,       # 0.5% stop loss
            'position_size': 100,   # $100 per trade
        }
        
        # Swing tracking
        self.swing_highs = []  # [(price, bar_index, pattern)]
        self.swing_lows = []   # [(price, bar_index, pattern)]
        self.last_swing_type = None  # 'high' or 'low'
        
        # Symbol rules
        self._set_rules()
        
        # Logs
        os.makedirs("logs", exist_ok=True)
        self.log_file = "logs/zigzag_sol_trades.log"
    
    def _set_rules(self):
        """Set trading rules for symbol."""
        if 'ETH' in self.symbol:
            self.qty_step, self.min_qty = '0.01', 0.01
        elif 'BTC' in self.symbol:
            self.qty_step, self.min_qty = '0.001', 0.001  
        elif 'SOL' in self.symbol:
            self.qty_step, self.min_qty = '0.01', 0.01
        elif 'ADA' in self.symbol:
            self.qty_step, self.min_qty = '1', 1.0
        else:
            self.qty_step, self.min_qty = '0.01', 0.01
    
    def connect(self):
        """Connect to MEXC exchange."""
        try:
            self.exchange = spot.HTTP(api_key=self.api_key, api_secret=self.api_secret)
            # Test connection
            test = self.exchange.ping()
            return True
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
    
    def detect_pivots(self, df):
        """Detect pivot highs and lows."""
        if len(df) < self.config['zigzag_period'] * 2 + 1:
            return None, None
        
        period = self.config['zigzag_period']
        
        # Check for pivot high at current position
        pivot_high = None
        pivot_low = None
        
        # Look back 'period' bars from current
        for i in range(period, len(df) - period):
            high_val = df['high'].iloc[i]
            low_val = df['low'].iloc[i]
            
            # Check if it's the highest in the period window
            is_pivot_high = True
            is_pivot_low = True
            
            # Check left and right sides
            for j in range(i - period, i + period + 1):
                if j != i:
                    if df['high'].iloc[j] >= high_val:
                        is_pivot_high = False
                    if df['low'].iloc[j] <= low_val:
                        is_pivot_low = False
            
            if is_pivot_high:
                pivot_high = (high_val, i)
            if is_pivot_low:
                pivot_low = (low_val, i)
        
        return pivot_high, pivot_low
    
    def update_swings(self, pivot_high, pivot_low):
        """Update swing highs and lows with patterns."""
        current_bar = len(self.price_data) - 1
        
        if pivot_high:
            price, bar_idx = pivot_high
            
            # Determine pattern (HH = Higher High, LH = Lower High)
            pattern = "H"
            if len(self.swing_highs) > 0:
                prev_high = self.swing_highs[-1][0]
                pattern = "HH" if price > prev_high else "LH"
            
            self.swing_highs.append((price, bar_idx, pattern))
            self.last_swing_type = 'high'
            
            # Keep only last 10 swings
            if len(self.swing_highs) > 10:
                self.swing_highs.pop(0)
        
        if pivot_low:
            price, bar_idx = pivot_low
            
            # Determine pattern (LL = Lower Low, HL = Higher Low)  
            pattern = "L"
            if len(self.swing_lows) > 0:
                prev_low = self.swing_lows[-1][0]
                pattern = "LL" if price < prev_low else "HL"
            
            self.swing_lows.append((price, bar_idx, pattern))
            self.last_swing_type = 'low'
            
            # Keep only last 10 swings
            if len(self.swing_lows) > 10:
                self.swing_lows.pop(0)
    
    def generate_zigzag_signal(self):
        """Generate BUY signals only for spot trading."""
        if len(self.swing_highs) < 2 or len(self.swing_lows) < 2:
            return None
        
        current_price = float(self.price_data['close'].iloc[-1])
        
        # Get latest swings
        latest_high = self.swing_highs[-1]
        latest_low = self.swing_lows[-1]
        
        # BUY SIGNALS ONLY (spot trading):
        # 1. HL (Higher Low) - bullish continuation
        # 2. Price above previous swing high (breakout)
        if latest_low[2] == "HL":  # Higher Low pattern
            if current_price > latest_low[0] * 1.005:  # 0.5% above swing low
                return {
                    'action': 'BUY',
                    'price': current_price,
                    'reason': 'HL_bounce',
                    'swing_low': latest_low[0],
                    'pattern': latest_low[2]
                }
        
        # Breakout above previous swing high
        if len(self.swing_highs) >= 2:
            prev_high = self.swing_highs[-2][0]
            if current_price > prev_high * 1.002:  # 0.2% above previous high
                return {
                    'action': 'BUY', 
                    'price': current_price,
                    'reason': 'breakout_high',
                    'break_level': prev_high,
                    'pattern': 'BREAKOUT'
                }
        
        return None
    
    async def get_market_data(self):
        """Get market data."""
        try:
            # Get klines data
            klines = self.exchange.klines(
                symbol=self.symbol,
                interval="1m",
                limit=100
            )
            
            if 'data' not in klines:
                return False
            
            df = pd.DataFrame(klines['data'], columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume'
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
        """Check current position (spot balance)."""
        try:
            account = self.exchange.account_info()
            
            if 'data' not in account:
                self.position = None
                return
            
            # Find SOL balance
            base_asset = self.symbol.replace('USDT', '')
            sol_balance = 0
            
            for balance in account['data']['balances']:
                if balance['asset'] == base_asset:
                    sol_balance = float(balance['free'])
                    break
            
            if sol_balance > self.min_qty:
                self.position = {
                    'symbol': self.symbol,
                    'size': sol_balance,
                    'side': 'Buy',
                    'avgPrice': 0  # We don't track avg price in spot
                }
            else:
                if self.position:
                    self.log_trade("CLOSE", float(self.price_data['close'].iloc[-1]), "position_sold")
                self.position = None
                
        except Exception as e:
            print(f"Position check error: {e}")
    
    def should_close(self):
        """Check if should close position."""
        if not self.position:
            return False, ""
        
        current_price = float(self.price_data['close'].iloc[-1])
        
        # Simple time-based or pattern-based exit
        # Since we don't track exact entry price in spot, use simple rules
        
        # Get latest swing high for exit
        if len(self.swing_highs) > 0:
            latest_high = self.swing_highs[-1]
            
            # If we're at a Lower High pattern, consider selling
            if latest_high[2] == "LH" and current_price < latest_high[0] * 0.995:
                return True, "LH_exit"
        
        # Simple profit taking at swing highs
        if len(self.swing_highs) >= 2:
            prev_high = self.swing_highs[-2][0]
            if current_price >= prev_high * 0.998:  # Near previous high
                return True, "swing_high_exit"
        
        return False, ""
    
    async def execute_trade(self, signal):
        """Execute BUY trade only."""
        current_price = signal['price']
        qty = self.config['position_size'] / current_price
        formatted_qty = self.format_qty(qty)
        
        if formatted_qty == "0":
            return
        
        try:
            order = self.exchange.new_order(
                symbol=self.symbol,
                side="BUY",
                type="MARKET",
                quantity=formatted_qty
            )
            
            if order.get('data'):
                self.trade_id += 1
                self.log_trade(signal['action'], current_price, f"{signal['reason']}_{signal['pattern']}")
                print(f"✅ {signal['action']}: {formatted_qty} @ ${current_price:.2f} | {signal['reason']}")
                
        except Exception as e:
            print(f"❌ Trade failed: {e}")
    
    async def close_position(self, reason):
        """Close position by selling all holdings."""
        if not self.position:
            return
        
        current_price = float(self.price_data['close'].iloc[-1])
        qty = float(self.position['size'])
        
        try:
            order = self.exchange.new_order(
                symbol=self.symbol,
                side="SELL",
                type="MARKET",
                quantity=self.format_qty(qty)
            )
            
            if order.get('data'):
                self.log_trade("SELL", current_price, f"{reason}")
                print(f"✅ Sold: {reason} @ ${current_price:.2f}")
                
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
        
        print(f"\n⚡ ZIGZAG SPOT BOT - {self.symbol}")
        print(f"💰 Price: ${price:,.2f}")
        
        # Show recent swings
        if len(self.swing_highs) > 0:
            latest_high = self.swing_highs[-1]
            print(f"📈 Last High: ${latest_high[0]:.2f} ({latest_high[2]})")
        
        if len(self.swing_lows) > 0:
            latest_low = self.swing_lows[-1]
            print(f"📉 Last Low: ${latest_low[0]:.2f} ({latest_low[2]})")
        
        if self.position:
            size = self.position.get('size', '0')
            print(f"🟢 Holdings: {size} SOL @ ${price:.2f}")
        else:
            print("⚡ Scanning for BUY signals...")
        
        print("-" * 50)
    
    async def run_cycle(self):
        """Main trading cycle."""
        # Get market data
        if not await self.get_market_data():
            return
        
        # Detect pivot points
        pivot_high, pivot_low = self.detect_pivots(self.price_data)
        
        # Update swing tracking
        self.update_swings(pivot_high, pivot_low)
        
        # Check position
        await self.check_position()
        
        # Close position if needed
        if self.position:
            should_close, reason = self.should_close()
            if should_close:
                await self.close_position(reason)
        
        # Look for BUY signals when no position
        if not self.position:
            signal = self.generate_zigzag_signal()
            if signal:
                await self.execute_trade(signal)
        
        self.show_status()
    
    async def run(self):
        """Main bot loop."""
        if not self.connect():
            print("❌ Failed to connect")
            return
        
        print(f"✅ Connected! Starting MEXC Spot ZigZag bot for {self.symbol}")
        print(f"📊 Config: Period={self.config['zigzag_period']} | {self.config['take_profit']}% TP | {self.config['stop_loss']}% SL")
        
        while True:
            try:
                await self.run_cycle()
                await asyncio.sleep(2)  # 2 second cycle
                
            except KeyboardInterrupt:
                print("\n🛑 Bot stopped")
                if self.position:
                    await self.close_position("manual_stop")
                break
                
            except Exception as e:
                print(f"❌ Error: {e}")
                await asyncio.sleep(5)

# Run the ZigZag trading bot
if __name__ == "__main__":
    bot = ZigZagTradingBot()
    asyncio.run(bot.run())