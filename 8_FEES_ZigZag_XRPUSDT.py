import os
import asyncio
import pandas as pd
import json
from datetime import datetime, timezone
from pybit.unified_trading import HTTP
from dotenv import load_dotenv

load_dotenv()

class ZigZagTradingBot:
    def __init__(self):
        self.symbol = 'XRPUSDT'
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
        self.consecutive_losses = 0
        self.daily_trades = 0
        self.daily_profit = 0
        self.last_trade_bar = 0
        self.last_reset_date = None
        
        # ZIG-ZAG CONFIG with Fee Calculations
        self.config = {
            'timeframe': '3',
            'lookback': 100,
            'zigzag_pct': 0.5,
            'min_swing_bars': 3,
            'risk_per_trade': 0.05,
            'max_position_pct': 0.30,
            'cooldown_bars': 3,
            'max_daily_trades': 30,
            'max_consecutive_losses': 5,
            'daily_profit_target': 0.02,
            'daily_loss_limit': -0.01,
            'maker_offset_pct': 0.01,
            # Fee structure
            'maker_fee_pct': -0.04,  # Negative = rebate
            # Gross TP/SL
            'gross_take_profit': 1.0,
            'gross_stop_loss': 0.5,
            # Net TP/SL (adjusted for 2x maker rebate)
            'net_take_profit': 1.08,  # 1.0 + 0.08 rebate
            'net_stop_loss': 0.42,    # 0.5 - 0.08 rebate
            # Trailing stop
            'trailing_activation': 0.4,  # Activate at 0.4% profit
            'trailing_distance': 0.32,   # Trail at 0.32% (0.4 - 0.08 rebate)
        }
        
        # Symbol rules
        self.qty_step = 0.1
        self.min_qty = 0.1
        
        # Capital management
        self.initial_capital = 1000
        self.current_capital = 1000
        
        # Performance tracking
        self.trades_history = []
        self.position_metadata = {}
        
        os.makedirs("logs", exist_ok=True)
        self.log_file = f"logs/zigzag_{datetime.now().strftime('%Y%m%d')}.log"
    
    def connect(self):
        """Connect to exchange."""
        try:
            self.exchange = HTTP(demo=self.demo_mode, api_key=self.api_key, api_secret=self.api_secret)
            return self.exchange.get_server_time().get('retCode') == 0
        except:
            return False
    
    def calculate_break_even(self, entry_price, side):
        """Calculate break-even price including fees."""
        fee_impact = 2 * abs(self.config['maker_fee_pct']) / 100
        
        # With rebate, break-even is better than entry
        if side == "Buy":
            return entry_price * (1 - fee_impact)
        else:
            return entry_price * (1 + fee_impact)
    
    def calculate_net_targets(self, entry_price, side):
        """Calculate net TP/SL accounting for round-trip fees."""
        if side == "Buy":
            net_tp = entry_price * (1 + self.config['net_take_profit'] / 100)
            net_sl = entry_price * (1 - self.config['net_stop_loss'] / 100)
        else:
            net_tp = entry_price * (1 - self.config['net_take_profit'] / 100)
            net_sl = entry_price * (1 + self.config['net_stop_loss'] / 100)
        
        return net_tp, net_sl
    
    def calculate_position_size(self, price):
        """Calculate position size with risk management."""
        risk_amount = self.current_capital * self.config['risk_per_trade']
        max_position = self.current_capital * self.config['max_position_pct']
        
        # Account for NET stop loss (tighter due to rebate)
        effective_stop = self.config['net_stop_loss']
        position_value = min(risk_amount / (effective_stop / 100), max_position)
        
        qty = position_value / price
        return self.format_qty(qty)
    
    def format_qty(self, qty):
        """Format quantity."""
        if qty < self.min_qty:
            return str(self.min_qty)
        return f"{round(qty / self.qty_step) * self.qty_step:.1f}"
    
    def identify_swings(self):
        """Identify zig-zag swings."""
        if len(self.price_data) < 10:
            return []
        
        df = self.price_data
        swings = []
        
        # Find local highs and lows
        for i in range(2, len(df) - 2):
            # Check for swing high
            is_high = (
                df['high'].iloc[i] > df['high'].iloc[i-1] and 
                df['high'].iloc[i] > df['high'].iloc[i-2] and
                df['high'].iloc[i] > df['high'].iloc[i+1] and 
                df['high'].iloc[i] > df['high'].iloc[i+2]
            )
            
            # Check for swing low
            is_low = (
                df['low'].iloc[i] < df['low'].iloc[i-1] and 
                df['low'].iloc[i] < df['low'].iloc[i-2] and
                df['low'].iloc[i] < df['low'].iloc[i+1] and 
                df['low'].iloc[i] < df['low'].iloc[i+2]
            )
            
            if is_high:
                swings.append({
                    'index': i,
                    'type': 'HIGH',
                    'price': df['high'].iloc[i],
                    'time': df['timestamp'].iloc[i]
                })
            elif is_low:
                swings.append({
                    'index': i,
                    'type': 'LOW',
                    'price': df['low'].iloc[i],
                    'time': df['timestamp'].iloc[i]
                })
        
        # Filter by minimum percentage
        filtered_swings = []
        for swing in swings:
            if not filtered_swings:
                filtered_swings.append(swing)
            else:
                price_change = abs(swing['price'] - filtered_swings[-1]['price']) / filtered_swings[-1]['price'] * 100
                if price_change >= self.config['zigzag_pct'] and swing['type'] != filtered_swings[-1]['type']:
                    filtered_swings.append(swing)
        
        return filtered_swings
    
    def generate_signal(self):
        """Generate trading signal."""
        if len(self.price_data) < 20:
            return None
        
        # Check trade filters
        if (self.daily_trades >= self.config['max_daily_trades'] or
            self.consecutive_losses >= self.config['max_consecutive_losses'] or
            self.daily_profit <= self.config['daily_loss_limit']):
            return None
        
        # Check cooldown
        current_bar = len(self.price_data) - 1
        if current_bar - self.last_trade_bar < self.config['cooldown_bars']:
            return None
        
        swings = self.identify_swings()
        if len(swings) < 3:
            return None
        
        current_price = float(self.price_data['close'].iloc[-1])
        last_swing = swings[-1]
        bars_since_swing = current_bar - last_swing['index']
        
        # Volume confirmation
        recent_vol = self.price_data['volume'].iloc[-3:].mean()
        avg_vol = self.price_data['volume'].iloc[-20:].mean()
        
        if recent_vol <= avg_vol * 0.8:
            return None
        
        # BUY signal at swing low reversal
        if (last_swing['type'] == 'LOW' and 
            bars_since_swing <= self.config['min_swing_bars'] and
            current_price > last_swing['price'] * 1.001):
            
            return {
                'action': 'BUY',
                'price': current_price,
                'reason': 'swing_low_reversal',
                'swing_price': last_swing['price']
            }
        
        # SELL signal at swing high reversal
        elif (last_swing['type'] == 'HIGH' and 
              bars_since_swing <= self.config['min_swing_bars'] and
              current_price < last_swing['price'] * 0.999):
            
            return {
                'action': 'SELL',
                'price': current_price,
                'reason': 'swing_high_reversal',
                'swing_price': last_swing['price']
            }
        
        # Breakout signals
        if last_swing['type'] == 'HIGH' and current_price > last_swing['price'] * 1.002:
            return {
                'action': 'BUY',
                'price': current_price,
                'reason': 'breakout_high',
                'swing_price': last_swing['price']
            }
        
        elif last_swing['type'] == 'LOW' and current_price < last_swing['price'] * 0.998:
            return {
                'action': 'SELL',
                'price': current_price,
                'reason': 'breakout_low',
                'swing_price': last_swing['price']
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
        except:
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
                    # Update capital on position close with NET PnL
                    gross_pnl = float(self.position.get('realisedPnl', 0))
                    entry_price = float(self.position.get('avgPrice', 0))
                    size = float(self.position.get('size', 0))
                    current_price = float(self.price_data['close'].iloc[-1])
                    
                    # Calculate fee rebate earned
                    fee_earned = (entry_price * size + current_price * size) * abs(self.config['maker_fee_pct']) / 100
                    net_pnl = gross_pnl + fee_earned
                    
                    self.current_capital += net_pnl
                    self.daily_profit += net_pnl / self.initial_capital
                self.position = None
            else:
                self.position = pos_list[0]
        except:
            pass
    
    def should_close(self):
        """Check if should close position with NET targets."""
        if not self.position:
            return False, ""
        
        current_price = float(self.price_data['close'].iloc[-1])
        entry_price = float(self.position.get('avgPrice', 0))
        side = self.position.get('side', '')
        
        if entry_price == 0:
            return False, ""
        
        # Check for new opposite swing
        swings = self.identify_swings()
        if swings:
            last_swing = swings[-1]
            if ((side == "Buy" and last_swing['type'] == 'HIGH' and current_price >= entry_price * 1.002) or
                (side == "Sell" and last_swing['type'] == 'LOW' and current_price <= entry_price * 0.998)):
                return True, "swing_exit"
        
        # Calculate NET targets
        net_tp, net_sl = self.calculate_net_targets(entry_price, side)
        
        # Check against NET targets
        if side == "Buy":
            if current_price >= net_tp:
                return True, f"take_profit_net_{self.config['net_take_profit']}%"
            if current_price <= net_sl:
                return True, f"stop_loss_net_{self.config['net_stop_loss']}%"
            
            # Trailing stop with NET adjustment
            pnl_pct = ((current_price - entry_price) / entry_price) * 100
            if pnl_pct > self.config['trailing_activation']:
                trailing_stop = entry_price * (1 + (pnl_pct - self.config['trailing_distance']) / 100)
                if current_price < trailing_stop:
                    return True, "trailing_stop_net"
        else:
            if current_price <= net_tp:
                return True, f"take_profit_net_{self.config['net_take_profit']}%"
            if current_price >= net_sl:
                return True, f"stop_loss_net_{self.config['net_stop_loss']}%"
            
            # Trailing stop with NET adjustment
            pnl_pct = ((entry_price - current_price) / entry_price) * 100
            if pnl_pct > self.config['trailing_activation']:
                trailing_stop = entry_price * (1 - (pnl_pct - self.config['trailing_distance']) / 100)
                if current_price > trailing_stop:
                    return True, "trailing_stop_net"
        
        return False, ""
    
    async def execute_trade(self, signal):
        """Execute maker-only trade with NET targets."""
        current_price = signal['price']
        formatted_qty = self.calculate_position_size(current_price)
        
        if float(formatted_qty) < self.min_qty:
            return
        
        # Calculate limit price
        limit_price = round(current_price * (1 - self.config['maker_offset_pct']/100 if signal['action'] == 'BUY' else 1 + self.config['maker_offset_pct']/100), 2)
        
        # Calculate NET targets
        break_even = self.calculate_break_even(limit_price, signal['action'])
        net_tp, net_sl = self.calculate_net_targets(limit_price, signal['action'])
        
        try:
            # Place main order without built-in TP/SL (will manage manually for NET targets)
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
                self.daily_trades += 1
                self.last_trade_bar = len(self.price_data) - 1
                
                # Store NET targets in metadata
                self.position_metadata = {
                    'breakEven': break_even,
                    'netTP': net_tp,
                    'netSL': net_sl,
                    'entryPrice': limit_price,
                    'side': signal['action']
                }
                
                print(f"✅ MAKER {signal['action']}: {formatted_qty} @ ${limit_price:.2f} | {signal['reason']}")
                print(f"   📊 Break-Even: ${break_even:.2f}")
                print(f"   🎯 Net TP: ${net_tp:.2f} ({self.config['net_take_profit']}%)")
                print(f"   🛡️ Net SL: ${net_sl:.2f} ({self.config['net_stop_loss']}%)")
                print(f"   💎 Swing: ${signal['swing_price']:.2f}")
                
                self.log_trade(signal['action'], limit_price, f"{signal['reason']}_BE:{break_even:.2f}_NetTP:{net_tp:.2f}")
            else:
                print(f"❌ Trade failed: {order.get('retMsg')}")
        except Exception as e:
            print(f"❌ Trade error: {e}")
    
    async def close_position(self, reason):
        """Close position with maker order and NET PnL."""
        if not self.position:
            return
        
        current_price = float(self.price_data['close'].iloc[-1])
        side = "Sell" if self.position.get('side') == "Buy" else "Buy"
        qty = float(self.position['size'])
        entry_price = float(self.position.get('avgPrice', 0))
        
        # Calculate limit price
        limit_price = round(current_price * (1 + self.config['maker_offset_pct']/100 if side == "Sell" else 1 - self.config['maker_offset_pct']/100), 2)
        
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
                # Calculate NET PnL including fees
                gross_pnl = float(self.position.get('unrealisedPnl', 0))
                fee_earned = (entry_price * qty + current_price * qty) * abs(self.config['maker_fee_pct']) / 100
                net_pnl = gross_pnl + fee_earned
                
                # Update tracking
                self.consecutive_losses = self.consecutive_losses + 1 if net_pnl < 0 else 0
                
                self.trades_history.append({
                    'gross_pnl': gross_pnl,
                    'net_pnl': net_pnl,
                    'fee_earned': fee_earned,
                    'reason': reason,
                    'timestamp': datetime.now()
                })
                
                print(f"✅ Closed: {reason}")
                print(f"   💵 Gross P&L: ${gross_pnl:.2f}")
                print(f"   💎 Fee Rebate: ${fee_earned:.2f}")
                print(f"   📊 Net P&L: ${net_pnl:.2f}")
                
                self.position = None
                self.position_metadata = {}
                
                self.log_trade("CLOSE", limit_price, f"{reason}_GrossPnL:${gross_pnl:.2f}_NetPnL:${net_pnl:.2f}")
        except Exception as e:
            print(f"❌ Close failed: {e}")
    
    def reset_daily_stats(self):
        """Reset daily statistics."""
        current_date = datetime.now(timezone.utc).date()
        if self.last_reset_date and current_date > self.last_reset_date:
            self.daily_trades = 0
            self.daily_profit = 0
            self.consecutive_losses = 0
        self.last_reset_date = current_date
    
    def log_trade(self, action, price, info):
        """Log trade."""
        with open(self.log_file, "a") as f:
            f.write(json.dumps({
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'id': self.trade_id,
                'action': action,
                'price': round(price, 2),
                'info': info,
                'capital': round(self.current_capital, 2)
            }) + "\n")
    
    def show_status(self):
        """Display status with NET calculations."""
        if not len(self.price_data):
            return
        
        price = float(self.price_data['close'].iloc[-1])
        swings = self.identify_swings()
        
        print(f"\n{'='*60}")
        print(f"📈 ZIG-ZAG TRADING BOT - {self.symbol}")
        print(f"{'='*60}")
        print(f"💰 Price: ${price:.2f}")
        print(f"📊 Capital: ${self.current_capital:.2f} ({(self.current_capital/self.initial_capital-1)*100:+.1f}%)")
        
        if swings:
            last_swing = swings[-1]
            print(f"🔄 Last Swing: {last_swing['type']} @ ${last_swing['price']:.2f}")
        
        if self.position:
            entry_price = float(self.position.get('avgPrice', 0))
            side = self.position.get('side', '')
            size = self.position.get('size', '0')
            
            # Calculate current NET PnL
            gross_pnl = float(self.position.get('unrealisedPnl', 0))
            fee_earned = (entry_price * float(size)) * abs(self.config['maker_fee_pct']) / 100
            current_fee = price * float(size) * abs(self.config['maker_fee_pct']) / 100
            net_pnl = gross_pnl + fee_earned + current_fee  # Include both entry and potential exit fee
            
            # Get stored targets
            break_even = self.position_metadata.get('breakEven', entry_price)
            net_tp = self.position_metadata.get('netTP', 0)
            net_sl = self.position_metadata.get('netSL', 0)
            
            emoji = "🟢" if side == "Buy" else "🔴"
            print(f"\n{emoji} POSITION: {side} {size} @ ${entry_price:.2f}")
            print(f"   💵 Gross P&L: ${gross_pnl:.2f}")
            print(f"   💎 Fee Rebate: ${fee_earned + current_fee:.2f}")
            print(f"   📊 Net P&L: ${net_pnl:.2f}")
            print(f"   🎯 BE: ${break_even:.2f} | TP: ${net_tp:.2f} | SL: ${net_sl:.2f}")
        else:
            print(f"\n🔍 Scanning for zig-zag patterns...")
            print(f"   Daily Trades: {self.daily_trades}/{self.config['max_daily_trades']}")
            print(f"   Consecutive Losses: {self.consecutive_losses}")
        
        print(f"{'='*60}")
    
    async def run_cycle(self):
        """Main cycle."""
        self.reset_daily_stats()
        
        if not await self.get_market_data():
            return
        
        await self.check_position()
        
        if self.position:
            should_close, reason = self.should_close()
            if should_close:
                await self.close_position(reason)
        elif signal := self.generate_signal():
            await self.execute_trade(signal)
        
        self.show_status()
    
    async def run(self):
        """Main loop."""
        if not self.connect():
            print("❌ Failed to connect")
            return
        
        print(f"\n{'='*60}")
        print(f"🚀 ZIG-ZAG TRADING BOT with NET Profit Tracking")
        print(f"{'='*60}")
        print(f"📊 Symbol: {self.symbol}")
        print(f"💰 Capital: ${self.initial_capital}")
        print(f"⏰ Timeframe: {self.config['timeframe']} minutes")
        print(f"📈 Min Swing: {self.config['zigzag_pct']}%")
        print(f"🎯 Net TP: {self.config['net_take_profit']}% | Net SL: {self.config['net_stop_loss']}%")
        print(f"💎 Using MAKER-ONLY orders for {self.config['maker_fee_pct']}% fee rebate")
        print(f"{'='*60}\n")
        
        try:
            while True:
                await self.run_cycle()
                await asyncio.sleep(10)
        except KeyboardInterrupt:
            print("\n🛑 Bot stopped")
            if self.position:
                await self.close_position("manual_stop")
            
            # Show final statistics
            if self.trades_history:
                total_net_pnl = sum(t['net_pnl'] for t in self.trades_history)
                total_fees_earned = sum(t['fee_earned'] for t in self.trades_history)
                print(f"\n📊 Final Statistics:")
                print(f"   Total Net P&L: ${total_net_pnl:.2f}")
                print(f"   Total Fees Earned: ${total_fees_earned:.2f}")
                print(f"   Final Capital: ${self.current_capital:.2f}")

if __name__ == "__main__":
    bot = ZigZagTradingBot()
    asyncio.run(bot.run())