import os
import asyncio
import pandas as pd
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
        
        # Strategy Config with Fee Calculations
        self.config = {
            'ema_fast': 5,
            'ema_slow': 13,
            'bb_period': 20,
            'bb_std': 2.0,
            'min_spread': 0.05,
            'volume_spike': 0.5,
            'position_size': 100,
            'maker_offset_pct': 0.01,
            # Fee structure
            'maker_fee_pct': -0.04,  # Negative = rebate
            # Gross TP/SL
            'gross_take_profit': 0.20,
            'gross_stop_loss': 0.15,
            # Net TP/SL (adjusted for fees)
            'net_take_profit': 0.28,  # 0.20 + 0.08 rebate
            'net_stop_loss': 0.07,    # 0.15 - 0.08 rebate
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
        except:
            return False
    
    def format_qty(self, qty):
        """Format quantity for DOGE."""
        return str(int(round(qty))) if qty >= self.min_qty else "0"
    
    def calculate_break_even(self, entry_price, side):
        """Calculate break-even price including fees.
        For maker orders with rebate, we earn on both entry and exit."""
        fee_impact = 2 * abs(self.config['maker_fee_pct']) / 100
        
        if self.config['maker_fee_pct'] < 0:  # Rebate case
            # We earn fees, so break-even is actually below entry for longs
            return entry_price * (1 - fee_impact) if side == "Buy" else entry_price * (1 + fee_impact)
        else:  # Regular fee case
            return entry_price * (1 + fee_impact) if side == "Buy" else entry_price * (1 - fee_impact)
    
    def calculate_net_targets(self, entry_price, side):
        """Calculate net TP/SL accounting for round-trip fees."""
        fee_adjustment = 2 * abs(self.config['maker_fee_pct']) / 100
        
        if side == "Buy":
            # For rebates, we earn more; for fees, we earn less
            net_tp = entry_price * (1 + self.config['net_take_profit'] / 100)
            net_sl = entry_price * (1 - self.config['net_stop_loss'] / 100)
        else:
            net_tp = entry_price * (1 - self.config['net_take_profit'] / 100)
            net_sl = entry_price * (1 + self.config['net_stop_loss'] / 100)
        
        return net_tp, net_sl
    
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
    
    def generate_signal(self, df):
        """Generate trading signal."""
        indicators = self.calculate_indicators(df)
        if not indicators:
            return None
        
        # Long signal
        if (indicators['price_momentum'] > 0.1 and
            indicators['volume_ratio'] > self.config['volume_spike'] and
            indicators['bb_position'] < 0.3 and
            indicators['spread'] > self.config['min_spread']):
            
            return {
                'action': 'BUY',
                'price': indicators['price'],
                'momentum': indicators['price_momentum'],
                'volume': indicators['volume_ratio'],
                'bb_pos': indicators['bb_position']
            }
        
        # Short signal
        if (indicators['price_momentum'] < -0.1 and
            indicators['volume_ratio'] > self.config['volume_spike'] and
            indicators['bb_position'] > 0.7 and
            indicators['spread'] > self.config['min_spread']):
            
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
            if positions.get('retCode') == 0:
                pos_list = positions['result']['list']
                self.position = pos_list[0] if pos_list and float(pos_list[0]['size']) > 0 else None
        except:
            pass
    
    def should_close(self):
        """Check if should close position with NET profit targets."""
        if not self.position:
            return False, ""
        
        current_price = float(self.price_data['close'].iloc[-1])
        entry_price = float(self.position.get('avgPrice', 0))
        side = self.position.get('side', '')
        
        if entry_price == 0:
            return False, ""
        
        # Calculate NET targets
        net_tp, net_sl = self.calculate_net_targets(entry_price, side)
        
        # Check against NET targets
        if side == "Buy":
            if current_price >= net_tp:
                return True, f"take_profit_net_{self.config['net_take_profit']}%"
            if current_price <= net_sl:
                return True, f"stop_loss_net_{self.config['net_stop_loss']}%"
        else:
            if current_price <= net_tp:
                return True, f"take_profit_net_{self.config['net_take_profit']}%"
            if current_price >= net_sl:
                return True, f"stop_loss_net_{self.config['net_stop_loss']}%"
        
        return False, ""
    
    async def execute_trade(self, signal):
        """Execute maker-only trade with PostOnly flag."""
        current_price = signal['price']
        qty = self.config['position_size'] / current_price
        formatted_qty = self.format_qty(qty)
        
        if formatted_qty == "0":
            return
        
        # Calculate limit price with offset for maker order
        limit_price = round(current_price * (1 - self.config['maker_offset_pct']/100 if signal['action'] == 'BUY' else 1 + self.config['maker_offset_pct']/100), 4)
        
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
                
                # Calculate and log break-even
                break_even = self.calculate_break_even(limit_price, signal['action'])
                net_tp, net_sl = self.calculate_net_targets(limit_price, signal['action'])
                
                self.log_trade(signal['action'], limit_price, 
                             f"BE:{break_even:.4f}_NetTP:{net_tp:.4f}_NetSL:{net_sl:.4f}")
                
                print(f"⚡ MAKER {signal['action']}: {formatted_qty} @ ${limit_price:.4f}")
                print(f"   📊 Break-Even: ${break_even:.4f} | Net TP: ${net_tp:.4f} | Net SL: ${net_sl:.4f}")
                print(f"   📈 Momentum:{signal['momentum']:.2f}% | Volume:{signal['volume']:.1f}x")
        except Exception as e:
            print(f"❌ Trade failed: {e}")
    
    async def close_position(self, reason):
        """Close position with maker order."""
        if not self.position:
            return
        
        current_price = float(self.price_data['close'].iloc[-1])
        side = "Sell" if self.position.get('side') == "Buy" else "Buy"
        qty = float(self.position['size'])
        entry_price = float(self.position.get('avgPrice', 0))
        
        # Calculate limit price for maker close
        limit_price = round(current_price * (1 + self.config['maker_offset_pct']/100 if side == "Sell" else 1 - self.config['maker_offset_pct']/100), 4)
        
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
                
                self.log_trade("CLOSE", limit_price, f"{reason}_GrossPnL:${gross_pnl:.2f}_NetPnL:${net_pnl:.2f}")
                print(f"💰 Closed: {reason} | Gross PnL: ${gross_pnl:.2f} | Net PnL: ${net_pnl:.2f}")
        except Exception as e:
            print(f"❌ Close failed: {e}")
    
    def log_trade(self, action, price, info):
        """Log trade."""
        with open(self.log_file, "a") as f:
            f.write(json.dumps({
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'id': self.trade_id,
                'action': action,
                'price': round(price, 6),
                'info': info
            }) + "\n")
    
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
        print(f"📈 BB Position: {indicators['bb_position']:.2f}")
        
        if self.position:
            entry_price = float(self.position.get('avgPrice', 0))
            side = self.position.get('side', '')
            size = self.position.get('size', '0')
            
            # Calculate current NET PnL
            current_price = float(self.price_data['close'].iloc[-1])
            gross_pnl = float(self.position.get('unrealisedPnl', 0))
            fee_earned = (entry_price * float(size)) * abs(self.config['maker_fee_pct']) / 100
            net_pnl = gross_pnl + fee_earned
            
            # Calculate break-even and targets
            break_even = self.calculate_break_even(entry_price, side)
            net_tp, net_sl = self.calculate_net_targets(entry_price, side)
            
            emoji = "🟢" if side == "Buy" else "🔴"
            print(f"{emoji} {side}: {size} @ ${entry_price:.4f}")
            print(f"   💵 Gross PnL: ${gross_pnl:.2f} | Net PnL: ${net_pnl:.2f}")
            print(f"   🎯 BE: ${break_even:.4f} | TP: ${net_tp:.4f} | SL: ${net_sl:.4f}")
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
        print(f"🎯 Net TP: {self.config['net_take_profit']}% | Net SL: {self.config['net_stop_loss']}%")
        print(f"💎 Using MAKER-ONLY orders for {self.config['maker_fee_pct']}% fee rebate")
        
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