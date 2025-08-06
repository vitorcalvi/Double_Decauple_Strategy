import os
import asyncio
import pandas as pd
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
        
        # Strategy Config with Fee Calculations
        self.config = {
            'ema_fast': 5,
            'ema_slow': 13,
            'rsi_period': 5,
            'rsi_oversold': 25,
            'rsi_overbought': 75,
            'volume_threshold': 1.5,
            'position_size': 100,
            'min_rsi_diff': 2.0,
            'maker_offset_pct': 0.01,
            # Fee structure
            'maker_fee_pct': -0.04,  # Negative = rebate
            # Gross TP/SL
            'gross_take_profit': 0.35,
            'gross_stop_loss': 0.20,
            # Net TP/SL (adjusted for 2x maker rebate)
            'net_take_profit': 0.43,  # 0.35 + 0.08 rebate
            'net_stop_loss': 0.12,    # 0.20 - 0.08 rebate
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
        except:
            return False
    
    def format_qty(self, qty):
        """Format quantity for BTC."""
        if qty < self.min_qty:
            return "0"
        return f"{round(qty / 0.001) * 0.001:.3f}"
    
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
        """Execute maker-only trade."""
        current_price = signal['price']
        qty = self.config['position_size'] / current_price
        formatted_qty = self.format_qty(qty)
        
        if formatted_qty == "0":
            return
        
        # Calculate limit price
        offset_mult = 1 - self.config['maker_offset_pct']/100 if signal['action'] == 'BUY' else 1 + self.config['maker_offset_pct']/100
        limit_price = round(current_price * offset_mult, 2)
        
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
                             f"RSI:{signal['rsi']:.1f}_BE:{break_even:.0f}_NetTP:{net_tp:.0f}")
                
                print(f"✅ MAKER {signal['action']}: {formatted_qty} BTC @ ${limit_price:,.0f}")
                print(f"   📊 Break-Even: ${break_even:,.0f} | Net TP: ${net_tp:,.0f} | Net SL: ${net_sl:,.0f}")
                print(f"   📈 RSI: {signal['rsi']:.1f} | Trend: {signal['trend']} | Vol: {signal['volume_ratio']:.1f}x")
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
        
        # Calculate limit price
        offset_mult = 1 + self.config['maker_offset_pct']/100 if side == "Sell" else 1 - self.config['maker_offset_pct']/100
        limit_price = round(current_price * offset_mult, 2)
        
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
                print(f"✅ Closed: {reason} | Gross PnL: ${gross_pnl:.2f} | Net PnL: ${net_pnl:.2f}")
        except Exception as e:
            print(f"❌ Close failed: {e}")
    
    def log_trade(self, action, price, info):
        """Log trade."""
        with open(self.log_file, "a") as f:
            f.write(json.dumps({
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'id': self.trade_id,
                'action': action,
                'price': round(price, 2),
                'info': info
            }) + "\n")
    
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
            entry_price = float(self.position.get('avgPrice', 0))
            side = self.position.get('side', '')
            size = self.position.get('size', '0')
            
            # Calculate current NET PnL
            gross_pnl = float(self.position.get('unrealisedPnl', 0))
            fee_earned = (entry_price * float(size)) * abs(self.config['maker_fee_pct']) / 100
            net_pnl = gross_pnl + fee_earned
            
            # Calculate break-even and targets
            break_even = self.calculate_break_even(entry_price, side)
            net_tp, net_sl = self.calculate_net_targets(entry_price, side)
            
            emoji = "🟢" if side == "Buy" else "🔴"
            print(f"{emoji} {side}: {size} BTC @ ${entry_price:,.0f}")
            print(f"   💵 Gross PnL: ${gross_pnl:.2f} | Net PnL: ${net_pnl:.2f}")
            print(f"   🎯 BE: ${break_even:,.0f} | TP: ${net_tp:,.0f} | SL: ${net_sl:,.0f}")
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
    bot = EMARSIBot()
    asyncio.run(bot.run())