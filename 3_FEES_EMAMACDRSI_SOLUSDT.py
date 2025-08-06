import os
import asyncio
import pandas as pd
import json
from datetime import datetime
from pybit.unified_trading import HTTP
from dotenv import load_dotenv

load_dotenv()

class EMAMACDRSIBot:
    """Strategy 1: 5-Minute EMA + MACD + RSI Momentum (83% Win Rate)"""
    
    def __init__(self):
        self.symbol = 'SOLUSDT'
        self.demo_mode = os.getenv('DEMO_MODE', 'true').lower() == 'true'
        
        # API setup
        prefix = 'TESTNET_' if self.demo_mode else 'LIVE_'
        self.api_key = os.getenv(f'{prefix}BYBIT_API_KEY')
        self.api_secret = os.getenv(f'{prefix}BYBIT_API_SECRET')
        
        # Trading state
        self.exchange = None
        self.position = None
        self.price_data = pd.DataFrame()
        self.trade_id = 0
        
        # Strategy parameters with Fee Calculations
        self.config = {
            'timeframe': '5',
            'ema_short': 12,
            'ema_long': 26,
            'macd_signal': 9,
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'rsi_neutral_low': 50,
            'position_size': 100,
            'lookback': 100,
            'maker_offset_pct': 0.01,
            # Fee structure
            'maker_fee_pct': -0.04,  # Negative = rebate
            # Gross TP/SL
            'gross_take_profit': 1.0,
            'gross_stop_loss': 0.5,
            # Net TP/SL (adjusted for 2x maker rebate)
            'net_take_profit': 1.08,  # 1.0 + 0.08 rebate
            'net_stop_loss': 0.42,    # 0.5 - 0.08 rebate
        }
        
        os.makedirs("logs", exist_ok=True)
        self.log_file = "logs/ema_macd_rsi_trades.log"
    
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
    
    def calculate_indicators(self, df):
        """Calculate all indicators."""
        if len(df) < self.config['lookback']:
            return None
        
        close = df['close']
        
        # EMAs and MACD
        ema_short = close.ewm(span=self.config['ema_short']).mean()
        ema_long = close.ewm(span=self.config['ema_long']).mean()
        macd_line = ema_short - ema_long
        signal_line = macd_line.ewm(span=self.config['macd_signal']).mean()
        histogram = macd_line - signal_line
        
        # RSI
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(window=self.config['rsi_period']).mean()
        loss = -delta.clip(upper=0).rolling(window=self.config['rsi_period']).mean()
        rsi = 100 - (100 / (1 + gain / loss))
        
        return {
            'price': close.iloc[-1],
            'ema_aligned': ema_short.iloc[-1] > ema_long.iloc[-1],
            'histogram_flip': histogram.iloc[-2] < 0 and histogram.iloc[-1] > 0,
            'histogram_reversal': histogram.iloc[-2] > 0 and histogram.iloc[-1] < 0,
            'rsi': rsi.iloc[-1],
            'rsi_above_50': rsi.iloc[-1] > self.config['rsi_neutral_low']
        }
    
    def generate_signal(self, df):
        """Generate trading signal."""
        analysis = self.calculate_indicators(df)
        if not analysis:
            return None
        
        # LONG signal - histogram flip with RSI confirmation
        if analysis['histogram_flip'] and analysis['rsi_above_50']:
            return {'action': 'BUY', 'price': analysis['price'], 'rsi': analysis['rsi']}
        
        # SHORT signal - histogram reversal
        if analysis['histogram_reversal']:
            return {'action': 'SELL', 'price': analysis['price'], 'rsi': analysis['rsi']}
        
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
            
            df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
            
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
        
        if entry_price == 0:
            return False, ""
        
        side = self.position.get('side', '')
        
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
        
        # Check MACD reversal
        analysis = self.calculate_indicators(self.price_data)
        if analysis and analysis['histogram_reversal']:
            return True, "macd_reversal"
        
        return False, ""
    
    async def execute_trade(self, signal):
        """Execute maker-only trade."""
        qty = self.config['position_size'] / signal['price']
        
        if qty < 1.0:
            return
        
        # Calculate limit price with offset
        offset_mult = 1 - self.config['maker_offset_pct']/100 if signal['action'] == 'BUY' else 1 + self.config['maker_offset_pct']/100
        limit_price = round(signal['price'] * offset_mult, 4)
        
        try:
            order = self.exchange.place_order(
                category="linear",
                symbol=self.symbol,
                side="Buy" if signal['action'] == 'BUY' else "Sell",
                orderType="Limit",
                qty=str(int(round(qty))),
                price=str(limit_price),
                timeInForce="PostOnly"
            )
            
            if order.get('retCode') == 0:
                self.trade_id += 1
                
                # Calculate and log break-even
                break_even = self.calculate_break_even(limit_price, signal['action'])
                net_tp, net_sl = self.calculate_net_targets(limit_price, signal['action'])
                
                self.log_trade(signal['action'], limit_price, 
                             f"RSI:{signal['rsi']:.1f}_BE:{break_even:.4f}_NetTP:{net_tp:.4f}")
                
                print(f"📈 MAKER {signal['action']}: {int(qty)} DOGE @ ${limit_price:.4f}")
                print(f"   📊 Break-Even: ${break_even:.4f} | Net TP: ${net_tp:.4f} | Net SL: ${net_sl:.4f}")
                print(f"   📈 Strategy: EMA+MACD+RSI | RSI: {signal['rsi']:.1f}")
        except:
            pass
    
    async def close_position(self, reason):
        """Close position with maker order."""
        if not self.position:
            return
        
        current_price = float(self.price_data['close'].iloc[-1])
        side = "Sell" if self.position.get('side') == "Buy" else "Buy"
        qty = float(self.position['size'])
        entry_price = float(self.position.get('avgPrice', 0))
        
        # Calculate limit price with offset
        offset_mult = 1 + self.config['maker_offset_pct']/100 if side == "Sell" else 1 - self.config['maker_offset_pct']/100
        limit_price = round(current_price * offset_mult, 4)
        
        try:
            order = self.exchange.place_order(
                category="linear",
                symbol=self.symbol,
                side=side,
                orderType="Limit",
                qty=str(int(round(qty))),
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
        except:
            pass
    
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
        if self.price_data.empty:
            return
        
        analysis = self.calculate_indicators(self.price_data)
        if not analysis:
            return
        
        print(f"\n📊 EMA+MACD+RSI Strategy - {self.symbol}")
        print(f"💰 Price: ${analysis['price']:.4f}")
        print(f"📈 RSI: {analysis['rsi']:.1f} | EMA Aligned: {analysis['ema_aligned']}")
        
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
            print(f"{emoji} {side}: {size} DOGE @ ${entry_price:.4f}")
            print(f"   💵 Gross PnL: ${gross_pnl:.2f} | Net PnL: ${net_pnl:.2f}")
            print(f"   🎯 BE: ${break_even:.4f} | TP: ${net_tp:.4f} | SL: ${net_sl:.4f}")
        else:
            print("⏳ Waiting for MACD histogram flip...")
        
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
        
        print(f"🚀 Strategy 1: EMA+MACD+RSI Momentum Bot (83% Win Rate)")
        print(f"⏰ Timeframe: 5 minutes")
        print(f"🎯 Net TP: {self.config['net_take_profit']}% | Net SL: {self.config['net_stop_loss']}%")
        print(f"💎 Using MAKER-ONLY orders for {self.config['maker_fee_pct']}% fee rebate")
        
        while True:
            try:
                await self.run_cycle()
                await asyncio.sleep(5)
            except KeyboardInterrupt:
                print("\nBot stopped")
                if self.position:
                    await self.close_position("manual_stop")
                break
            except Exception as e:
                print(f"Error: {e}")
                await asyncio.sleep(5)

if __name__ == "__main__":
    bot = EMAMACDRSIBot()
    asyncio.run(bot.run())