import os
import asyncio
import pandas as pd
import json
from datetime import datetime
from pybit.unified_trading import HTTP
from dotenv import load_dotenv

load_dotenv()

class ETHScalpingBot:
    def __init__(self):
        self.symbol = 'ETHUSDT'
        self.demo_mode = os.getenv('DEMO_MODE', 'true').lower() == 'true'
        
        prefix = 'TESTNET_' if self.demo_mode else 'LIVE_'
        self.api_key = os.getenv(f'{prefix}BYBIT_API_KEY')
        self.api_secret = os.getenv(f'{prefix}BYBIT_API_SECRET')
        self.exchange = None
        
        self.position = None
        self.price_data = pd.DataFrame()
        self.trade_id = 0
        self.pending_order = None
        self.last_signal = None
        self.last_order_time = None
        self.order_timeout = 60  # Cancel after 60 seconds
        self.order_cooldown = 30  # 30 second cooldown between orders
        
        # FIXED CONFIG - Proper position sizing and risk/reward
        self.config = {
            'ema_fast': 5,
            'ema_slow': 13,
            'bb_period': 20,
            'bb_std': 2.0,
            'position_size': 500,  # FIXED: Minimum for ETHUSDT
            'maker_offset_pct': 0.02,  # Slightly higher offset to ensure PostOnly
            'net_take_profit': 1.05,  # FIXED: 1% net after fees (was 0.28%)
            'net_stop_loss': 0.45,    # FIXED: Tighter stop with fee rebate (was 0.07%)
        }
        
        os.makedirs("logs", exist_ok=True)
        self.log_file = "logs/1_FEES_EMA_BB_ETHUSDT_FIXED.log"
    
    def connect(self):
        try:
            self.exchange = HTTP(demo=self.demo_mode, api_key=self.api_key, api_secret=self.api_secret)
            return self.exchange.get_server_time().get('retCode') == 0
        except:
            return False
    
    def format_qty(self, qty):
        """FIXED: Proper quantity formatting for ETHUSDT"""
        # ETHUSDT requires 3 decimals
        return f"{round(qty / 0.001) * 0.001:.3f}"
    
    async def check_pending_orders(self):
        """FIXED: Proper order lifecycle management"""
        try:
            orders = self.exchange.get_open_orders(category="linear", symbol=self.symbol)
            if orders.get('retCode') != 0:
                return False
            
            order_list = orders['result']['list']
            if not order_list:
                self.pending_order = None
                return False
            
            order = order_list[0]
            age = datetime.now().timestamp() - int(order['createdTime']) / 1000
            
            if age > self.order_timeout:
                self.exchange.cancel_order(category="linear", symbol=self.symbol, orderId=order['orderId'])
                print(f"❌ Cancelled stale order (aged {age:.0f}s)")
                self.pending_order = None
                self.last_signal = None
                return False
            
            self.pending_order = order
            return True
        except:
            return False
    
    def is_valid_signal(self, signal):
        """FIXED: Signal validation to prevent duplicates"""
        if not signal:
            return False
            
        # Check cooldown
        if self.last_order_time:
            elapsed = (datetime.now() - self.last_order_time).total_seconds()
            if elapsed < self.order_cooldown:
                return False
        
        # Check for duplicate signal
        if self.last_signal:
            price_change = abs(signal['price'] - self.last_signal['price']) / self.last_signal['price']
            if price_change < 0.001:  # Less than 0.1% change
                return False
        
        return True
    
    def calculate_indicators(self, df):
        if len(df) < 20:
            return None
        
        close = df['close']
        ema_fast = close.ewm(span=self.config['ema_fast']).mean().iloc[-1]
        ema_slow = close.ewm(span=self.config['ema_slow']).mean().iloc[-1]
        
        sma = close.rolling(window=self.config['bb_period']).mean()
        std = close.rolling(window=self.config['bb_period']).std()
        upper_band = (sma + std * self.config['bb_std']).iloc[-1]
        lower_band = (sma - std * self.config['bb_std']).iloc[-1]
        
        bb_range = upper_band - lower_band
        bb_position = (close.iloc[-1] - lower_band) / bb_range if bb_range != 0 else 0.5
        
        return {
            'price': close.iloc[-1],
            'ema_fast': ema_fast,
            'ema_slow': ema_slow,
            'bb_position': bb_position,
            'trend': 'UP' if ema_fast > ema_slow else 'DOWN'
        }
    
    def generate_signal(self, df):
        indicators = self.calculate_indicators(df)
        if not indicators:
            return None
        
        # Long signal - EMA bullish and price oversold
        if indicators['trend'] == 'UP' and indicators['bb_position'] <= 0.3:
            signal = {'action': 'BUY', 'price': indicators['price'], 'bb_pos': indicators['bb_position']}
            if self.is_valid_signal(signal):
                return signal
        
        # Short signal - EMA bearish and price overbought
        if indicators['trend'] == 'DOWN' and indicators['bb_position'] >= 0.7:
            signal = {'action': 'SELL', 'price': indicators['price'], 'bb_pos': indicators['bb_position']}
            if self.is_valid_signal(signal):
                return signal
        
        return None
    
    async def get_market_data(self):
        try:
            klines = self.exchange.get_kline(category="linear", symbol=self.symbol, interval="1", limit=50)
            if klines.get('retCode') != 0:
                return False
            
            df = pd.DataFrame(klines['result']['list'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
            
            self.price_data = df.sort_values('timestamp').reset_index(drop=True)
            return True
        except:
            return False
    
    async def check_position(self):
        try:
            positions = self.exchange.get_positions(category="linear", symbol=self.symbol)
            if positions.get('retCode') == 0:
                pos_list = positions['result']['list']
                self.position = pos_list[0] if pos_list and float(pos_list[0]['size']) > 0 else None
        except:
            pass
    
    def should_close(self):
        """FIXED: Proper risk/reward calculation"""
        if not self.position:
            return False, ""
        
        current_price = float(self.price_data['close'].iloc[-1])
        entry_price = float(self.position.get('avgPrice', 0))
        side = self.position.get('side', '')
        
        if entry_price == 0:
            return False, ""
        
        profit_pct = ((current_price - entry_price) / entry_price * 100) if side == "Buy" else ((entry_price - current_price) / entry_price * 100)
        
        # Fixed risk/reward with proper targets
        if profit_pct >= self.config['net_take_profit']:
            return True, "take_profit"
        if profit_pct <= -self.config['net_stop_loss']:
            return True, "stop_loss"
        
        return False, ""
    
    async def execute_trade(self, signal):
        """FIXED: Proper order execution with all safeguards"""
        # Check all conditions
        if await self.check_pending_orders():
            return
        if self.position:
            return
        if not self.is_valid_signal(signal):
            return
        
        qty = self.config['position_size'] / signal['price']
        formatted_qty = self.format_qty(qty)
        
        # Verify minimum quantity
        if float(formatted_qty) < 0.001:
            print(f"⚠️ Quantity too small: {formatted_qty}")
            return
        
        # Calculate limit price with proper offset
        offset = 1 - self.config['maker_offset_pct']/100 if signal['action'] == 'BUY' else 1 + self.config['maker_offset_pct']/100
        limit_price = round(signal['price'] * offset, 2)
        
        try:
            # FIXED: Limit order with PostOnly for maker rebate
            order = self.exchange.place_order(
                category="linear",
                symbol=self.symbol,
                side="Buy" if signal['action'] == 'BUY' else "Sell",
                orderType="Limit",
                qty=formatted_qty,
                price=str(limit_price),
                timeInForce="PostOnly"  # CRITICAL: Earn rebate
            )
            
            if order.get('retCode') == 0:
                self.trade_id += 1
                self.last_signal = signal
                self.last_order_time = datetime.now()
                self.pending_order = order['result']
                
                print(f"✅ {signal['action']} Order Placed:")
                print(f"   📊 Quantity: {formatted_qty} ETH @ ${limit_price:.2f}")
                print(f"   📈 BB Position: {signal['bb_pos']:.2f}")
                print(f"   🎯 Risk/Reward: 1:{self.config['net_take_profit']/self.config['net_stop_loss']:.1f}")
                
                self.log_trade(signal['action'], limit_price, f"BB:{signal['bb_pos']:.2f}")
            else:
                print(f"❌ Order failed: {order.get('retMsg', 'Unknown error')}")
        except Exception as e:
            print(f"❌ Trade failed: {e}")
    
    async def close_position(self, reason):
        """FIXED: Market order for immediate exit"""
        if not self.position:
            return
        
        side = "Sell" if self.position.get('side') == "Buy" else "Buy"
        qty = float(self.position['size'])
        formatted_qty = self.format_qty(qty)
        
        try:
            # FIXED: Market order for quick exit
            order = self.exchange.place_order(
                category="linear",
                symbol=self.symbol,
                side=side,
                orderType="Market",  # Quick exit
                qty=formatted_qty,
                reduceOnly=True  # Only close position
            )
            
            if order.get('retCode') == 0:
                pnl = float(self.position.get('unrealisedPnl', 0))
                entry = float(self.position.get('avgPrice', 0))
                current = float(self.price_data['close'].iloc[-1])
                
                print(f"\n💰 Position Closed: {reason}")
                print(f"   Entry: ${entry:.2f} → Exit: ${current:.2f}")
                print(f"   PnL: ${pnl:.2f}")
                
                self.log_trade("CLOSE", current, f"{reason}_PnL:${pnl:.2f}")
                self.position = None
                self.last_signal = None
        except Exception as e:
            print(f"❌ Close failed: {e}")
    
    def log_trade(self, action, price, info):
        with open(self.log_file, "a") as f:
            f.write(json.dumps({
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'action': action,
                'price': round(price, 2),
                'info': info
            }) + "\n")
    
    def show_status(self):
        if len(self.price_data) == 0:
            return
        
        current_price = float(self.price_data['close'].iloc[-1])
        indicators = self.calculate_indicators(self.price_data)
        
        status = f"📊 ETH: ${current_price:,.2f}"
        
        if self.position:
            entry = float(self.position.get('avgPrice', 0))
            pnl = float(self.position.get('unrealisedPnl', 0))
            side = self.position.get('side', '')
            pct = ((current_price - entry) / entry * 100) if side == "Buy" else ((entry - current_price) / entry * 100)
            status += f" | 📍 {side} @ ${entry:.2f} | {pct:+.2f}% | PnL: ${pnl:+.2f}"
        elif self.pending_order:
            order_price = float(self.pending_order.get('price', 0))
            order_side = self.pending_order.get('side', '')
            age = int(datetime.now().timestamp() - int(self.pending_order.get('createdTime', 0)) / 1000)
            status += f" | ⏳ Pending {order_side} @ ${order_price:.2f} ({age}s)"
        elif indicators:
            status += f" | BB: {indicators['bb_position']:.2f} | Trend: {indicators['trend']}"
        
        print(status, end='\r')
    
    async def run_cycle(self):
        if not await self.get_market_data():
            return
        
        await self.check_position()
        await self.check_pending_orders()
        
        if self.position:
            should_close, reason = self.should_close()
            if should_close:
                await self.close_position(reason)
        elif not self.pending_order:
            signal = self.generate_signal(self.price_data)
            if signal:
                await self.execute_trade(signal)
        
        self.show_status()
    
    async def run(self):
        if not self.connect():
            print("❌ Failed to connect")
            return
        
        print(f"🚀 FIXED EMA + BB Bot for {self.symbol}")
        print(f"💰 Position Size: ${self.config['position_size']} (Minimum for ETH)")
        print(f"🎯 TP: {self.config['net_take_profit']}% | SL: {self.config['net_stop_loss']}%")
        print(f"⚖️ Risk/Reward Ratio: 1:{self.config['net_take_profit']/self.config['net_stop_loss']:.1f}")
        print(f"⏱️ Cooldown: {self.order_cooldown}s | Timeout: {self.order_timeout}s")
        print(f"✅ Connected! Starting bot with proper risk management...")
        print("-" * 50)
        
        while True:
            try:
                await self.run_cycle()
                
                # FIXED: Proper loop timing based on state
                if self.pending_order:
                    await asyncio.sleep(5)   # Check pending less often
                elif self.position:
                    await asyncio.sleep(3)   # Monitor position
                else:
                    await asyncio.sleep(10)  # Scan for entries slower
                    
            except KeyboardInterrupt:
                print("\n" + "=" * 50)
                print("🛑 Shutting down bot...")
                
                # Cancel all pending orders
                try:
                    cancelled = self.exchange.cancel_all_orders(category="linear", symbol=self.symbol)
                    if cancelled.get('retCode') == 0:
                        result = cancelled.get('result', {})
                        if result.get('list'):
                            print(f"✅ Cancelled {len(result['list'])} pending orders")
                except:
                    pass
                
                # Close position if exists
                if self.position:
                    print("📍 Closing open position...")
                    await self.close_position("manual_stop")
                
                print("✅ Bot stopped successfully")
                break
                
            except Exception as e:
                print(f"\n⚠️ Error: {e}")
                await asyncio.sleep(5)

if __name__ == "__main__":
    bot = ETHScalpingBot()
    asyncio.run(bot.run())