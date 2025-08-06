import os
import asyncio
import pandas as pd
import json
from datetime import datetime
from pybit.unified_trading import HTTP
from dotenv import load_dotenv

load_dotenv()

class PivotReversalBot:
    def __init__(self):
        self.symbol = 'BTCUSDT'
        self.demo_mode = os.getenv('DEMO_MODE', 'true').lower() == 'true'
        
        prefix = 'TESTNET_' if self.demo_mode else 'LIVE_'
        self.api_key = os.getenv(f'{prefix}BYBIT_API_KEY')
        self.api_secret = os.getenv(f'{prefix}BYBIT_API_SECRET')
        self.exchange = None
        
        self.position = None
        self.price_data = pd.DataFrame()
        self.trade_id = 0
        
        # Order management - IMPROVED (NO COOLDOWN)
        self.pending_order = None
        self.last_order_time = None
        self.order_timeout = 180  # Cancel orders older than 180 seconds
        self.last_signal = None  # Track last signal to avoid duplicates
        
        # SIMPLIFIED config
        self.config = {
            'rsi_period': 7,
            'rsi_oversold': 40,  # Relaxed from 30
            'rsi_overbought': 60, # Relaxed from 70
            'position_size': 100,
            'maker_offset_pct': 0.01,
            'net_take_profit': 0.68,
            'net_stop_loss': 0.07,
        }
        
        # Status tracking
        self.last_status_time = None
        self.status_interval = 5  # Show status every 5 seconds
        
        os.makedirs("logs", exist_ok=True)
        self.log_file = "logs/5_FEES_MACD_VWAP_BTCUSDT.log"
    
    def connect(self):
        try:
            self.exchange = HTTP(demo=self.demo_mode, api_key=self.api_key, api_secret=self.api_secret)
            return self.exchange.get_server_time().get('retCode') == 0
        except:
            return False
    
    async def check_pending_orders(self):
        """Check and manage pending orders - IMPROVED"""
        try:
            orders = self.exchange.get_open_orders(
                category="linear",
                symbol=self.symbol
            )
            
            if orders.get('retCode') == 0:
                order_list = orders['result']['list']
                
                if order_list:
                    for order in order_list:
                        order_time = int(order['createdTime']) / 1000
                        age = datetime.now().timestamp() - order_time
                        
                        # Only cancel if REALLY old
                        if age > self.order_timeout:
                            self.exchange.cancel_order(
                                category="linear",
                                symbol=self.symbol,
                                orderId=order['orderId']
                            )
                            print(f"❌ Cancelled stale order (aged {age:.0f}s): {order['orderId']}")
                            self.pending_order = None
                            self.last_signal = None  # Reset signal tracking
                        else:
                            self.pending_order = order
                            # Don't spam the console with pending order status
                            return True
                else:
                    self.pending_order = None
                    return False
            return False
        except Exception as e:
            print(f"⚠️ Order check error: {e}")
            return False
    
    def format_qty(self, qty):
        # BTCUSDT uses 0.001 minimum - FIXED formatting
        return f"{round(qty / 0.001) * 0.001:.3f}"
    
    def calculate_indicators(self, df):
        if len(df) < self.config['rsi_period'] + 1:
            return None
        
        close = df['close']
        
        # RSI only (simplified)
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(window=self.config['rsi_period']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.config['rsi_period']).mean()
        
        epsilon = 1e-10
        rs = gain / (loss + epsilon)
        rsi = 100 - (100 / (1 + rs))
        
        return {
            'rsi': rsi.iloc[-1] if not rsi.empty else 50,
            'rsi_prev': rsi.iloc[-2] if len(rsi) > 1 else 50
        }
    
    def generate_signal(self, df):
        if len(df) < 20:
            return None
        
        current_price = float(df['close'].iloc[-1])
        
        indicators = self.calculate_indicators(df)
        if not indicators:
            return None
        
        rsi = indicators['rsi']
        rsi_prev = indicators['rsi_prev']
        
        # Check if signal is duplicate of last one
        if self.last_signal:
            price_change = abs(current_price - self.last_signal['price']) / self.last_signal['price']
            if price_change < 0.001:  # Less than 0.1% price change
                return None  # Skip duplicate signal
        
        # SIMPLIFIED: Just RSI levels with momentum
        # BUY Signal
        if rsi <= self.config['rsi_oversold'] and rsi > rsi_prev:
            signal = {'action': 'BUY', 'price': current_price, 'rsi': rsi}
            return signal
        
        # SELL Signal
        if rsi >= self.config['rsi_overbought'] and rsi < rsi_prev:
            signal = {'action': 'SELL', 'price': current_price, 'rsi': rsi}
            return signal
        
        return None
    
    async def get_market_data(self):
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
        try:
            positions = self.exchange.get_positions(category="linear", symbol=self.symbol)
            if positions.get('retCode') == 0:
                pos_list = positions['result']['list']
                self.position = pos_list[0] if pos_list and float(pos_list[0]['size']) > 0 else None
        except:
            pass
    
    def should_close(self):
        if not self.position:
            return False, ""
        
        current_price = float(self.price_data['close'].iloc[-1])
        entry_price = float(self.position.get('avgPrice', 0))
        side = self.position.get('side', '')
        
        if entry_price == 0:
            return False, ""
        
        if side == "Buy":
            profit_pct = ((current_price - entry_price) / entry_price) * 100
            if profit_pct >= self.config['net_take_profit']:
                return True, "take_profit"
            if profit_pct <= -self.config['net_stop_loss']:
                return True, "stop_loss"
        else:
            profit_pct = ((entry_price - current_price) / entry_price) * 100
            if profit_pct >= self.config['net_take_profit']:
                return True, "take_profit"
            if profit_pct <= -self.config['net_stop_loss']:
                return True, "stop_loss"
        
        return False, ""
    
    async def execute_trade(self, signal):
        # NO COOLDOWN - Removed
        
        # Check for pending orders
        if await self.check_pending_orders():
            # Silent return - don't spam console about pending orders
            return
        
        # Check if already in position
        if self.position:
            return
        
        qty = self.config['position_size'] / signal['price']
        formatted_qty = self.format_qty(qty)
        
        if float(formatted_qty) < 0.001:
            return
        
        offset_mult = 1 - self.config['maker_offset_pct']/100 if signal['action'] == 'BUY' else 1 + self.config['maker_offset_pct']/100
        limit_price = round(signal['price'] * offset_mult, 2)
        
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
                self.last_order_time = datetime.now()
                self.last_signal = signal  # Track the signal
                self.pending_order = order['result']
                print(f"\n✅ {signal['action']} Order Placed:")
                print(f"   📊 Quantity: {formatted_qty} BTC @ ${limit_price:,.2f}")
                print(f"   📈 RSI: {signal['rsi']:.1f}")
                print(f"   ⏱️ Order timeout: {self.order_timeout}s")
                self.log_trade(signal['action'], limit_price, f"RSI:{signal['rsi']:.1f}")
        except Exception as e:
            print(f"❌ Trade failed: {e}")
    
    async def close_position(self, reason):
        if not self.position:
            return
        
        side = "Sell" if self.position.get('side') == "Buy" else "Buy"
        qty = float(self.position['size'])
        
        try:
            order = self.exchange.place_order(
                category="linear",
                symbol=self.symbol,
                side=side,
                orderType="Market",
                qty=self.format_qty(qty),
                reduceOnly=True
            )
            
            if order.get('retCode') == 0:
                pnl = float(self.position.get('unrealisedPnl', 0))
                entry_price = float(self.position.get('avgPrice', 0))
                current_price = float(self.price_data['close'].iloc[-1])
                
                print(f"\n💰 Position Closed: {reason}")
                print(f"   Entry: ${entry_price:,.2f} → Exit: ${current_price:,.2f}")
                print(f"   PnL: ${pnl:.2f}")
                self.log_trade("CLOSE", current_price, f"{reason}_PnL:${pnl:.2f}")
                self.position = None
                self.last_signal = None  # Reset signal tracking
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
        """Show status periodically, not every cycle"""
        now = datetime.now()
        
        # Only show status every N seconds
        if self.last_status_time:
            if (now - self.last_status_time).total_seconds() < self.status_interval:
                return
        
        self.last_status_time = now
        
        if len(self.price_data) == 0:
            return
        
        current_price = float(self.price_data['close'].iloc[-1])
        
        # Build status line
        status_parts = [f"📊 BTC: ${current_price:,.2f}"]
        
        if self.position:
            entry = float(self.position.get('avgPrice', 0))
            pnl = float(self.position.get('unrealisedPnl', 0))
            side = self.position.get('side', '')
            status_parts.append(f"| 📍 {side} from ${entry:,.2f} | PnL: ${pnl:+.2f}")
        elif self.pending_order:
            order_price = float(self.pending_order.get('price', 0))
            order_side = self.pending_order.get('side', '')
            order_time = int(self.pending_order.get('createdTime', 0)) / 1000
            age = int(datetime.now().timestamp() - order_time)
            status_parts.append(f"| ⏳ Pending {order_side} @ ${order_price:,.2f} ({age}s)")
        else:
            indicators = self.calculate_indicators(self.price_data)
            if indicators:
                status_parts.append(f"| RSI: {indicators['rsi']:.1f}")
                if indicators['rsi'] <= self.config['rsi_oversold']:
                    status_parts.append(f"| 🟢 Oversold")
                elif indicators['rsi'] >= self.config['rsi_overbought']:
                    status_parts.append(f"| 🔴 Overbought")
        
        # Print single line status
        print(" ".join(status_parts), end='\r')
    
    async def run_cycle(self):
        if not await self.get_market_data():
            return
        
        await self.check_position()
        await self.check_pending_orders()  # Check and clean up orders
        
        if self.position:
            should_close, reason = self.should_close()
            if should_close:
                await self.close_position(reason)
        elif not self.pending_order:  # Only generate signal if no pending order
            signal = self.generate_signal(self.price_data)
            if signal:
                await self.execute_trade(signal)
        
        # Show status periodically
        self.show_status()
    
    async def run(self):
        if not self.connect():
            print("❌ Failed to connect")
            return
        
        print(f"🚀 Starting Pivot Reversal Bot for {self.symbol}")
        print(f"📊 Strategy: RSI-based reversal trading")
        print(f"🎯 Take Profit: {self.config['net_take_profit']}% | Stop Loss: {self.config['net_stop_loss']}%")
        print(f"⏱️ No cooldown | Order timeout: {self.order_timeout}s")
        print(f"{'='*50}")
        
        while True:
            try:
                await self.run_cycle()
                await asyncio.sleep(2)  # 2 second cycle for less spam
            except KeyboardInterrupt:
                print(f"\n{'='*50}")
                print("🛑 Shutting down bot...")
                # Cancel all pending orders on shutdown
                try:
                    cancelled = self.exchange.cancel_all_orders(category="linear", symbol=self.symbol)
                    if cancelled.get('retCode') == 0:
                        result = cancelled.get('result', {})
                        if result.get('list'):
                            print(f"✅ Cancelled {len(result['list'])} pending orders")
                except:
                    pass
                
                if self.position:
                    print("📍 Closing open position...")
                    await self.close_position("manual_stop")
                
                print("✅ Bot stopped successfully")
                break
            except Exception as e:
                print(f"\n⚠️ Error: {e}")
                await asyncio.sleep(5)

if __name__ == "__main__":
    bot = PivotReversalBot()
    asyncio.run(bot.run())