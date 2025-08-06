import os
import asyncio
import pandas as pd
import json
from datetime import datetime
from pybit.unified_trading import HTTP
from dotenv import load_dotenv

load_dotenv()

class EMAMACDRSIBot:
    def __init__(self):
        self.symbol = 'SOLUSDT'
        self.demo_mode = os.getenv('DEMO_MODE', 'true').lower() == 'true'
        
        prefix = 'TESTNET_' if self.demo_mode else 'LIVE_'
        self.api_key = os.getenv(f'{prefix}BYBIT_API_KEY')
        self.api_secret = os.getenv(f'{prefix}BYBIT_API_SECRET')
        
        self.exchange = None
        self.position = None
        self.price_data = pd.DataFrame()
        self.trade_id = 0
        
        # Order management - ADDED (NO COOLDOWN)
        self.pending_order = None
        self.last_order_time = None
        self.order_timeout = 180  # cancel orders older than this
        self.last_signal = None  # track last signal to avoid duplicates
        
        self.config = {
            'timeframe': '5',
            'ema_short': 12,
            'ema_long': 26,
            'macd_signal': 9,
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_neutral_low': 50,
            'position_size': 100,
            'lookback': 100,
            'maker_offset_pct': 0.01,
            'net_take_profit': 1.08,
            'net_stop_loss': 0.42,
        }
        
        # Status tracking
        self.last_status_time = None
        self.status_interval = 5  # Show status every 5 seconds
        
        os.makedirs("logs", exist_ok=True)
        self.log_file = "logs/3_FEES_EMAMACDRSI_SOLUSDT.log"
    
    def connect(self):
        try:
            self.exchange = HTTP(demo=self.demo_mode, api_key=self.api_key, api_secret=self.api_secret)
            return self.exchange.get_server_time().get('retCode') == 0
        except:
            return False
    
    async def check_pending_orders(self):
        """Check and manage pending orders - ADDED"""
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
                        
                        if age > self.order_timeout:
                            self.exchange.cancel_order(
                                category="linear",
                                symbol=self.symbol,
                                orderId=order['orderId']
                            )
                            print(f"❌ Cancelled stale order (aged {age:.0f}s)")
                            self.pending_order = None
                            self.last_signal = None
                        else:
                            self.pending_order = order
                            return True  # Has pending order
                else:
                    self.pending_order = None
                    return False
            return False
        except Exception as e:
            return False
    
    def calculate_indicators(self, df):
        if len(df) < self.config['lookback']:
            return None
        
        close = df['close']
        
        ema_short = close.ewm(span=self.config['ema_short']).mean()
        ema_long = close.ewm(span=self.config['ema_long']).mean()
        macd_line = ema_short - ema_long
        signal_line = macd_line.ewm(span=self.config['macd_signal']).mean()
        histogram = macd_line - signal_line
        
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
        analysis = self.calculate_indicators(df)
        if not analysis:
            return None
        
        # Check if signal is duplicate of last one - ADDED
        if self.last_signal:
            price_change = abs(analysis['price'] - self.last_signal['price']) / self.last_signal['price']
            if price_change < 0.001:  # Less than 0.1% price change
                return None  # Skip duplicate signal
        
        if analysis['histogram_flip'] and analysis['rsi_above_50']:
            return {'action': 'BUY', 'price': analysis['price'], 'rsi': analysis['rsi']}
        
        if analysis['histogram_reversal']:
            return {'action': 'SELL', 'price': analysis['price'], 'rsi': analysis['rsi']}
        
        return None
    
    async def get_market_data(self):
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
        
        analysis = self.calculate_indicators(self.price_data)
        if analysis and analysis['histogram_reversal']:
            return True, "macd_reversal"
        
        return False, ""
    
    async def execute_trade(self, signal):
        # NO COOLDOWN - Removed
        
        # Check for pending orders - ADDED
        if await self.check_pending_orders():
            return  # Silent return if pending order exists
        
        # Check if already in position - ADDED
        if self.position:
            return
        
        qty = self.config['position_size'] / signal['price']
        
        # SOLUSDT uses integer quantities
        formatted_qty = str(int(round(qty)))
        if int(formatted_qty) == 0:
            return
            
        # LIMIT order for entry
        offset_mult = 1 - self.config['maker_offset_pct']/100 if signal['action'] == 'BUY' else 1 + self.config['maker_offset_pct']/100
        limit_price = round(signal['price'] * offset_mult, 4)
        
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
                self.last_order_time = datetime.now()  # ADDED
                self.last_signal = signal  # ADDED
                self.pending_order = order['result']  # ADDED
                
                print(f"\n✅ {signal['action']} Order Placed:")
                print(f"   📊 Quantity: {formatted_qty} SOL @ ${limit_price:.4f}")
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
        
        # MARKET order for exit
        try:
            order = self.exchange.place_order(
                category="linear",
                symbol=self.symbol,
                side=side,
                orderType="Market",
                qty=str(int(round(qty))),
                reduceOnly=True
            )
            
            if order.get('retCode') == 0:
                pnl = float(self.position.get('unrealisedPnl', 0))
                entry_price = float(self.position.get('avgPrice', 0))
                current_price = float(self.price_data['close'].iloc[-1])
                
                print(f"\n💰 Position Closed: {reason}")
                print(f"   Entry: ${entry_price:.4f} → Exit: ${current_price:.4f}")
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
                'price': round(price, 6),
                'info': info
            }) + "\n")
    
    def show_status(self):
        """Show status periodically - ADDED"""
        now = datetime.now()
        
        if self.last_status_time:
            if (now - self.last_status_time).total_seconds() < self.status_interval:
                return
        
        self.last_status_time = now
        
        if len(self.price_data) == 0:
            return
        
        current_price = float(self.price_data['close'].iloc[-1])
        
        # Build status line
        status_parts = [f"📊 SOL: ${current_price:.2f}"]
        
        if self.position:
            entry = float(self.position.get('avgPrice', 0))
            pnl = float(self.position.get('unrealisedPnl', 0))
            side = self.position.get('side', '')
            status_parts.append(f"| 📍 {side} from ${entry:.4f} | PnL: ${pnl:+.2f}")
        elif self.pending_order:
            order_price = float(self.pending_order.get('price', 0))
            order_side = self.pending_order.get('side', '')
            order_time = int(self.pending_order.get('createdTime', 0)) / 1000
            age = int(datetime.now().timestamp() - order_time)
            status_parts.append(f"| ⏳ Pending {order_side} @ ${order_price:.4f} ({age}s)")
        else:
            indicators = self.calculate_indicators(self.price_data)
            if indicators:
                status_parts.append(f"| RSI: {indicators['rsi']:.1f}")
                trend = "UP" if indicators['ema_aligned'] else "DOWN"
                status_parts.append(f"| Trend: {trend}")
        
        print(" ".join(status_parts), end='\r')
    
    async def run_cycle(self):
        if not await self.get_market_data():
            return
        
        await self.check_position()
        await self.check_pending_orders()  # ADDED - Check and clean up orders
        
        if self.position:
            should_close, reason = self.should_close()
            if should_close:
                await self.close_position(reason)
        elif not self.pending_order:  # MODIFIED - Only generate signal if no pending order
            signal = self.generate_signal(self.price_data)
            if signal:
                await self.execute_trade(signal)
        
        self.show_status()  # ADDED - Show periodic status
    
    async def run(self):
        if not self.connect():
            print("❌ Failed to connect")
            return
        
        print(f"🚀 Starting EMA+MACD+RSI Bot for {self.symbol}")
        print(f"📊 Strategy: MACD histogram + RSI confirmation")
        print(f"⏰ Timeframe: {self.config['timeframe']} minutes")
        print(f"🎯 Take Profit: {self.config['net_take_profit']}% | Stop Loss: {self.config['net_stop_loss']}%")
        print(f"⏱️ No cooldown | Order timeout: {self.order_timeout}s")
        print(f"{'='*50}")
        
        while True:
            try:
                await self.run_cycle()
                await asyncio.sleep(5)
            except KeyboardInterrupt:
                print(f"\n{'='*50}")
                print("🛑 Shutting down bot...")
                
                # Cancel all pending orders on shutdown - ADDED
                try:
                    cancelled = self.exchange.cancel_all_orders(category="linear", symbol=self.symbol)
                    if cancelled.get('retCode') == 0:
                        result = cancelled.get('result', {})
                        if result.get('list'):
                            print(f"✅ Cancelled {len(result['list'])} pending orders")
                except:
                    pass
                
                # Close position if exists - IMPROVED
                if self.position:
                    print("📍 Closing open position...")
                    await self.close_position("manual_stop")
                
                print("✅ Bot stopped successfully")
                break
                
            except Exception as e:
                print(f"\n⚠️ Error: {e}")
                await asyncio.sleep(5)

if __name__ == "__main__":
    bot = EMAMACDRSIBot()
    asyncio.run(bot.run())