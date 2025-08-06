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
        
        prefix = 'TESTNET_' if self.demo_mode else 'LIVE_'
        self.api_key = os.getenv(f'{prefix}BYBIT_API_KEY')
        self.api_secret = os.getenv(f'{prefix}BYBIT_API_SECRET')
        self.exchange = None
        
        self.position = None
        self.price_data = pd.DataFrame()
        self.trade_id = 0
        self.pending_order = None
        self.last_signal = None
        self.order_timeout = 180
        
        self.config = {
            'ema_fast': 5,
            'ema_slow': 13,
            'bb_period': 20,
            'bb_std': 2.0,
            'position_size': 400,  # Increased to meet minimum requirements
            'maker_offset_pct': 0.01,
            'net_take_profit': 0.28,
            'net_stop_loss': 0.07,
            'min_qty': 0.1,  # Minimum quantity for ETHUSDT on testnet
        }
        
        os.makedirs("logs", exist_ok=True)
        self.log_file = "logs/1_FEES_EMA_BB_ETHUSDT.log"
    
    def connect(self):
        try:
            self.exchange = HTTP(demo=self.demo_mode, api_key=self.api_key, api_secret=self.api_secret)
            if self.exchange.get_server_time().get('retCode') != 0:
                return False
            
            # Get instrument info to determine correct quantity rules
            try:
                instruments = self.exchange.get_instruments_info(category="linear", symbol=self.symbol)
                if instruments.get('retCode') == 0:
                    inst_info = instruments['result']['list'][0]
                    self.config['min_qty'] = float(inst_info.get('lotSizeFilter', {}).get('minOrderQty', 0.1))
                    qty_step = float(inst_info.get('lotSizeFilter', {}).get('qtyStep', 0.001))
                    print(f"📋 {self.symbol} Rules: Min Qty={self.config['min_qty']}, Step={qty_step}")
            except:
                print(f"⚠️ Using default quantity rules")
            
            return True
        except:
            return False
    
    async def check_pending_orders(self):
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
                print(f"\n❌ Cancelled stale order (aged {age:.0f}s)")
                self.pending_order = None
                self.last_signal = None
                return False
            
            self.pending_order = order
            return True
        except:
            return False
    
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
            'bb_position': bb_position
        }
    
    def generate_signal(self, df):
        indicators = self.calculate_indicators(df)
        if not indicators:
            return None
        
        # Skip duplicate signals
        if self.last_signal:
            price_change = abs(indicators['price'] - self.last_signal['price']) / self.last_signal['price']
            if price_change < 0.002:  # 0.2% threshold
                return None
        
        # Long signal - EMA bullish and price near lower band
        if indicators['ema_fast'] > indicators['ema_slow'] and indicators['bb_position'] <= 0.5:
            return {'action': 'BUY', 'price': indicators['price'], 'bb_pos': indicators['bb_position']}
        
        # Short signal - EMA bearish and price near upper band  
        if indicators['ema_fast'] < indicators['ema_slow'] and indicators['bb_position'] >= 0.5:
            return {'action': 'SELL', 'price': indicators['price'], 'bb_pos': indicators['bb_position']}
        
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
        if not self.position:
            return False, ""
        
        current_price = float(self.price_data['close'].iloc[-1])
        entry_price = float(self.position.get('avgPrice', 0))
        side = self.position.get('side', '')
        
        if entry_price == 0:
            return False, ""
        
        profit_pct = ((current_price - entry_price) / entry_price * 100) if side == "Buy" else ((entry_price - current_price) / entry_price * 100)
        
        if profit_pct >= self.config['net_take_profit']:
            return True, "take_profit"
        if profit_pct <= -self.config['net_stop_loss']:
            return True, "stop_loss"
        
        return False, ""
    
    async def execute_trade(self, signal):
        if await self.check_pending_orders() or self.position:
            return
        
        qty = self.config['position_size'] / signal['price']
        
        # ETHUSDT on testnet often requires minimum 0.1 ETH
        if qty < self.config['min_qty']:
            qty = self.config['min_qty']
        
        # Format with proper decimals
        formatted_qty = f"{qty:.3f}"
        
        # Verify quantity is valid
        if float(formatted_qty) < self.config['min_qty']:
            print(f"⚠️ Quantity too small: {formatted_qty} < {self.config['min_qty']}")
            return
        
        offset = 1 - self.config['maker_offset_pct']/100 if signal['action'] == 'BUY' else 1 + self.config['maker_offset_pct']/100
        limit_price = round(signal['price'] * offset, 2)
        
        try:
            print(f"📝 Placing order: {signal['action']} {formatted_qty} ETH @ ${limit_price:.2f}")
            
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
                self.last_signal = signal
                self.pending_order = order['result']
                print(f"✅ {signal['action']} Order Placed: {formatted_qty} ETH @ ${limit_price:.2f} | BB: {signal['bb_pos']:.2f}")
                self.log_trade(signal['action'], limit_price, f"BB:{signal['bb_pos']:.2f}")
            else:
                print(f"❌ Order failed: {order.get('retMsg', 'Unknown error')}")
        except Exception as e:
            print(f"❌ Trade failed: {e}")
    
    async def close_position(self, reason):
        if not self.position:
            return
        
        side = "Sell" if self.position.get('side') == "Buy" else "Buy"
        qty = float(self.position['size'])
        
        # Format quantity properly
        formatted_qty = f"{qty:.3f}"
        
        try:
            order = self.exchange.place_order(
                category="linear",
                symbol=self.symbol,
                side=side,
                orderType="Market",
                qty=formatted_qty,
                reduceOnly=True
            )
            
            if order.get('retCode') == 0:
                pnl = float(self.position.get('unrealisedPnl', 0))
                print(f"\n💰 Closed: {reason} | PnL: ${pnl:.2f}")
                self.log_trade("CLOSE", float(self.price_data['close'].iloc[-1]), f"{reason}_PnL:${pnl:.2f}")
                self.position = None
                self.last_signal = None
            else:
                print(f"\n❌ Close failed: {order.get('retMsg', 'Unknown error')}")
        except Exception as e:
            print(f"\n❌ Close failed: {e}")
    
    def log_trade(self, action, price, info):
        with open(self.log_file, "a") as f:
            f.write(json.dumps({
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'action': action,
                'price': round(price, 6),
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
            status += f" | 📍 {side} @ ${entry:.2f} PnL: ${pnl:+.2f}"
        elif self.pending_order:
            order_price = float(self.pending_order.get('price', 0))
            order_side = self.pending_order.get('side', '')
            status += f" | ⏳ {order_side} @ ${order_price:.2f}"
        elif indicators:
            status += f" | BB: {indicators['bb_position']:.2f}"
            trend = "UP" if indicators['ema_fast'] > indicators['ema_slow'] else "DOWN"
            status += f" | Trend: {trend}"
        
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
        
        print(f"🚀 EMA + BB Bot for {self.symbol}")
        print(f"💰 Position Size: ${self.config['position_size']}")
        print(f"🎯 TP: {self.config['net_take_profit']}% | SL: {self.config['net_stop_loss']}%")
        print(f"⏱️ Order timeout: {self.order_timeout}s")
        print(f"✅ Connected successfully! Monitoring market...")
        print("-" * 50)
        
        cycle_count = 0
        while True:
            try:
                cycle_count += 1
                if cycle_count % 30 == 1:  # Show debug every 30 cycles
                    print(f"\n🔄 Cycle {cycle_count} - Running...")
                
                await self.run_cycle()
                await asyncio.sleep(1)
            except KeyboardInterrupt:
                print("\n🛑 Shutting down...")
                try:
                    self.exchange.cancel_all_orders(category="linear", symbol=self.symbol)
                except:
                    pass
                if self.position:
                    await self.close_position("manual_stop")
                print("✅ Bot stopped")
                break
            except Exception as e:
                print(f"\n⚠️ Error: {e}")
                await asyncio.sleep(5)

if __name__ == "__main__":
    bot = DOGEScalpingBot()
    asyncio.run(bot.run())