import os
import asyncio
import pandas as pd
import json
from datetime import datetime
from pybit.unified_trading import HTTP
from dotenv import load_dotenv

load_dotenv()

class ZigZagTradingBot:
    def __init__(self):
        self.symbol = 'DOGEUSDT'
        self.demo_mode = os.getenv('DEMO_MODE', 'true').lower() == 'true'
        
        prefix = 'TESTNET_' if self.demo_mode else 'LIVE_'
        self.api_key = os.getenv(f'{prefix}BYBIT_API_KEY')
        self.api_secret = os.getenv(f'{prefix}BYBIT_API_SECRET')
        self.exchange = None
        
        self.position = None
        self.price_data = pd.DataFrame()
        self.trade_id = 0
        
        # Order management - NO COOLDOWN
        self.pending_order = None
        self.last_order_time = None
        self.order_timeout = 180  # Cancel orders older than 180 seconds
        
        self.config = {
            'timeframe': '1',
            'lookback': 50,
            'zigzag_pct': 0.2,
            'maker_offset_pct': 0.01,
            'net_take_profit': 1.08,
            'net_stop_loss': 0.42,
            'position_size': 100,
        }
        
        os.makedirs("logs", exist_ok=True)
        self.log_file = f"logs/4_FEES_LIQUIDITYSWEEPBOT_DOGEUSDT.log"
    
    def connect(self):
        try:
            self.exchange = HTTP(demo=self.demo_mode, api_key=self.api_key, api_secret=self.api_secret)
            return self.exchange.get_server_time().get('retCode') == 0
        except:
            return False
    
    def format_qty(self, qty):
        return str(int(round(qty)))
    
    async def check_pending_orders(self):
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
                            print(f"❌ Cancelled old order: {order['orderId']}")
                            self.pending_order = None
                        else:
                            self.pending_order = order
                            return True
                else:
                    self.pending_order = None
                    return False
            return False
        except:
            return False
    
    def identify_swings(self):
        if len(self.price_data) < 5:
            return []
        
        df = self.price_data
        swings = []
        
        for i in range(1, len(df) - 1):
            if df['high'].iloc[i] > df['high'].iloc[i-1] and df['high'].iloc[i] > df['high'].iloc[i+1]:
                swings.append({'index': i, 'type': 'HIGH', 'price': df['high'].iloc[i]})
            elif df['low'].iloc[i] < df['low'].iloc[i-1] and df['low'].iloc[i] < df['low'].iloc[i+1]:
                swings.append({'index': i, 'type': 'LOW', 'price': df['low'].iloc[i]})
        
        return swings
    
    def generate_signal(self):
        if len(self.price_data) < 10:
            return None
        
        swings = self.identify_swings()
        if len(swings) < 2:
            return None
        
        current_price = float(self.price_data['close'].iloc[-1])
        last_swing = swings[-1]
        
        if last_swing['type'] == 'LOW' and current_price > last_swing['price']:
            return {'action': 'BUY', 'price': current_price, 'reason': 'swing_low'}
        elif last_swing['type'] == 'HIGH' and current_price < last_swing['price']:
            return {'action': 'SELL', 'price': current_price, 'reason': 'swing_high'}
        
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
            if current_price >= entry_price * (1 + self.config['net_take_profit'] / 100):
                return True, "take_profit"
            if current_price <= entry_price * (1 - self.config['net_stop_loss'] / 100):
                return True, "stop_loss"
        else:
            if current_price <= entry_price * (1 - self.config['net_take_profit'] / 100):
                return True, "take_profit"
            if current_price >= entry_price * (1 + self.config['net_stop_loss'] / 100):
                return True, "stop_loss"
        
        return False, ""
    
    async def execute_trade(self, signal):
        # NO COOLDOWN - Check for pending orders only
        if await self.check_pending_orders():
            return
        
        if self.position:
            return
        
        qty = self.config['position_size'] / signal['price']
        formatted_qty = self.format_qty(qty)
        
        if float(formatted_qty) < 1:
            return
        
        limit_price = round(signal['price'] * (1 - self.config['maker_offset_pct']/100 if signal['action'] == 'BUY' else 1 + self.config['maker_offset_pct']/100), 4)
        
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
                self.pending_order = order['result']
                print(f"✅ {signal['action']}: {formatted_qty} @ ${limit_price:.4f} | {signal['reason']}")
                self.log_trade(signal['action'], limit_price, signal['reason'])
        except Exception as e:
            print(f"❌ Trade error: {e}")
    
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
                print(f"✅ Closed: {reason} | PnL: ${pnl:.2f}")
                self.log_trade("CLOSE", 0, f"{reason}_PnL:${pnl:.2f}")
        except Exception as e:
            print(f"❌ Close failed: {e}")
    
    def log_trade(self, action, price, info):
        with open(self.log_file, "a") as f:
            f.write(json.dumps({
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'action': action,
                'price': round(price, 4),
                'info': info
            }) + "\n")
    
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
            signal = self.generate_signal()
            if signal:
                await self.execute_trade(signal)
    
    async def run(self):
        if not self.connect():
            print("❌ Failed to connect")
            return
        
        print(f"✅ ZigZag Trading Bot - {self.symbol}")
        print(f"⏰ Timeframe: {self.config['timeframe']} minute")
        print(f"🎯 TP: {self.config['net_take_profit']}% | SL: {self.config['net_stop_loss']}%")
        print(f"⏱️ No cooldown | Order timeout: {self.order_timeout}s")
        
        try:
            while True:
                await self.run_cycle()
                await asyncio.sleep(2)
        except KeyboardInterrupt:
            print("\n🛑 Bot stopped")
            try:
                self.exchange.cancel_all_orders(category="linear", symbol=self.symbol)
            except:
                pass
            if self.position:
                await self.close_position("manual_stop")

if __name__ == "__main__":
    bot = ZigZagTradingBot()
    asyncio.run(bot.run())