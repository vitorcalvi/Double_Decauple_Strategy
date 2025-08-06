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
        self.pending_order = None
        self.last_signal = None
        self.order_timeout = 180
        
        self.config = {
            'timeframe': '5',
            'ema_short': 12,
            'ema_long': 26,
            'macd_signal': 9,
            'rsi_period': 14,
            'rsi_neutral_low': 50,
            'position_size': 100,
            'lookback': 100,
            'maker_offset_pct': 0.01,
            'net_take_profit': 1.08,
            'net_stop_loss': 0.42,
        }
        
        os.makedirs("logs", exist_ok=True)
        self.log_file = "logs/3_FEES_EMAMACDRSI_SOLUSDT.log"
    
    def connect(self):
        try:
            self.exchange = HTTP(demo=self.demo_mode, api_key=self.api_key, api_secret=self.api_secret)
            return self.exchange.get_server_time().get('retCode') == 0
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
                print(f"❌ Cancelled stale order (aged {age:.0f}s)")
                self.pending_order = None
                self.last_signal = None
                return False
            
            self.pending_order = order
            return True
        except:
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
        epsilon = 1e-10
        rsi = 100 - (100 / (1 + gain / (loss + epsilon)))
        
        return {
            'price': close.iloc[-1],
            'ema_aligned': ema_short.iloc[-1] > ema_long.iloc[-1],
            'histogram': histogram.iloc[-1],
            'histogram_prev': histogram.iloc[-2] if len(histogram) > 1 else 0,
            'rsi': rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50,
            'rsi_above_50': rsi.iloc[-1] > self.config['rsi_neutral_low']
        }
    
    def generate_signal(self, df):
        analysis = self.calculate_indicators(df)
        if not analysis:
            return None
        
        # Skip duplicate signals (less strict check)
        if self.last_signal:
            price_change = abs(analysis['price'] - self.last_signal['price']) / self.last_signal['price']
            if price_change < 0.002:  # 0.2% price change threshold
                return None
        
        # Simplified conditions for more frequent signals
        histogram_positive = analysis['histogram'] > 0
        histogram_turning_positive = analysis['histogram_prev'] <= 0 and analysis['histogram'] > 0
        histogram_turning_negative = analysis['histogram_prev'] >= 0 and analysis['histogram'] < 0
        
        # BUY signal - more relaxed
        if (histogram_positive and analysis['rsi'] < 60) or (histogram_turning_positive and analysis['rsi'] < 70):
            return {'action': 'BUY', 'price': analysis['price'], 'rsi': analysis['rsi']}
        
        # SELL signal - more relaxed
        if (not histogram_positive and analysis['rsi'] > 40) or (histogram_turning_negative and analysis['rsi'] > 30):
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
            
            df = pd.DataFrame(klines['result']['list'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
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
        
        profit_pct = ((current_price - entry_price) / entry_price * 100) if side == "Buy" else ((entry_price - current_price) / entry_price * 100)
        
        if profit_pct >= self.config['net_take_profit']:
            return True, "take_profit"
        if profit_pct <= -self.config['net_stop_loss']:
            return True, "stop_loss"
        
        # Simplified MACD reversal check
        analysis = self.calculate_indicators(self.price_data)
        if analysis:
            histogram_turning_negative = analysis['histogram_prev'] >= 0 and analysis['histogram'] < 0
            if side == "Buy" and histogram_turning_negative and analysis['rsi'] > 60:
                return True, "macd_reversal"
            
            histogram_turning_positive = analysis['histogram_prev'] <= 0 and analysis['histogram'] > 0
            if side == "Sell" and histogram_turning_positive and analysis['rsi'] < 40:
                return True, "macd_reversal"
        
        return False, ""
    
    async def execute_trade(self, signal):
        if await self.check_pending_orders() or self.position:
            return
        
        qty = self.config['position_size'] / signal['price']
        formatted_qty = str(int(round(qty)))
        
        if int(formatted_qty) == 0:
            return
        
        offset = 1 - self.config['maker_offset_pct']/100 if signal['action'] == 'BUY' else 1 + self.config['maker_offset_pct']/100
        limit_price = round(signal['price'] * offset, 4)
        
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
                self.last_signal = signal
                self.pending_order = order['result']
                print(f"\n✅ {signal['action']}: {formatted_qty} SOL @ ${limit_price:.4f} | RSI: {signal['rsi']:.1f}")
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
                qty=str(int(round(qty))),
                reduceOnly=True
            )
            
            if order.get('retCode') == 0:
                pnl = float(self.position.get('unrealisedPnl', 0))
                print(f"\n💰 Closed: {reason} | PnL: ${pnl:.2f}")
                self.log_trade("CLOSE", float(self.price_data['close'].iloc[-1]), f"{reason}_PnL:${pnl:.2f}")
                self.position = None
                self.last_signal = None
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
        if len(self.price_data) == 0:
            return
        
        current_price = float(self.price_data['close'].iloc[-1])
        indicators = self.calculate_indicators(self.price_data)
        
        status_parts = []
        status_parts.append(f"📊 SOL: ${current_price:.2f}")
        
        if self.position:
            entry = float(self.position.get('avgPrice', 0))
            pnl = float(self.position.get('unrealisedPnl', 0))
            side = self.position.get('side', '')
            status_parts.append(f"📍 {side} @ ${entry:.4f}")
            status_parts.append(f"PnL: ${pnl:+.2f}")
        elif self.pending_order:
            order_price = float(self.pending_order.get('price', 0))
            order_side = self.pending_order.get('side', '')
            status_parts.append(f"⏳ {order_side} @ ${order_price:.4f}")
        elif indicators:
            status_parts.append(f"RSI: {indicators['rsi']:.1f}")
            status_parts.append(f"MACD: {'↑' if indicators['histogram'] > 0 else '↓'}")
            status_parts.append(f"Trend: {'UP' if indicators['ema_aligned'] else 'DOWN'}")
        
        print(" | ".join(status_parts) + "    ", end='\r')
    
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
        
        print(f"🚀 EMA+MACD+RSI Bot for {self.symbol}")
        print(f"⏰ Timeframe: {self.config['timeframe']} minutes")
        print(f"🎯 TP: {self.config['net_take_profit']}% | SL: {self.config['net_stop_loss']}%")
        print(f"⏱️ Order timeout: {self.order_timeout}s")
        print(f"✅ Connected! Starting bot...")
        print("=" * 50)
        
        cycle_count = 0
        while True:
            try:
                cycle_count += 1
                if cycle_count % 60 == 1:  # Debug every 60 cycles (5 minutes)
                    print(f"\n🔄 Heartbeat - Cycle {cycle_count}")
                
                await self.run_cycle()
                await asyncio.sleep(5)
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
    bot = EMAMACDRSIBot()
    asyncio.run(bot.run())