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
        
        prefix = 'TESTNET_' if self.demo_mode else 'LIVE_'
        self.api_key = os.getenv(f'{prefix}BYBIT_API_KEY')
        self.api_secret = os.getenv(f'{prefix}BYBIT_API_SECRET')
        self.exchange = None
        
        self.position = None
        self.price_data = pd.DataFrame()
        self.trade_id = 0
        self.daily_trades = 0
        self.last_trade_bar = 0
        
        self.config = {
            'timeframe': '3',
            'lookback': 100,
            'zigzag_pct': 0.5,
            'min_swing_bars': 3,
            'cooldown_bars': 3,
            'max_daily_trades': 30,
            'maker_offset_pct': 0.01,
            'net_take_profit': 1.08,
            'net_stop_loss': 0.42,
            'trailing_activation': 0.4,
            'trailing_distance': 0.32,
            'position_size': 100,
        }
        
        self.swing_highs = []
        self.swing_lows = []
        
        os.makedirs("logs", exist_ok=True)
        self.log_file = f"logs/4_FEES_LIQUIDITYSWEEPBOT_DOGEUSDT.log"
    
    def connect(self):
        try:
            self.exchange = HTTP(demo=self.demo_mode, api_key=self.api_key, api_secret=self.api_secret)
            return self.exchange.get_server_time().get('retCode') == 0
        except:
            return False
    
    def format_qty(self, qty):
        # XRPUSDT uses integer quantities  
        return str(int(round(qty)))
    
    def identify_swings(self):
        if len(self.price_data) < 10:
            return []
        
        df = self.price_data
        swings = []
        
        for i in range(2, len(df) - 2):
            is_high = (df['high'].iloc[i] > df['high'].iloc[i-1] and 
                      df['high'].iloc[i] > df['high'].iloc[i-2] and
                      df['high'].iloc[i] > df['high'].iloc[i+1] and 
                      df['high'].iloc[i] > df['high'].iloc[i+2])
            
            is_low = (df['low'].iloc[i] < df['low'].iloc[i-1] and 
                     df['low'].iloc[i] < df['low'].iloc[i-2] and
                     df['low'].iloc[i] < df['low'].iloc[i+1] and 
                     df['low'].iloc[i] < df['low'].iloc[i+2])
            
            if is_high:
                swings.append({'index': i, 'type': 'HIGH', 'price': df['high'].iloc[i]})
            elif is_low:
                swings.append({'index': i, 'type': 'LOW', 'price': df['low'].iloc[i]})
        
        # Filter by minimum percentage
        filtered = []
        for swing in swings:
            if not filtered:
                filtered.append(swing)
            else:
                price_change = abs(swing['price'] - filtered[-1]['price']) / filtered[-1]['price'] * 100
                if price_change >= self.config['zigzag_pct'] and swing['type'] != filtered[-1]['type']:
                    filtered.append(swing)
        
        return filtered
    
    def generate_signal(self):
        if len(self.price_data) < 20:
            return None
        
        if self.daily_trades >= self.config['max_daily_trades']:
            return None
        
        current_bar = len(self.price_data) - 1
        if current_bar - self.last_trade_bar < self.config['cooldown_bars']:
            return None
        
        swings = self.identify_swings()
        if len(swings) < 3:
            return None
        
        current_price = float(self.price_data['close'].iloc[-1])
        last_swing = swings[-1]
        bars_since_swing = current_bar - last_swing['index']
        
        recent_vol = self.price_data['volume'].iloc[-3:].mean()
        avg_vol = self.price_data['volume'].iloc[-20:].mean()
        
        if recent_vol <= avg_vol * 0.8:
            return None
        
        # BUY signal at swing low reversal
        if (last_swing['type'] == 'LOW' and 
            bars_since_swing <= self.config['min_swing_bars'] and
            current_price > last_swing['price'] * 1.001):
            return {'action': 'BUY', 'price': current_price, 'reason': 'swing_low'}
        
        # SELL signal at swing high reversal
        elif (last_swing['type'] == 'HIGH' and 
              bars_since_swing <= self.config['min_swing_bars'] and
              current_price < last_swing['price'] * 0.999):
            return {'action': 'SELL', 'price': current_price, 'reason': 'swing_high'}
        
        # Breakout signals
        if last_swing['type'] == 'HIGH' and current_price > last_swing['price'] * 1.002:
            return {'action': 'BUY', 'price': current_price, 'reason': 'breakout_high'}
        
        elif last_swing['type'] == 'LOW' and current_price < last_swing['price'] * 0.998:
            return {'action': 'SELL', 'price': current_price, 'reason': 'breakout_low'}
        
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
            
            if positions.get('retCode') != 0:
                return
            
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
        
        swings = self.identify_swings()
        if swings:
            last_swing = swings[-1]
            if ((side == "Buy" and last_swing['type'] == 'HIGH' and current_price >= entry_price * 1.002) or
                (side == "Sell" and last_swing['type'] == 'LOW' and current_price <= entry_price * 0.998)):
                return True, "swing_exit"
        
        if side == "Buy":
            net_tp = entry_price * (1 + self.config['net_take_profit'] / 100)
            net_sl = entry_price * (1 - self.config['net_stop_loss'] / 100)
            if current_price >= net_tp:
                return True, "take_profit"
            if current_price <= net_sl:
                return True, "stop_loss"
            
            # Trailing stop
            pnl_pct = ((current_price - entry_price) / entry_price) * 100
            if pnl_pct > self.config['trailing_activation']:
                trailing_stop = entry_price * (1 + (pnl_pct - self.config['trailing_distance']) / 100)
                if current_price < trailing_stop:
                    return True, "trailing_stop"
        else:
            net_tp = entry_price * (1 - self.config['net_take_profit'] / 100)
            net_sl = entry_price * (1 + self.config['net_stop_loss'] / 100)
            if current_price <= net_tp:
                return True, "take_profit"
            if current_price >= net_sl:
                return True, "stop_loss"
            
            # Trailing stop
            pnl_pct = ((entry_price - current_price) / entry_price) * 100
            if pnl_pct > self.config['trailing_activation']:
                trailing_stop = entry_price * (1 - (pnl_pct - self.config['trailing_distance']) / 100)
                if current_price > trailing_stop:
                    return True, "trailing_stop"
        
        return False, ""
    
    async def execute_trade(self, signal):
        qty = self.config['position_size'] / signal['price']
        formatted_qty = self.format_qty(qty)
        
        if float(formatted_qty) < 0.1:
            return
        
        # LIMIT order for entry
        limit_price = round(signal['price'] * (1 - self.config['maker_offset_pct']/100 if signal['action'] == 'BUY' else 1 + self.config['maker_offset_pct']/100), 2)
        
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
                self.daily_trades += 1
                self.last_trade_bar = len(self.price_data) - 1
                
                print(f"✅ {signal['action']}: {formatted_qty} @ ${limit_price:.2f} | {signal['reason']}")
                self.log_trade(signal['action'], limit_price, signal['reason'])
        except Exception as e:
            print(f"❌ Trade error: {e}")
    
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
                'price': round(price, 2),
                'info': info
            }) + "\n")
    
    async def run_cycle(self):
        if not await self.get_market_data():
            return
        
        await self.check_position()
        
        if self.position:
            should_close, reason = self.should_close()
            if should_close:
                await self.close_position(reason)
        elif signal := self.generate_signal():
            await self.execute_trade(signal)
    
    async def run(self):
        if not self.connect():
            print("❌ Failed to connect")
            return
        
        print(f"✅ ZigZag Trading Bot - {self.symbol}")
        print(f"⏰ Timeframe: {self.config['timeframe']} minutes")
        print(f"🎯 TP: {self.config['net_take_profit']}% | SL: {self.config['net_stop_loss']}%")
        
        try:
            while True:
                await self.run_cycle()
                await asyncio.sleep(10)
        except KeyboardInterrupt:
            print("\n🛑 Bot stopped")
            if self.position:
                await self.close_position("manual_stop")

if __name__ == "__main__":
    bot = ZigZagTradingBot()
    asyncio.run(bot.run())