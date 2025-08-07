import os
import asyncio
import pandas as pd
import json
from datetime import datetime, timezone
from pybit.unified_trading import HTTP
from dotenv import load_dotenv

load_dotenv()

# Integrated Trade Logger
class TradeLogger:
    def __init__(self, bot_name, symbol):
        self.bot_name = bot_name
        self.symbol = symbol
        self.currency = "USDT"
        self.open_trades = {}
        self.trade_id = 1000
        
        # Ensure logs directory exists
        os.makedirs("logs", exist_ok=True)
        self.log_file = f"logs/{bot_name}_{symbol}.log"
        
    def generate_trade_id(self):
        self.trade_id += 1
        return self.trade_id
    
    def log_trade_open(self, side, expected_price, actual_price, qty, stop_loss, take_profit, info=""):
        """Log position opening with slippage"""
        trade_id = self.generate_trade_id()
        slippage = actual_price - expected_price if side == "BUY" else expected_price - actual_price
        
        log_entry = {
            "id": trade_id,
            "bot": self.bot_name,
            "symbol": self.symbol,
            "side": "LONG" if side == "BUY" else "SHORT",
            "action": "OPEN",
            "ts": datetime.now(timezone.utc).isoformat(),
            "expected_price": round(expected_price, 4),
            "actual_price": round(actual_price, 4),
            "slippage": round(slippage, 4),
            "qty": round(qty, 6),
            "stop_loss": round(stop_loss, 4),
            "take_profit": round(take_profit, 4),
            "currency": self.currency,
            "info": info
        }
        
        self.open_trades[trade_id] = {
            "entry_time": datetime.now(),
            "entry_price": actual_price,
            "side": side,
            "qty": qty,
            "stop_loss": stop_loss,
            "take_profit": take_profit
        }
        
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        
        return trade_id, log_entry
    
    def log_trade_close(self, trade_id, expected_exit, actual_exit, reason, fees_entry=0.1, fees_exit=0.25):
        """Log position closing with slippage and PnL calculation"""
        if trade_id not in self.open_trades:
            return None
            
        trade = self.open_trades[trade_id]
        duration = (datetime.now() - trade["entry_time"]).total_seconds()
        
        slippage = actual_exit - expected_exit if trade["side"] == "SELL" else expected_exit - actual_exit
        
        if trade["side"] == "BUY":
            gross_pnl = (actual_exit - trade["entry_price"]) * trade["qty"]
        else:
            gross_pnl = (trade["entry_price"] - actual_exit) * trade["qty"]
        
        total_fees = fees_entry + fees_exit
        net_pnl = gross_pnl - total_fees
        
        log_entry = {
            "id": trade_id,
            "bot": self.bot_name,
            "symbol": self.symbol,
            "side": "LONG" if trade["side"] == "BUY" else "SHORT",
            "action": "CLOSE",
            "ts": datetime.now(timezone.utc).isoformat(),
            "duration_sec": int(duration),
            "entry_price": round(trade["entry_price"], 4),
            "expected_exit": round(expected_exit, 4),
            "actual_exit": round(actual_exit, 4),
            "slippage": round(slippage, 4),
            "qty": round(trade["qty"], 6),
            "gross_pnl": round(gross_pnl, 2),
            "fees": {"entry": fees_entry, "exit": fees_exit, "total": total_fees},
            "net_pnl": round(net_pnl, 2),
            "reason": reason,
            "currency": self.currency
        }
        
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        
        del self.open_trades[trade_id]
        return log_entry

class ZigZagTradingBot:
    def __init__(self):
        self.symbol = 'XRPUSDT'
        self.demo_mode = os.getenv('DEMO_MODE', 'true').lower() == 'true'
        
        # API setup
        prefix = 'TESTNET_' if self.demo_mode else 'LIVE_'
        self.api_key = os.getenv(f'{prefix}BYBIT_API_KEY')
        self.api_secret = os.getenv(f'{prefix}BYBIT_API_SECRET')
        self.exchange = None
        
        # Trading state
        self.position = None
        self.price_data = pd.DataFrame()
        self.daily_trades = 0
        self.last_trade_bar = 0
        
        # ZigZag configuration
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
        
        # Logging setup
        self.logger = TradeLogger("ZIGZAG", self.symbol)
        self.current_trade_id = None
    
    def connect(self):
        """Connect to exchange API"""
        try:
            self.exchange = HTTP(demo=self.demo_mode, api_key=self.api_key, api_secret=self.api_secret)
            return self.exchange.get_server_time().get('retCode') == 0
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False
    
    def format_qty(self, qty):
        """Format quantity according to exchange requirements"""
        return f"{round(qty / 0.1) * 0.1:.1f}" if qty >= 0.1 else str(0.1)
    
    def identify_swings(self):
        """Identify swing highs and lows in price data"""
        if len(self.price_data) < 10:
            return []
        
        df = self.price_data
        swings = []
        
        # Identify pivot points using 2-bar look ahead/behind
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
        
        # Filter swings by minimum percentage change
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
        """Generate trading signal based on zigzag pattern"""
        if len(self.price_data) < 20:
            return None
        
        # Check daily trade limit
        if self.daily_trades >= self.config['max_daily_trades']:
            return None
        
        # Check cooldown period
        current_bar = len(self.price_data) - 1
        if current_bar - self.last_trade_bar < self.config['cooldown_bars']:
            return None
        
        # Get swing points
        swings = self.identify_swings()
        if len(swings) < 3:
            return None
        
        current_price = float(self.price_data['close'].iloc[-1])
        last_swing = swings[-1]
        bars_since_swing = current_bar - last_swing['index']
        
        # Volume filter
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
        """Retrieve market data from exchange"""
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
            
            # Convert data types
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
            
            # Sort data by time
            self.price_data = df.sort_values('timestamp').reset_index(drop=True)
            return True
        except Exception as e:
            print(f"❌ Market data error: {e}")
            return False
    
    async def check_position(self):
        """Check current position status"""
        try:
            positions = self.exchange.get_positions(category="linear", symbol=self.symbol)
            
            if positions.get('retCode') != 0:
                return
            
            pos_list = positions['result']['list']
            self.position = pos_list[0] if pos_list and float(pos_list[0]['size']) > 0 else None
        except Exception as e:
            print(f"❌ Position check error: {e}")
            pass
    
    def should_close(self):
        """Determine if position should be closed"""
        if not self.position:
            return False, ""
        
        current_price = float(self.price_data['close'].iloc[-1])
        entry_price = float(self.position.get('avgPrice', 0))
        side = self.position.get('side', '')
        
        if entry_price == 0:
            return False, ""
        
        # Check for swing-based exit
        swings = self.identify_swings()
        if swings:
            last_swing = swings[-1]
            if ((side == "Buy" and last_swing['type'] == 'HIGH' and current_price >= entry_price * 1.002) or
                (side == "Sell" and last_swing['type'] == 'LOW' and current_price <= entry_price * 0.998)):
                return True, "swing_exit"
        
        # Take profit and stop loss
        if side == "Buy":
            net_tp = entry_price * (1 + self.config['net_take_profit'] / 100)
            net_sl = entry_price * (1 - self.config['net_stop_loss'] / 100)
            
            if current_price >= net_tp:
                return True, "take_profit"
            if current_price <= net_sl:
                return True, "stop_loss"
            
            # Trailing stop for long positions
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
            
            # Trailing stop for short positions
            pnl_pct = ((entry_price - current_price) / entry_price) * 100
            if pnl_pct > self.config['trailing_activation']:
                trailing_stop = entry_price * (1 - (pnl_pct - self.config['trailing_distance']) / 100)
                if current_price > trailing_stop:
                    return True, "trailing_stop"
        
        return False, ""
    
    async def execute_trade(self, signal):
        """Execute a trade based on the signal"""
        qty = self.config['position_size'] / signal['price']
        formatted_qty = self.format_qty(qty)
        
        if float(formatted_qty) < 0.1:
            return
        
        # Calculate limit price with maker offset
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
                self.daily_trades += 1
                self.last_trade_bar = len(self.price_data) - 1
                
                # Calculate stop loss and take profit levels
                stop_loss = limit_price * (1 - self.config['net_stop_loss']/100) if signal['action'] == 'BUY' else limit_price * (1 + self.config['net_stop_loss']/100)
                take_profit = limit_price * (1 + self.config['net_take_profit']/100) if signal['action'] == 'BUY' else limit_price * (1 - self.config['net_take_profit']/100)
                
                # Log the trade
                self.current_trade_id, _ = self.logger.log_trade_open(
                    side=signal['action'],
                    expected_price=signal['price'],
                    actual_price=limit_price,
                    qty=float(formatted_qty),
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    info=signal['reason']
                )
                
                print(f"✅ {signal['action']}: {formatted_qty} XRP @ ${limit_price:.4f} | {signal['reason']}")
        except Exception as e:
            print(f"❌ Trade error: {e}")
    
    async def close_position(self, reason):
        """Close the current position"""
        if not self.position:
            return
        
        side = "Sell" if self.position.get('side') == "Buy" else "Buy"
        qty = float(self.position['size'])
        current_price = float(self.price_data['close'].iloc[-1])
        
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
                # Log the trade close
                if self.current_trade_id:
                    self.logger.log_trade_close(
                        trade_id=self.current_trade_id,
                        expected_exit=current_price,
                        actual_exit=current_price,
                        reason=reason
                    )
                    self.current_trade_id = None
                
                pnl = float(self.position.get('unrealisedPnl', 0))
                print(f"✅ Closed: {reason} | PnL: ${pnl:.2f}")
        except Exception as e:
            print(f"❌ Close failed: {e}")
    
    async def run_cycle(self):
        """Run one trading cycle"""
        if not await self.get_market_data():
            return
        
        await self.check_position()
        
        if self.position:
            should_close, reason = self.should_close()
            if should_close:
                await self.close_position(reason)
        elif signal := self.generate_signal():  # Python 3.8+ assignment expression
            await self.execute_trade(signal)
    
    async def run(self):
        """Main bot loop"""
        if not self.connect():
            print("❌ Failed to connect to exchange")
            return
        
        print(f"🚀 ZigZag Trading Bot - {self.symbol}")
        print(f"⏰ Timeframe: {self.config['timeframe']} minutes")
        print(f"🎯 TP: {self.config['net_take_profit']}% | SL: {self.config['net_stop_loss']}%")
        print(f"📈 Swing %: {self.config['zigzag_pct']}% | Max daily trades: {self.config['max_daily_trades']}")
        
        try:
            while True:
                await self.run_cycle()
                await asyncio.sleep(10)
        except KeyboardInterrupt:
            print("\n🛑 Bot stopped by user")
            if self.position:
                await self.close_position("manual_stop")
        except Exception as e:
            print(f"⚠️ Runtime error: {e}")
            if self.position:
                await self.close_position("error_stop")

if __name__ == "__main__":
    bot = ZigZagTradingBot()
    asyncio.run(bot.run())