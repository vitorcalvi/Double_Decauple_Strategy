import os
import asyncio
import pandas as pd
import json
from datetime import datetime, timezone
from pybit.unified_trading import HTTP
from dotenv import load_dotenv

load_dotenv()

class TradeLogger:
    def __init__(self, bot_name, symbol):
        self.bot_name = bot_name
        self.symbol = symbol
        self.currency = "USDT"
        self.open_trades = {}
        self.trade_id = 1000
        
        os.makedirs("logs", exist_ok=True)
        self.log_file = f"logs/{bot_name}_{symbol}.log"
        
    def generate_trade_id(self):
        self.trade_id += 1
        return self.trade_id
    
    def log_trade_open(self, side, expected_price, actual_price, qty, stop_loss, take_profit, info=""):
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
    
    def log_trade_close(self, trade_id, expected_exit, actual_exit, reason, fees_entry=-0.04, fees_exit=-0.04):
        if trade_id not in self.open_trades:
            return None
            
        trade = self.open_trades[trade_id]
        duration = (datetime.now() - trade["entry_time"]).total_seconds()
        
        slippage = actual_exit - expected_exit if trade["side"] == "SELL" else expected_exit - actual_exit
        
        if trade["side"] == "BUY":
            gross_pnl = (actual_exit - trade["entry_price"]) * trade["qty"]
        else:
            gross_pnl = (trade["entry_price"] - actual_exit) * trade["qty"]
        
        entry_fee_pct = abs(fees_entry) if fees_entry < 0 else fees_entry
        exit_fee_pct = abs(fees_exit) if fees_exit < 0 else fees_exit
        
        if fees_entry < 0:
            entry_rebate = trade["entry_price"] * trade["qty"] * entry_fee_pct / 100
        else:
            entry_rebate = -(trade["entry_price"] * trade["qty"] * entry_fee_pct / 100)
            
        if fees_exit < 0:
            exit_rebate = actual_exit * trade["qty"] * exit_fee_pct / 100
        else:
            exit_rebate = -(actual_exit * trade["qty"] * exit_fee_pct / 100)
        
        total_fee_impact = entry_rebate + exit_rebate
        net_pnl = gross_pnl + total_fee_impact
        
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
            "fee_impact": {
                "entry": round(entry_rebate, 2), 
                "exit": round(exit_rebate, 2), 
                "total": round(total_fee_impact, 2)
            },
            "net_pnl": round(net_pnl, 2),
            "reason": reason,
            "currency": self.currency
        }
        
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        
        del self.open_trades[trade_id]
        return log_entry

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
        self.pending_order = None
        self.last_signal = None
        
        self.order_timeout = 180
        
        self.config = {
            'timeframe': '5',
            'ema_short': 12,
            'ema_long': 26,
            'macd_signal': 9,
            'rsi_period': 14,
            'position_size': 100,
            'lookback': 100,
            'maker_offset_pct': 0.01,
            'net_take_profit': 1.08,
            'net_stop_loss': 0.42,
        }
        
        self.logger = TradeLogger("EMA_MACD_RSI", self.symbol)
        self.current_trade_id = None
    
    def connect(self):
        try:
            self.exchange = HTTP(demo=self.demo_mode, api_key=self.api_key, api_secret=self.api_secret)
            return self.exchange.get_server_time().get('retCode') == 0
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False
    
    def format_qty(self, qty):
        return str(int(round(qty)))
    
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
                return False
            
            self.pending_order = order
            return True
        except Exception as e:
            print(f"❌ Order check error: {e}")
            return False
    
    def calculate_indicators(self, df):
        if len(df) < self.config['lookback']:
            return None
        
        try:
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
                'rsi': rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
            }
        except Exception as e:
            print(f"❌ Indicator calculation error: {e}")
            return None
    
    def generate_signal(self, df):
        analysis = self.calculate_indicators(df)
        if not analysis:
            return None
        
        if self.last_signal:
            price_change = abs(analysis['price'] - self.last_signal['price']) / self.last_signal['price']
            if price_change < 0.002:
                return None
        
        histogram_positive = analysis['histogram'] > 0
        histogram_turning_positive = analysis['histogram_prev'] <= 0 and analysis['histogram'] > 0
        histogram_turning_negative = analysis['histogram_prev'] >= 0 and analysis['histogram'] < 0
        
        if (histogram_positive and analysis['rsi'] < 60) or (histogram_turning_positive and analysis['rsi'] < 70):
            return {
                'action': 'BUY', 
                'price': analysis['price'], 
                'rsi': analysis['rsi'],
                'reason': 'macd_bullish'
            }
        
        if (not histogram_positive and analysis['rsi'] > 40) or (histogram_turning_negative and analysis['rsi'] > 30):
            return {
                'action': 'SELL', 
                'price': analysis['price'], 
                'rsi': analysis['rsi'],
                'reason': 'macd_bearish'
            }
        
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
            
            df = pd.DataFrame(klines['result']['list'], 
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df.dropna(subset=['open', 'high', 'low', 'close'])
            self.price_data = df.sort_values('timestamp').reset_index(drop=True)
            
            return len(self.price_data) >= 20
        except Exception as e:
            print(f"❌ Market data error: {e}")
            return False
    
    async def check_position(self):
        try:
            positions = self.exchange.get_positions(category="linear", symbol=self.symbol)
            if positions.get('retCode') == 0:
                pos_list = positions['result']['list']
                self.position = pos_list[0] if pos_list and float(pos_list[0]['size']) > 0 else None
        except Exception as e:
            print(f"❌ Position check error: {e}")
            self.position = None
    
    def should_close(self):
        if not self.position:
            return False, ""
        
        try:
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
            
            analysis = self.calculate_indicators(self.price_data)
            if analysis:
                histogram_turning_negative = analysis['histogram_prev'] >= 0 and analysis['histogram'] < 0
                if side == "Buy" and histogram_turning_negative and analysis['rsi'] > 60:
                    return True, "macd_reversal"
                
                histogram_turning_positive = analysis['histogram_prev'] <= 0 and analysis['histogram'] > 0
                if side == "Sell" and histogram_turning_positive and analysis['rsi'] < 40:
                    return True, "macd_reversal"
            
            return False, ""
        except Exception as e:
            print(f"❌ Position evaluation error: {e}")
            return False, ""
    
    async def execute_trade(self, signal):
        if await self.check_pending_orders() or self.position:
            return
        
        qty = self.config['position_size'] / signal['price']
        formatted_qty = self.format_qty(qty)
        
        if float(formatted_qty) < 1:
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
                self.last_signal = signal
                self.pending_order = order['result']
                
                stop_loss = limit_price * (1 - self.config['net_stop_loss']/100) if signal['action'] == 'BUY' else limit_price * (1 + self.config['net_stop_loss']/100)
                take_profit = limit_price * (1 + self.config['net_take_profit']/100) if signal['action'] == 'BUY' else limit_price * (1 - self.config['net_take_profit']/100)
                
                self.current_trade_id, _ = self.logger.log_trade_open(
                    side=signal['action'],
                    expected_price=signal['price'],
                    actual_price=limit_price,
                    qty=float(formatted_qty),
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    info=f"RSI:{signal['rsi']:.1f}_{signal['reason']}"
                )
                
                print(f"✅ {signal['action']}: {formatted_qty} SOL @ ${limit_price:.4f} | RSI: {signal['rsi']:.1f}")
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
                current_price = float(self.price_data['close'].iloc[-1])
                
                if self.current_trade_id:
                    self.logger.log_trade_close(
                        trade_id=self.current_trade_id,
                        expected_exit=current_price,
                        actual_exit=current_price,
                        reason=reason,
                        fees_entry=-0.04,
                        fees_exit=0.1
                    )
                    self.current_trade_id = None
                
                print(f"💰 Position closed: {reason}")
                self.position = None
                self.last_signal = None
        except Exception as e:
            print(f"❌ Close failed: {e}")
    
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
    
    async def run(self):
        if not self.connect():
            print("❌ Failed to connect to exchange")
            return
        
        print(f"🚀 EMA+MACD+RSI Bot for {self.symbol}")
        print(f"⏰ Timeframe: {self.config['timeframe']} min")
        print(f"🎯 TP: {self.config['net_take_profit']}% | SL: {self.config['net_stop_loss']}%")
        
        try:
            while True:
                await self.run_cycle()
                await asyncio.sleep(5)
        except KeyboardInterrupt:
            print("\n🛑 Bot stopped by user")
            try:
                self.exchange.cancel_all_orders(category="linear", symbol=self.symbol)
            except:
                pass
            if self.position:
                await self.close_position("manual_stop")
        except Exception as e:
            print(f"⚠️ Runtime error: {e}")

if __name__ == "__main__":
    bot = EMAMACDRSIBot()
    asyncio.run(bot.run())