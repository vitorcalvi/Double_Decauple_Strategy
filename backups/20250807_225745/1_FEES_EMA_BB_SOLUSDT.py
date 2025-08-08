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
        
        # Trade cooldown mechanism
        self.last_trade_time = 0
        self.trade_cooldown = 30  # 30 seconds between trades
        
        
        # Emergency stop tracking
        self.daily_pnl = 0
        self.consecutive_losses = 0
        self.max_daily_loss = 50  # $50 max daily loss
        
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

    def log_trade_close(self, trade_id, expected_exit, actual_exit, reason):
        if trade_id not in self.open_trades:
            return None
            
        trade = self.open_trades[trade_id]
        duration = (datetime.now() - trade["entry_time"]).total_seconds()
        
        # FIXED: Proper slippage calculation
        slippage = actual_exit - expected_exit if trade["side"] == "SELL" else expected_exit - actual_exit
        
        # FIXED: Proper PnL calculation
        if trade["side"] == "BUY":
            gross_pnl = (actual_exit - trade["entry_price"]) * trade["qty"]
        else:
            gross_pnl = (trade["entry_price"] - actual_exit) * trade["qty"]
        
        # FIXED: Proper maker rebate calculation
        entry_rebate = trade["entry_price"] * trade["qty"] * 0.0004  # 0.04% rebate
        exit_rebate = actual_exit * trade["qty"] * 0.0004           # 0.04% rebate
        total_rebates = entry_rebate + exit_rebate
        
        # FIXED: Net PnL includes rebates earned
        net_pnl = gross_pnl + total_rebates
        
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
            "rebates_earned": round(total_rebates, 2),
            "net_pnl": round(net_pnl, 2),
            "reason": reason,
            "currency": self.currency
        }
        
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        
        del self.open_trades[trade_id]
        return log_entry

class EMABBFixedBot:
    def __init__(self):
        
        # Trade cooldown mechanism
        self.last_trade_time = 0
        self.trade_cooldown = 30  # 30 seconds between trades
        
        
        # Emergency stop tracking
        self.daily_pnl = 0
        self.consecutive_losses = 0
        self.max_daily_loss = 50  # $50 max daily loss
        
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
        
        # FIXED CONFIG
        self.config = {
            'timeframe': '5',
            'ema_fast': 9,
            'ema_slow': 21,
            'ema_trend': 50,
            'macd_fast': 5,
            'macd_slow': 13,
            'macd_signal': 9,  # FIXED: proper signal line
            'rsi_period': 9,
            'bb_period': 20,
            'bb_std': 2,
            'risk_per_trade': 0.01,  # FIXED: 1% risk per trade
            'max_position_pct': 0.05,  # FIXED: max 5% of account
            'lookback': 100,
            'maker_offset_pct': 0.01,
            'base_slippage': 0.02,  # FIXED: 0.02% base slippage
            'stop_loss_pct': 0.50,
            'take_profit_pct': 1.00,
        }
        
        self.logger = TradeLogger("EMA_BB_FIXED", self.symbol)
        self.current_trade_id = None
    
    def connect(self):
        try:
            self.exchange = HTTP(demo=self.demo_mode, api_key=self.api_key, api_secret=self.api_secret)
            return self.exchange.get_server_time().get('retCode') == 0
        except:
            return False
    
    # FIXED: Dynamic position sizing
    def calculate_position_size(self, price, stop_loss_price):
        try:
            account = self.exchange.get_wallet_balance(category="linear", coin="USDT")
            balance = float(account['result']['list'][0]['totalEquity'])
            
            risk_amount = balance * self.config['risk_per_trade']
            price_diff = abs(price - stop_loss_price)
            risk_per_unit = price_diff
            
            if risk_per_unit > 0:
                position_size = risk_amount / risk_per_unit
            else:
                position_size = balance * 0.02  # Fallback 2%
            
            max_size = balance * self.config['max_position_pct']
            return min(position_size, max_size)
        except:
            return 50  # Fallback for demo/errors
    
    # FIXED: Proper slippage modeling
    def apply_slippage(self, price, side, volatility=None):
        if volatility is None:
            if len(self.price_data) >= 20:
                returns = self.price_data['close'].pct_change().dropna()
                volatility = returns.rolling(20).std().iloc[-1]
            else:
                volatility = 0.01
        
        base_slippage = self.config['base_slippage'] / 100
        vol_multiplier = min(volatility * 100, 3)  # Cap at 3x
        total_slippage = base_slippage * (1 + vol_multiplier)
        
        if side in ["Buy", "BUY"]:
            return price * (1 + total_slippage)
        else:
            return price * (1 - total_slippage)
    
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
                self.pending_order = None
                return False
            
            self.pending_order = order
            return True
        except:
            return False
    
    # FIXED: Enhanced filters with trend, volatility, volume
    def enhanced_filters(self, df):
        if len(df) < 50:
            return False, False, False
        
        # Trend filter
        ema_20 = df['close'].ewm(span=20).mean()
        trend_up = df['close'].iloc[-1] > ema_20.iloc[-1]
        
        # Volatility filter
        returns = df['close'].pct_change().dropna()
        if len(returns) >= 20:
            volatility = returns.rolling(20).std().iloc[-1]
            vol_normal = 0.005 < volatility < 0.03
        else:
            vol_normal = True
        
        # Volume filter
        if len(df) >= 20:
            volume_avg = df['volume'].rolling(20).mean()
            volume_ok = df['volume'].iloc[-1] > volume_avg.iloc[-1] * 0.8
        else:
            volume_ok = True
        
        return trend_up, vol_normal, volume_ok
    
    def calculate_indicators(self, df):
        if len(df) < self.config['lookback']:
            return None
        
        try:
            close = df['close']
            
            # EMAs with trend filter
            ema_fast = close.ewm(span=self.config['ema_fast']).mean()
            ema_slow = close.ewm(span=self.config['ema_slow']).mean()
            ema_trend = close.ewm(span=self.config['ema_trend']).mean()
            
            # FIXED MACD with proper signal line
            exp1 = close.ewm(span=self.config['macd_fast']).mean()
            exp2 = close.ewm(span=self.config['macd_slow']).mean()
            macd_line = exp1 - exp2
            signal_line = macd_line.ewm(span=self.config['macd_signal']).mean()
            histogram = macd_line - signal_line
            
            # RSI
            delta = close.diff()
            gain = delta.clip(lower=0).rolling(window=self.config['rsi_period']).mean()
            loss = -delta.clip(upper=0).rolling(window=self.config['rsi_period']).mean()
            rsi = 100 - (100 / (1 + gain / (loss + 1e-10)))
            
            # Bollinger Bands
            bb_middle = close.rolling(window=self.config['bb_period']).mean()
            bb_std = close.rolling(window=self.config['bb_period']).std()
            bb_upper = bb_middle + (bb_std * self.config['bb_std'])
            bb_lower = bb_middle - (bb_std * self.config['bb_std'])
            
            return {
                'price': close.iloc[-1],
                'ema_fast': ema_fast.iloc[-1],
                'ema_slow': ema_slow.iloc[-1],
                'ema_trend': ema_trend.iloc[-1],
                'trend_bullish': ema_fast.iloc[-1] > ema_slow.iloc[-1] and close.iloc[-1] > ema_trend.iloc[-1],
                'trend_bearish': ema_fast.iloc[-1] < ema_slow.iloc[-1] and close.iloc[-1] < ema_trend.iloc[-1],
                'histogram': histogram.iloc[-1],
                'histogram_prev': histogram.iloc[-2] if len(histogram) > 1 else 0,
                'rsi': rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50,
                'bb_upper': bb_upper.iloc[-1],
                'bb_lower': bb_lower.iloc[-1],
                'bb_middle': bb_middle.iloc[-1]
            }
        except Exception as e:
            print(f"Indicator error: {e}")
            return None
    
    def generate_signal(self, df):
        ind = self.calculate_indicators(df)
        if not ind:
            return None
        
        # FIXED: Enhanced filters
        trend_up, vol_normal, volume_ok = self.enhanced_filters(df)
        if not (vol_normal and volume_ok):
            return None
        
        # Avoid duplicate signals
        if self.last_signal:
            price_change = abs(ind['price'] - self.last_signal['price']) / self.last_signal['price']
            if price_change < 0.002:
                return None
        
        # BUY signal: Enhanced conditions
        if (ind['trend_bullish'] and trend_up and 
            ind['rsi'] >= 60 and ind['rsi'] < 75 and
            ind['histogram'] > 0 and ind['histogram_prev'] <= 0 and
            ind['price'] > ind['bb_lower']):
            return {
                'action': 'BUY',
                'price': ind['price'],
                'rsi': ind['rsi'],
                'reason': 'enhanced_long'
            }
        
        # SELL signal: Enhanced conditions
        if (ind['trend_bearish'] and not trend_up and
            ind['rsi'] <= 40 and ind['rsi'] > 25 and
            ind['histogram'] < 0 and ind['histogram_prev'] >= 0 and
            ind['price'] < ind['bb_upper']):
            return {
                'action': 'SELL',
                'price': ind['price'],
                'rsi': ind['rsi'],
                'reason': 'enhanced_short'
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
        except:
            return False
    
    async def check_position(self):
        try:
            positions = self.exchange.get_positions(category="linear", symbol=self.symbol)
            if positions.get('retCode') == 0:
                pos_list = positions['result']['list']
                if pos_list and float(pos_list[0]['size']) > 0:
                    self.position = pos_list[0]
                else:
                    self.position = None
        except:
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
            
            if profit_pct >= self.config['take_profit_pct']:
                return True, "take_profit"
            if profit_pct <= -self.config['stop_loss_pct']:
                return True, "stop_loss"
            
            return False, ""
        except:
            return False, ""
    
    async def execute_trade(self, signal):
        
        # Check trade cooldown
        import time
        if time.time() - self.last_trade_time < self.trade_cooldown:
            remaining = self.trade_cooldown - (time.time() - self.last_trade_time)
            print(f"⏰ Trade cooldown: wait {remaining:.0f}s")
            return
        if await self.check_pending_orders() or self.position:
            return
        
        # FIXED: Calculate stop loss first for position sizing
        if signal['action'] == 'BUY':
            stop_loss_price = signal['price'] * (1 - self.config['stop_loss_pct']/100)
        else:
            stop_loss_price = signal['price'] * (1 + self.config['stop_loss_pct']/100)
        
        # FIXED: Dynamic position sizing
        position_value = self.calculate_position_size(signal['price'], stop_loss_price)
        qty = position_value / signal['price']
        formatted_qty = self.format_qty(qty)
        
        if float(formatted_qty) < 1:
            return
        
        # FIXED: Apply slippage to execution price
        actual_price = self.apply_slippage(signal['price'], signal['action'])
        
        offset = 1 - self.config['maker_offset_pct']/100 if signal['action'] == 'BUY' else 1 + self.config['maker_offset_pct']/100
        limit_price = round(actual_price * offset, 4)
        
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
                self.last_trade_time = time.time()  # Update last trade time
                self.last_signal = signal
                self.pending_order = order['result']
                
                stop_loss = stop_loss_price
                take_profit = signal['price'] * (1 + self.config['take_profit_pct']/100) if signal['action'] == 'BUY' else signal['price'] * (1 - self.config['take_profit_pct']/100)
                
                self.current_trade_id, _ = self.logger.log_trade_open(
                    side=signal['action'],
                    expected_price=signal['price'],
                    actual_price=limit_price,
                    qty=float(formatted_qty),
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    info=f"RSI:{signal['rsi']:.1f}_{signal['reason']}_FIXED"
                )
                
                print(f"✅ FIXED {signal['action']}: {formatted_qty} @ ${limit_price:.4f} | RSI: {signal['rsi']:.1f}")
                print(f"   💰 Position: ${position_value:.2f} | Risk: 1%")
        except Exception as e:
            print(f"❌ Trade failed: {e}")
    
    async def close_position(self, reason):
        if not self.position:
            return
        
        side = "Sell" if self.position.get('side') == "Buy" else "Buy"
        qty = float(self.position['size'])
        current_price = float(self.price_data['close'].iloc[-1])
        
        # FIXED: Apply slippage to close price
        actual_exit_price = self.apply_slippage(current_price, side)
        
        try:
            order = self.exchange.place_order(
                category="linear",
                symbol=self.symbol,
                side=side,
                orderType="Limit",
                qty=self.format_qty(qty),
                reduceOnly=True
            )
            
            if order.get('retCode') == 0:
                if self.current_trade_id:
                    self.logger.log_trade_close(
                        trade_id=self.current_trade_id,
                        expected_exit=current_price,
                        actual_exit=actual_exit_price,
                        reason=reason
                    )
                    self.current_trade_id = None
                
                print(f"💰 Closed: {reason}")
                self.position = None
                self.last_signal = None
        except:
            pass
    
    async def run_cycle(self):
        
        # Emergency stop check
        if self.daily_pnl < -self.max_daily_loss:
            print(f"🔴 EMERGENCY STOP: Daily loss ${abs(self.daily_pnl):.2f} exceeded limit")
            if self.position:
                await self.close_position("emergency_stop")
            return
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
            print("❌ Failed to connect")
            return
        
        print(f"🔧 FIXED EMA+BB Bot for {self.symbol}")
        print(f"✅ FIXED: Dynamic position sizing (1% risk)")
        print(f"✅ FIXED: Proper fee calculations (+rebates)")
        print(f"✅ FIXED: Slippage modeling")
        print(f"✅ FIXED: Enhanced filters")
        print(f"🎯 TP: {self.config['take_profit_pct']}% | SL: {self.config['stop_loss_pct']}%")
        
        try:
            while True:
                await self.run_cycle()
                await asyncio.sleep(5)
        except KeyboardInterrupt:
            print("\n🛑 Bot stopped")
            try:
                self.exchange.cancel_all_orders(category="linear", symbol=self.symbol)
            except:
                pass
            if self.position:
                await self.close_position("manual_stop")

if __name__ == "__main__":
    bot = EMABBFixedBot()
    asyncio.run(bot.run())