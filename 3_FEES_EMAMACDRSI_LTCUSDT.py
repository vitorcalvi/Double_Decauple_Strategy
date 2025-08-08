import os
import time
import asyncio
import pandas as pd
import json
from datetime import datetime, timezone
from pybit.unified_trading import HTTP
from dotenv import load_dotenv

load_dotenv()

class TradeLogger:
    def __init__(self, bot_name, symbol):
        self.demo_mode = os.getenv('DEMO_MODE', 'true').lower() == 'true'
        self.bot_name = bot_name
        self.symbol = symbol
        self.currency = "USDT"
        self.open_trades = {}
        self.trade_id = 1000
        
        os.makedirs("logs", exist_ok=True)
        self.log_file = f"logs/3_FEES_EMAMACDRSI_LTCUSDT.log"
        
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
        
        gross_pnl = ((actual_exit - trade["entry_price"]) * trade["qty"] 
                    if trade["side"] == "BUY" 
                    else (trade["entry_price"] - actual_exit) * trade["qty"])
        
        entry_fee = trade["entry_price"] * trade["qty"] * fees_entry / 100
        exit_fee = actual_exit * trade["qty"] * fees_exit / 100
        total_fees = entry_fee + exit_fee
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
            "total_fees": round(total_fees, 2),
            "net_pnl": round(net_pnl, 2),
            "reason": reason,
            "currency": self.currency
        }
        
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        
        del self.open_trades[trade_id]
        return log_entry

class EMARSIBot:
    def __init__(self):
        self.symbol = 'LTCUSDT'
        self.demo_mode = os.getenv('DEMO_MODE', 'true').lower() == 'true'
        
        prefix = 'TESTNET_' if self.demo_mode else 'LIVE_'
        self.api_key = os.getenv(f'{prefix}BYBIT_API_KEY')
        self.api_secret = os.getenv(f'{prefix}BYBIT_API_SECRET')
        
        self.exchange = None
        self.position = None
        self.pending_order = None
        self.price_data = pd.DataFrame()
        self.account_balance = 1000
        
        self.config = {
            'ema_fast': 5,
            'ema_slow': 13,
            'rsi_period': 5,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'risk_per_trade': 1.0,
            'maker_offset_pct': 0.01,
            'slippage_pct': 0.02,
            'net_take_profit': 0.86,
            'net_stop_loss': 0.43,
            'order_timeout': 180,
            'min_notional': 5,
        }
        
        self.tick_size = 0.01
        self.qty_step = 0.01
        
        self.last_trade_time = 0
        self.trade_cooldown = 30
        
        self.logger = TradeLogger("EMA_RSI_FIXED", self.symbol)
        self.current_trade_id = None
    
    def connect(self):
        try:
            self.exchange = HTTP(demo=self.demo_mode, api_key=self.api_key, api_secret=self.api_secret)
            return self.exchange.get_server_time().get('retCode') == 0
        except:
            return False
    
    async def get_account_balance(self):
        try:
            result = self.exchange.get_wallet_balance(accountType="UNIFIED", coin="USDT")
            if result.get('retCode') == 0:
                balance_list = result['result']['list']
                if balance_list:
                    for coin in balance_list[0]['coin']:
                        if coin['coin'] == 'USDT':
                            self.account_balance = float(coin['availableToWithdraw'])
                            return True
        except:
            self.account_balance = 1000
        return False
    
    async def get_instrument_info(self):
        try:
            result = self.exchange.get_instruments_info(category="linear", symbol=self.symbol)
            if result.get('retCode') == 0:
                info = result['result']['list'][0]
                self.tick_size = float(info['priceFilter']['tickSize'])
                self.qty_step = float(info['lotSizeFilter']['qtyStep'])
                return True
        except:
            pass
        return False
    
    def calculate_position_size(self, price, stop_loss_price):
        if self.account_balance <= 0:
            return 0
        
        risk_amount = self.account_balance * self.config['risk_per_trade'] / 100
        stop_distance = abs(price - stop_loss_price)
        
        if stop_distance == 0:
            return 0
        
        position_size = risk_amount / stop_distance
        notional = position_size * price
        
        return position_size if notional >= self.config['min_notional'] else 0
    
    def format_price(self, price):
        return round(price / self.tick_size) * self.tick_size
    
    def format_qty(self, qty):
        formatted = round(qty / self.qty_step) * self.qty_step
        return f"{formatted:.2f}"
    
    def estimate_execution_price(self, market_price, side, is_limit=True):
        if is_limit:
            return self.format_price(
                market_price * (1 - self.config['maker_offset_pct']/100) if side == 'BUY'
                else market_price * (1 + self.config['maker_offset_pct']/100)
            )
        else:
            return self.format_price(
                market_price * (1 + self.config['slippage_pct']/100) if side == 'BUY'
                else market_price * (1 - self.config['slippage_pct']/100)
            )
    
    async def check_pending_orders(self):
        # Clear pending orders after timeout
        if self.pending_order and time.time() - self.last_trade_time > 30:
            self.pending_order = False
            print("✓ Cleared stale pending order")
            
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
            
            if age > self.config['order_timeout']:
                self.exchange.cancel_order(category="linear", symbol=self.symbol, orderId=order['orderId'])
                self.pending_order = None
                return False
            
            self.pending_order = order
            return True
        except:
            return False
    
    def calculate_indicators(self, df):
        if len(df) < max(self.config['ema_slow'], self.config['rsi_period']) + 1:
            return None
        
        close = df['close']
        
        ema_fast = close.ewm(span=self.config['ema_fast']).mean().iloc[-1]
        ema_slow = close.ewm(span=self.config['ema_slow']).mean().iloc[-1]
        
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(window=self.config['rsi_period']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.config['rsi_period']).mean()
        rsi = 100 - (100 / (1 + gain / loss)).iloc[-1]
        
        # Handle flat market
        if pd.isna(rsi) or rsi == 0:
            rsi = 50.0
        
        return {
            'trend': 'UP' if ema_fast > ema_slow else 'DOWN',
            'rsi': rsi if pd.notna(rsi) else 50,
            'ema_fast': ema_fast,
            'ema_slow': ema_slow
        }
    
    def generate_signal(self, df):
        if self.position:
            return None
        
        time_since_last = datetime.now().timestamp() - self.last_trade_time
        if time_since_last < self.trade_cooldown:
            return None
        
        indicators = self.calculate_indicators(df)
        if not indicators:
            return None
        
        price = float(df['close'].iloc[-1])
        
        # Strong signal requirements
        if indicators['trend'] == 'UP' and indicators['rsi'] < self.config['rsi_oversold']:
            return {'action': 'BUY', 'price': price, 'rsi': indicators['rsi']}
        elif indicators['trend'] == 'DOWN' and indicators['rsi'] > self.config['rsi_overbought']:
            return {'action': 'SELL', 'price': price, 'rsi': indicators['rsi']}
        
        return None
    
    async def get_market_data(self):
        try:
            klines = self.exchange.get_kline(category="linear", symbol=self.symbol, interval="1", limit=50)
            if klines.get('retCode') != 0:
                return False
            
            df = pd.DataFrame(klines['result']['list'], 
                            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
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
        
        if entry_price == 0:
            return False, ""
        
        is_long = self.position.get('side') == "Buy"
        profit_pct = ((current_price - entry_price) / entry_price * 100 if is_long 
                     else (entry_price - current_price) / entry_price * 100)
        
        if profit_pct >= self.config['net_take_profit']:
            return True, "take_profit"
        if profit_pct <= -self.config['net_stop_loss']:
            return True, "stop_loss"
        
        # Exit on opposite EMA crossover
        indicators = self.calculate_indicators(self.price_data)
        if indicators:
            if (is_long and indicators['trend'] == 'DOWN') or (not is_long and indicators['trend'] == 'UP'):
                return True, "ema_crossover"
        
        return False, ""
    
    async def execute_trade(self, signal):
        await self.check_position()
        if self.position:
            print("⚠️ Position already exists, skipping trade")
            return
            
        if await self.check_pending_orders():
            print("⚠️ Pending order exists, skipping trade")
            return
        
        time_since_last = datetime.now().timestamp() - self.last_trade_time
        if time_since_last < self.trade_cooldown:
            print(f"⚠️ Trade cooldown active, wait {self.trade_cooldown - time_since_last:.0f}s")
            return
        
        await self.get_account_balance()
        
        market_price = signal['price']
        is_buy = signal['action'] == 'BUY'
        stop_loss_price = (market_price * (1 - self.config['net_stop_loss']/100) if is_buy 
                          else market_price * (1 + self.config['net_stop_loss']/100))
        
        qty = self.calculate_position_size(market_price, stop_loss_price)
        formatted_qty = self.format_qty(qty)
        
        if float(formatted_qty) < self.qty_step:
            print(f"⚠️ Position size too small: {formatted_qty}")
            return
        
        limit_price = self.estimate_execution_price(market_price, signal['action'], is_limit=True)
        
        try:
            order = self.exchange.place_order(category="linear", symbol=self.symbol, side="Buy" if is_buy else "Sell", orderType="Limit", qty=formatted_qty, price=str(limit_price), timeInForce="PostOnly")
            
            if order.get('retCode') == 0:
                self.pending_order = order['result']
                self.last_trade_time = datetime.now().timestamp()
                
                take_profit = (limit_price * (1 + self.config['net_take_profit']/100) if is_buy 
                             else limit_price * (1 - self.config['net_take_profit']/100))
                
                self.current_trade_id, _ = self.logger.log_trade_open(
                    side=signal['action'],
                    expected_price=market_price,
                    actual_price=limit_price,
                    qty=float(formatted_qty),
                    stop_loss=stop_loss_price,
                    take_profit=take_profit,
                    info=f"RSI:{signal['rsi']:.1f}_Trend:{signal['action']}"
                )
                
                print(f"✅ {signal['action']}: {formatted_qty} @ ${limit_price:.2f} | RSI: {signal['rsi']:.1f} | Balance: ${self.account_balance:.0f}")
                
        except Exception as e:
            print(f"❌ Trade failed: {e}")
    
    async def close_position(self, reason):
        if not self.position:
            return
        
        qty = float(self.position['size'])
        side = "Sell" if self.position.get('side') == "Buy" else "Buy"
        current_price = float(self.price_data['close'].iloc[-1])
        
        execution_price = self.estimate_execution_price(current_price, side, is_limit=False)
        
        try:
            order = self.exchange.place_order(category="linear", symbol=self.symbol, side=side, orderType="Limit", reduceOnly=True, timeInForce="PostOnly")
            
            if order.get('retCode') == 0:
                if self.current_trade_id:
                    self.logger.log_trade_close(
                        trade_id=self.current_trade_id,
                        expected_exit=current_price,
                        actual_exit=execution_price,
                        reason=reason,
                        fees_entry=-0.04,
                        fees_exit=0.1
                    )
                    self.current_trade_id = None
                
                print(f"✅ Closed: {reason} @ ${execution_price:.2f}")
        except Exception as e:
            print(f"❌ Close failed: {e}")
    
    def show_status(self):
        if len(self.price_data) == 0:
            return
        
        current_price = float(self.price_data['close'].iloc[-1])
        status_parts = [f"📊 BNB: ${current_price:.2f}", f"💰 Balance: ${self.account_balance:.0f}"]
        
        if self.position:
            entry = float(self.position.get('avgPrice', 0))
            side = self.position.get('side', '')
            size = float(self.position.get('size', 0))
            pnl = float(self.position.get('unrealisedPnl', 0))
            status_parts.append(f"📍 {side}: {size:.2f} @ ${entry:.2f} | PnL: ${pnl:.2f}")
        elif self.pending_order:
            order_price = float(self.pending_order.get('price', 0))
            order_side = self.pending_order.get('side', '')
            age = int(datetime.now().timestamp() - int(self.pending_order.get('createdTime', 0)) / 1000)
            status_parts.append(f"⏳ {order_side} @ ${order_price:.2f} ({age}s)")
        else:
            indicators = self.calculate_indicators(self.price_data)
            if indicators:
                if indicators['trend'] == 'UP' and indicators['rsi'] < self.config['rsi_oversold']:
                    signal_status = "🟢 BUY SIGNAL"
                elif indicators['trend'] == 'DOWN' and indicators['rsi'] > self.config['rsi_overbought']:
                    signal_status = "🔴 SELL SIGNAL"
                else:
                    signal_status = "⚪ NO SIGNAL"
                
                status_parts.append(f"RSI: {indicators['rsi']:.1f} | Trend: {indicators['trend']} | {signal_status}")
        
        if self.last_trade_time > 0:
            time_since_last = datetime.now().timestamp() - self.last_trade_time
            if time_since_last < self.trade_cooldown:
                status_parts.append(f"⏰ Cooldown: {self.trade_cooldown - time_since_last:.0f}s")
        
        print(" | ".join(status_parts), end='\r')
    
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
        
        await self.get_account_balance()
        await self.get_instrument_info()
        
        print(f"🔧 EMA + RSI bot for {self.symbol} (FIXED)")
        print(f"✅ FIXES APPLIED:")
        print(f"   • Strong RSI signals: <{self.config['rsi_oversold']} for BUY, >{self.config['rsi_overbought']} for SELL")
        print(f"   • Position tracking: No duplicate trades")
        print(f"   • Trade cooldown: {self.trade_cooldown}s between trades")
        print(f"   • Exit on EMA crossover")
        print(f"💰 Account Balance: ${self.account_balance:.2f}")
        print(f"🎯 TP: {self.config['net_take_profit']:.2f}% | SL: {self.config['net_stop_loss']:.2f}%")
        
        while True:
            try:
                await self.run_cycle()
                await asyncio.sleep(1)
            except KeyboardInterrupt:
                print("\n🛑 Bot stopped")
                try:
                    self.exchange.cancel_all_orders(category="linear", symbol=self.symbol)
                except:
                    pass
                if self.position:
                    await self.close_position("manual_stop")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                await asyncio.sleep(5)

if __name__ == "__main__":
    bot = EMARSIBot()
    asyncio.run(bot.run())