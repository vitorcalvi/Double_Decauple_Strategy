import os
import asyncio
import pandas as pd
import numpy as np
import json
from datetime import datetime, timezone
from pybit.unified_trading import HTTP
from dotenv import load_dotenv

load_dotenv()

# Use the same TradeLogger class from above
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
        
        entry_rebate = trade["entry_price"] * trade["qty"] * abs(fees_entry) / 100
        exit_rebate = actual_exit * trade["qty"] * abs(fees_exit) / 100
        total_rebates = entry_rebate + exit_rebate
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
            "fee_rebates": {"entry": round(entry_rebate, 2), "exit": round(exit_rebate, 2), "total": round(total_rebates, 2)},
            "net_pnl": round(net_pnl, 2),
            "reason": reason,
            "currency": self.currency
        }
        
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        
        del self.open_trades[trade_id]
        return log_entry

class RMISuperTrendBot:
    def __init__(self):
        
        # Trade cooldown mechanism
        self.last_trade_time = 0
        self.trade_cooldown = 30  # 30 seconds between trades
        
        
        # Emergency stop tracking
        self.daily_pnl = 0
        self.consecutive_losses = 0
        self.max_daily_loss = 50  # $50 max daily loss
        
        self.symbol = 'ADAUSDT'
        self.demo_mode = os.getenv('DEMO_MODE', 'true').lower() == 'true'
        
        prefix = 'TESTNET_' if self.demo_mode else 'LIVE_'
        self.api_key = os.getenv(f'{prefix}BYBIT_API_KEY')
        self.api_secret = os.getenv(f'{prefix}BYBIT_API_SECRET')
        self.exchange = None
        
        self.position = None
        self.price_data = pd.DataFrame()
        self.account_balance = 1000
        
        # 🔴 CRITICAL FIX: Order management state
        self.pending_order = False
        self.last_order_time = None
        self.active_order_id = None
        self.min_order_interval = 30  # Minimum seconds between orders
        
        self.config = {
            'timeframe': '3',
            'rmi_period': 14,
            'rmi_momentum': 5,
            'rmi_threshold_long': 55,
            'rmi_threshold_short': 45,
            'supertrend_period': 10,
            'supertrend_multiplier': 2,
            'risk_pct': 2.0,
            'maker_offset_pct': 0.01,
            'maker_fee_pct': -0.04,
            'net_take_profit': 0.70,
            'net_stop_loss': 0.35,
            'slippage_pct': 0.02,
            'min_notional': 5,
            'qty_precision': 0,
        }
        
        self.logger = TradeLogger("RMI_SUPERTREND_FIXED", self.symbol)
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
    
    async def update_account_balance(self):
        try:
            wallet = self.exchange.get_wallet_balance(accountType="UNIFIED")
            if wallet.get('retCode') == 0:
                for coin in wallet['result']['list'][0]['coin']:
                    if coin['coin'] == 'USDT':
                        self.account_balance = float(coin['availableToWithdraw'])
                        break
        except Exception as e:
            print(f"⚠️ Balance update error: {e}")
    
    def calculate_position_size(self, price, stop_loss_price):
        if self.account_balance <= 0:
            return 0
        
        risk_amount = self.account_balance * (self.config['risk_pct'] / 100)
        price_diff = abs(price - stop_loss_price)
        
        if price_diff == 0:
            return 0
        
        qty = risk_amount / price_diff
        
        notional = qty * price
        if notional < self.config['min_notional']:
            qty = self.config['min_notional'] / price
        
        return qty
    
    def calculate_limit_price(self, market_price, side, include_slippage=True):
        slippage_mult = 1 + (self.config['slippage_pct'] / 100) if include_slippage else 1
        
        if side == 'BUY':
            price_with_slippage = market_price * slippage_mult
            limit_price = price_with_slippage * (1 - self.config['maker_offset_pct'] / 100)
        else:
            price_with_slippage = market_price / slippage_mult
            limit_price = price_with_slippage * (1 + self.config['maker_offset_pct'] / 100)
        
        return round(limit_price, 4)
    
    # 🔴 CRITICAL FIX: Check and cancel pending orders
    async def check_pending_orders(self):
        """Check for any unfilled orders and cancel old ones"""
        try:
            orders = self.exchange.get_open_orders(
                category="linear",
                symbol=self.symbol,
                limit=50
            )
            
            if orders.get('retCode') == 0:
                open_orders = orders['result']['list']
                
                for order in open_orders:
                    order_time = datetime.fromtimestamp(int(order['createdTime'])/1000, tz=timezone.utc)
                    time_diff = (datetime.now(timezone.utc) - order_time).total_seconds()
                    
                    # Cancel orders older than 60 seconds
                    if time_diff > 60:
                        try:
                            self.exchange.cancel_order(
                                category="linear",
                                symbol=self.symbol,
                                orderId=order['orderId']
                            )
                            print(f"❌ Cancelled stale order: {order['orderId']}")
                            self.pending_order = False
                            self.active_order_id = None
                        except:
                            pass
                    else:
                        # We have a pending order
                        self.pending_order = True
                        self.active_order_id = order['orderId']
                        return True
                
                # No pending orders
                self.pending_order = False
                self.active_order_id = None
                return False
        except Exception as e:
            print(f"⚠️ Order check error: {e}")
            return False
    
    def calculate_rmi(self, prices):
        if len(prices) < self.config['rmi_period'] + self.config['rmi_momentum']:
            return None
        
        momentum = prices.diff(self.config['rmi_momentum'])
        
        gain = momentum.where(momentum > 0, 0)
        loss = -momentum.where(momentum < 0, 0)
        
        avg_gain = gain.rolling(window=self.config['rmi_period']).mean()
        avg_loss = loss.rolling(window=self.config['rmi_period']).mean()
        
        rs = avg_gain / (avg_loss + 1e-10)
        rmi = 100 - (100 / (1 + rs))
        
        return rmi
    
    def calculate_supertrend(self, df):
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=self.config['supertrend_period']).mean()
        
        hl_avg = (high + low) / 2
        multiplier = self.config['supertrend_multiplier']
        
        upper_band = hl_avg + (multiplier * atr)
        lower_band = hl_avg - (multiplier * atr)
        
        supertrend = pd.Series(index=df.index, dtype=float)
        direction = pd.Series(index=df.index, dtype=int)
        
        for i in range(self.config['supertrend_period'], len(df)):
            if pd.isna(upper_band.iloc[i]) or pd.isna(lower_band.iloc[i]):
                continue
                
            if i == self.config['supertrend_period']:
                if close.iloc[i] <= upper_band.iloc[i]:
                    direction.iloc[i] = -1
                    supertrend.iloc[i] = upper_band.iloc[i]
                else:
                    direction.iloc[i] = 1
                    supertrend.iloc[i] = lower_band.iloc[i]
            else:
                prev_dir = direction.iloc[i-1]
                
                if close.iloc[i] <= upper_band.iloc[i]:
                    if prev_dir == -1:
                        if upper_band.iloc[i] < supertrend.iloc[i-1]:
                            supertrend.iloc[i] = upper_band.iloc[i]
                        else:
                            supertrend.iloc[i] = supertrend.iloc[i-1]
                        direction.iloc[i] = -1
                    else:
                        supertrend.iloc[i] = upper_band.iloc[i]
                        direction.iloc[i] = -1
                else:
                    if prev_dir == 1:
                        if lower_band.iloc[i] > supertrend.iloc[i-1]:
                            supertrend.iloc[i] = lower_band.iloc[i]
                        else:
                            supertrend.iloc[i] = supertrend.iloc[i-1]
                        direction.iloc[i] = 1
                    else:
                        supertrend.iloc[i] = lower_band.iloc[i]
                        direction.iloc[i] = 1
        
        return supertrend, direction
    
    def generate_signal(self, df):
        # 🔴 CRITICAL FIX: Don't generate signals if we have pending orders or position
        if self.pending_order or self.position:
            return None
        
        # 🔴 CRITICAL FIX: Check minimum time between orders
        if self.last_order_time:
            time_since_last = (datetime.now() - self.last_order_time).total_seconds()
            if time_since_last < self.min_order_interval:
                return None
        
        if len(df) < 50:
            return None
        
        current_price = float(df['close'].iloc[-1])
        
        rmi = self.calculate_rmi(df['close'])
        if rmi is None or pd.isna(rmi.iloc[-1]):
            return None
        
        current_rmi = rmi.iloc[-1]
        
        supertrend, direction = self.calculate_supertrend(df)
        if pd.isna(supertrend.iloc[-1]) or pd.isna(direction.iloc[-1]):
            return None
        
        current_supertrend = supertrend.iloc[-1]
        current_direction = direction.iloc[-1]
        
        if current_rmi > self.config['rmi_threshold_long'] and current_direction == 1 and current_price > current_supertrend:
            return {
                'action': 'BUY',
                'price': current_price,
                'rmi': current_rmi,
                'supertrend': current_supertrend,
                'trend': 'UP'
            }
        elif current_rmi < self.config['rmi_threshold_short'] and current_direction == -1 and current_price < current_supertrend:
            return {
                'action': 'SELL',
                'price': current_price,
                'rmi': current_rmi,
                'supertrend': current_supertrend,
                'trend': 'DOWN'
            }
        
        return None
    
    async def get_market_data(self):
        try:
            klines = self.exchange.get_kline(
                category="linear",
                symbol=self.symbol,
                interval=self.config['timeframe'],
                limit=100
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
            profit_pct = (current_price - entry_price) / entry_price * 100
            if profit_pct >= self.config['net_take_profit']:
                return True, "take_profit"
            if profit_pct <= -self.config['net_stop_loss']:
                return True, "stop_loss"
        else:
            profit_pct = (entry_price - current_price) / entry_price * 100
            if profit_pct >= self.config['net_take_profit']:
                return True, "take_profit"
            if profit_pct <= -self.config['net_stop_loss']:
                return True, "stop_loss"
        
        supertrend, direction = self.calculate_supertrend(self.price_data)
        if not pd.isna(supertrend.iloc[-1]):
            if side == "Buy" and current_price < supertrend.iloc[-1]:
                return True, "supertrend_exit"
            elif side == "Sell" and current_price > supertrend.iloc[-1]:
                return True, "supertrend_exit"
        
        rmi = self.calculate_rmi(self.price_data['close'])
        if rmi is not None and not pd.isna(rmi.iloc[-1]):
            if side == "Buy" and rmi.iloc[-1] < self.config['rmi_threshold_short']:
                return True, "rmi_reversal"
            elif side == "Sell" and rmi.iloc[-1] > self.config['rmi_threshold_long']:
                return True, "rmi_reversal"
        
        return False, ""
    
    async def execute_trade(self, signal):
        
        # Check trade cooldown
        import time
        if time.time() - self.last_trade_time < self.trade_cooldown:
            remaining = self.trade_cooldown - (time.time() - self.last_trade_time)
            print(f"⏰ Trade cooldown: wait {remaining:.0f}s")
            return
        # 🔴 CRITICAL FIX: Double-check no pending orders
        if self.pending_order:
            print("⚠️ Order already pending, skipping signal")
            return
        
        # 🔴 CRITICAL FIX: Set pending flag immediately
        self.pending_order = True
        self.last_order_time = datetime.now()
        
        await self.update_account_balance()
        
        if signal['action'] == 'BUY':
            stop_loss_price = signal['supertrend'] * 0.995
        else:
            stop_loss_price = signal['supertrend'] * 1.005
        
        qty = self.calculate_position_size(signal['price'], stop_loss_price)
        formatted_qty = self.format_qty(qty)
        
        if int(formatted_qty) < (self.config['min_notional'] / signal['price']):
            print(f"⚠️ Position size too small: {formatted_qty}")
            self.pending_order = False  # Reset flag
            return
        
        limit_price = self.calculate_limit_price(signal['price'], signal['action'])
        
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
                self.active_order_id = order['result']['orderId']
                
                net_tp = limit_price * (1 + self.config['net_take_profit']/100) if signal['action'] == 'BUY' else limit_price * (1 - self.config['net_take_profit']/100)
                net_sl = limit_price * (1 - self.config['net_stop_loss']/100) if signal['action'] == 'BUY' else limit_price * (1 + self.config['net_stop_loss']/100)
                
                self.current_trade_id, _ = self.logger.log_trade_open(
                    side=signal['action'],
                    expected_price=signal['price'],
                    actual_price=limit_price,
                    qty=float(formatted_qty),
                    stop_loss=stop_loss_price,
                    take_profit=net_tp,
                    info=f"rmi:{signal['rmi']:.1f}_st:{signal['supertrend']:.4f}_trend:{signal['trend']}_risk:{self.config['risk_pct']}%"
                )
                
                position_value = float(formatted_qty) * limit_price
                print(f"✅ ORDER PLACED {signal['action']}: {formatted_qty} @ ${limit_price:.4f}")
                print(f"   💰 Position Value: ${position_value:.2f}")
                print(f"   📊 RMI: {signal['rmi']:.1f} | SuperTrend: ${signal['supertrend']:.4f}")
                print(f"   🆔 Order ID: {self.active_order_id}")
            else:
                print(f"❌ Order failed: {order.get('retMsg')}")
                self.pending_order = False  # Reset flag on failure
                
        except Exception as e:
            print(f"❌ Trade failed: {e}")
            self.pending_order = False  # Reset flag on error
    
    async def close_position(self, reason):
        if not self.position:
            return
        
        # 🔴 CRITICAL FIX: Set pending flag for close orders too
        self.pending_order = True
        
        current_price = float(self.price_data['close'].iloc[-1])
        side = "Sell" if self.position.get('side') == "Buy" else "Buy"
        qty = float(self.position['size'])
        
        limit_price = self.calculate_limit_price(current_price, side)
        
        try:
            order = self.exchange.place_order(
                category="linear",
                symbol=self.symbol,
                side=side,
                orderType="Limit",
                qty=self.format_qty(qty),
                price=str(limit_price),
                timeInForce="PostOnly",
                reduceOnly=True
            )
            
            if order.get('retCode') == 0:
                if self.current_trade_id:
                    self.logger.log_trade_close(
                        trade_id=self.current_trade_id,
                        expected_exit=current_price,
                        actual_exit=limit_price,
                        reason=reason,
                        fees_entry=self.config['maker_fee_pct'],
                        fees_exit=self.config['maker_fee_pct']
                    )
                    self.current_trade_id = None
                
                print(f"✅ CLOSE ORDER PLACED: {reason}")
                self.position = None
            else:
                print(f"❌ Close failed: {order.get('retMsg')}")
                self.pending_order = False  # Reset on failure
                
        except Exception as e:
            print(f"❌ Close failed: {e}")
            self.pending_order = False  # Reset on error
    
    def show_status(self):
        if len(self.price_data) == 0:
            return
        
        current_price = float(self.price_data['close'].iloc[-1])
        
        print(f"\n📈 RMI + SuperTrend Bot - {self.symbol}")
        print(f"💰 Price: ${current_price:.4f} | Balance: ${self.account_balance:.2f}")
        
        # 🔴 CRITICAL FIX: Show order status
        if self.pending_order:
            print(f"⏳ PENDING ORDER: {self.active_order_id}")
        
        rmi = self.calculate_rmi(self.price_data['close'])
        supertrend, direction = self.calculate_supertrend(self.price_data)
        
        if rmi is not None and not pd.isna(rmi.iloc[-1]):
            print(f"📊 RMI: {rmi.iloc[-1]:.1f}")
        
        if not pd.isna(supertrend.iloc[-1]):
            trend = "UP" if direction.iloc[-1] == 1 else "DOWN"
            print(f"📈 SuperTrend: ${supertrend.iloc[-1]:.4f} | Trend: {trend}")
        
        if self.position:
            entry_price = float(self.position.get('avgPrice', 0))
            side = self.position.get('side', '')
            size = self.position.get('size', '0')
            
            pnl = float(self.position.get('unrealisedPnl', 0))
            
            emoji = "🟢" if side == "Buy" else "🔴"
            print(f"{emoji} {side}: {size} ADA @ ${entry_price:.4f} | PnL: ${pnl:.2f}")
        else:
            print("🔍 Scanning...")
        
        print("-" * 60)
    
    async def run_cycle(self):
        
        # Emergency stop check
        if self.daily_pnl < -self.max_daily_loss:
            print(f"🔴 EMERGENCY STOP: Daily loss ${abs(self.daily_pnl):.2f} exceeded limit")
            if self.position:
                await self.close_position("emergency_stop")
            return
        if not await self.get_market_data():
            return
        
        # 🔴 CRITICAL FIX: Check pending orders first
        await self.check_pending_orders()
        
        await self.check_position()
        
        if self.position:
            should_close, reason = self.should_close()
            if should_close and not self.pending_order:
                await self.close_position(reason)
        else:
            if not self.pending_order:
                signal = self.generate_signal(self.price_data)
                if signal:
                    await self.execute_trade(signal)
        
        self.show_status()
    
    async def run(self):
        if not self.connect():
            print("❌ Failed to connect")
            return
        
        print(f"📈 RMI + SuperTrend Bot - ORDER MANAGEMENT FIXED")
        print(f"✅ CRITICAL FIXES:")
        print(f"   • Pending order tracking")
        print(f"   • Minimum {self.min_order_interval}s between orders")
        print(f"   • Stale order cancellation")
        print(f"   • Race condition prevention")
        
        try:
            while True:
                await self.run_cycle()
                await asyncio.sleep(5)
        except KeyboardInterrupt:
            print("\n🛑 Bot stopped")
            if self.position:
                await self.close_position("manual_stop")
        except Exception as e:
            print(f"⚠️ Runtime error: {e}")
            await asyncio.sleep(5)

if __name__ == "__main__":
    bot = RMISuperTrendBot()
    asyncio.run(bot.run())