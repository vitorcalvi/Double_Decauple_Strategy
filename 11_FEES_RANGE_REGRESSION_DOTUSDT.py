import os
import asyncio
import pandas as pd
import numpy as np
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

class RangeBalancingBot:
    def __init__(self):
        self.symbol = 'DOTUSDT'
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
            'timeframe': '5',
            'regression_period': 50,
            'bb_period': 20,
            'bb_std': 2.0,
            'channel_width_pct': 1.5,
            'risk_pct': 2.0,
            'maker_offset_pct': 0.01,
            'maker_fee_pct': -0.04,
            'net_take_profit': 0.6,
            'net_stop_loss': 0.3,
            'slippage_pct': 0.02,
            'min_notional': 5,
            'qty_precision': 1,
        }
        
        self.regression_channel = None
        self.last_channel_update = None
        
        self.logger = TradeLogger("RANGE_REGRESSION_FIXED", self.symbol)
        self.current_trade_id = None
    
    def connect(self):
        try:
            self.exchange = HTTP(demo=self.demo_mode, api_key=self.api_key, api_secret=self.api_secret)
            return self.exchange.get_server_time().get('retCode') == 0
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False
    
    def format_qty(self, qty):
        precision = self.config['qty_precision']
        step = 10 ** (-precision)
        rounded_qty = round(qty / step) * step
        return f"{rounded_qty:.{precision}f}"
    
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
    
    def calculate_linear_regression(self, prices):
        if len(prices) < self.config['regression_period']:
            return None
        
        recent_prices = prices.tail(self.config['regression_period'])
        x = np.arange(len(recent_prices))
        y = recent_prices.values
        
        coefficients = np.polyfit(x, y, 1)
        slope = coefficients[0]
        intercept = coefficients[1]
        
        regression_line = slope * x + intercept
        
        residuals = y - regression_line
        std_dev = np.std(residuals)
        
        channel_width = std_dev * self.config['channel_width_pct']
        upper_channel = regression_line[-1] + channel_width
        lower_channel = regression_line[-1] - channel_width
        midline = regression_line[-1]
        
        angle = np.degrees(np.arctan(slope))
        
        return {
            'upper': upper_channel,
            'lower': lower_channel,
            'midline': midline,
            'slope': slope,
            'angle': angle,
            'std_dev': std_dev
        }
    
    def calculate_bollinger_bands(self, prices):
        if len(prices) < self.config['bb_period']:
            return None
        
        sma = prices.rolling(window=self.config['bb_period']).mean()
        std = prices.rolling(window=self.config['bb_period']).std()
        
        upper_band = sma + (std * self.config['bb_std'])
        lower_band = sma - (std * self.config['bb_std'])
        
        return {
            'upper': upper_band.iloc[-1],
            'lower': lower_band.iloc[-1],
            'middle': sma.iloc[-1],
            'bandwidth': (upper_band.iloc[-1] - lower_band.iloc[-1]) / sma.iloc[-1] * 100
        }
    
    def generate_signal(self, df):
        # 🔴 CRITICAL FIX: Don't generate signals if we have pending orders or position
        if self.pending_order or self.position:
            return None
        
        # 🔴 CRITICAL FIX: Check minimum time between orders
        if self.last_order_time:
            time_since_last = (datetime.now() - self.last_order_time).total_seconds()
            if time_since_last < self.min_order_interval:
                return None
        
        if len(df) < self.config['regression_period']:
            return None
        
        current_price = float(df['close'].iloc[-1])
        
        if not self.last_channel_update or (datetime.now() - self.last_channel_update).total_seconds() > 600:
            self.regression_channel = self.calculate_linear_regression(df['close'])
            self.last_channel_update = datetime.now()
        
        if not self.regression_channel:
            return None
        
        bb = self.calculate_bollinger_bands(df['close'])
        if not bb:
            return None
        
        reg_position = (current_price - self.regression_channel['lower']) / (self.regression_channel['upper'] - self.regression_channel['lower'])
        bb_position = (current_price - bb['lower']) / (bb['upper'] - bb['lower'])
        
        if reg_position <= 0.1 and bb_position <= 0.2:
            return {
                'action': 'BUY',
                'price': current_price,
                'reg_channel': self.regression_channel['lower'],
                'bb_band': bb['lower'],
                'trend_angle': self.regression_channel['angle']
            }
        elif reg_position >= 0.9 and bb_position >= 0.8:
            return {
                'action': 'SELL',
                'price': current_price,
                'reg_channel': self.regression_channel['upper'],
                'bb_band': bb['upper'],
                'trend_angle': self.regression_channel['angle']
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
        if not self.position or not self.regression_channel:
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
            
            if current_price >= self.regression_channel['midline']:
                return True, "channel_midline"
        else:
            profit_pct = (entry_price - current_price) / entry_price * 100
            if profit_pct >= self.config['net_take_profit']:
                return True, "take_profit"
            if profit_pct <= -self.config['net_stop_loss']:
                return True, "stop_loss"
            
            if current_price <= self.regression_channel['midline']:
                return True, "channel_midline"
        
        bb = self.calculate_bollinger_bands(self.price_data['close'])
        if bb:
            if side == "Buy" and current_price >= bb['upper']:
                return True, "opposite_bb_band"
            elif side == "Sell" and current_price <= bb['lower']:
                return True, "opposite_bb_band"
        
        return False, ""
    
    async def execute_trade(self, signal):
        # 🔴 CRITICAL FIX: Double-check no pending orders
        if self.pending_order:
            print("⚠️ Order already pending, skipping signal")
            return
        
        # 🔴 CRITICAL FIX: Set pending flag immediately
        self.pending_order = True
        self.last_order_time = datetime.now()
        
        await self.update_account_balance()
        
        if signal['action'] == 'BUY':
            stop_loss_price = signal['reg_channel'] * 0.995
        else:
            stop_loss_price = signal['reg_channel'] * 1.005
        
        qty = self.calculate_position_size(signal['price'], stop_loss_price)
        formatted_qty = self.format_qty(qty)
        
        if float(formatted_qty) < (self.config['min_notional'] / signal['price']):
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
                    info=f"reg:{signal['reg_channel']:.4f}_bb:{signal['bb_band']:.4f}_angle:{signal['trend_angle']:.1f}_risk:{self.config['risk_pct']}%"
                )
                
                position_value = float(formatted_qty) * limit_price
                print(f"✅ ORDER PLACED {signal['action']}: {formatted_qty} @ ${limit_price:.4f}")
                print(f"   💰 Position Value: ${position_value:.2f}")
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
        
        print(f"\n📊 Range Balancing Bot - {self.symbol}")
        print(f"💰 Price: ${current_price:.4f} | Balance: ${self.account_balance:.2f}")
        
        # 🔴 CRITICAL FIX: Show order status
        if self.pending_order:
            print(f"⏳ PENDING ORDER: {self.active_order_id}")
        
        if self.regression_channel:
            print(f"📈 Regression: L:${self.regression_channel['lower']:.4f} | M:${self.regression_channel['midline']:.4f} | U:${self.regression_channel['upper']:.4f}")
        
        if self.position:
            entry_price = float(self.position.get('avgPrice', 0))
            side = self.position.get('side', '')
            size = self.position.get('size', '0')
            
            pnl = float(self.position.get('unrealisedPnl', 0))
            
            emoji = "🟢" if side == "Buy" else "🔴"
            print(f"{emoji} {side}: {size} DOT @ ${entry_price:.4f} | PnL: ${pnl:.2f}")
        else:
            print("🔍 Scanning...")
        
        print("-" * 60)
    
    async def run_cycle(self):
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
        
        print(f"📊 Range Balancing Bot - ORDER MANAGEMENT FIXED")
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
    bot = RangeBalancingBot()
    asyncio.run(bot.run())