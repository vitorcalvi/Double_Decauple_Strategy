import os
import asyncio
import pandas as pd
import numpy as np
import json
import time
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
        
        # Emergency stop tracking
        self.daily_pnl = 0
        self.consecutive_losses = 0
        self.max_daily_loss = 50
        
        os.makedirs("logs", exist_ok=True)
        self.log_file = f"logs/9_FEES_PIVOT_REVERSAL_LINKUSDT.log"
        
    def generate_trade_id(self):
        self.trade_id += 1
        return self.trade_id
    
    def log_trade_open(self, side, expected_price, actual_price, qty, stop_loss, take_profit, info=""):
        trade_id = self.generate_trade_id()
        slippage = 0  # PostOnly = zero slippage
        
        log_entry = {
            "id": trade_id,
            "bot": self.bot_name,
            "symbol": self.symbol,
            "side": "LONG" if side == "BUY" else "SHORT",
            "action": "OPEN",
            "ts": datetime.now(timezone.utc).isoformat(),
            "expected_price": round(expected_price, 4),
            "actual_price": round(actual_price, 4),
            "slippage": round(slippage, 6),
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
    
    def log_trade_close(self, trade_id, expected_exit, actual_exit, reason, fees_entry=-0.01, fees_exit=-0.01):
        if trade_id not in self.open_trades:
            return None
            
        trade = self.open_trades[trade_id]
        duration = (datetime.now() - trade["entry_time"]).total_seconds()
        slippage = 0  # PostOnly = zero slippage
        
        gross_pnl = ((actual_exit - trade["entry_price"]) * trade["qty"] 
                    if trade["side"] == "BUY" 
                    else (trade["entry_price"] - actual_exit) * trade["qty"])
        
        # Maker rebates (negative fees = rebate)
        entry_rebate = abs(trade["entry_price"] * trade["qty"] * fees_entry / 100)
        exit_rebate = abs(actual_exit * trade["qty"] * fees_exit / 100)
        total_rebates = entry_rebate + exit_rebate
        net_pnl = gross_pnl + total_rebates
        
        # Update daily PnL
        self.daily_pnl += net_pnl
        self.consecutive_losses = self.consecutive_losses + 1 if net_pnl < 0 else 0
        
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
            "slippage": round(slippage, 6),
            "qty": round(trade["qty"], 6),
            "gross_pnl": round(gross_pnl, 2),
            "fee_rebates": {
                "entry": round(entry_rebate, 2),
                "exit": round(exit_rebate, 2),
                "total": round(total_rebates, 2)
            },
            "net_pnl": round(net_pnl, 2),
            "reason": reason,
            "currency": self.currency
        }
        
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        
        del self.open_trades[trade_id]
        return log_entry

class PivotReversalBot:
    def __init__(self):
        self.symbol = 'LINKUSDT'
        self.demo_mode = os.getenv('DEMO_MODE', 'true').lower() == 'true'
        
        prefix = 'TESTNET_' if self.demo_mode else 'LIVE_'
        self.api_key = os.getenv(f'{prefix}BYBIT_API_KEY')
        self.api_secret = os.getenv(f'{prefix}BYBIT_API_SECRET')
        self.exchange = None
        
        # Trading state
        self.position = None
        self.pending_order = None
        self.price_data = pd.DataFrame()
        self.account_balance = 1000
        
        # Failed pivot tracking
        self.failed_pivots = {}
        self.pivot_cooldown = 600  # 10 minutes cooldown
        
        # Configuration with ZERO SLIPPAGE settings
        self.config = {
            'timeframe': '3',
            'rsi_period': 14,
            'mfi_period': 14,
            'rsi_oversold': 25,
            'rsi_overbought': 75,
            'mfi_oversold': 25,
            'mfi_overbought': 75,
            'risk_pct': 2.0,
            'maker_offset_pct': 0.02,
            'maker_fee': -0.01,
            'net_take_profit': 0.6,
            'net_stop_loss': 0.3,
            'min_notional': 5,
            'qty_precision': 1,
            'pivot_distance_pct': 0.3,
            'order_timeout': 30,
            'limit_order_retries': 3
        }
        
        # Pivot levels tracking
        self.pivot_levels = {}
        self.last_pivot_update = None
        
        # Trade cooldown
        self.last_trade_time = 0
        self.trade_cooldown = 30
        
        # Trade logging
        self.logger = TradeLogger("PIVOT_REVERSAL_FIXED", self.symbol)
        self.current_trade_id = None
        self.last_entry_pivot = None
    
    def connect(self):
        try:
            self.exchange = HTTP(demo=self.demo_mode, api_key=self.api_key, api_secret=self.api_secret)
            return self.exchange.get_server_time().get('retCode') == 0
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False
    
    async def execute_limit_order(self, side, qty, price, is_reduce=False):
        """Execute limit order with PostOnly for ZERO slippage"""
        formatted_qty = self.format_qty(qty)
        
        for retry in range(self.config['limit_order_retries']):
            # Calculate limit price with offset
            limit_price = (price * (1 - self.config['maker_offset_pct']/100) if side == "Buy"
                          else price * (1 + self.config['maker_offset_pct']/100))
            limit_price = round(limit_price, 4)
            
            params = {
                "category": "linear",
                "symbol": self.symbol,
                "side": side,
                "orderType": "Limit",
                "qty": formatted_qty,
                "price": str(limit_price),
                "timeInForce": "PostOnly"
            }
            
            if is_reduce:
                params["reduceOnly"] = True
            
            try:
                order = self.exchange.place_order(**params)
                
                if order.get('retCode') == 0:
                    order_id = order['result']['orderId']
                    
                    # Wait for fill
                    start_time = time.time()
                    while time.time() - start_time < self.config['order_timeout']:
                        await asyncio.sleep(1)
                        
                        # Check order status
                        order_status = self.exchange.get_open_orders(
                            category="linear",
                            symbol=self.symbol,
                            orderId=order_id
                        )
                        
                        if order_status['retCode'] == 0 and not order_status['result']['list']:
                            return limit_price  # Order filled with ZERO slippage
                    
                    # Cancel unfilled order
                    try:
                        self.exchange.cancel_order(
                            category="linear",
                            symbol=self.symbol,
                            orderId=order_id
                        )
                    except:
                        pass
            except Exception as e:
                print(f"❌ Order attempt {retry+1} failed: {e}")
            
            # Get fresh price for next attempt
            if retry < self.config['limit_order_retries'] - 1:
                await asyncio.sleep(2)
                await self.get_market_data()
                price = float(self.price_data['close'].iloc[-1])
        
        return None
    
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
    
    async def check_pending_orders(self):
        try:
            orders = self.exchange.get_open_orders(category="linear", symbol=self.symbol)
            if orders.get('retCode') != 0:
                self.pending_order = None
                return False
            
            order_list = orders['result']['list']
            self.pending_order = order_list[0] if order_list else None
            return bool(self.pending_order)
        except Exception as e:
            print(f"❌ Order check error: {e}")
            return False
    
    async def check_position(self):
        try:
            positions = self.exchange.get_positions(category="linear", symbol=self.symbol)
            if positions.get('retCode') == 0:
                pos_list = positions['result']['list']
                if pos_list:
                    for pos in pos_list:
                        if float(pos.get('size', 0)) > 0:
                            self.position = pos
                            return True
                self.position = None
        except Exception as e:
            print(f"❌ Position check error: {e}")
            self.position = None
        return False
    
    def calculate_pivot_points(self, df):
        if len(df) < 2:
            return None
        
        prev_high = df['high'].iloc[-2]
        prev_low = df['low'].iloc[-2]
        prev_close = df['close'].iloc[-2]
        
        pivot = (prev_high + prev_low + prev_close) / 3
        
        r1 = 2 * pivot - prev_low
        r2 = pivot + (prev_high - prev_low)
        r3 = r1 + (prev_high - prev_low)
        
        s1 = 2 * pivot - prev_high
        s2 = pivot - (prev_high - prev_low)
        s3 = s1 - (prev_high - prev_low)
        
        return {
            'pivot': pivot,
            'r1': r1, 'r2': r2, 'r3': r3,
            's1': s1, 's2': s2, 's3': s3
        }
    
    def calculate_rsi(self, prices):
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=self.config['rsi_period']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.config['rsi_period']).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        # Handle flat market
        return rsi if not pd.isna(rsi.iloc[-1]) and rsi.iloc[-1] != 0 else pd.Series([50] * len(prices))
    
    def calculate_mfi(self, df):
        if len(df) < self.config['mfi_period'] + 1:
            return None
        
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']
        
        positive_flow = []
        negative_flow = []
        
        for i in range(1, len(typical_price)):
            if typical_price.iloc[i] > typical_price.iloc[i-1]:
                positive_flow.append(money_flow.iloc[i])
                negative_flow.append(0)
            elif typical_price.iloc[i] < typical_price.iloc[i-1]:
                positive_flow.append(0)
                negative_flow.append(money_flow.iloc[i])
            else:
                positive_flow.append(0)
                negative_flow.append(0)
        
        positive_mf = pd.Series(positive_flow).rolling(window=self.config['mfi_period']).sum()
        negative_mf = pd.Series(negative_flow).rolling(window=self.config['mfi_period']).sum()
        
        mfi = 100 - (100 / (1 + positive_mf / (negative_mf + 1e-10)))
        return mfi.iloc[-1] if not pd.isna(mfi.iloc[-1]) else 50
    
    def find_nearest_pivot(self, price, pivots):
        if not pivots:
            return None, None
        
        levels = [
            ('pivot', pivots['pivot']),
            ('s1', pivots['s1']), ('s2', pivots['s2']), ('s3', pivots['s3']),
            ('r1', pivots['r1']), ('r2', pivots['r2']), ('r3', pivots['r3'])
        ]
        
        for name, level in levels:
            distance_pct = abs(price - level) / level * 100
            if distance_pct < self.config['pivot_distance_pct']:
                return name, level
        
        return None, None
    
    def is_pivot_failed(self, pivot_name):
        """Check if pivot recently failed"""
        if pivot_name not in self.failed_pivots:
            return False
        
        time_since_fail = datetime.now().timestamp() - self.failed_pivots[pivot_name]
        return time_since_fail < self.pivot_cooldown
    
    def mark_pivot_failed(self, pivot_name):
        """Mark pivot as failed"""
        self.failed_pivots[pivot_name] = datetime.now().timestamp()
        print(f"⚠️ Marked {pivot_name} as failed, cooldown for {self.pivot_cooldown}s")
    
    def generate_signal(self, df):
        if len(df) < 30 or self.position:
            return None
        
        # Check trade cooldown
        if time.time() - self.last_trade_time < self.trade_cooldown:
            return None
        
        current_price = float(df['close'].iloc[-1])
        
        # Update pivot points every hour
        if not self.last_pivot_update or (datetime.now() - self.last_pivot_update).total_seconds() > 3600:
            self.pivot_levels = self.calculate_pivot_points(df)
            self.last_pivot_update = datetime.now()
        
        if not self.pivot_levels:
            return None
        
        pivot_name, pivot_level = self.find_nearest_pivot(current_price, self.pivot_levels)
        if not pivot_name or self.is_pivot_failed(pivot_name):
            return None
        
        rsi = self.calculate_rsi(df['close']).iloc[-1]
        mfi = self.calculate_mfi(df)
        
        if not mfi:
            return None
        
        # Long signal: Near support with STRONG oversold conditions
        if pivot_name in ['s1', 's2', 's3']:
            if rsi < self.config['rsi_oversold'] and mfi < self.config['mfi_oversold']:
                return {
                    'action': 'BUY',
                    'price': current_price,
                    'pivot': pivot_name,
                    'pivot_level': pivot_level,
                    'rsi': rsi,
                    'mfi': mfi
                }
        
        # Short signal: Near resistance with STRONG overbought conditions
        elif pivot_name in ['r1', 'r2', 'r3']:
            if rsi > self.config['rsi_overbought'] and mfi > self.config['mfi_overbought']:
                return {
                    'action': 'SELL',
                    'price': current_price,
                    'pivot': pivot_name,
                    'pivot_level': pivot_level,
                    'rsi': rsi,
                    'mfi': mfi
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
    
    def should_close(self):
        if not self.position or not self.pivot_levels:
            return False, ""
        
        current_price = float(self.price_data['close'].iloc[-1])
        entry_price = float(self.position.get('avgPrice', 0))
        side = self.position.get('side', '')
        
        if entry_price == 0:
            return False, ""
        
        # Check profit/loss
        profit_pct = ((current_price - entry_price) / entry_price * 100 if side == "Buy"
                     else (entry_price - current_price) / entry_price * 100)
        
        if profit_pct >= self.config['net_take_profit']:
            return True, "take_profit"
        
        if profit_pct <= -self.config['net_stop_loss']:
            # Mark pivot as failed on stop loss
            if self.last_entry_pivot:
                self.mark_pivot_failed(self.last_entry_pivot)
            return True, "stop_loss"
        
        # Exit at next resistance/support pivot
        if side == "Buy" and current_price >= self.pivot_levels['r1']:
            return True, "next_pivot_r1"
        elif side == "Sell" and current_price <= self.pivot_levels['s1']:
            return True, "next_pivot_s1"
        
        return False, ""
    
    async def execute_trade(self, signal):
        # Check trade cooldown
        if time.time() - self.last_trade_time < self.trade_cooldown:
            remaining = self.trade_cooldown - (time.time() - self.last_trade_time)
            print(f"⏰ Trade cooldown: wait {remaining:.0f}s")
            return
        
        # Double-check no position exists
        await self.check_position()
        if self.position:
            print("⚠️ Position already exists, skipping trade")
            return
        
        if await self.check_pending_orders():
            print("⚠️ Pending order exists, skipping trade")
            return
        
        await self.update_account_balance()
        
        # Calculate stop loss beyond pivot level
        stop_loss_price = (signal['pivot_level'] * 0.995 if signal['action'] == 'BUY'
                          else signal['pivot_level'] * 1.005)
        
        qty = self.calculate_position_size(signal['price'], stop_loss_price)
        
        if qty < (self.config['min_notional'] / signal['price']):
            print(f"⚠️ Position size too small: {qty}")
            return
        
        # Execute with ZERO slippage
        actual_price = await self.execute_limit_order(
            "Buy" if signal['action'] == 'BUY' else "Sell",
            qty,
            signal['price']
        )
        
        if actual_price:
            self.last_trade_time = time.time()
            self.last_entry_pivot = signal['pivot']
            
            net_tp = (actual_price * (1 + self.config['net_take_profit']/100) if signal['action'] == 'BUY'
                     else actual_price * (1 - self.config['net_take_profit']/100))
            
            self.current_trade_id, _ = self.logger.log_trade_open(
                side=signal['action'],
                expected_price=signal['price'],
                actual_price=actual_price,
                qty=qty,
                stop_loss=stop_loss_price,
                take_profit=net_tp,
                info=f"pivot:{signal['pivot']}_{signal['pivot_level']:.4f}_rsi:{signal['rsi']:.1f}_mfi:{signal['mfi']:.1f}"
            )
            
            position_value = qty * actual_price
            print(f"✅ PIVOT {signal['action']}: {self.format_qty(qty)} @ ${actual_price:.4f}")
            print(f"   📍 Pivot: {signal['pivot']} @ ${signal['pivot_level']:.4f}")
            print(f"   📊 RSI: {signal['rsi']:.1f} | MFI: {signal['mfi']:.1f}")
            print(f"   ✅ ZERO SLIPPAGE with PostOnly")
    
    async def close_position(self, reason):
        if not self.position:
            return
        
        current_price = float(self.price_data['close'].iloc[-1])
        side = "Sell" if self.position.get('side') == "Buy" else "Buy"
        qty = float(self.position['size'])
        
        actual_price = await self.execute_limit_order(side, qty, current_price, is_reduce=True)
        
        if actual_price and self.current_trade_id:
            self.logger.log_trade_close(
                trade_id=self.current_trade_id,
                expected_exit=current_price,
                actual_exit=actual_price,
                reason=reason,
                fees_entry=self.config['maker_fee'],
                fees_exit=self.config['maker_fee']
            )
            self.current_trade_id = None
            
            print(f"✅ Closed: {reason} @ ${actual_price:.4f} | ZERO SLIPPAGE")
            self.position = None
    
    def show_status(self):
        if len(self.price_data) == 0:
            return
        
        current_price = float(self.price_data['close'].iloc[-1])
        
        print(f"\n🎯 Pivot Reversal Bot - {self.symbol}")
        print(f"💰 Price: ${current_price:.4f} | Balance: ${self.account_balance:.2f}")
        
        if self.pivot_levels:
            print(f"📊 Pivots: S1:${self.pivot_levels['s1']:.4f} | P:${self.pivot_levels['pivot']:.4f} | R1:${self.pivot_levels['r1']:.4f}")
            
            rsi = self.calculate_rsi(self.price_data['close']).iloc[-1] if len(self.price_data) > 14 else 50
            mfi = self.calculate_mfi(self.price_data)
            if mfi:
                status = ("🟢" if (rsi < self.config['rsi_oversold'] and mfi < self.config['mfi_oversold']) 
                         else "🔴" if (rsi > self.config['rsi_overbought'] and mfi > self.config['mfi_overbought']) 
                         else "⚪")
                print(f"📈 RSI: {rsi:.1f} | MFI: {mfi:.1f} {status}")
        
        if self.position:
            entry_price = float(self.position.get('avgPrice', 0))
            side = self.position.get('side', '')
            size = self.position.get('size', '0')
            pnl = float(self.position.get('unrealisedPnl', 0))
            
            emoji = "🟢" if side == "Buy" else "🔴"
            print(f"{emoji} {side}: {size} LINK @ ${entry_price:.4f} | PnL: ${pnl:.2f}")
        elif self.pending_order:
            print(f"⏳ Pending order: {self.pending_order.get('side')} @ ${self.pending_order.get('price')}")
        else:
            print("🔍 Waiting for strong pivot reversal signals...")
        
        # Show failed pivots
        if self.failed_pivots:
            active_fails = []
            current_time = datetime.now().timestamp()
            for pivot, fail_time in self.failed_pivots.items():
                remaining = self.pivot_cooldown - (current_time - fail_time)
                if remaining > 0:
                    active_fails.append(f"{pivot}({int(remaining)}s)")
            if active_fails:
                print(f"❌ Failed pivots on cooldown: {', '.join(active_fails)}")
        
        print("-" * 60)
    
    async def run_cycle(self):
        # Emergency stop check
        if self.logger.daily_pnl < -self.logger.max_daily_loss:
            print(f"🔴 EMERGENCY STOP: Daily loss ${abs(self.logger.daily_pnl):.2f} exceeded limit")
            if self.position:
                await self.close_position("emergency_stop")
            return
        
        if not await self.get_market_data():
            return
        
        await self.check_position()
        
        if self.position:
            should_close, reason = self.should_close()
            if should_close:
                await self.close_position(reason)
        else:
            signal = self.generate_signal(self.price_data)
            if signal:
                await self.execute_trade(signal)
        
        self.show_status()
    
    async def run(self):
        if not self.connect():
            print("❌ Failed to connect")
            return
        
        print(f"🎯 Pivot Point Reversal Bot - ZERO SLIPPAGE VERSION")
        print(f"✅ FEATURES:")
        print(f"   • PostOnly Limit Orders = 0 Slippage")
        print(f"   • Maker Rebate: {abs(self.config['maker_fee'])}%")
        print(f"   • Strong momentum: RSI<{self.config['rsi_oversold']} or >{self.config['rsi_overbought']}")
        print(f"   • MFI check: <{self.config['mfi_oversold']} or >{self.config['mfi_overbought']}")
        print(f"   • Failed pivot tracking: {self.pivot_cooldown}s cooldown")
        
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
    bot = PivotReversalBot()
    asyncio.run(bot.run())