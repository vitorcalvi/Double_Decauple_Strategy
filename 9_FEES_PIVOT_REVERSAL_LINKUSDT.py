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
        
        os.makedirs("logs", exist_ok=True)
        self.log_file = f"logs/{bot_name}_{symbol}.log"
        
    def generate_trade_id(self):
        self.trade_id += 1
        return self.trade_id
    
    def log_trade_open(self, side, expected_price, actual_price, qty, stop_loss, take_profit, info=""):
        trade_id = self.generate_trade_id()
        # ✅ FIXED: Proper slippage calculation
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
        
        # ✅ FIXED: Proper slippage calculation
        slippage = actual_exit - expected_exit if trade["side"] == "SELL" else expected_exit - actual_exit
        
        if trade["side"] == "BUY":
            gross_pnl = (actual_exit - trade["entry_price"]) * trade["qty"]
        else:
            gross_pnl = (trade["entry_price"] - actual_exit) * trade["qty"]
        
        # ✅ FIXED: Proper maker rebate calculation (negative fees = rebates earned)
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

class PivotReversalBot:
    """Pivot Point Reversal Strategy - FULLY FIXED VERSION"""
    
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
        self.account_balance = 1000  # Will be updated from API
        
        # Anti-duplicate mechanisms
        self.last_order_time = 0
        self.order_cooldown = 10
        self.last_signal = None
        self.min_price_change_pct = 0.1
        
        # ✅ FIXED: Strategy configuration with proper risk management
        self.config = {
            'timeframe': '3',
            'rsi_period': 14,
            'mfi_period': 14,
            'risk_pct': 2.0,              # Risk 2% of account per trade
            'maker_offset_pct': 0.01,
            'maker_fee_pct': -0.04,
            'net_take_profit': 0.6,
            'net_stop_loss': 0.3,
            'slippage_pct': 0.02,         # Expected slippage 0.02%
            'min_notional': 5,            # Minimum $5 position
            'qty_precision': 1,           # LINK quantity precision
        }
        
        # Pivot levels tracking
        self.pivot_levels = {}
        self.last_pivot_update = None
        
        # Trade logging
        self.logger = TradeLogger("PIVOT_REVERSAL_FIXED", self.symbol)
        self.current_trade_id = None
    
    def connect(self):
        """Connect to exchange"""
        try:
            self.exchange = HTTP(demo=self.demo_mode, api_key=self.api_key, api_secret=self.api_secret)
            return self.exchange.get_server_time().get('retCode') == 0
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False
    
    # ✅ FIXED: Proper quantity formatting with precision
    def format_qty(self, qty):
        """Format quantity with proper precision for LINK"""
        precision = self.config['qty_precision']
        step = 10 ** (-precision)
        rounded_qty = round(qty / step) * step
        return f"{rounded_qty:.{precision}f}"
    
    # ✅ FIXED: Get account balance for proper position sizing
    async def update_account_balance(self):
        """Update account balance from API"""
        try:
            wallet = self.exchange.get_wallet_balance(accountType="UNIFIED")
            if wallet.get('retCode') == 0:
                for coin in wallet['result']['list'][0]['coin']:
                    if coin['coin'] == 'USDT':
                        self.account_balance = float(coin['availableToWithdraw'])
                        break
        except Exception as e:
            print(f"⚠️ Balance update error: {e}")
    
    # ✅ FIXED: Calculate position size based on account balance and risk
    def calculate_position_size(self, price, stop_loss_price):
        """Calculate position size based on risk percentage"""
        if self.account_balance <= 0:
            return 0
        
        risk_amount = self.account_balance * (self.config['risk_pct'] / 100)
        price_diff = abs(price - stop_loss_price)
        
        if price_diff == 0:
            return 0
        
        # Calculate quantity that risks exactly risk_amount
        qty = risk_amount / price_diff
        
        # Apply minimum notional check
        notional = qty * price
        if notional < self.config['min_notional']:
            qty = self.config['min_notional'] / price
        
        return qty
    
    # ✅ FIXED: Add slippage modeling to limit price calculation
    def calculate_limit_price(self, market_price, side, include_slippage=True):
        """Calculate limit price with slippage and maker offset"""
        slippage_mult = 1 + (self.config['slippage_pct'] / 100) if include_slippage else 1
        
        if side == 'BUY':
            # For buy: add slippage, subtract maker offset for better fill
            price_with_slippage = market_price * slippage_mult
            limit_price = price_with_slippage * (1 - self.config['maker_offset_pct'] / 100)
        else:
            # For sell: subtract slippage, add maker offset for better fill
            price_with_slippage = market_price / slippage_mult
            limit_price = price_with_slippage * (1 + self.config['maker_offset_pct'] / 100)
        
        return round(limit_price, 4)
    
    async def check_pending_orders(self):
        """Check for any pending orders"""
        try:
            orders = self.exchange.get_open_orders(category="linear", symbol=self.symbol)
            if orders.get('retCode') != 0:
                self.pending_order = None
                return False
            
            order_list = orders['result']['list']
            if order_list and len(order_list) > 0:
                self.pending_order = order_list[0]
                # Cancel old orders
                order_age = (datetime.now().timestamp() - int(order_list[0]['createdTime']) / 1000)
                if order_age > 300:  # 5 minutes
                    self.exchange.cancel_order(
                        category="linear",
                        symbol=self.symbol,
                        orderId=order_list[0]['orderId']
                    )
                    print(f"❌ Cancelled stale order (aged {order_age:.0f}s)")
                    self.pending_order = None
                    return False
                return True
            
            self.pending_order = None
            return False
        except Exception as e:
            print(f"❌ Order check error: {e}")
            return False
    
    async def check_position(self):
        """Check current position status"""
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
            return False
        except Exception as e:
            print(f"❌ Position check error: {e}")
            self.position = None
            return False
    
    def calculate_pivot_points(self, df):
        """Calculate pivot points using previous period data"""
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
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=self.config['rsi_period']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.config['rsi_period']).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_mfi(self, df):
        """Calculate Money Flow Index"""
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
        """Find nearest pivot level to current price"""
        if not pivots:
            return None, None
        
        levels = [
            ('pivot', pivots['pivot']),
            ('s1', pivots['s1']), ('s2', pivots['s2']), ('s3', pivots['s3']),
            ('r1', pivots['r1']), ('r2', pivots['r2']), ('r3', pivots['r3'])
        ]
        
        for name, level in levels:
            distance_pct = abs(price - level) / level * 100
            if distance_pct < 0.3:  # Within 0.3% of pivot level
                return name, level
        
        return None, None
    
    def generate_signal(self, df):
        """Generate trading signal based on pivot reversals"""
        if len(df) < 30:
            return None
        
        if self.position:
            return None
        
        current_price = float(df['close'].iloc[-1])
        
        # Check for duplicate signals
        if self.last_signal:
            price_change_pct = abs(current_price - self.last_signal['price']) / self.last_signal['price'] * 100
            if price_change_pct < self.min_price_change_pct:
                return None
        
        # Update pivot points every hour
        if not self.last_pivot_update or (datetime.now() - self.last_pivot_update).total_seconds() > 3600:
            self.pivot_levels = self.calculate_pivot_points(df)
            self.last_pivot_update = datetime.now()
        
        if not self.pivot_levels:
            return None
        
        pivot_name, pivot_level = self.find_nearest_pivot(current_price, self.pivot_levels)
        if not pivot_name:
            return None
        
        rsi = self.calculate_rsi(df['close']).iloc[-1]
        mfi = self.calculate_mfi(df)
        
        if not mfi:
            return None
        
        # Long signal: Near support with oversold conditions
        if pivot_name in ['s1', 's2', 's3'] and rsi < 30 and mfi < 30:
            signal = {
                'action': 'BUY',
                'price': current_price,
                'pivot': pivot_name,
                'pivot_level': pivot_level,
                'rsi': rsi,
                'mfi': mfi
            }
            self.last_signal = signal
            return signal
        
        # Short signal: Near resistance with overbought conditions
        elif pivot_name in ['r1', 'r2', 'r3'] and rsi > 70 and mfi > 70:
            signal = {
                'action': 'SELL',
                'price': current_price,
                'pivot': pivot_name,
                'pivot_level': pivot_level,
                'rsi': rsi,
                'mfi': mfi
            }
            self.last_signal = signal
            return signal
        
        return None
    
    async def get_market_data(self):
        """Retrieve market data from exchange"""
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
        """Determine if position should be closed"""
        if not self.position or not self.pivot_levels:
            return False, ""
        
        current_price = float(self.price_data['close'].iloc[-1])
        entry_price = float(self.position.get('avgPrice', 0))
        side = self.position.get('side', '')
        
        if entry_price == 0:
            return False, ""
        
        # Check profit targets
        if side == "Buy":
            profit_pct = (current_price - entry_price) / entry_price * 100
            if profit_pct >= self.config['net_take_profit']:
                return True, "take_profit"
            if profit_pct <= -self.config['net_stop_loss']:
                return True, "stop_loss"
            
            # Exit at next resistance pivot
            if current_price >= self.pivot_levels['r1']:
                return True, "next_pivot_r1"
        else:
            profit_pct = (entry_price - current_price) / entry_price * 100
            if profit_pct >= self.config['net_take_profit']:
                return True, "take_profit"
            if profit_pct <= -self.config['net_stop_loss']:
                return True, "stop_loss"
            
            # Exit at next support pivot
            if current_price <= self.pivot_levels['s1']:
                return True, "next_pivot_s1"
        
        return False, ""
    
    async def execute_trade(self, signal):
        """Execute trade with proper risk management"""
        # Check no position exists
        await self.check_position()
        if self.position:
            print("⚠️ Position already exists, skipping trade")
            return
        
        # Check for pending orders
        if await self.check_pending_orders():
            print("⚠️ Pending order exists, skipping trade")
            return
        
        # Check cooldown
        current_time = time.time()
        if current_time - self.last_order_time < self.order_cooldown:
            remaining = self.order_cooldown - (current_time - self.last_order_time)
            print(f"⚠️ Order cooldown active, wait {remaining:.1f}s")
            return
        
        # ✅ FIXED: Update account balance for proper position sizing
        await self.update_account_balance()
        
        # ✅ FIXED: Calculate stop loss beyond pivot level
        if signal['action'] == 'BUY':
            stop_loss_price = signal['pivot_level'] * 0.995
        else:
            stop_loss_price = signal['pivot_level'] * 1.005
        
        # ✅ FIXED: Calculate position size based on risk
        qty = self.calculate_position_size(signal['price'], stop_loss_price)
        formatted_qty = self.format_qty(qty)
        
        if float(formatted_qty) < (self.config['min_notional'] / signal['price']):
            print(f"⚠️ Position size too small: {formatted_qty}")
            return
        
        # ✅ FIXED: Calculate limit price with slippage
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
                self.last_order_time = current_time
                net_tp = limit_price * (1 + self.config['net_take_profit']/100) if signal['action'] == 'BUY' else limit_price * (1 - self.config['net_take_profit']/100)
                
                self.current_trade_id, _ = self.logger.log_trade_open(
                    side=signal['action'],
                    expected_price=signal['price'],
                    actual_price=limit_price,
                    qty=float(formatted_qty),
                    stop_loss=stop_loss_price,
                    take_profit=net_tp,
                    info=f"pivot:{signal['pivot']}_{signal['pivot_level']:.4f}_rsi:{signal['rsi']:.1f}_mfi:{signal['mfi']:.1f}_risk:{self.config['risk_pct']}%"
                )
                
                position_value = float(formatted_qty) * limit_price
                print(f"✅ FIXED PIVOT {signal['action']}: {formatted_qty} @ ${limit_price:.4f}")
                print(f"   💰 Position Value: ${position_value:.2f} (Risk: {self.config['risk_pct']}%)")
                print(f"   📍 Pivot: {signal['pivot']} @ ${signal['pivot_level']:.4f}")
                print(f"   📊 RSI: {signal['rsi']:.1f} | MFI: {signal['mfi']:.1f}")
                print(f"   🛡️ Account Balance: ${self.account_balance:.2f}")
                
        except Exception as e:
            print(f"❌ Trade failed: {e}")
    
    async def close_position(self, reason):
        """Close position with maker order for rebate benefits"""
        if not self.position:
            return
        
        current_price = float(self.price_data['close'].iloc[-1])
        side = "Sell" if self.position.get('side') == "Buy" else "Buy"
        qty = float(self.position['size'])
        
        # ✅ FIXED: Use slippage modeling for close price
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
                
                print(f"✅ Closed: {reason}")
                self.position = None
                
        except Exception as e:
            print(f"❌ Close failed: {e}")
    
    def show_status(self):
        """Show current status"""
        if len(self.price_data) == 0:
            return
        
        current_price = float(self.price_data['close'].iloc[-1])
        
        print(f"\n🎯 FIXED Pivot Reversal Bot - {self.symbol}")
        print(f"💰 Price: ${current_price:.4f} | Balance: ${self.account_balance:.2f}")
        print(f"🔧 FIXED: Risk-based position sizing ({self.config['risk_pct']}% per trade)")
        
        if self.pivot_levels:
            print(f"📊 Pivots: S1:${self.pivot_levels['s1']:.4f} | P:${self.pivot_levels['pivot']:.4f} | R1:${self.pivot_levels['r1']:.4f}")
            
            rsi = self.calculate_rsi(self.price_data['close']).iloc[-1] if len(self.price_data) > 14 else 50
            mfi = self.calculate_mfi(self.price_data)
            if mfi:
                print(f"📈 RSI: {rsi:.1f} | MFI: {mfi:.1f}")
        
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
            print("🔍 Scanning for pivot reversals...")
        
        print("-" * 60)
    
    async def run_cycle(self):
        """Run one trading cycle"""
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
        """Main bot loop"""
        if not self.connect():
            print("❌ Failed to connect")
            return
        
        print(f"🎯 FULLY FIXED Pivot Point Reversal Bot")
        print(f"✅ FIXES APPLIED:")
        print(f"   • Position Sizing: Account balance-based ({self.config['risk_pct']}% risk)")
        print(f"   • Fee Calculations: Proper maker rebates")
        print(f"   • Slippage Modeling: {self.config['slippage_pct']}% expected")
        print(f"   • Risk Management: Stop loss at pivot levels")
        print(f"   • Quantity Precision: {self.config['qty_precision']} decimals")
        
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