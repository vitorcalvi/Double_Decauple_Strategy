#2_FEES_EMA_RSI_BNBUSDT.py

import os
import asyncio
import pandas as pd
import json
import time
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
        
        # Trading limits
        self.daily_pnl = 0
        self.consecutive_losses = 0
        self.max_daily_loss = 50
        
        os.makedirs("logs", exist_ok=True)
        self.log_file = f"logs/2_FEES_EMA_RSI_BNBUSDT.log"
        
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

class EMARSIBot:
    def __init__(self):
        self.symbol = 'BNBUSDT'
        self.demo_mode = os.getenv('DEMO_MODE', 'true').lower() == 'true'
        
        prefix = 'TESTNET_' if self.demo_mode else 'LIVE_'
        self.api_key = os.getenv(f'{prefix}BYBIT_API_KEY')
        self.api_secret = os.getenv(f'{prefix}BYBIT_API_SECRET')
        
        self.exchange = None
        self.position = None
        self.pending_order = None
        self.price_data = pd.DataFrame()
        self.account_balance = 0  # NO FALLBACK - start at 0
        
        # Configuration with ZERO SLIPPAGE settings
        self.config = {
            'ema_fast': 5,
            'ema_slow': 13,
            'rsi_period': 5,
            'rsi_oversold': 35,
            'rsi_overbought': 65,
            'risk_per_trade': 1.0,
            'maker_offset_pct': 0.02,
            'maker_fee': -0.01,
            'net_take_profit': 0.86,
            'net_stop_loss': 0.43,
            'order_timeout': 30,
            'min_notional': 5,
            'limit_order_retries': 3
        }
        
        self.tick_size = 0.01
        self.qty_step = 0.01
        
        # Trade cooldown
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

    async def execute_limit_order(self, side, qty, price, is_reduce=False):
        """Execute limit order with PostOnly for ZERO slippage - FIXED"""
        formatted_qty = self.format_qty(qty)
        
        # Verify position exists before placing reduce-only order
        if is_reduce:
            await self.check_position()
            if not self.position:
                print("⚠️ No position to reduce")
                return None
        
        for retry in range(self.config['limit_order_retries']):
            # Calculate limit price with offset
            limit_price = (price * (1 - self.config['maker_offset_pct']/100) if side == "Buy" 
                        else price * (1 + self.config['maker_offset_pct']/100))
            limit_price = self.format_price(limit_price)
            
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
                        
                elif order.get('retCode') == 110017:
                    # Position doesn't exist error
                    print("ℹ️ Position already closed")
                    return None
                    
            except Exception as e:
                if "110017" in str(e):
                    print("ℹ️ Position already closed")
                    return None
                print(f"❌ Order attempt {retry+1} failed: {e}")
            
            # Get fresh price for next attempt
            if retry < self.config['limit_order_retries'] - 1:
                await asyncio.sleep(2)
                await self.get_market_data()
                price = float(self.price_data['close'].iloc[-1])
        
        return None

    
    async def get_account_balance(self):
        """Get account balance - NO FALLBACK"""
        try:
            result = self.exchange.get_wallet_balance(accountType="UNIFIED", coin="USDT")
            if result.get('retCode') != 0:
                print(f"❌ Failed to get wallet balance: {result.get('retMsg')}")
                return False
                
            balance_list = result.get('result', {}).get('list', [])
            if not balance_list:
                print("❌ No wallet data returned")
                return False
                
            # Try multiple possible balance fields (testnet may use different ones)
            for coin_data in balance_list[0].get('coin', []):
                if coin_data.get('coin') == 'USDT':
                    # Check multiple balance fields in order of preference
                    balance_fields = [
                        'availableToWithdraw',
                        'walletBalance',
                        'equity',
                        'availableBalance',
                        'balance'
                    ]
                    
                    for field in balance_fields:
                        balance_value = coin_data.get(field)
                        if balance_value is not None:
                            try:
                                # Convert to float and validate
                                balance = float(balance_value)
                                if balance >= 0:  # Valid balance
                                    self.account_balance = balance
                                    print(f"✅ Balance from {field}: ${balance:.2f}")
                                    return True
                            except (ValueError, TypeError):
                                continue
                    
                    # If we get here, no valid balance field was found
                    print(f"❌ No valid balance field in USDT data")
                    print(f"   Available fields: {list(coin_data.keys())}")
                    print(f"   Values: {coin_data}")
                    return False
                    
            print("❌ USDT not found in wallet")
            return False
            
        except Exception as e:
            print(f"❌ Balance error: {e}")
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
        """Updated position sizing that checks for valid balance"""
        if self.account_balance <= 0:
            print("❌ No valid account balance for position sizing")
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
    
    async def check_pending_orders(self):
        """Fixed pending orders check without repeated messages"""
        try:
            orders = self.exchange.get_open_orders(category="linear", symbol=self.symbol)
            if orders.get('retCode') != 0:
                self.pending_order = None
                return False
            
            order_list = orders.get('result', {}).get('list', [])
            
            if not order_list:
                # Only clear and notify if we previously had a pending order
                if self.pending_order:
                    print("✓ Pending order cleared")
                    self.pending_order = None
                return False
            
            # Check if order is stale
            if order_list:
                order = order_list[0]
                created_time = int(order.get('createdTime', 0)) / 1000
                current_time = time.time()
                
                if created_time > 0 and (current_time - created_time) > 30:
                    # Cancel stale order
                    try:
                        self.exchange.cancel_order(
                            category="linear",
                            symbol=self.symbol,
                            orderId=order['orderId']
                        )
                        print(f"✓ Cancelled stale order (age: {current_time - created_time:.0f}s)")
                    except:
                        pass
                    self.pending_order = None
                    return False
                
                self.pending_order = order
                return True
                
        except Exception as e:
            print(f"❌ Error checking pending orders: {e}")
            self.pending_order = None
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
        
        # Check trade cooldown
        time_since_last = datetime.now().timestamp() - self.last_trade_time
        if time_since_last < self.trade_cooldown:
            return None
        
        indicators = self.calculate_indicators(df)
        if not indicators:
            return None
        
        price = float(df['close'].iloc[-1])
        
        # SIMPLIFIED: Use RSI extremes as primary signal
        if indicators['rsi'] < 30:  # Strong oversold
            return {'action': 'BUY', 'price': price, 'rsi': indicators['rsi']}
        elif indicators['rsi'] > 70:  # Strong overbought
            return {'action': 'SELL', 'price': price, 'rsi': indicators['rsi']}
        
        # SECONDARY: Use trend confirmation with relaxed RSI
        if indicators['trend'] == 'UP' and indicators['rsi'] < 45:
            return {'action': 'BUY', 'price': price, 'rsi': indicators['rsi']}
        elif indicators['trend'] == 'DOWN' and indicators['rsi'] > 55:
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
        """Enhanced position check that properly syncs with exchange state"""
        try:
            positions = self.exchange.get_positions(category="linear", symbol=self.symbol)
            if positions.get('retCode') != 0:
                print(f"⚠️ Failed to get positions: {positions.get('retMsg')}")
                self.position = None
                return False
                
            pos_list = positions['result']['list']
            
            # Check if we have a position with size > 0
            current_has_position = False
            for pos in pos_list:
                if float(pos.get('size', 0)) > 0:
                    self.position = pos
                    current_has_position = True
                    break
            
            # If no position found but we thought we had one
            if not current_has_position and self.position:
                print("✅ Position closed externally - clearing state")
                self.position = None
                self.pending_order = None
                # Log the external close if we had a trade ID
                if self.current_trade_id:
                    # Log with unknown exit price since it was closed externally
                    print(f"ℹ️ Logging external close for trade {self.current_trade_id}")
                    self.current_trade_id = None
            
            # Clear position reference if no position exists
            if not current_has_position:
                self.position = None
                return False
                
            return True
            
        except Exception as e:
            print(f"❌ Position check error: {e}")
            self.position = None
            self.pending_order = None
            return False

    
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
            if (is_long and indicators['trend'] == 'UP') or (not is_long and indicators['trend'] == 'DOWN'):
                return True, "ema_crossover"
        
        return False, ""


    async def execute_trade(self, signal):
        """Updated execute_trade that requires valid balance"""
        # Check position
        await self.check_position()
        if self.position:
            print("⚠️ Position already exists, skipping trade")
            return
            
        if await self.check_pending_orders():
            print("⚠️ Pending order exists, skipping trade")
            return
        
        # Check cooldown
        time_since_last = datetime.now().timestamp() - self.last_trade_time
        if time_since_last < self.trade_cooldown:
            print(f"⚠️ Trade cooldown active, wait {self.trade_cooldown - time_since_last:.0f}s")
            return
        
        # Update balance - MUST succeed to continue
        if not await self.get_account_balance():
            print("❌ Cannot execute trade - failed to get account balance")
            return
        
        # Verify we have a valid balance
        if self.account_balance <= 0:
            print("❌ Cannot execute trade - account balance is 0 or invalid")
            return
        
        market_price = signal['price']
        is_buy = signal['action'] == 'BUY'
        stop_loss_price = (market_price * (1 - self.config['net_stop_loss']/100) if is_buy 
                        else market_price * (1 + self.config['net_stop_loss']/100))
        
        qty = self.calculate_position_size(market_price, stop_loss_price)
        
        if qty < self.qty_step:
            print(f"⚠️ Position size too small: {qty}")
            return
        
        # Execute with ZERO slippage
        actual_price = await self.execute_limit_order(
            "Buy" if is_buy else "Sell",
            qty,
            market_price
        )
        
        if actual_price:
            self.last_trade_time = datetime.now().timestamp()
            
            take_profit = (actual_price * (1 + self.config['net_take_profit']/100) if is_buy 
                        else actual_price * (1 - self.config['net_take_profit']/100))
            
            self.current_trade_id, _ = self.logger.log_trade_open(
                side=signal['action'],
                expected_price=market_price,
                actual_price=actual_price,
                qty=qty,
                stop_loss=stop_loss_price,
                take_profit=take_profit,
                info=f"RSI:{signal['rsi']:.1f}_Trend:{signal['action']}_Balance:{self.account_balance:.2f}"
            )
            
            print(f"✅ {signal['action']}: {self.format_qty(qty)} @ ${actual_price:.2f}")
            print(f"   📊 RSI: {signal['rsi']:.1f} | Balance: ${self.account_balance:.2f}")
            print(f"   ✅ ZERO SLIPPAGE with PostOnly")

    async def close_position(self, reason):
        """Fixed close_position that verifies position exists before closing"""
        # First, re-check if position actually exists on exchange
        has_position = await self.check_position()
        
        if not has_position or not self.position:
            print(f"ℹ️ No position to close (reason: {reason})")
            # Clean up any stale trade ID
            if self.current_trade_id:
                self.current_trade_id = None
            return
        
        qty = float(self.position.get('size', 0))
        if qty <= 0:
            print("⚠️ Position size is 0, nothing to close")
            self.position = None
            self.current_trade_id = None
            return
            
        side = "Sell" if self.position.get('side') == "Buy" else "Buy"
        current_price = float(self.price_data['close'].iloc[-1])
        
        # Try to close with PostOnly limit order
        actual_price = await self.execute_limit_order(side, qty, current_price, is_reduce=True)
        
        if actual_price:
            # Successfully closed
            if self.current_trade_id:
                self.logger.log_trade_close(
                    trade_id=self.current_trade_id,
                    expected_exit=current_price,
                    actual_exit=actual_price,
                    reason=reason,
                    fees_entry=self.config['maker_fee'],
                    fees_exit=self.config['maker_fee']
                )
                self.current_trade_id = None
                
            print(f"✅ Closed: {reason} @ ${actual_price:.2f} | ZERO SLIPPAGE")
            self.position = None
        else:
            # If limit order fails, check if position was already closed
            await self.check_position()
            if not self.position:
                print(f"ℹ️ Position was already closed externally")
                self.current_trade_id = None

    
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
        """Updated run_cycle with better position state management"""
        if not await self.get_market_data():
            return
        
        # Always check position state first
        await self.check_position()
        await self.check_pending_orders()
        
        if self.position:
            # We have a position, check if we should close it
            should_close, reason = self.should_close()
            if should_close:
                await self.close_position(reason)
        elif not self.pending_order:
            # No position and no pending order, look for signals
            signal = self.generate_signal(self.price_data)
            if signal:
                await self.execute_trade(signal)
        
        self.show_status()

    async def run(self):
        """Updated run method that requires successful balance retrieval"""
        if not self.connect():
            print("❌ Failed to connect to exchange")
            return
        
        # Get balance - exit if it fails
        if not await self.get_account_balance():
            print("❌ Failed to get account balance - cannot continue")
            print("   Please check:")
            print("   1. API keys are correct and have proper permissions")
            print("   2. Unified trading account is enabled")
            print("   3. You have USDT in your account")
            return
        
        if not await self.get_instrument_info():
            print("⚠️ Using default instrument info")
        
        # Only proceed if we have a valid balance
        if self.account_balance <= 0:
            print("❌ Account balance is 0 - cannot trade")
            return
        
        print(f"🔧 EMA + RSI Bot for {self.symbol} - ZERO SLIPPAGE VERSION")
        print(f"✅ FEATURES:")
        print(f"   • PostOnly Limit Orders = 0 Slippage")
        print(f"   • Maker Rebate: {abs(self.config['maker_fee'])}%")
        print(f"   • Strong RSI signals: <{self.config['rsi_oversold']} BUY, >{self.config['rsi_overbought']} SELL")
        print(f"   • Trade cooldown: {self.trade_cooldown}s")
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