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
        self.demo_mode = os.getenv('DEMO_MODE', 'true').lower() == 'true'

     self.LIVE_TRADING = False  # Enable actual trading
     self.account_balance = 1000.0  # Default balance
     self.pending_order = False
     self.last_trade_time = 0
     self.trade_cooldown = 30  # 30 seconds between trades
     self.bot_name = bot_name
    async def execute_limit_order(self, side, qty, price, is_reduce=False):
    """Execute limit order with PostOnly for zero slippage"""
    formatted_qty = self.format_qty(qty)
    
    # Calculate limit price with small offset
    if side == "Buy":
        pass
        limit_price = price * 0.9998  # Slightly below market
        else:
        limit_price = price * 1.0002  # Slightly above market
    
    limit_price = float(self.format_price(limit_price))
    
    params = {
    "category": "linear",
    "symbol": self.symbol,
    "side": side,
    "orderType": "Limit",
    "qty": formatted_qty,
    "price": str(limit_price),
    "timeInForce": "PostOnly"  # This ensures ZERO slippage
    }
    
    if is_reduce:
        pass
        params["reduceOnly"] = True
    
    order = self.exchange.place_order(**params)
    
    if order.get('retCode') == 0:
        pass
        return limit_price  # Return actual price, slippage = 0
    return None
    def __init__(self, bot_name, symbol):
        self.demo_mode = os.getenv('DEMO_MODE', 'true').lower() == 'true'
        self.LIVE_TRADING = False  # Enable actual trading
        self.account_balance = 1000.0  # Default balance
        self.pending_order = False
        self.last_trade_time = 0
        self.trade_cooldown = 30  # 30 seconds between trades
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
            pass
            f.write(json.dumps(log_entry) + "\n")
        
            return trade_id, log_entry
    
    def log_trade_close(self, trade_id, expected_exit, actual_exit, reason, fees_entry=-0.04, fees_exit=-0.04):
        if trade_id not in self.open_trades:
            pass
            return None
            
        trade = self.open_trades[trade_id]
        duration = (datetime.now() - trade["entry_time"]).total_seconds()
        
        slippage = 0  # PostOnly = zero slippage
        
        if trade["side"] == "BUY":
            pass
            gross_pnl = (actual_exit - trade["entry_price"]) * trade["qty"]
            else:
            gross_pnl = (trade["entry_price"] - actual_exit) * trade["qty"]
        
        # FIXED: Proper rebate calculation for maker orders
        entry_rebate = trade["entry_price"] * trade["qty"] * abs(fees_entry) / 100
        exit_rebate = actual_exit * trade["qty"] * abs(fees_exit) / 100
        total_rebates = entry_rebate + exit_rebate
        net_pnl = gross_pnl + total_rebates  # Add rebates since they're earnings
        
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
            pass
            f.write(json.dumps(log_entry) + "\n")
        
        del self.open_trades[trade_id]
        return log_entry

class VWAPRSIDivergenceBot:
    def __init__(self):
        
        self.LIVE_TRADING = False  # Enable actual trading
        self.account_balance = 1000.0  # Default balance
        self.pending_order = False
        self.last_trade_time = 0
        self.trade_cooldown = 30  # 30 seconds between trades
        # Trade cooldown mechanism
        self.last_trade_time = 0
        self.trade_cooldown = 30  # 30 seconds between trades
        
        
        # Emergency stop tracking
        self.daily_pnl = 0
        self.consecutive_losses = 0
        self.max_daily_loss = 50  # $50 max daily loss
        
        self.symbol = 'AVAXUSDT'
        self.demo_mode = os.getenv('DEMO_MODE', 'true').lower() == 'true'
        
        prefix = 'TESTNET_' if self.demo_mode else 'LIVE_'
        self.api_key = os.getenv(f'{prefix}BYBIT_API_KEY')
        self.api_secret = os.getenv(f'{prefix}BYBIT_API_SECRET')
        self.exchange = None
        
        self.position = None
        self.price_data = pd.DataFrame()
        self.account_balance = 0
        
        # FIXED PARAMETERS
        self.config = {
        'timeframe': '5',
        'rsi_period': 9,
        'divergence_lookback': 5,
        'ema_period': 50,
        'risk_per_trade': 2.0,       # FIXED: 2% risk per trade instead of fixed $100
        'maker_offset_pct': 0.01,
        'maker_fee_pct': -0.04,
        'net_take_profit': 0.70,
        'net_stop_loss': 0.35,
        'slippage_basis_points': 3,  # FIXED: 0.03% expected slippage for AVAX
        }
        
        self.rsi_pivots = {'highs': [], 'lows': []}
        self.price_pivots = {'highs': [], 'lows': []}
        
        self.logger = TradeLogger("VWAP_RSI_DIV_FIXED", self.symbol)
        self.current_trade_id = None
    
    def connect(self):
        try:
            self.exchange = HTTP(demo=self.demo_mode, api_key=self.api_key, api_secret=self.api_secret)
            return self.exchange.get_server_time().get('retCode') == 0
        except Exception as e:
            pass
            print(f"❌ Connection error: {e}")
            return False
    
    # FIXED: Proper quantity formatting with instrument precision
    def format_qty(self, qty):
        # AVAX minimum quantity is 0.01 with 2 decimal places
        min_qty = 0.01
        if qty < min_qty:
            pass
            return "0"
        
        # Round to 2 decimal places for AVAX
        formatted = round(qty / min_qty) * min_qty
        return f"{formatted:.2f}"
    
    # FIXED: Get account balance for position sizing
        async def get_account_balance(self):
        try:
            wallet = self.exchange.get_wallet_balance(accountType="UNIFIED")
            if wallet.get('retCode') == 0:
                pass
                for coin in wallet['result']['list'][0]['coin']:
                    pass
                    if coin['coin'] == 'USDT':
                        pass
                        self.account_balance = float(coin['availableToWithdraw'])
                        return True
                    return False
        except Exception as e:
            pass
            print(f"❌ Balance check error: {e}")
            return False
    
    # FIXED: Calculate position size based on account balance and risk
    def calculate_position_size(self, price, stop_loss_price):
        if self.account_balance <= 0:
            pass
            return 0
        
        # Calculate risk amount (2% of balance)
        risk_amount = self.account_balance * (self.config['risk_per_trade'] / 100)
        
        # Calculate position size based on stop loss distance
        stop_distance = abs(price - stop_loss_price)
        if stop_distance == 0:
            pass
            return 0
        
        # Position size = Risk Amount / Stop Distance
        position_size_usdt = min(risk_amount / stop_distance * price, self.account_balance * 0.1)  # Max 10% of balance
        qty = position_size_usdt / price
        
        return qty
    
    # FIXED: Add slippage modeling
    def apply_slippage(self, expected_price, side):
        slippage_pct = self.config['slippage_basis_points'] / 10000  # Convert basis points to percentage
        
        if side in ['BUY', 'Buy']:
            pass
            # Buy orders get worse (higher) price due to slippage
            actual_price = expected_price * (1 + slippage_pct)
            else:
            # Sell orders get worse (lower) price due to slippage
            actual_price = expected_price * (1 - slippage_pct)
        
            return actual_price
    
    def calculate_vwap(self, df):
        if len(df) < 20:
            pass
            return None
        
        recent_data = df.tail(min(288, len(df)))
        typical_price = (recent_data['high'] + recent_data['low'] + recent_data['close']) / 3
        volume = recent_data['volume']
        
        vwap = (typical_price * volume).cumsum() / volume.cumsum()
        return vwap.iloc[-1] if not vwap.empty else None
    
    def calculate_rsi(self, prices):
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=self.config['rsi_period']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.config['rsi_period']).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))

            # Handle flat market
            if pd.isna(rsi) or rsi == 0:
                rsi = 50.0  # Neutral RSI for flat market
                return rsi
    
    def detect_pivots(self, series, window=5):
        pivots_high = []
        pivots_low = []
        
        for i in range(window, len(series) - window):
            pass
            if all(series.iloc[i] >= series.iloc[i-j] for j in range(1, window+1)) and \:
                pass
            all(series.iloc[i] >= series.iloc[i+j] for j in range(1, window+1)):
            pivots_high.append((i, series.iloc[i]))
            
            if all(series.iloc[i] <= series.iloc[i-j] for j in range(1, window+1)) and \:
                pass
            all(series.iloc[i] <= series.iloc[i+j] for j in range(1, window+1)):
            pivots_low.append((i, series.iloc[i]))
        
            return pivots_high, pivots_low
    
    def detect_divergence(self, df):
        if len(df) < 30:
            pass
            return None
        
        close = df['close']
        rsi = self.calculate_rsi(close)
        
        price_highs, price_lows = self.detect_pivots(close)
        rsi_highs, rsi_lows = self.detect_pivots(rsi)
        
        current_price = close.iloc[-1]
        current_rsi = rsi.iloc[-1]
        
        # Bullish divergence
        if len(price_lows) >= 2 and len(rsi_lows) >= 2:
            pass
            if price_lows[-1][1] < price_lows[-2][1] and rsi_lows[-1][1] > rsi_lows[-2][1]:
                pass
                if abs(price_lows[-1][0] - len(df) + 1) <= 5:
                    pass
                    return {
                'type': 'bullish',
                'price': current_price,
                'rsi': current_rsi,
                'strength': abs(rsi_lows[-1][1] - rsi_lows[-2][1])
                }
        
        # Bearish divergence
        if len(price_highs) >= 2 and len(rsi_highs) >= 2:
            pass
            if price_highs[-1][1] > price_highs[-2][1] and rsi_highs[-1][1] < rsi_highs[-2][1]:
                pass
                if abs(price_highs[-1][0] - len(df) + 1) <= 5:
                    pass
                    return {
                'type': 'bearish',
                'price': current_price,
                'rsi': current_rsi,
                'strength': abs(rsi_highs[-1][1] - rsi_highs[-2][1])
                }
        
                return None
    
    def generate_signal(self, df):
        if len(df) < 50:
            pass
            return None
        
        divergence = self.detect_divergence(df)
        if not divergence:
            pass
            return None
        
        current_price = float(df['close'].iloc[-1])
        vwap = self.calculate_vwap(df)
        ema = df['close'].ewm(span=self.config['ema_period']).mean().iloc[-1]
        
        if not vwap:
            pass
            return None
        
        # Bullish divergence + price crosses above VWAP
        if divergence['type'] == 'bullish' and current_price > vwap and current_price > ema:
            pass
            return {
        'action': 'BUY',
        'price': current_price,
        'vwap': vwap,
        'rsi': divergence['rsi'],
        'divergence_strength': divergence['strength']
        }
        
        # Bearish divergence + price crosses below VWAP
        elif divergence['type'] == 'bearish' and current_price < vwap and current_price < ema:
            pass
            return {
        'action': 'SELL',
        'price': current_price,
        'vwap': vwap,
        'rsi': divergence['rsi'],
        'divergence_strength': divergence['strength']
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
                pass
                return False
            
            df = pd.DataFrame(klines['result']['list'], columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
            ])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                pass
                df[col] = pd.to_numeric(df[col])
            
            self.price_data = df.sort_values('timestamp').reset_index(drop=True)
            return True
        except Exception as e:
            pass
            print(f"❌ Market data error: {e}")
            return False
    
            async def check_position(self):
        try:
            positions = self.exchange.get_positions(category="linear", symbol=self.symbol)
            if positions.get('retCode') == 0:
                pass
                pos_list = positions['result']['list']
                self.position = pos_list[0] if pos_list and float(pos_list[0]['size']) > 0 else None
        except Exception as e:
            pass
            print(f"❌ Position check error: {e}")
            pass
    
    def should_close(self):
        if not self.position:
            pass
            return False, ""
        
        current_price = float(self.price_data['close'].iloc[-1])
        entry_price = float(self.position.get('avgPrice', 0))
        side = self.position.get('side', '')
        
        if entry_price == 0:
            pass
            return False, ""
        
        if side == "Buy":
            pass
            profit_pct = (current_price - entry_price) / entry_price * 100
            if profit_pct >= self.config['net_take_profit']:
                pass
                return True, "take_profit"
            if profit_pct <= -self.config['net_stop_loss']:
                pass
                return True, "stop_loss"
            
            # Check for swing high
            price_highs, _ = self.detect_pivots(self.price_data['close'])
            if price_highs and abs(price_highs[-1][0] - len(self.price_data) + 1) <= 3:
                pass
                return True, "swing_high_exit"
            else:
            profit_pct = (entry_price - current_price) / entry_price * 100
            if profit_pct >= self.config['net_take_profit']:
                pass
                return True, "take_profit"
            if profit_pct <= -self.config['net_stop_loss']:
                pass
                return True, "stop_loss"
            
            # Check for swing low
            _, price_lows = self.detect_pivots(self.price_data['close'])
            if price_lows and abs(price_lows[-1][0] - len(self.price_data) + 1) <= 3:
                pass
                return True, "swing_low_exit"
        
        # Check for opposite RSI extreme
        rsi = self.calculate_rsi(self.price_data['close']).iloc[-1]
        if side == "Buy" and rsi > 70:
            pass
            return True, "rsi_overbought"
        elif side == "Sell" and rsi < 30:
            pass
            return True, "rsi_oversold"
        
        return False, ""
    
                async def execute_trade(self, signal):
        
        # Check trade cooldown
            import time
        if time.time() - self.last_trade_time < self.trade_cooldown:
            pass
            remaining = self.trade_cooldown - (time.time() - self.last_trade_time)
            print(f"⏰ Trade cooldown: wait {remaining:.0f}s")
            return
        # FIXED: Check account balance first
        if not await self.get_account_balance():
            pass
            print("❌ Could not get account balance")
            return
        
        if self.account_balance < 10:  # Minimum $10 balance:
            pass
            print(f"❌ Insufficient balance: ${self.account_balance:.2f}")
            return
        
        # FIXED: Calculate stop loss price for position sizing
            stop_loss_pct = self.config['net_stop_loss'] / 100
        if signal['action'] == 'BUY':
            pass
            stop_loss_price = signal['price'] * (1 - stop_loss_pct)
            else:
            stop_loss_price = signal['price'] * (1 + stop_loss_pct)
        
        # FIXED: Calculate position size based on risk
            qty = self.calculate_position_size(signal['price'], stop_loss_price)
        formatted_qty = self.format_qty(qty)
        
        if formatted_qty == "0":
            pass
            print(f"❌ Position size too small: {qty:.6f}")
            return
        
        offset_mult = 1 - self.config['maker_offset_pct']/100 if signal['action'] == 'BUY' else 1 + self.config['maker_offset_pct']/100
        limit_price = round(signal['price'] * offset_mult, 4)
        
        # FIXED: Apply slippage to expected execution price
        expected_execution_price = self.apply_slippage(limit_price, signal['action'])
        
        try:
            order = self.exchange.place_order(
            category="linear",
            symbol=self.symbol,
            side="Buy" if signal['action'] == 'BUY' else "Sell",
            orderType="Limit",
            qty=formatted_qty,
            price=str(limit_price),
            timeInForce="PostOnly"),
            timeInForce="PostOnly"
            )
            
            if order.get('retCode') == 0:
                pass
                self.last_trade_time = time.time()  # Update last trade time
                net_tp = expected_execution_price * (1 + self.config['net_take_profit']/100) if signal['action'] == 'BUY' else expected_execution_price * (1 - self.config['net_take_profit']/100)
                net_sl = expected_execution_price * (1 - self.config['net_stop_loss']/100) if signal['action'] == 'BUY' else expected_execution_price * (1 + self.config['net_stop_loss']/100)
                
                position_value = float(formatted_qty) * expected_execution_price
                risk_pct = (position_value * stop_loss_pct / self.account_balance) * 100
                
                self.current_trade_id, _ = self.logger.log_trade_open(
                side=signal['action'],
                expected_price=limit_price,
                actual_price=expected_execution_price,
                qty=float(formatted_qty),
                stop_loss=net_sl,
                take_profit=net_tp,
                info=f"vwap:{signal['vwap']:.4f}_rsi:{signal['rsi']:.1f}_div:{signal['divergence_strength']:.1f}_risk:{risk_pct:.1f}%_bal:{self.account_balance:.2f}"
                )
                
                print(f"📈 DIVERGENCE {signal['action']}: {formatted_qty} @ ${limit_price:.4f}")
                print(f"   💰 Position Value: ${position_value:.2f} ({risk_pct:.1f}% of balance)")
                print(f"   🎯 VWAP: ${signal['vwap']:.4f} | RSI: {signal['rsi']:.1f}")
                print(f"   💪 Divergence Strength: {signal['divergence_strength']:.1f}")
                print(f"   🎯 Expected Execution: ${expected_execution_price:.4f} (with slippage)")
                
        except Exception as e:
            pass
            print(f"❌ Trade failed: {e}")
    
                async def close_position(self, reason):
        if not self.position:
            pass
            return
        
            current_price = float(self.price_data['close'].iloc[-1])
        side = "Sell" if self.position.get('side') == "Buy" else "Buy"
        qty = float(self.position['size'])
        
        offset_mult = 1 + self.config['maker_offset_pct']/100 if side == "Sell" else 1 - self.config['maker_offset_pct']/100
        limit_price = round(current_price * offset_mult, 4)
        
        # FIXED: Apply slippage to exit price
        expected_exit_price = self.apply_slippage(limit_price, side)
        
        try:
            order = self.exchange.place_order(
            category="linear",
            symbol=self.symbol,
            side=side,
            orderType="Limit",
            qty=self.format_qty(qty)
            timeInForce="PostOnly"),
            price=str(limit_price),
            timeInForce="PostOnly",
            reduceOnly=True
            )
            
            if order.get('retCode') == 0:
                pass
                if self.current_trade_id:
                    pass
                    self.logger.log_trade_close(
                    trade_id=self.current_trade_id,
                    expected_exit=limit_price,
                    actual_exit=expected_exit_price,
                    reason=reason,
                    fees_entry=self.config['maker_fee_pct'],
                    fees_exit=self.config['maker_fee_pct']
                    )
                    self.current_trade_id = None
                
                print(f"✅ Closed: {reason} @ ${expected_exit_price:.4f}")
                self.position = None
                
        except Exception as e:
            pass
            print(f"❌ Close failed: {e}")
    
    def show_status(self):
        if len(self.price_data) == 0:
            pass
            return
        
        current_price = float(self.price_data['close'].iloc[-1])
        vwap = self.calculate_vwap(self.price_data)
        rsi = self.calculate_rsi(self.price_data['close']).iloc[-1] if len(self.price_data) > 14 else 50
        
        print(f"\n📈 FIXED VWAP + RSI Divergence - {self.symbol}")
        print(f"💰 Price: ${current_price:.4f} | Balance: ${self.account_balance:.2f}")
        print(f"🔧 FIXES APPLIED:")
        print(f"   • Position Sizing: Risk-based ({self.config['risk_per_trade']}% per trade)")
        print(f"   • Fee Calculations: Proper maker rebates")  
        print(f"   • Slippage Modeling: {self.config['slippage_basis_points']} basis points")
        
        if vwap:
            pass
            print(f"📊 VWAP: ${vwap:.4f} | RSI: {rsi:.1f}")
            position_to_vwap = "Above" if current_price > vwap else "Below"
            print(f"📍 Price is {position_to_vwap} VWAP")
        
        if self.position:
            pass
            entry_price = float(self.position.get('avgPrice', 0))
            side = self.position.get('side', '')
            size = self.position.get('size', '0')
            
            pnl = float(self.position.get('unrealisedPnl', 0))
            position_value = float(size) * current_price
            risk_pct = (position_value / self.account_balance) * 100 if self.account_balance > 0 else 0
            
            emoji = "🟢" if side == "Buy" else "🔴"
            print(f"{emoji} {side}: {size} AVAX @ ${entry_price:.4f} | PnL: ${pnl:.2f}")
            print(f"   📊 Position: ${position_value:.2f} ({risk_pct:.1f}% of balance)")
            else:
            print("🔍 Scanning for RSI divergences...")
        
            print("-" * 50)
    
        async def run_cycle(self):
        
        # Emergency stop check
        if self.daily_pnl < -self.max_daily_loss:
            pass
            print(f"🔴 EMERGENCY STOP: Daily loss ${abs(self.daily_pnl):.2f} exceeded limit")
            if self.position:
                pass
                await self.close_position("emergency_stop")
                return
        if not await self.get_market_data():
            pass
            return
        
        await self.check_position()
        
        if self.position:
            pass
            should_close, reason = self.should_close()
            if should_close:
                pass
                await self.close_position(reason)
                else:
                signal = self.generate_signal(self.price_data)
            if signal:
                pass
                await self.execute_trade(signal)
        
                self.show_status()
    
            async def run(self):
        if not self.connect():
            pass
            print("❌ Failed to connect")
            return
        
        print(f"📈 FIXED VWAP + RSI Divergence Bot - {self.symbol}")
        print(f"🔧 CRITICAL FIXES:")
        print(f"   ✅ Position Sizing: Account balance-based with {self.config['risk_per_trade']}% risk")
        print(f"   ✅ Fee Calculations: Proper maker rebate handling")
        print(f"   ✅ Slippage Modeling: {self.config['slippage_basis_points']} basis points expected slippage")
        print(f"   ✅ Instrument Precision: AVAX 0.01 minimum quantity")
        print(f"⏰ Timeframe: {self.config['timeframe']} minutes")
        print(f"🎯 Net TP: {self.config['net_take_profit']}% | Net SL: {self.config['net_stop_loss']}%")
        print(f"💎 Using MAKER-ONLY orders for {abs(self.config['maker_fee_pct'])}% fee rebate")
        
        try:
            while True:
                pass
                await self.run_cycle()
                await asyncio.sleep(10)
        except KeyboardInterrupt:
            pass
            print("\n🛑 Bot stopped")
            if self.position:
                pass
        # Check for position closing conditions
        if self.position:
            pass
            pnl = self.position.get('unrealisedPnl', 0)
            if pnl > 20 or pnl < -10:  # Close on profit/loss:
                pass
                await self.close_position("pnl_threshold")
                elif time.time() - self.last_trade_time > 3600:  # Close after 1 hour:
                    pass
                await self.close_position("timeout")
                pass
                await self.close_position("manual_stop")
        except Exception as e:
            pass
            print(f"⚠️ Runtime error: {e}")
            await asyncio.sleep(5)

if __name__ == "__main__":
    pass
    bot = VWAPRSIDivergenceBot()
    asyncio.run(bot.run())