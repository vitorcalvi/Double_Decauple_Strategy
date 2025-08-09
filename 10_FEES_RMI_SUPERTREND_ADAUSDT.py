#!/usr/bin/env python3
"""
RMI + Supertrend Momentum Bot - ADAUSDT
Combines Relative Momentum Index with Supertrend for trend-following
"""

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
        self.daily_pnl = 0
        self.consecutive_losses = 0
        self.max_daily_loss = 50
        
        os.makedirs("logs", exist_ok=True)
        self.log_file = f"logs/10_FEES_RMI_SUPERTREND_ADAUSDT.log"
        
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
            "expected_price": round(expected_price, 6),
            "actual_price": round(actual_price, 6),
            "slippage": round(slippage, 6),
            "qty": round(qty, 2),
            "stop_loss": round(stop_loss, 6),
            "take_profit": round(take_profit, 6),
            "currency": self.currency,
            "info": info
        }
        
        self.open_trades[trade_id] = {
            "entry_time": datetime.now(),
            "entry_price": actual_price,
            "side": side,
            "qty": qty
        }
        
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        
        return trade_id, log_entry
    
    def log_trade_close(self, trade_id, expected_exit, actual_exit, reason, fees_entry=-0.01, fees_exit=-0.01):
        if trade_id not in self.open_trades:
            return None
            
        trade = self.open_trades[trade_id]
        duration = (datetime.now() - trade["entry_time"]).total_seconds()
        
        gross_pnl = ((actual_exit - trade["entry_price"]) * trade["qty"] if trade["side"] == "BUY"
                    else (trade["entry_price"] - actual_exit) * trade["qty"])
        
        # Maker rebates
        entry_fee = trade["entry_price"] * trade["qty"] * abs(fees_entry) / 100
        exit_fee = actual_exit * trade["qty"] * abs(fees_exit) / 100
        net_pnl = gross_pnl + entry_fee + exit_fee  # Add rebates
        
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
            "entry_price": round(trade["entry_price"], 6),
            "expected_exit": round(expected_exit, 6),
            "actual_exit": round(actual_exit, 6),
            "slippage": 0,
            "qty": round(trade["qty"], 2),
            "gross_pnl": round(gross_pnl, 2),
            "net_pnl": round(net_pnl, 2),
            "reason": reason,
            "currency": self.currency
        }
        
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        
        del self.open_trades[trade_id]
        return log_entry

class RMISupertrendBot:
    def __init__(self):
        self.symbol = 'ADAUSDT'
        self.demo_mode = os.getenv('DEMO_MODE', 'true').lower() == 'true'
        
        prefix = 'TESTNET_' if self.demo_mode else 'LIVE_'
        self.api_key = os.getenv(f'{prefix}BYBIT_API_KEY')
        self.api_secret = os.getenv(f'{prefix}BYBIT_API_SECRET')
        self.exchange = None
        
        self.position = None
        self.pending_order = False
        self.last_order_time = 0
        
        self.price_data = pd.DataFrame()
        self.indicators = {}
        self.account_balance = 1000
        
        self.config = {
            'timeframe': '15',
            'lookback_bars': 100,
            'rmi_period': 14,
            'momentum_period': 4,
            'supertrend_period': 10,
            'supertrend_multiplier': 3.0,
            'rmi_oversold': 30,
            'rmi_overbought': 70,
            'risk_per_trade_pct': 2.0,
            'stop_loss_pct': 0.45,
            'take_profit_pct': 2.0,
            'maker_fee': -0.01,
            'taker_fee': 0.055,
            'maker_offset_pct': 0.02,
            'net_stop_loss': 0.505,
            'net_take_profit': 1.945,
            'max_position_size': 5000,
            'min_trade_interval': 300,
            'order_timeout': 30,
            'volume_filter': 1.2
        }
        
        self.logger = TradeLogger("RMI_SUPERTREND", self.symbol)
        self.current_trade_id = None
        self.last_trade_time = 0
        self.last_supertrend_direction = None
    
    def connect(self):
        try:
            self.exchange = HTTP(demo=self.demo_mode, api_key=self.api_key, api_secret=self.api_secret)
            return self.exchange.get_server_time().get('retCode') == 0
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False
    
    async def fetch_market_data(self):
        try:
            klines = self.exchange.get_kline(
                category="linear",
                symbol=self.symbol,
                interval=self.config['timeframe'],
                limit=self.config['lookback_bars']
            )
            
            if klines['retCode'] == 0:
                df = pd.DataFrame(klines['result']['list'],
                                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
                df = df.astype(float)
                df = df.sort_values('timestamp').reset_index(drop=True)
                self.price_data = df
                return True
        except Exception as e:
            print(f"❌ Data fetch error: {e}")
        return False
    
    def calculate_rmi(self, prices, period=None, momentum_period=None):
        """Calculate Relative Momentum Index"""
        if period is None:
            period = self.config['rmi_period']
        if momentum_period is None:
            momentum_period = self.config['momentum_period']
        
        if len(prices) < period + momentum_period:
            return pd.Series([50] * len(prices))
        
        # Calculate momentum (change over momentum_period)
        momentum = prices.diff(momentum_period)
        
        # Separate gains and losses
        gains = momentum.where(momentum > 0, 0)
        losses = -momentum.where(momentum < 0, 0)
        
        # Calculate average gains and losses
        avg_gain = gains.rolling(window=period, min_periods=1).mean()
        avg_loss = losses.rolling(window=period, min_periods=1).mean()
        
        # Calculate RMI
        rs = avg_gain / (avg_loss + 1e-10)
        rmi = 100 - (100 / (1 + rs))
        
        # Fill NaN values with 50 (neutral)
        rmi = rmi.fillna(50)
        
        return rmi
    
    def calculate_supertrend(self, df):
        """Calculate Supertrend indicator"""
        period = self.config['supertrend_period']
        multiplier = self.config['supertrend_multiplier']
        
        # Calculate ATR
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        atr = ranges.max(axis=1).rolling(period).mean()
        
        # Calculate basic bands
        hl_avg = (df['high'] + df['low']) / 2
        upper_band = hl_avg + (multiplier * atr)
        lower_band = hl_avg - (multiplier * atr)
        
        # Initialize Supertrend
        supertrend = pd.Series(index=df.index, dtype=float)
        direction = pd.Series(index=df.index, dtype=float)
        
        for i in range(len(df)):
            if i == 0:
                supertrend.iloc[i] = upper_band.iloc[i]
                direction.iloc[i] = -1
            else:
                # Current close price
                curr_close = df['close'].iloc[i]
                
                # Previous values
                prev_supertrend = supertrend.iloc[i-1]
                
                # Determine current supertrend and direction
                if curr_close <= upper_band.iloc[i]:
                    curr_up = upper_band.iloc[i]
                else:
                    curr_up = prev_supertrend if prev_supertrend > upper_band.iloc[i] else upper_band.iloc[i]
                
                if curr_close >= lower_band.iloc[i]:
                    curr_down = lower_band.iloc[i]
                else:
                    curr_down = prev_supertrend if prev_supertrend < lower_band.iloc[i] else lower_band.iloc[i]
                
                # Determine direction
                if prev_supertrend == supertrend.iloc[i-1]:
                    if curr_close <= curr_up:
                        supertrend.iloc[i] = curr_up
                        direction.iloc[i] = -1
                    else:
                        supertrend.iloc[i] = curr_down
                        direction.iloc[i] = 1
                else:
                    if curr_close >= curr_down:
                        supertrend.iloc[i] = curr_down
                        direction.iloc[i] = 1
                    else:
                        supertrend.iloc[i] = curr_up
                        direction.iloc[i] = -1
        
        return supertrend, direction, atr
    
    def calculate_indicators(self):
        """Calculate all indicators"""
        if len(self.price_data) < self.config['rmi_period'] + self.config['momentum_period']:
            return
        
        df = self.price_data
        
        # Calculate RMI
        rmi = self.calculate_rmi(df['close'])
        
        # Calculate Supertrend
        supertrend, direction, atr = self.calculate_supertrend(df)
        
        # Volume analysis
        volume_ma = df['volume'].rolling(20).mean()
        volume_ratio = df['volume'] / volume_ma
        
        # Price momentum
        price_change = df['close'].pct_change(5).fillna(0)
        
        self.indicators = {
            'rmi': rmi.iloc[-1] if not pd.isna(rmi.iloc[-1]) else 50,
            'rmi_prev': rmi.iloc[-2] if len(rmi) > 1 and not pd.isna(rmi.iloc[-2]) else 50,
            'supertrend': supertrend.iloc[-1] if not pd.isna(supertrend.iloc[-1]) else df['close'].iloc[-1],
            'supertrend_direction': direction.iloc[-1] if not pd.isna(direction.iloc[-1]) else 0,
            'supertrend_prev_direction': direction.iloc[-2] if len(direction) > 1 and not pd.isna(direction.iloc[-2]) else 0,
            'atr': atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else 0,
            'volume_ratio': volume_ratio.iloc[-1] if not pd.isna(volume_ratio.iloc[-1]) else 1,
            'price_momentum': price_change.iloc[-1] if not pd.isna(price_change.iloc[-1]) else 0,
            'current_price': df['close'].iloc[-1]
        }
    
    def generate_signal(self):
        """Generate trading signal based on RMI and Supertrend"""
        if not self.indicators or self.indicators['atr'] == 0:
            return None
        
        current_price = self.indicators['current_price']
        rmi = self.indicators['rmi']
        rmi_prev = self.indicators['rmi_prev']
        supertrend_dir = self.indicators['supertrend_direction']
        supertrend_prev_dir = self.indicators['supertrend_prev_direction']
        volume_ratio = self.indicators['volume_ratio']
        momentum = self.indicators['price_momentum']
        
        # Volume filter
        if volume_ratio < self.config['volume_filter']:
            return None
        
        # Detect Supertrend direction change
        trend_flip_up = supertrend_prev_dir == -1 and supertrend_dir == 1
        trend_flip_down = supertrend_prev_dir == 1 and supertrend_dir == -1
        
        # Long signal: Supertrend flips bullish + RMI oversold or rising
        if trend_flip_up and (rmi < self.config['rmi_oversold'] or (rmi > rmi_prev and rmi < 50)):
            return {
                'action': 'BUY',
                'price': current_price,
                'rmi': rmi,
                'supertrend_dir': supertrend_dir,
                'momentum': momentum,
                'reason': 'supertrend_bullish_rmi_oversold'
            }
        
        # Additional long: Strong uptrend + RMI not overbought
        elif supertrend_dir == 1 and rmi < 60 and momentum > 0.01 and rmi > rmi_prev:
            return {
                'action': 'BUY',
                'price': current_price,
                'rmi': rmi,
                'supertrend_dir': supertrend_dir,
                'momentum': momentum,
                'reason': 'uptrend_momentum_rmi_rising'
            }
        
        # Short signal: Supertrend flips bearish + RMI overbought or falling
        elif trend_flip_down and (rmi > self.config['rmi_overbought'] or (rmi < rmi_prev and rmi > 50)):
            return {
                'action': 'SELL',
                'price': current_price,
                'rmi': rmi,
                'supertrend_dir': supertrend_dir,
                'momentum': momentum,
                'reason': 'supertrend_bearish_rmi_overbought'
            }
        
        # Additional short: Strong downtrend + RMI not oversold
        elif supertrend_dir == -1 and rmi > 40 and momentum < -0.01 and rmi < rmi_prev:
            return {
                'action': 'SELL',
                'price': current_price,
                'rmi': rmi,
                'supertrend_dir': supertrend_dir,
                'momentum': momentum,
                'reason': 'downtrend_momentum_rmi_falling'
            }
        
        return None
    
    def calculate_position_size(self, price, stop_price):
        risk_amount = self.account_balance * (self.config['risk_per_trade_pct'] / 100)
        price_diff = abs(price - stop_price)
        
        if price_diff > 0:
            position_size = risk_amount / price_diff
            max_size = min(position_size, self.config['max_position_size'])
            return max_size
        return 0
    
    def format_qty(self, qty):
        # ADA typically trades in whole units
        return str(int(qty))
    
    async def get_account_balance(self):
        try:
            wallet = self.exchange.get_wallet_balance(accountType="UNIFIED", coin="USDT")
            if wallet['retCode'] == 0:
                for coin in wallet['result']['list'][0]['coin']:
                    if coin['coin'] == 'USDT':
                        self.account_balance = float(coin['walletBalance'])
                        return True
        except:
            pass
        return False
    
    async def update_position(self):
        try:
            pos = self.exchange.get_positions(category="linear", symbol=self.symbol)
            if pos['retCode'] == 0 and pos['result']['list']:
                position_data = pos['result']['list'][0]
                if float(position_data['size']) > 0:
                    self.position = position_data
                else:
                    self.position = None
        except:
            self.position = None
    
    async def execute_trade(self, signal):
        if self.position:
            return
        
        current_time = time.time()
        if current_time - self.last_trade_time < self.config['min_trade_interval']:
            return
        
        await self.get_account_balance()
        
        # Calculate stop loss using ATR
        atr_stop = self.indicators['atr'] * 1.5
        stop_loss = (signal['price'] - atr_stop if signal['action'] == 'BUY'
                    else signal['price'] + atr_stop)
        
        # Ensure minimum stop loss
        min_stop = (signal['price'] * (1 - self.config['net_stop_loss']/100) if signal['action'] == 'BUY'
                   else signal['price'] * (1 + self.config['net_stop_loss']/100))
        
        if signal['action'] == 'BUY':
            stop_loss = min(stop_loss, min_stop)
        else:
            stop_loss = max(stop_loss, min_stop)
        
        qty = self.calculate_position_size(signal['price'], stop_loss)
        formatted_qty = self.format_qty(qty)
        
        if int(formatted_qty) < 1:
            return
        
        # PostOnly limit order for rebate
        limit_price = (signal['price'] * (1 - self.config['maker_offset_pct']/100) if signal['action'] == 'BUY'
                      else signal['price'] * (1 + self.config['maker_offset_pct']/100))
        
        # Calculate take profit
        take_profit = (limit_price * (1 + self.config['net_take_profit']/100) if signal['action'] == 'BUY'
                      else limit_price * (1 - self.config['net_take_profit']/100))
        
        try:
            order = self.exchange.place_order(
                category="linear",
                symbol=self.symbol,
                side="Buy" if signal['action'] == 'BUY' else "Sell",
                orderType="Limit",
                qty=formatted_qty,
                price=str(round(limit_price, 6)),
                timeInForce="PostOnly"
            )
            
            if order.get('retCode') == 0:
                self.last_trade_time = current_time
                self.last_order_time = current_time
                self.pending_order = True
                self.last_supertrend_direction = signal['supertrend_dir']
                
                self.current_trade_id, _ = self.logger.log_trade_open(
                    side=signal['action'],
                    expected_price=signal['price'],
                    actual_price=limit_price,
                    qty=float(formatted_qty),
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    info=f"rmi:{signal['rmi']:.1f}_st_dir:{signal['supertrend_dir']}_mom:{signal['momentum']:.3f}_{signal['reason']}"
                )
                
                direction = "📈" if signal['action'] == 'BUY' else "📉"
                print(f"{direction} RMI+ST {signal['action']}: {formatted_qty} ADA @ ${limit_price:.6f}")
                print(f"   RMI: {signal['rmi']:.1f} | Supertrend: {'UP' if signal['supertrend_dir'] == 1 else 'DOWN'}")
                print(f"   Momentum: {signal['momentum']:.3f} | Reason: {signal['reason']}")
                
        except Exception as e:
            print(f"❌ Order error: {e}")
    
    async def check_pending_orders(self):
        if not self.pending_order:
            return
        
        if time.time() - self.last_order_time > self.config['order_timeout']:
            try:
                self.exchange.cancel_all_orders(category="linear", symbol=self.symbol)
                self.pending_order = False
                print("⏰ Order timeout - cancelled")
            except:
                pass
    
    async def check_exit_conditions(self):
        if not self.position:
            return
        
        current_price = float(self.price_data['close'].iloc[-1])
        entry_price = float(self.position.get('avgPrice', 0))
        side = self.position.get('side')
        
        # Update indicators
        self.calculate_indicators()
        
        if not self.indicators:
            return
        
        supertrend_dir = self.indicators['supertrend_direction']
        rmi = self.indicators['rmi']
        
        if side == "Buy":
            pnl_pct = ((current_price - entry_price) / entry_price) * 100
            
            # Stop loss
            if pnl_pct <= -self.config['net_stop_loss']:
                await self.close_position("STOP_LOSS")
            # Take profit
            elif pnl_pct >= self.config['net_take_profit']:
                await self.close_position("TAKE_PROFIT")
            # Supertrend flip
            elif supertrend_dir == -1:
                await self.close_position("SUPERTREND_FLIP")
            # RMI extreme overbought
            elif rmi > 80:
                await self.close_position("RMI_OVERBOUGHT")
                
        elif side == "Sell":
            pnl_pct = ((entry_price - current_price) / entry_price) * 100
            
            # Stop loss
            if pnl_pct <= -self.config['net_stop_loss']:
                await self.close_position("STOP_LOSS")
            # Take profit
            elif pnl_pct >= self.config['net_take_profit']:
                await self.close_position("TAKE_PROFIT")
            # Supertrend flip
            elif supertrend_dir == 1:
                await self.close_position("SUPERTREND_FLIP")
            # RMI extreme oversold
            elif rmi < 20:
                await self.close_position("RMI_OVERSOLD")
    
    async def close_position(self, reason):
        if not self.position:
            return
        
        current_price = float(self.price_data['close'].iloc[-1])
        side = "Sell" if self.position.get('side') == "Buy" else "Buy"
        qty = self.position.get('size')
        
        try:
            # Try PostOnly first for rebate
            limit_price = (current_price * (1 + self.config['maker_offset_pct']/100) if side == "Sell"
                          else current_price * (1 - self.config['maker_offset_pct']/100))
            
            order = self.exchange.place_order(
                category="linear",
                symbol=self.symbol,
                side=side,
                orderType="Limit",
                qty=qty,
                price=str(round(limit_price, 6)),
                timeInForce="PostOnly",
                reduceOnly=True
            )
            
            # If PostOnly fails, use market order
            if order.get('retCode') != 0:
                order = self.exchange.place_order(
                    category="linear",
                    symbol=self.symbol,
                    side=side,
                    orderType="Market",
                    qty=qty,
                    reduceOnly=True
                )
                exit_price = current_price
                exit_fee = self.config['taker_fee']
            else:
                exit_price = limit_price
                exit_fee = self.config['maker_fee']
            
            if order.get('retCode') == 0 and self.current_trade_id:
                self.logger.log_trade_close(
                    trade_id=self.current_trade_id,
                    expected_exit=current_price,
                    actual_exit=exit_price,
                    reason=reason,
                    fees_entry=self.config['maker_fee'],
                    fees_exit=exit_fee
                )
                
                print(f"✅ Closed: {reason} @ ${exit_price:.6f}")
                self.position = None
                self.current_trade_id = None
                
        except Exception as e:
            print(f"❌ Close error: {e}")
    
    def show_status(self):
        if len(self.price_data) == 0 or not self.indicators:
            return
        
        current_price = self.indicators['current_price']
        rmi = self.indicators['rmi']
        supertrend_dir = self.indicators['supertrend_direction']
        atr = self.indicators['atr']
        momentum = self.indicators['price_momentum']
        
        print(f"\n🎯 RMI+Supertrend Bot - {self.symbol}")
        print(f"💰 Price: ${current_price:.6f} | Balance: ${self.account_balance:.2f}")
        
        # Indicators
        trend = "📈 BULLISH" if supertrend_dir == 1 else "📉 BEARISH"
        rmi_status = "🔴" if rmi > self.config['rmi_overbought'] else "🟢" if rmi < self.config['rmi_oversold'] else "⚪"
        print(f"📊 Supertrend: {trend} | RMI: {rmi:.1f} {rmi_status}")
        print(f"📈 Momentum: {momentum:+.3f} | ATR: ${atr:.6f}")
        
        if self.position:
            entry_price = float(self.position.get('avgPrice', 0))
            side = self.position.get('side', '')
            size = self.position.get('size', '0')
            pnl = float(self.position.get('unrealisedPnl', 0))
            
            emoji = "🟢" if side == "Buy" else "🔴"
            print(f"{emoji} {side}: {size} ADA @ ${entry_price:.6f} | PnL: ${pnl:.2f}")
        elif self.pending_order:
            print(f"⏳ Pending order...")
        else:
            print("🔍 Waiting for RMI+Supertrend signal...")
    
    async def run(self):
        if not self.connect():
            print("❌ Failed to connect")
            return
        
        print(f"🚀 Starting RMI+Supertrend Bot - {self.symbol}")
        print(f"📊 Mode: {'DEMO' if self.demo_mode else 'LIVE'}")
        print(f"⚙️ RMI Period: {self.config['rmi_period']} | Supertrend: {self.config['supertrend_period']}x{self.config['supertrend_multiplier']}")
        print(f"📈 RMI Levels: OS={self.config['rmi_oversold']} | OB={self.config['rmi_overbought']}")
        
        iteration = 0
        while True:
            try:
                
                
                await self.fetch_market_data()
                self.calculate_indicators()
                await self.update_position()
                await self.check_pending_orders()
                
                if self.position:
                    await self.check_exit_conditions()
                else:
                    signal = self.generate_signal()
                    if signal:
                        await self.execute_trade(signal)
                
                if iteration % 5 == 0:
                    self.show_status()
                
                iteration += 1
                await asyncio.sleep(5)
                
            except Exception as e:
                print(f"❌ Error: {e}")
                await asyncio.sleep(10)

if __name__ == "__main__":
    bot = RMISupertrendBot()
    asyncio.run(bot.run())