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
        self.LIVE_TRADING = False  # Enable actual trading
        self.account_balance = 1000.0  # Default balance
        self.pending_order = False
        self.last_trade_time = 0
        self.trade_cooldown = 30  # 30 seconds between trades
        self.bot_name = bot_name
        self.symbol = symbol
        self.currency = "USDT"
        self.open_trades = {}
        self.trade_id = 1000
        
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
            "qty": round(qty, 6),
            "stop_loss": round(stop_loss, 6),
            "take_profit": round(take_profit, 6),
            "currency": self.currency,
            "info": info
        }
        
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        self.open_trades[trade_id] = log_entry
        return trade_id, log_entry

class RMISupertrendBot:
    def __init__(self):
        self.LIVE_TRADING = False  # Enable actual trading
        self.account_balance = 1000.0  # Default balance
        self.pending_order = False
        self.last_trade_time = 0
        self.trade_cooldown = 30  # 30 seconds between trades
        self.config = {
            'symbol': 'ADAUSDT',
            'interval': '15',
            'rmi_period': 14,
            'momentum_period': 4,
            'atr_period': 10,
            'atr_multiplier': 3,
            'risk_percent': 2.0,
            'stop_loss_pct': 0.45,
            'take_profit_pct': 2.0,
            'maker_fee': -0.01,  # Rebate
            'taker_fee': 0.055,
            'maker_offset_pct': 0.02,
            'net_stop_loss': 0.505,
            'net_take_profit': 1.945,
            'max_position_size': 5000,  # ADA position limit
            'min_trade_interval': 300,
            'limit_order_timeout': 30,
            'limit_order_retries': 3
        }
        
        self.symbol = self.config['symbol']
        self.logger = TradeLogger("RMI_SUPERTREND_FIXED", self.symbol)
        
        prefix = "TESTNET_" if not self.LIVE_TRADING else "LIVE_"
        api_key = os.getenv(f"{prefix}BYBIT_API_KEY", "")
        api_secret = os.getenv(f"{prefix}BYBIT_API_SECRET", "")
    
        self.exchange = HTTP(
        testnet=(not self.LIVE_TRADING),  # Use testnet if not live
        api_key=api_key,
        api_secret=api_secret
        )
        self.position = None
        self.current_trade_id = None
        self.account_balance = 10000
        self.price_data = pd.DataFrame()
        self.last_trade_time = 0
        
    def format_qty(self, qty):
        info = self.exchange.get_instruments_info(category="linear", symbol=self.symbol)
        if info['retCode'] == 0:
            qty_step = float(info['result']['list'][0]['lotSizeFilter']['qtyStep'])
            return str(int(qty / qty_step) * qty_step)
        return str(qty)
    
    def format_price(self, price):
        info = self.exchange.get_instruments_info(category="linear", symbol=self.symbol)
        if info['retCode'] == 0:
            tick_size = float(info['result']['list'][0]['priceFilter']['tickSize'])
            return str(round(price / tick_size) * tick_size)
        return str(price)
    
    async def execute_limit_order(self, side, qty, base_price, is_reduce=False):
        """Execute limit order with PostOnly for zero slippage"""
        formatted_qty = self.format_qty(qty)
        
        for retry in range(self.config['limit_order_retries']):
            # Calculate limit price
            if side == "Buy":
                limit_price = base_price * (1 - self.config['maker_offset_pct']/100)
            else:
                limit_price = base_price * (1 + self.config['maker_offset_pct']/100)
            
            limit_price = float(self.format_price(limit_price))
            
            try:
                params = {
                    "category": "linear",
                    "symbol": self.symbol,
                    "side": side,
                    "orderType": "Limit",
                    "qty": formatted_qty,
                    "price": str(limit_price),
                    "timeInForce": "PostOnly"  # Zero slippage
                }
                
                if is_reduce:
                    params["reduceOnly"] = True
                
                order = self.exchange.place_order(**params)
                
                if order.get('retCode') == 0:
                    order_id = order['result']['orderId']
                    
                    # Wait for fill
                    start_time = time.time()
                    while time.time() - start_time < self.config['limit_order_timeout']:
                        await asyncio.sleep(1)
                        
                        # Check order status
                        order_status = self.exchange.get_open_orders(
                            category="linear",
                            symbol=self.symbol,
                            orderId=order_id
                        )
                        
                        if order_status['retCode'] == 0:
                            if not order_status['result']['list']:
                                # Order filled
                                return limit_price
                    
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
                kline = self.exchange.get_kline(
                    category="linear",
                    symbol=self.symbol,
                    interval=self.config['interval'],
                    limit=1
                )
                if kline['retCode'] == 0:
                    base_price = float(kline['result']['list'][0][4])
        
        return None
    
    def calculate_rmi(self, prices, period=14, momentum_period=4):
        """Calculate Relative Momentum Index"""
        momentum = prices.diff(momentum_period)
        
        gains = momentum.where(momentum > 0, 0)
        losses = -momentum.where(momentum < 0, 0)
        
        avg_gain = gains.rolling(window=period).mean()
        avg_loss = losses.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rmi = 100 - (100 / (1 + rs))
        
        return rmi
    
    def calculate_supertrend(self, df, period=10, multiplier=3):
        """Calculate Supertrend indicator"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Calculate ATR
        hl = high - low
        hc = abs(high - close.shift(1))
        lc = abs(low - close.shift(1))
        
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        # Calculate basic bands
        hl_avg = (high + low) / 2
        upper_band = hl_avg + (multiplier * atr)
        lower_band = hl_avg - (multiplier * atr)
        
        # Initialize Supertrend
        supertrend = pd.Series(index=df.index, dtype=float)
        direction = pd.Series(index=df.index, dtype=float)
        
        for i in range(period, len(df)):
            if close.iloc[i] <= upper_band.iloc[i]:
                supertrend.iloc[i] = upper_band.iloc[i]
            else:
                supertrend.iloc[i] = lower_band.iloc[i]
            
            if i > period:
                if close.iloc[i] > supertrend.iloc[i-1]:
                    direction.iloc[i] = 1  # Uptrend
                else:
                    direction.iloc[i] = -1  # Downtrend
        
        return supertrend, direction
    
    async def fetch_price_data(self):
        try:
            kline = self.exchange.get_kline(
                category="linear",
                symbol=self.symbol,
                interval=self.config['interval'],
                limit=100
            )
            
            if kline['retCode'] == 0:
                df = pd.DataFrame(kline['result']['list'],
                                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
                df = df.astype(float)
                df = df.sort_values('timestamp')
                self.price_data = df
                
        except Exception as e:
            print(f"❌ Price fetch error: {e}")
    
    async def check_signals(self):
        if len(self.price_data) < 30:
            return None
        
        # Check minimum trade interval
        if time.time() - self.last_trade_time < self.config['min_trade_interval']:
            return None
        
        # Calculate indicators
        rmi = self.calculate_rmi(
            self.price_data['close'],
            self.config['rmi_period'],
            self.config['momentum_period']
        )
        
        supertrend, direction = self.calculate_supertrend(
            self.price_data,
            self.config['atr_period'],
            self.config['atr_multiplier']
        )
        
        current_price = float(self.price_data['close'].iloc[-1])
        current_rmi = rmi.iloc[-1]
        current_st = supertrend.iloc[-1]
        current_direction = direction.iloc[-1]
        
        # Generate signals
        if current_rmi < 30 and current_direction == 1:
            return {
                'action': 'BUY',
                'price': current_price,
                'rmi': current_rmi,
                'supertrend': current_st,
                'trend': 'UP'
            }
        elif current_rmi > 70 and current_direction == -1:
            return {
                'action': 'SELL',
                'price': current_price,
                'rmi': current_rmi,
                'supertrend': current_st,
                'trend': 'DOWN'
            }
        
        # Also check for trend reversals
        if len(direction) > 2:
            if direction.iloc[-2] == -1 and direction.iloc[-1] == 1 and current_rmi < 50:
                return {
                    'action': 'BUY',
                    'price': current_price,
                    'rmi': current_rmi,
                    'supertrend': current_st,
                    'trend': 'REVERSAL_UP'
                }
            elif direction.iloc[-2] == 1 and direction.iloc[-1] == -1 and current_rmi > 50:
                return {
                    'action': 'SELL',
                    'price': current_price,
                    'rmi': current_rmi,
                    'supertrend': current_st,
                    'trend': 'REVERSAL_DOWN'
                }
        
        return None
    
    async def update_position(self):
        try:
            pos = self.exchange.get_positions(category="linear", symbol=self.symbol)
            if pos['retCode'] == 0 and pos['result']['list']:
                position = pos['result']['list'][0]
                if float(position['size']) > 0:
                    self.position = position
                else:
                    self.position = None
            else:
                self.position = None
        except:
            self.position = None
    
    async def execute_trade(self, signal):
        if not signal or self.position:
            return
        
        # Calculate position size
        risk_amount = self.account_balance * self.config['risk_percent'] / 100
        stop_distance = self.config['net_stop_loss'] / 100
        position_size = risk_amount / (signal['price'] * stop_distance)
        
        # Apply maximum position size limit
        position_size = min(position_size, self.config['max_position_size'])
        
        # Calculate stop loss
        if signal['action'] == 'BUY':
            stop_loss_price = signal['price'] * (1 - self.config['net_stop_loss']/100)
        else:
            stop_loss_price = signal['price'] * (1 + self.config['net_stop_loss']/100)
        
        stop_loss_price = float(self.format_price(stop_loss_price))
        
        try:
            # Execute with limit order
            actual_price = await self.execute_limit_order(
                "Buy" if signal['action'] == 'BUY' else "Sell",
                position_size,
                signal['price']
            )
            
            if actual_price:
                self.last_trade_time = time.time()
                
                # Calculate take profit
                if signal['action'] == 'BUY':
                    take_profit = actual_price * (1 + self.config['net_take_profit']/100)
                else:
                    take_profit = actual_price * (1 - self.config['net_take_profit']/100)
                
                # Log with zero slippage
                self.current_trade_id, _ = self.logger.log_trade_open(
                    side=signal['action'],
                    expected_price=signal['price'],
                    actual_price=actual_price,
                    qty=position_size,
                    stop_loss=stop_loss_price,
                    take_profit=take_profit,
                    info=f"rmi:{signal['rmi']:.1f}_st:{signal['supertrend']:.4f}_trend:{signal['trend']}_risk:{self.config['risk_percent']}%"
                )
                
                position_value = position_size * actual_price
                
                print(f"📊 RMI-ST {signal['action']}: {position_size:.0f} @ ${actual_price:.4f}")
                print(f"   RMI: {signal['rmi']:.1f} | ST: ${signal['supertrend']:.4f} | Trend: {signal['trend']}")
                print(f"   💰 Position: ${position_value:.2f}")
                print(f"   ✅ ZERO SLIPPAGE with PostOnly")
                
        except Exception as e:
            print(f"❌ Trade failed: {e}")
    
    async def close_position(self, reason):
        if not self.position:
            return
        
        current_price = float(self.price_data['close'].iloc[-1])
        side = "Sell" if self.position.get('side') == "Buy" else "Buy"
        qty = float(self.position['size'])
        
        try:
            actual_price = await self.execute_limit_order(
                side,
                qty,
                current_price,
                is_reduce=True
            )
            
            if actual_price:
                print(f"✅ Closed: {reason} @ ${actual_price:.4f} | ZERO SLIPPAGE")
            
        except Exception as e:
            print(f"❌ Close failed: {e}")
    
    async def check_exit_conditions(self):
        if not self.position:
            return
        
        current_price = float(self.price_data['close'].iloc[-1])
        entry_price = float(self.position.get('avgPrice', 0))
        side = self.position.get('side', '')
        
        if side == "Buy":
            pnl_pct = ((current_price - entry_price) / entry_price) * 100
            
            if pnl_pct <= -self.config['net_stop_loss']:
                await self.close_position("STOP_LOSS")
            elif pnl_pct >= self.config['net_take_profit']:
                await self.close_position("TAKE_PROFIT")
        
        elif side == "Sell":
            pnl_pct = ((entry_price - current_price) / entry_price) * 100
            
            if pnl_pct <= -self.config['net_stop_loss']:
                await self.close_position("STOP_LOSS")
            elif pnl_pct >= self.config['net_take_profit']:
                await self.close_position("TAKE_PROFIT")
    
    def show_status(self):
        if len(self.price_data) == 0:
            return
        
        current_price = float(self.price_data['close'].iloc[-1])
        
        rmi = self.calculate_rmi(
            self.price_data['close'],
            self.config['rmi_period'],
            self.config['momentum_period']
        )
        
        supertrend, direction = self.calculate_supertrend(
            self.price_data,
            self.config['atr_period'],
            self.config['atr_multiplier']
        )
        
        print(f"\n📊 RMI-Supertrend - {self.symbol}")
        print(f"💰 Price: ${current_price:.4f} | Balance: ${self.account_balance:.2f}")
        print(f"📈 RMI: {rmi.iloc[-1]:.1f} | ST: ${supertrend.iloc[-1]:.4f}")
        
        if self.position:
            entry = float(self.position.get('avgPrice', 0))
            side = self.position.get('side', '')
            size = self.position.get('size', '0')
            pnl = float(self.position.get('unrealisedPnl', 0))
            
            print(f"📈 Position: {side} {size} @ ${entry:.4f}")
            print(f"   P&L: ${pnl:.2f}")
    
    async def run(self):
        print(f"🚀 Starting RMI-Supertrend Bot - {self.symbol}")
        print(f"✅ ZERO SLIPPAGE MODE:")
        print(f"   • PostOnly Limit Orders")
        print(f"   • Maker Rebate: {abs(self.config['maker_fee'])}%")
        
        iteration = 0
        while True:
            try:
                await self.fetch_price_data()
                await self.update_position()
                
                # Check for exit conditions
                await self.check_exit_conditions()
                
                # Check for new signals
                if not self.position:
                    signal = await self.check_signals()
                    if signal:
                        await self.execute_trade(signal)
                
                # Show status every 5 iterations
                if iteration % 5 == 0:
                    self.show_status()
                
                iteration += 1
                await asyncio.sleep(60)
                
            except Exception as e:
                print(f"❌ Error: {e}")
                await asyncio.sleep(30)

if __name__ == "__main__":
    bot = RMISupertrendBot()
    asyncio.run(bot.run())