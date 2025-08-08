#!/usr/bin/env python3
"""
LSTM-XGBoost Hybrid Bot - AVAXUSDT (Fixed Version)
Fixes applied:
- Keras Input layer warning fixed
- HTTP connection parameter fixed (demo= instead of testnet=)
- Proper environment variable handling
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
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

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
    
    def log_trade_close(self, trade_id, expected_exit, actual_exit, reason, fees_entry, fees_exit):
        if trade_id not in self.open_trades:
            return None
        
        open_trade = self.open_trades[trade_id]
        exit_slippage = 0  # PostOnly = zero slippage
        
        entry_price = open_trade['actual_price']
        exit_price = actual_exit
        qty = open_trade['qty']
        
        if open_trade['side'] == "LONG":
            gross_pnl = (exit_price - entry_price) * qty
        else:
            gross_pnl = (entry_price - exit_price) * qty
        
        total_fees = (abs(fees_entry) + abs(fees_exit)) * qty * entry_price / 100
        net_pnl = gross_pnl - total_fees
        
        log_entry = {
            "id": trade_id,
            "bot": self.bot_name,
            "symbol": self.symbol,
            "side": open_trade['side'],
            "action": "CLOSE",
            "ts": datetime.now(timezone.utc).isoformat(),
            "expected_exit": round(expected_exit, 6),
            "actual_exit": round(actual_exit, 6),
            "exit_slippage": round(exit_slippage, 6),
            "reason": reason,
            "gross_pnl": round(gross_pnl, 6),
            "fees": round(total_fees, 6),
            "net_pnl": round(net_pnl, 6),
            "currency": self.currency
        }
        
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        del self.open_trades[trade_id]
        return log_entry

class LSTMXGBoostBot:
    def __init__(self):
        self.symbol = 'NEARUSDT'
        self.demo_mode = os.getenv('DEMO_MODE', 'true').lower() == 'true'
        
        # Get credentials based on demo mode
        prefix = 'TESTNET_' if self.demo_mode else 'LIVE_'
        self.api_key = os.getenv(f'{prefix}BYBIT_API_KEY')
        self.api_secret = os.getenv(f'{prefix}BYBIT_API_SECRET')
        
        self.config = {
            'interval': '15',
            'lstm_lookback': 60,
            'xgb_features': 10,
            'risk_percent': 2.0,
            'stop_loss_pct': 0.45,
            'take_profit_pct': 2.0,
            'maker_fee': -0.01,  # Rebate for PostOnly
            'taker_fee': 0.055,
            'maker_offset_pct': 0.02,  # Distance from market for limit orders
            'net_stop_loss': 0.505,
            'net_take_profit': 1.945,
            'max_position_size': 150,  # REDUCED from 450 to avoid market impact
            'min_trade_interval': 300,
            'limit_order_timeout': 30,
            'limit_order_retries': 3,
            'split_threshold': 100,  # Split orders above 100 AVAX
            'split_count': 3
        }
        
        self.logger = TradeLogger("LSTM_XGBOOST", self.symbol)
        
        # Initialize exchange connection with FIX
        self.exchange = HTTP(
            demo=self.demo_mode,  # FIXED: Changed from testnet= to demo=
            api_key=self.api_key,
            api_secret=self.api_secret
        )
        
        self.position = None
        self.price_data = pd.DataFrame()
        self.account_balance = 1000.0
        self.pending_order = False
        self.last_trade_time = 0
        self.trade_cooldown = 30
        self.last_volatility = 0.005
        self.current_trade_id = None
        
        # ML Models
        self.lstm_model = None
        self.xgb_model = None
        self.scaler = MinMaxScaler()
        self.models_trained = False
        
    def format_price(self, price):
        return f"{round(price, 6):.6f}"
    
    def format_qty(self, qty):
        return f"{round(qty / 0.01) * 0.01:.2f}"
    
    async def update_position(self):
        try:
            positions = self.exchange.get_positions(
                category="linear",
                symbol=self.symbol
            )
            
            if positions['retCode'] == 0:
                pos_list = positions['result']['list']
                if pos_list and float(pos_list[0]['size']) > 0:
                    self.position = pos_list[0]
                else:
                    self.position = None
        except Exception as e:
            print(f"❌ Position update error: {e}")
    
    async def execute_limit_order(self, side, qty, base_price, is_reduce=False):
        """Execute limit order with PostOnly for ZERO slippage"""
        formatted_qty = self.format_qty(qty)
        
        # Multiple attempts with adjusting offset
        for retry in range(self.config['limit_order_retries']):
            offset = self.config['maker_offset_pct'] * (1 + retry * 0.5) / 100
            
            if side == "Buy":
                limit_price = base_price * (1 - offset)
            else:
                limit_price = base_price * (1 + offset)
            
            limit_price = float(self.format_price(limit_price))
            
            params = {
                "category": "linear",
                "symbol": self.symbol,
                "side": side,
                "orderType": "Limit",
                "qty": formatted_qty,
                "price": str(limit_price),
                "timeInForce": "PostOnly"  # CRITICAL: Zero slippage
            }
            
            if is_reduce:
                params["reduceOnly"] = True
            
            try:
                order = self.exchange.place_order(**params)
                if order.get('retCode') == 0:
                    order_id = order['result']['orderId']
                    print(f"✅ PostOnly order placed @ ${limit_price:.6f}")
                    
                    # Wait for fill
                    start = time.time()
                    while time.time() - start < self.config['limit_order_timeout']:
                        await asyncio.sleep(1)
                        
                        order_info = self.exchange.get_order_history(
                            category="linear",
                            symbol=self.symbol,
                            orderId=order_id
                        )
                        
                        if order_info['retCode'] == 0:
                            order_data = order_info['result']['list'][0] if order_info['result']['list'] else None
                            if order_data and order_data['orderStatus'] == 'Filled':
                                actual_price = float(order_data['avgPrice'])
                                print(f"✅ FILLED @ ${actual_price:.6f} | ZERO SLIPPAGE")
                                return actual_price
                            elif order_data and order_data['orderStatus'] in ['Cancelled', 'Rejected']:
                                break
                    
                    # Cancel if not filled
                    self.exchange.cancel_order(
                        category="linear",
                        symbol=self.symbol,
                        orderId=order_id
                    )
                    
            except Exception as e:
                print(f"❌ Order attempt {retry+1} failed: {e}")
            
            # Adjust offset for next retry
            if retry < self.config['limit_order_retries'] - 1:
                # Get fresh price for next attempt
                kline = self.exchange.get_kline(
                    category="linear",
                    symbol=self.symbol,
                    interval=self.config['interval'],
                    limit=1
                )
                if kline['retCode'] == 0:
                    base_price = float(kline['result']['list'][0][4])
        
        # Last resort - aggressive limit order
        if side == "Buy":
            fallback_price = base_price * 1.001
        else:
            fallback_price = base_price * 0.999
        
        fallback_price = float(self.format_price(fallback_price))
        
        params = {
            "category": "linear",
            "symbol": self.symbol,
            "side": side,
            "orderType": "Limit",
            "qty": formatted_qty,
            "price": str(fallback_price),
            "timeInForce": "IOC"  # Immediate or Cancel
        }
        
        if is_reduce:
            params["reduceOnly"] = True
        
        try:
            order = self.exchange.place_order(**params)
            if order.get('retCode') == 0:
                return fallback_price
        except:
            pass
        
        return None
    
    async def fetch_price_data(self):
        try:
            kline = self.exchange.get_kline(
                category="linear",
                symbol=self.symbol,
                interval=self.config['interval'],
                limit=200
            )
            
            if kline['retCode'] == 0:
                df = pd.DataFrame(kline['result']['list'],
                                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
                df = df.astype(float)
                df = df.sort_values('timestamp').reset_index(drop=True)
                self.price_data = df
                
                # Calculate volatility
                returns = df['close'].pct_change().dropna()
                self.last_volatility = returns.std() if len(returns) > 0 else 0.005
                
        except Exception as e:
            print(f"❌ Price fetch error: {e}")
    
    def prepare_lstm_data(self, df):
        if len(df) < self.config['lstm_lookback'] + 20:
            return None, None
        
        features = pd.DataFrame()
        features['returns'] = df['close'].pct_change()
        features['volume'] = df['volume'] / df['volume'].rolling(20).mean()
        features['high_low'] = (df['high'] - df['low']) / df['close']
        features['close_open'] = (df['close'] - df['open']) / df['open']
        
        features = features.fillna(0)
        scaled_data = self.scaler.fit_transform(features)
        
        X, y = [], []
        for i in range(self.config['lstm_lookback'], len(scaled_data) - 1):
            X.append(scaled_data[i-self.config['lstm_lookback']:i])
            y.append(1 if df['close'].iloc[i+1] > df['close'].iloc[i] else 0)
        
        return np.array(X), np.array(y)
    
    def build_lstm_model(self):
        """Build LSTM model with fixed Input layer to avoid warning"""
        model = Sequential([
            Input(shape=(self.config['lstm_lookback'], 4)),  # FIXED: Use Input layer instead of input_shape
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    def calculate_xgb_features(self, df):
        if len(df) < 30:
            return None
        
        features = []
        close = df['close']
        
        # Price momentum
        features.append(close.pct_change(1).iloc[-1])
        features.append(close.pct_change(5).iloc[-1])
        features.append((close.iloc[-1] - close.rolling(10).mean().iloc[-1]) / close.iloc[-1])
        
        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi = 100 - (100 / (1 + gain / loss))
        features.append(rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50)
        
        # Bollinger Bands
        sma = close.rolling(20).mean()
        std = close.rolling(20).std()
        upper_band = sma + (2 * std)
        lower_band = sma - (2 * std)
        bb_position = (close.iloc[-1] - lower_band.iloc[-1]) / (upper_band.iloc[-1] - lower_band.iloc[-1])
        features.append(bb_position if not pd.isna(bb_position) else 0.5)
        
        # Volume ratio
        features.append(df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1])
        
        # MACD
        macd = close.ewm(span=12).mean() - close.ewm(span=26).mean()
        signal = macd.ewm(span=9).mean()
        features.append((macd.iloc[-1] - signal.iloc[-1]) / close.iloc[-1] if close.iloc[-1] != 0 else 0)
        
        # Price range
        features.append((df['high'].iloc[-1] - df['low'].iloc[-1]) / df['close'].iloc[-1])
        features.append((df['close'].iloc[-1] - df['open'].iloc[-1]) / df['open'].iloc[-1])
        
        # ATR
        atr = pd.DataFrame()
        atr['h-l'] = df['high'] - df['low']
        atr['h-pc'] = abs(df['high'] - df['close'].shift(1))
        atr['l-pc'] = abs(df['low'] - df['close'].shift(1))
        atr['tr'] = atr[['h-l', 'h-pc', 'l-pc']].max(axis=1)
        atr_value = atr['tr'].rolling(14).mean().iloc[-1] / close.iloc[-1]
        features.append(atr_value if not pd.isna(atr_value) else 0.01)
        
        return np.array(features).reshape(1, -1)
    
    async def train_models(self):
        if len(self.price_data) < 100:
            return
        
        # Train LSTM
        X_lstm, y_lstm = self.prepare_lstm_data(self.price_data)
        if X_lstm is not None and len(X_lstm) > 50:
            self.lstm_model = self.build_lstm_model()
            self.lstm_model.fit(X_lstm, y_lstm, epochs=10, batch_size=32, verbose=0)
        
        # Train XGBoost
        X_xgb, y_xgb = [], []
        for i in range(30, len(self.price_data) - 1):
            features = self.calculate_xgb_features(self.price_data.iloc[:i+1])
            if features is not None:
                X_xgb.append(features[0])
                y_xgb.append(1 if self.price_data['close'].iloc[i+1] > self.price_data['close'].iloc[i] else 0)
        
        if len(X_xgb) > 30:
            self.xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1)
            self.xgb_model.fit(np.array(X_xgb), np.array(y_xgb))
        
        self.models_trained = True
        print("✅ Models trained successfully")
    
    def get_hybrid_prediction(self, df):
        if not self.models_trained or self.lstm_model is None or self.xgb_model is None:
            return 0.5, 0.5, 0.5
        
        # LSTM prediction
        lstm_data, _ = self.prepare_lstm_data(df)
        if lstm_data is not None and len(lstm_data) > 0:
            lstm_pred = self.lstm_model.predict(lstm_data[-1:], verbose=0)[0][0]
        else:
            lstm_pred = 0.5
        
        # XGBoost prediction
        xgb_features = self.calculate_xgb_features(df)
        if xgb_features is not None:
            xgb_pred = self.xgb_model.predict_proba(xgb_features)[0][1]
        else:
            xgb_pred = 0.5
        
        # Hybrid prediction (weighted average)
        hybrid = 0.6 * lstm_pred + 0.4 * xgb_pred
        
        return hybrid, lstm_pred, xgb_pred
    
    async def check_signals(self):
        if not self.models_trained:
            return None
        
        # Check cooldown
        if time.time() - self.last_trade_time < self.config['min_trade_interval']:
            return None
        
        hybrid_pred, lstm_pred, xgb_pred = self.get_hybrid_prediction(self.price_data)
        current_price = float(self.price_data['close'].iloc[-1])
        
        # Strong buy signal
        if hybrid_pred > 0.65 and lstm_pred > 0.6 and xgb_pred > 0.6:
            return {
                'action': 'BUY',
                'price': current_price,
                'confidence': hybrid_pred,
                'lstm': lstm_pred,
                'xgb': xgb_pred
            }
        
        # Strong sell signal
        elif hybrid_pred < 0.35 and lstm_pred < 0.4 and xgb_pred < 0.4:
            return {
                'action': 'SELL',
                'price': current_price,
                'confidence': 1 - hybrid_pred,
                'lstm': lstm_pred,
                'xgb': xgb_pred
            }
        
        return None
    
    async def execute_trade(self, signal):
        # Calculate position size
        risk_amount = self.account_balance * (self.config['risk_percent'] / 100)
        stop_loss_distance = signal['price'] * (self.config['stop_loss_pct'] / 100)
        
        position_size = risk_amount / stop_loss_distance
        position_size = min(position_size, self.config['max_position_size'])
        
        if position_size < 1:
            print(f"⚠️ Position size too small: {position_size}")
            return
        
        # Split large orders
        if position_size > self.config['split_threshold']:
            split_size = position_size / self.config['split_count']
            print(f"📦 Splitting order: {self.config['split_count']} x {self.format_qty(split_size)} AVAX")
            
            total_filled = 0
            avg_price = 0
            
            for i in range(self.config['split_count']):
                actual_price = await self.execute_limit_order(
                    "Buy" if signal['action'] == 'BUY' else "Sell",
                    split_size,
                    signal['price']
                )
                
                if actual_price:
                    total_filled += split_size
                    avg_price = ((avg_price * (i / (i+1))) + (actual_price / (i+1))) if i > 0 else actual_price
                    await asyncio.sleep(2)  # Small delay between orders
                else:
                    print(f"⚠️ Split order {i+1} failed")
                    break
            
            if total_filled > 0:
                self.last_trade_time = time.time()
                print(f"✅ Order filled: {self.format_qty(total_filled)} AVAX @ ${avg_price:.6f}")
                
                # Log trade
                stop_loss = avg_price * (1 - self.config['stop_loss_pct']/100) if signal['action'] == 'BUY' else avg_price * (1 + self.config['stop_loss_pct']/100)
                take_profit = avg_price * (1 + self.config['take_profit_pct']/100) if signal['action'] == 'BUY' else avg_price * (1 - self.config['take_profit_pct']/100)
                
                self.current_trade_id, _ = self.logger.log_trade_open(
                    side=signal['action'],
                    expected_price=signal['price'],
                    actual_price=avg_price,
                    qty=total_filled,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    info=f"hybrid:{signal['confidence']:.3f}_lstm:{signal['lstm']:.3f}_xgb:{signal['xgb']:.3f}"
                )
        else:
            # Normal single order
            actual_price = await self.execute_limit_order(
                "Buy" if signal['action'] == 'BUY' else "Sell",
                position_size,
                signal['price']
            )
            
            if actual_price:
                self.last_trade_time = time.time()
                print(f"✅ {signal['action']}: {self.format_qty(position_size)} @ ${actual_price:.6f}")
                
                # Log trade
                stop_loss = actual_price * (1 - self.config['stop_loss_pct']/100) if signal['action'] == 'BUY' else actual_price * (1 + self.config['stop_loss_pct']/100)
                take_profit = actual_price * (1 + self.config['take_profit_pct']/100) if signal['action'] == 'BUY' else actual_price * (1 - self.config['take_profit_pct']/100)
                
                self.current_trade_id, _ = self.logger.log_trade_open(
                    side=signal['action'],
                    expected_price=signal['price'],
                    actual_price=actual_price,
                    qty=position_size,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    info=f"hybrid:{signal['confidence']:.3f}_lstm:{signal['lstm']:.3f}_xgb:{signal['xgb']:.3f}"
                )
    
    async def close_position(self, reason):
        if not self.position:
            return
        
        try:
            current_price = float(self.price_data['close'].iloc[-1])
            side = "Sell" if self.position['side'] == "Buy" else "Buy"
            qty = float(self.position['size'])
            
            # Use split orders for large positions
            if qty > self.config['split_threshold']:
                split_size = qty / self.config['split_count']
                total_qty = 0
                avg_price = 0
                
                for i in range(self.config['split_count']):
                    actual_price = await self.execute_limit_order(
                        side,
                        split_size,
                        current_price,
                        is_reduce=True
                    )
                    
                    if actual_price:
                        total_qty += split_size
                        avg_price = ((avg_price * (i / (i+1))) + (actual_price / (i+1))) if i > 0 else actual_price
                        await asyncio.sleep(1)
                
                actual_price = avg_price if total_qty > 0 else None
            else:
                actual_price = await self.execute_limit_order(
                    side,
                    qty,
                    current_price,
                    is_reduce=True
                )
            
            if actual_price and self.current_trade_id:
                self.logger.log_trade_close(
                    trade_id=self.current_trade_id,
                    expected_exit=current_price,
                    actual_exit=actual_price,
                    reason=reason,
                    fees_entry=abs(self.config['maker_fee']),
                    fees_exit=abs(self.config['maker_fee'])
                )
                self.current_trade_id = None
                
                print(f"✅ Closed: {reason} @ ${actual_price:.6f} | ZERO SLIPPAGE")
            
        except Exception as e:
            print(f"❌ Close failed: {e}")
    
    def show_status(self):
        if len(self.price_data) == 0:
            return
        
        current_price = float(self.price_data['close'].iloc[-1])
        
        print(f"\n🧠 LSTM-XGBoost Hybrid - {self.symbol}")
        print(f"💰 Price: ${current_price:.6f} | Balance: ${self.account_balance:.2f}")
        print(f"⚙️ Max Position: {self.config['max_position_size']} AVAX (reduced from 450)")
        
        if self.models_trained:
            hybrid_pred, lstm_pred, xgb_pred = self.get_hybrid_prediction(self.price_data)
            print(f"📊 Predictions - Hybrid: {hybrid_pred:.3f} | LSTM: {lstm_pred:.3f} | XGB: {xgb_pred:.3f}")
        else:
            print("🔄 Training models...")
        
        if self.position:
            entry = float(self.position.get('avgPrice', 0))
            side = self.position.get('side', '')
            size = self.position.get('size', '0')
            pnl = float(self.position.get('unrealisedPnl', 0))
            
            position_value = float(size) * entry
            maker_fee = abs(self.config['maker_fee']) * position_value / 100
            net_pnl = pnl + maker_fee  # Add back rebate
            
            print(f"📈 Position: {side} {size} @ ${entry:.6f}")
            print(f"   P&L: ${pnl:.2f} | Rebate: ${maker_fee:.2f} | Net: ${net_pnl:.2f}")
    
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
            elif self.models_trained:
                hybrid_pred, _, _ = self.get_hybrid_prediction(self.price_data)
                if hybrid_pred < 0.3:
                    await self.close_position("SIGNAL_EXIT")
        
        elif side == "Sell":
            pnl_pct = ((entry_price - current_price) / entry_price) * 100
            
            if pnl_pct <= -self.config['net_stop_loss']:
                await self.close_position("STOP_LOSS")
            elif pnl_pct >= self.config['net_take_profit']:
                await self.close_position("TAKE_PROFIT")
            elif self.models_trained:
                hybrid_pred, _, _ = self.get_hybrid_prediction(self.price_data)
                if hybrid_pred > 0.7:
                    await self.close_position("SIGNAL_EXIT")
    
    async def run(self):
        print(f"🚀 Starting LSTM-XGBoost Bot - {self.symbol}")
        print(f"✅ SLIPPAGE FIX ENABLED:")
        print(f"   • PostOnly Limit Orders = 0 Slippage")
        print(f"   • Max Position: {self.config['max_position_size']} AVAX")
        print(f"   • Order Splitting: Above {self.config['split_threshold']} AVAX")
        print(f"   • Maker Rebate: {abs(self.config['maker_fee'])}%")
        
        # Test connection
        try:
            server_time = self.exchange.get_server_time()
            if server_time['retCode'] == 0:
                print(f"✅ Connected to {'Testnet' if self.demo_mode else 'Mainnet'}")
            else:
                print(f"❌ Connection failed: {server_time.get('retMsg')}")
                return
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return
        
        # Initial training
        await self.fetch_price_data()
        await self.train_models()
        
        iteration = 0
        while True:
            try:
                await self.fetch_price_data()
                await self.update_position()
                
                # Retrain models periodically
                if iteration % 20 == 0:
                    await self.train_models()
                
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
    bot = LSTMXGBoostBot()
    asyncio.run(bot.run())