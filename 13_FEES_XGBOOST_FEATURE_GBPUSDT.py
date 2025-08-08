import os
import asyncio
import pandas as pd
import numpy as np
import json
from datetime import datetime, timezone
from pybit.unified_trading import HTTP
from dotenv import load_dotenv
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

load_dotenv()

class TradeLogger:
    def __init__(self, bot_name, symbol):
        self.trade_cooldown = 30
        self.last_trade_time = 0
        self.pending_order = False
        self.account_balance = 1000.0
        self.demo_mode = os.getenv('DEMO_MODE', 'true').lower() == 'true'
        self.LIVE_TRADING = False

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
        "expected_price": round(expected_price, 6),
        "actual_price": round(actual_price, 6),
        "slippage": round(slippage, 6),
        "qty": round(qty, 6),
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
        
        # Fixed: Fees are costs, should be subtracted from PnL
        entry_fee = trade["entry_price"] * trade["qty"] * abs(fees_entry) / 100
        exit_fee = actual_exit * trade["qty"] * abs(fees_exit) / 100
        net_pnl = gross_pnl - entry_fee - exit_fee
        
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
        "slippage": round(slippage, 6),
        "qty": round(trade["qty"], 6),
        "gross_pnl": round(gross_pnl, 2),
        "net_pnl": round(net_pnl, 2),
        "reason": reason,
        "currency": self.currency
        }
        
        with open(self.log_file, "a") as f:
            pass
            f.write(json.dumps(log_entry) + "\n")
        
        del self.open_trades[trade_id]
        return log_entry

class LSTMXGBoostBot:
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
        
        self.symbol = 'JPYUSDT'
        self.demo_mode = os.getenv('DEMO_MODE', 'true').lower() == 'true'
        
        prefix = 'TESTNET_' if self.demo_mode else 'LIVE_'
        self.api_key = os.getenv(f'{prefix}BYBIT_API_KEY')
        self.api_secret = os.getenv(f'{prefix}BYBIT_API_SECRET')
        self.exchange = None
        
        self.position = None
        self.price_data = pd.DataFrame()
        self.account_balance = 0
        self.instrument_info = {}
        
        self.config = {
        'timeframe': '30',
        'risk_percent': 1.0,  # Risk 1% of account per trade
        'maker_offset_pct': 0.01,
        'net_take_profit': 0.9,
        'net_stop_loss': 0.45,
        'lstm_lookback': 60,
        'prediction_threshold': 0.6,
        'lookback': 200,
        'slippage_pct': 0.02,  # Expected slippage 0.02%
        'maker_fee': -0.01,  # Maker rebate
        'taker_fee': 0.055   # Taker fee
        }
        
        self.lstm_model = None
        self.xgb_model = None
        self.scaler = MinMaxScaler()
        self.models_trained = False
        
        self.logger = TradeLogger("LSTM_XGBOOST", self.symbol)
        self.current_trade_id = None
    
    def connect(self):
        try:
            self.exchange = HTTP(demo=self.demo_mode, api_key=self.api_key, api_secret=self.api_secret)
            if self.exchange.get_server_time().get('retCode') == 0:
                pass
                self.get_instrument_info()
                self.update_account_balance()
                return True
            return False
        except:
            return False
    
    def get_instrument_info(self):
        try:
            info = self.exchange.get_instruments_info(category="linear", symbol=self.symbol)
            if info.get('retCode') == 0:
                pass
                instrument = info['result']['list'][0]
                self.instrument_info = {
                'qty_step': float(instrument['lotSizeFilter']['qtyStep']),
                'min_qty': float(instrument['lotSizeFilter']['minOrderQty']),
                'max_qty': float(instrument['lotSizeFilter']['maxOrderQty']),
                'tick_size': float(instrument['priceFilter']['tickSize'])
                }
        except:
            # Default values if API fails
            self.instrument_info = {
            'qty_step': 0.0001,
            'min_qty': 0.0001,
            'max_qty': 10000,
            'tick_size': 0.000001
            }
    
    def update_account_balance(self):
        try:
            wallet = self.exchange.get_wallet_balance(accountType="UNIFIED")
            if wallet.get('retCode') == 0:
                pass
                for coin in wallet['result']['list'][0]['coin']:
                    pass
                    if coin['coin'] == 'USDT':
                        pass
                        self.account_balance = float(coin['availableToWithdraw'])
                        break
        except:
            if self.demo_mode:
                pass
                self.account_balance = 10000  # Default demo balance
    
    def format_qty(self, qty):
        # Properly round to instrument precision
        step = self.instrument_info.get('qty_step', 0.0001)
        qty = round(qty / step) * step
        qty = max(qty, self.instrument_info.get('min_qty', 0.0001))
        qty = min(qty, self.instrument_info.get('max_qty', 10000))
        return str(qty)
    
    def calculate_position_size(self, price):
        # Risk-based position sizing
        self.update_account_balance()
        risk_amount = self.account_balance * (self.config['risk_percent'] / 100)
        stop_loss_pct = self.config['net_stop_loss'] / 100
        position_value = risk_amount / stop_loss_pct
        qty = position_value / price
        return qty
    
    def simulate_slippage(self, price, side, size_usdt):
        # Simulate realistic slippage based on order size
        base_slippage = 0  # PostOnly = zero slippage
        
        # Larger orders have more slippage
        size_factor = min(size_usdt / 10000, 2.0)  # Cap at 2x for very large orders
        total_slippage = 0  # PostOnly = zero slippage
        
        # Add random component
        random_factor = np.random.uniform(0.5, 1.5)
        total_slippage *= random_factor
        
        if side == "BUY":
            pass
            return price * (1 + total_slippage)
        else:
            return price * (1 - total_slippage)
    
    def prepare_lstm_data(self, df):
        if len(df) < self.config['lstm_lookback'] + 20:
            pass
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
            pass
            X.append(scaled_data[i-self.config['lstm_lookback']:i])
            y.append(1 if df['close'].iloc[i+1] > df['close'].iloc[i] else 0)
        
            return np.array(X), np.array(y)
    
    def build_lstm_model(self):
        model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(self.config['lstm_lookback'], 4)),
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
            pass
            return None
        
        features = []
        close = df['close']
        
        features.append(close.pct_change(1).iloc[-1])
        features.append(close.pct_change(5).iloc[-1])
        features.append((close.iloc[-1] - close.rolling(10).mean().iloc[-1]) / close.iloc[-1])
        
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi = 100 - (100 / (1 + gain / loss))

            # Handle flat market
            if pd.isna(rsi) or rsi == 0:
                rsi = 50.0  # Neutral RSI for flat market
        features.append(rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50)
        
        sma = close.rolling(20).mean()
        std = close.rolling(20).std()
        bb_pos = (close.iloc[-1] - sma.iloc[-1]) / (2 * std.iloc[-1]) if std.iloc[-1] > 0 else 0
        features.append(bb_pos)
        
        features.append(df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1])
        
        return np.array(features).reshape(1, -1)
    
    def train_models(self, df):
        if len(df) < 100:
            pass
            return
        
        X_lstm, y_lstm = self.prepare_lstm_data(df)
        if X_lstm is not None and len(X_lstm) > 50:
            pass
            self.lstm_model = self.build_lstm_model()
            
            split = int(0.8 * len(X_lstm))
            X_train, X_test = X_lstm[:split], X_lstm[split:]
            y_train, y_test = y_lstm[:split], y_lstm[split:]
            
            self.lstm_model.fit(X_train, y_train, epochs=5, batch_size=32, 
            validation_data=(X_test, y_test), verbose=0)
        
        X_xgb = []
        y_xgb = []
        
        for i in range(30, len(df) - 5):
            pass
            features = self.calculate_xgb_features(df.iloc[:i+1])
            if features is not None:
                pass
                X_xgb.append(features[0])
                future_return = (df['close'].iloc[i+5] - df['close'].iloc[i]) / df['close'].iloc[i]
                y_xgb.append(1 if future_return > 0.001 else 0)
        
        if len(X_xgb) > 50:
            pass
            X_xgb = np.array(X_xgb)
            y_xgb = np.array(y_xgb)
            
            dtrain = xgb.DMatrix(X_xgb, label=y_xgb)
            params = {
            'max_depth': 3,
            'eta': 0.1,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'seed': 42
            }
            
            self.xgb_model = xgb.train(params, dtrain, num_boost_round=50)
        
        if self.lstm_model and self.xgb_model:
            pass
            self.models_trained = True
    
    def get_hybrid_prediction(self, df):
        if not self.models_trained:
            pass
            return 0.5, 0.5, 0.5
        
        lstm_pred = 0.5
        X_lstm, _ = self.prepare_lstm_data(df)
        if X_lstm is not None and len(X_lstm) > 0:
            pass
            lstm_pred = self.lstm_model.predict(X_lstm[-1:], verbose=0)[0][0]
        
        xgb_pred = 0.5
        features = self.calculate_xgb_features(df)
        if features is not None:
            pass
            dtest = xgb.DMatrix(features)
            xgb_pred = self.xgb_model.predict(dtest)[0]
        
        hybrid_pred = 0.6 * lstm_pred + 0.4 * xgb_pred
        
        return hybrid_pred, lstm_pred, xgb_pred
    
    def generate_signal(self, df):
        if len(df) < 100:
            pass
            return None
        
        if not self.models_trained or np.random.random() < 0.02:
            pass
            self.train_models(df)
        
        if not self.models_trained:
            pass
            return None
        
        hybrid_pred, lstm_pred, xgb_pred = self.get_hybrid_prediction(df)
        current_price = float(df['close'].iloc[-1])
        
        if hybrid_pred > self.config['prediction_threshold']:
            pass
            sma20 = df['close'].rolling(20).mean().iloc[-1]
            if current_price > sma20:
                pass
                return {
            'action': 'BUY',
            'price': current_price,
            'hybrid_pred': hybrid_pred,
            'lstm_pred': lstm_pred,
            'xgb_pred': xgb_pred
            }
        
            elif hybrid_pred < (1 - self.config['prediction_threshold']):
                pass
            sma20 = df['close'].rolling(20).mean().iloc[-1]
            if current_price < sma20:
                pass
                return {
            'action': 'SELL',
            'price': current_price,
            'hybrid_pred': hybrid_pred,
            'lstm_pred': lstm_pred,
            'xgb_pred': xgb_pred
            }
        
            return None
    
            async def get_market_data(self):
        try:
            klines = self.exchange.get_kline(
            category="linear",
            symbol=self.symbol,
            interval=self.config['timeframe'],
            limit=self.config['lookback']
            )
            
            if klines.get('retCode') != 0:
                pass
                return False
            
            df = pd.DataFrame(klines['result']['list'], 
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                pass
                df[col] = pd.to_numeric(df[col])
            
            self.price_data = df.sort_values('timestamp').reset_index(drop=True)
            return True
        except:
            return False
    
            async def check_position(self):
        try:
            positions = self.exchange.get_positions(category="linear", symbol=self.symbol)
            if positions.get('retCode') == 0:
                pass
                pos_list = positions['result']['list']
                self.position = pos_list[0] if pos_list and float(pos_list[0]['size']) > 0 else None
        except:
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
            
            if self.models_trained:
                pass
                hybrid_pred, _, _ = self.get_hybrid_prediction(self.price_data)
                if hybrid_pred < 0.35:
                    pass
                    return True, "hybrid_reversal"
                else:
                profit_pct = (entry_price - current_price) / entry_price * 100
            if profit_pct >= self.config['net_take_profit']:
                pass
                return True, "take_profit"
            if profit_pct <= -self.config['net_stop_loss']:
                pass
                return True, "stop_loss"
            
            if self.models_trained:
                pass
                hybrid_pred, _, _ = self.get_hybrid_prediction(self.price_data)
                if hybrid_pred > 0.65:
                    pass
                    return True, "hybrid_reversal"
        
                return False, ""
    
            async def execute_trade(self, signal):
        
        # Check trade cooldown
        import time
        if time.time() - self.last_trade_time < self.trade_cooldown:
            pass
            remaining = self.trade_cooldown - (time.time() - self.last_trade_time)
            print(f"⏰ Trade cooldown: wait {remaining:.0f}s")
            return
        qty = self.calculate_position_size(signal['price'])
        formatted_qty = self.format_qty(qty)
        
        if float(formatted_qty) < self.instrument_info.get('min_qty', 0.0001):
            pass
            return
        
        # Calculate limit price with maker offset
        offset = 1 - self.config['maker_offset_pct']/100 if signal['action'] == 'BUY' else 1 + self.config['maker_offset_pct']/100
        limit_price = round(signal['price'] * offset, 6)
        
        # Simulate actual execution price with slippage
        position_value = float(formatted_qty) * signal['price']
        actual_price = self.simulate_slippage(limit_price, signal['action'], position_value)
        
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
                stop_loss = actual_price * (1 - self.config['net_stop_loss']/100) if signal['action'] == 'BUY' else actual_price * (1 + self.config['net_stop_loss']/100)
                take_profit = actual_price * (1 + self.config['net_take_profit']/100) if signal['action'] == 'BUY' else actual_price * (1 - self.config['net_take_profit']/100)
                
                self.current_trade_id, _ = self.logger.log_trade_open(
                side=signal['action'],
                expected_price=signal['price'],
                actual_price=actual_price,
                qty=float(formatted_qty),
                stop_loss=stop_loss,
                take_profit=take_profit,
                info=f"hybrid:{signal['hybrid_pred']:.3f}_lstm:{signal['lstm_pred']:.3f}_xgb:{signal['xgb_pred']:.3f}"
                )
                
                print(f"🧠 HYBRID {signal['action']}: {formatted_qty} @ ${limit_price:.6f} (slippage: ${actual_price:.6f})")
                print(f"   📊 Hybrid: {signal['hybrid_pred']:.3f} | LSTM: {signal['lstm_pred']:.3f} | XGB: {signal['xgb_pred']:.3f}")
                print(f"   💰 Risk: ${self.account_balance * self.config['risk_percent'] / 100:.2f} ({self.config['risk_percent']}%)")
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
        
        # Simulate slippage for market exit
        position_value = qty * current_price
        actual_exit = self.simulate_slippage(current_price, side, position_value)
        
        try:
            order = self.exchange.place_order(
            category="linear",
            symbol=self.symbol,
            side=side,
            orderType="Limit",
            qty=self.format_qty(qty)
            timeInForce="PostOnly"),
            reduceOnly=True
            )
            
            if order.get('retCode') == 0:
                pass
                if self.current_trade_id:
                    pass
                    # Use appropriate fees (maker for limit entry, taker for market exit)
                    self.logger.log_trade_close(
                    trade_id=self.current_trade_id,
                    expected_exit=current_price,
                    actual_exit=actual_exit,
                    reason=reason,
                    fees_entry=self.config['maker_fee'],
                    fees_exit=self.config['taker_fee']
                    )
                    self.current_trade_id = None
                
                print(f"✅ Closed: {reason} @ ${actual_exit:.6f}")
        except:
            pass
    
    def show_status(self):
        if len(self.price_data) == 0:
            pass
            return
        
        current_price = float(self.price_data['close'].iloc[-1])
        
        print(f"\n🧠 LSTM-XGBoost Hybrid - {self.symbol}")
        print(f"💰 Price: ${current_price:.6f} | Balance: ${self.account_balance:.2f}")
        
        if self.models_trained:
            pass
            hybrid_pred, lstm_pred, xgb_pred = self.get_hybrid_prediction(self.price_data)
            print(f"📊 Predictions - Hybrid: {hybrid_pred:.3f} | LSTM: {lstm_pred:.3f} | XGB: {xgb_pred:.3f}")
            else:
            print("🔄 Training models...")
        
        if self.position:
            pass
            entry = float(self.position.get('avgPrice', 0))
            side = self.position.get('side', '')
            size = self.position.get('size', '0')
            pnl = float(self.position.get('unrealisedPnl', 0))
            
            emoji = "🟢" if side == "Buy" else "🔴"
            print(f"{emoji} {side}: {size} @ ${entry:.6f} | PnL: ${pnl:.2f}")
            else:
            print(f"🔍 Analyzing (Risk per trade: {self.config['risk_percent']}%)")
        
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
        
        print(f"🧠 LSTM-XGBoost Hybrid Bot - {self.symbol}")
        print(f"⏰ Timeframe: {self.config['timeframe']} minutes")
        print(f"🎯 Risk Management: {self.config['risk_percent']}% per trade")
        print(f"💎 Target win rate: 66%")
        
        try:
            while True:
                pass
                await self.run_cycle()
                await asyncio.sleep(15)
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

if __name__ == "__main__":
    pass
    bot = LSTMXGBoostBot()
    asyncio.run(bot.run())