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
        net_pnl = gross_pnl + entry_rebate + exit_rebate
        
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
            f.write(json.dumps(log_entry) + "\n")
        
        del self.open_trades[trade_id]
        return log_entry

class LSTMXGBoostBot:
    def __init__(self):
        self.symbol = 'JPYUSDT'
        self.demo_mode = os.getenv('DEMO_MODE', 'true').lower() == 'true'
        
        prefix = 'TESTNET_' if self.demo_mode else 'LIVE_'
        self.api_key = os.getenv(f'{prefix}BYBIT_API_KEY')
        self.api_secret = os.getenv(f'{prefix}BYBIT_API_SECRET')
        self.exchange = None
        
        self.position = None
        self.price_data = pd.DataFrame()
        
        self.config = {
            'timeframe': '30',
            'position_size': 100,
            'maker_offset_pct': 0.01,
            'net_take_profit': 0.9,
            'net_stop_loss': 0.45,
            'lstm_lookback': 60,
            'prediction_threshold': 0.6,
            'lookback': 200
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
            return self.exchange.get_server_time().get('retCode') == 0
        except:
            return False
    
    def format_qty(self, qty):
        return str(int(round(qty * 10000)) / 10000)
    
    def prepare_lstm_data(self, df):
        if len(df) < self.config['lstm_lookback'] + 20:
            return None, None
        
        # Prepare features
        features = pd.DataFrame()
        features['returns'] = df['close'].pct_change()
        features['volume'] = df['volume'] / df['volume'].rolling(20).mean()
        features['high_low'] = (df['high'] - df['low']) / df['close']
        features['close_open'] = (df['close'] - df['open']) / df['open']
        
        features = features.fillna(0)
        
        # Scale data
        scaled_data = self.scaler.fit_transform(features)
        
        # Create sequences
        X, y = [], []
        for i in range(self.config['lstm_lookback'], len(scaled_data) - 1):
            X.append(scaled_data[i-self.config['lstm_lookback']:i])
            # Target: next return
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
            return None
        
        features = []
        close = df['close']
        
        # Short-term features
        features.append(close.pct_change(1).iloc[-1])
        features.append(close.pct_change(5).iloc[-1])
        features.append((close.iloc[-1] - close.rolling(10).mean().iloc[-1]) / close.iloc[-1])
        
        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi = 100 - (100 / (1 + gain / loss))
        features.append(rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50)
        
        # Bollinger position
        sma = close.rolling(20).mean()
        std = close.rolling(20).std()
        bb_pos = (close.iloc[-1] - sma.iloc[-1]) / (2 * std.iloc[-1]) if std.iloc[-1] > 0 else 0
        features.append(bb_pos)
        
        # Volume
        features.append(df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1])
        
        return np.array(features).reshape(1, -1)
    
    def train_models(self, df):
        if len(df) < 100:
            return
        
        # Train LSTM
        X_lstm, y_lstm = self.prepare_lstm_data(df)
        if X_lstm is not None and len(X_lstm) > 50:
            # Build and train LSTM
            self.lstm_model = self.build_lstm_model()
            
            # Split data
            split = int(0.8 * len(X_lstm))
            X_train, X_test = X_lstm[:split], X_lstm[split:]
            y_train, y_test = y_lstm[:split], y_lstm[split:]
            
            # Train with minimal epochs for speed
            self.lstm_model.fit(X_train, y_train, epochs=5, batch_size=32, 
                              validation_data=(X_test, y_test), verbose=0)
        
        # Train XGBoost
        X_xgb = []
        y_xgb = []
        
        for i in range(30, len(df) - 5):
            features = self.calculate_xgb_features(df.iloc[:i+1])
            if features is not None:
                X_xgb.append(features[0])
                # Target
                future_return = (df['close'].iloc[i+5] - df['close'].iloc[i]) / df['close'].iloc[i]
                y_xgb.append(1 if future_return > 0.001 else 0)
        
        if len(X_xgb) > 50:
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
            self.models_trained = True
    
    def get_hybrid_prediction(self, df):
        if not self.models_trained:
            return 0.5, 0.5, 0.5
        
        # LSTM prediction
        lstm_pred = 0.5
        X_lstm, _ = self.prepare_lstm_data(df)
        if X_lstm is not None and len(X_lstm) > 0:
            lstm_pred = self.lstm_model.predict(X_lstm[-1:], verbose=0)[0][0]
        
        # XGBoost prediction
        xgb_pred = 0.5
        features = self.calculate_xgb_features(df)
        if features is not None:
            dtest = xgb.DMatrix(features)
            xgb_pred = self.xgb_model.predict(dtest)[0]
        
        # Hybrid prediction (weighted average)
        hybrid_pred = 0.6 * lstm_pred + 0.4 * xgb_pred
        
        return hybrid_pred, lstm_pred, xgb_pred
    
    def generate_signal(self, df):
        if len(df) < 100:
            return None
        
        # Train models periodically
        if not self.models_trained or np.random.random() < 0.02:
            self.train_models(df)
        
        if not self.models_trained:
            return None
        
        hybrid_pred, lstm_pred, xgb_pred = self.get_hybrid_prediction(df)
        current_price = float(df['close'].iloc[-1])
        
        # Buy signal
        if hybrid_pred > self.config['prediction_threshold']:
            # Confirm trend
            sma20 = df['close'].rolling(20).mean().iloc[-1]
            if current_price > sma20:
                return {
                    'action': 'BUY',
                    'price': current_price,
                    'hybrid_pred': hybrid_pred,
                    'lstm_pred': lstm_pred,
                    'xgb_pred': xgb_pred
                }
        
        # Sell signal
        elif hybrid_pred < (1 - self.config['prediction_threshold']):
            sma20 = df['close'].rolling(20).mean().iloc[-1]
            if current_price < sma20:
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
                return False
            
            df = pd.DataFrame(klines['result']['list'], 
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
            
            self.price_data = df.sort_values('timestamp').reset_index(drop=True)
            return True
        except:
            return False
    
    async def check_position(self):
        try:
            positions = self.exchange.get_positions(category="linear", symbol=self.symbol)
            if positions.get('retCode') == 0:
                pos_list = positions['result']['list']
                self.position = pos_list[0] if pos_list and float(pos_list[0]['size']) > 0 else None
        except:
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
            
            # Check hybrid prediction
            if self.models_trained:
                hybrid_pred, _, _ = self.get_hybrid_prediction(self.price_data)
                if hybrid_pred < 0.35:
                    return True, "hybrid_reversal"
        else:
            profit_pct = (entry_price - current_price) / entry_price * 100
            if profit_pct >= self.config['net_take_profit']:
                return True, "take_profit"
            if profit_pct <= -self.config['net_stop_loss']:
                return True, "stop_loss"
            
            if self.models_trained:
                hybrid_pred, _, _ = self.get_hybrid_prediction(self.price_data)
                if hybrid_pred > 0.65:
                    return True, "hybrid_reversal"
        
        return False, ""
    
    async def execute_trade(self, signal):
        qty = self.config['position_size'] / signal['price']
        formatted_qty = self.format_qty(qty)
        
        if float(formatted_qty) < 0.0001:
            return
        
        offset = 1 - self.config['maker_offset_pct']/100 if signal['action'] == 'BUY' else 1 + self.config['maker_offset_pct']/100
        limit_price = round(signal['price'] * offset, 6)
        
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
                stop_loss = limit_price * (1 - self.config['net_stop_loss']/100) if signal['action'] == 'BUY' else limit_price * (1 + self.config['net_stop_loss']/100)
                take_profit = limit_price * (1 + self.config['net_take_profit']/100) if signal['action'] == 'BUY' else limit_price * (1 - self.config['net_take_profit']/100)
                
                self.current_trade_id, _ = self.logger.log_trade_open(
                    side=signal['action'],
                    expected_price=signal['price'],
                    actual_price=limit_price,
                    qty=float(formatted_qty),
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    info=f"hybrid:{signal['hybrid_pred']:.3f}_lstm:{signal['lstm_pred']:.3f}_xgb:{signal['xgb_pred']:.3f}"
                )
                
                print(f"🧠 HYBRID {signal['action']}: {formatted_qty} @ ${limit_price:.6f}")
                print(f"   📊 Hybrid: {signal['hybrid_pred']:.3f} | LSTM: {signal['lstm_pred']:.3f} | XGB: {signal['xgb_pred']:.3f}")
        except Exception as e:
            print(f"❌ Trade failed: {e}")
    
    async def close_position(self, reason):
        if not self.position:
            return
        
        current_price = float(self.price_data['close'].iloc[-1])
        side = "Sell" if self.position.get('side') == "Buy" else "Buy"
        qty = float(self.position['size'])
        
        try:
            order = self.exchange.place_order(
                category="linear",
                symbol=self.symbol,
                side=side,
                orderType="Market",
                qty=self.format_qty(qty),
                reduceOnly=True
            )
            
            if order.get('retCode') == 0:
                if self.current_trade_id:
                    self.logger.log_trade_close(
                        trade_id=self.current_trade_id,
                        expected_exit=current_price,
                        actual_exit=current_price,
                        reason=reason
                    )
                    self.current_trade_id = None
                
                print(f"✅ Closed: {reason}")
        except:
            pass
    
    def show_status(self):
        if len(self.price_data) == 0:
            return
        
        current_price = float(self.price_data['close'].iloc[-1])
        
        print(f"\n🧠 LSTM-XGBoost Hybrid - {self.symbol}")
        print(f"💰 Price: ${current_price:.6f}")
        
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
            
            emoji = "🟢" if side == "Buy" else "🔴"
            print(f"{emoji} {side}: {size} @ ${entry:.6f} | PnL: ${pnl:.2f}")
        else:
            print("🔍 Analyzing long-term patterns...")
        
        print("-" * 50)
    
    async def run_cycle(self):
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
        
        print(f"🧠 LSTM-XGBoost Hybrid Bot - {self.symbol}")
        print(f"⏰ Timeframe: {self.config['timeframe']} minutes")
        print(f"🎯 LSTM for long-term + XGBoost for short-term")
        print(f"💎 Target win rate: 66%")
        
        try:
            while True:
                await self.run_cycle()
                await asyncio.sleep(15)
        except KeyboardInterrupt:
            print("\n🛑 Bot stopped")
            if self.position:
                await self.close_position("manual_stop")

if __name__ == "__main__":
    bot = LSTMXGBoostBot()
    asyncio.run(bot.run())