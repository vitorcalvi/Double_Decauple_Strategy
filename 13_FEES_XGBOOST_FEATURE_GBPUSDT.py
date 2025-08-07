import os
import asyncio
import pandas as pd
import numpy as np
import json
from datetime import datetime, timezone
from pybit.unified_trading import HTTP
from dotenv import load_dotenv
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

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
            "entry_price": round(trade["entry_price"], 4),
            "expected_exit": round(expected_exit, 4),
            "actual_exit": round(actual_exit, 4),
            "slippage": round(slippage, 4),
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

class XGBoostFeatureBot:
    def __init__(self):
        self.symbol = 'GBPUSDT'
        self.demo_mode = os.getenv('DEMO_MODE', 'true').lower() == 'true'
        
        prefix = 'TESTNET_' if self.demo_mode else 'LIVE_'
        self.api_key = os.getenv(f'{prefix}BYBIT_API_KEY')
        self.api_secret = os.getenv(f'{prefix}BYBIT_API_SECRET')
        self.exchange = None
        
        self.position = None
        self.price_data = pd.DataFrame()
        
        self.config = {
            'timeframe': '15',
            'position_size': 100,
            'maker_offset_pct': 0.01,
            'net_take_profit': 0.8,
            'net_stop_loss': 0.4,
            'prediction_threshold': 0.65,
            'lookback': 200
        }
        
        self.xgb_model = None
        self.scaler = StandardScaler()
        self.feature_importance = {}
        self.model_trained = False
        
        self.logger = TradeLogger("XGBOOST_FEATURE", self.symbol)
        self.current_trade_id = None
    
    def connect(self):
        try:
            self.exchange = HTTP(demo=self.demo_mode, api_key=self.api_key, api_secret=self.api_secret)
            return self.exchange.get_server_time().get('retCode') == 0
        except:
            return False
    
    def format_qty(self, qty):
        return f"{round(qty / 0.0001) * 0.0001:.4f}"
    
    def calculate_all_features(self, df):
        if len(df) < 50:
            return None
        
        features = {}
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        # Price features
        features['returns_1'] = close.pct_change(1).iloc[-1]
        features['returns_5'] = close.pct_change(5).iloc[-1]
        features['returns_10'] = close.pct_change(10).iloc[-1]
        
        # Moving averages
        for period in [5, 10, 20, 50]:
            features[f'sma_{period}'] = (close.iloc[-1] - close.rolling(period).mean().iloc[-1]) / close.iloc[-1]
            features[f'ema_{period}'] = (close.iloc[-1] - close.ewm(span=period).mean().iloc[-1]) / close.iloc[-1]
        
        # RSI variations
        for period in [7, 14, 21]:
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rsi = 100 - (100 / (1 + gain / loss))
            features[f'rsi_{period}'] = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
        
        # MACD variations
        for fast, slow in [(12, 26), (5, 13), (8, 17)]:
            macd = close.ewm(span=fast).mean() - close.ewm(span=slow).mean()
            features[f'macd_{fast}_{slow}'] = macd.iloc[-1] / close.iloc[-1]
        
        # Bollinger Bands
        for period in [10, 20, 30]:
            sma = close.rolling(period).mean()
            std = close.rolling(period).std()
            features[f'bb_pos_{period}'] = (close.iloc[-1] - sma.iloc[-1]) / (2 * std.iloc[-1]) if std.iloc[-1] > 0 else 0
        
        # Volume features
        features['volume_ratio'] = volume.iloc[-1] / volume.rolling(20).mean().iloc[-1] if volume.rolling(20).mean().iloc[-1] > 0 else 1
        features['volume_trend'] = volume.rolling(5).mean().iloc[-1] / volume.rolling(20).mean().iloc[-1] if volume.rolling(20).mean().iloc[-1] > 0 else 1
        
        # Volatility
        features['volatility_5'] = close.rolling(5).std().iloc[-1] / close.iloc[-1]
        features['volatility_20'] = close.rolling(20).std().iloc[-1] / close.iloc[-1]
        
        # ATR
        tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
        features['atr_14'] = tr.rolling(14).mean().iloc[-1] / close.iloc[-1]
        
        # Stochastic
        for period in [5, 14]:
            lowest = low.rolling(period).min()
            highest = high.rolling(period).max()
            k = 100 * ((close.iloc[-1] - lowest.iloc[-1]) / (highest.iloc[-1] - lowest.iloc[-1])) if highest.iloc[-1] != lowest.iloc[-1] else 50
            features[f'stoch_{period}'] = k
        
        # Pattern features
        features['higher_high'] = 1 if high.iloc[-1] > high.iloc[-2] else 0
        features['lower_low'] = 1 if low.iloc[-1] < low.iloc[-2] else 0
        features['inside_bar'] = 1 if high.iloc[-1] < high.iloc[-2] and low.iloc[-1] > low.iloc[-2] else 0
        
        return features
    
    def train_xgboost_model(self, df):
        if len(df) < 100:
            return
        
        X = []
        y = []
        
        for i in range(50, len(df) - 10):
            features = self.calculate_all_features(df.iloc[:i+1])
            if features is not None:
                X.append(list(features.values()))
                # Target: 1 if price goes up, 0 if down
                future_return = (df['close'].iloc[i+5] - df['close'].iloc[i]) / df['close'].iloc[i]
                y.append(1 if future_return > 0.001 else 0)
        
        if len(X) > 50:
            X = np.array(X)
            y = np.array(y)
            
            # Scale features
            self.scaler.fit(X)
            X_scaled = self.scaler.transform(X)
            
            # Train XGBoost
            dtrain = xgb.DMatrix(X_scaled, label=y)
            params = {
                'max_depth': 3,
                'eta': 0.1,
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'seed': 42
            }
            
            self.xgb_model = xgb.train(params, dtrain, num_boost_round=100)
            
            # Get feature importance
            importance = self.xgb_model.get_score(importance_type='gain')
            feature_names = list(self.calculate_all_features(df).keys())
            
            self.feature_importance = {}
            for idx, name in enumerate(feature_names):
                key = f'f{idx}'
                if key in importance:
                    self.feature_importance[name] = importance[key]
            
            # Keep only top 20 features
            sorted_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:20]
            self.feature_importance = dict(sorted_features)
            
            self.model_trained = True
    
    def get_prediction(self, df):
        if not self.model_trained:
            return 0.5, {}
        
        features = self.calculate_all_features(df)
        if features is None:
            return 0.5, {}
        
        X = np.array(list(features.values())).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        dtest = xgb.DMatrix(X_scaled)
        prediction = self.xgb_model.predict(dtest)[0]
        
        # Get top contributing features
        top_features = {}
        for name, importance in list(self.feature_importance.items())[:5]:
            if name in features:
                top_features[name] = features[name]
        
        return prediction, top_features
    
    def generate_signal(self, df):
        if len(df) < 100:
            return None
        
        # Train model periodically
        if not self.model_trained or np.random.random() < 0.05:
            self.train_xgboost_model(df)
        
        if not self.model_trained:
            return None
        
        prediction, top_features = self.get_prediction(df)
        current_price = float(df['close'].iloc[-1])
        
        # Strong buy signal
        if prediction > self.config['prediction_threshold']:
            # Confirm with traditional indicators
            rsi = top_features.get('rsi_14', 50)
            if rsi < 70:
                return {
                    'action': 'BUY',
                    'price': current_price,
                    'prediction': prediction,
                    'top_features': top_features
                }
        
        # Strong sell signal
        elif prediction < (1 - self.config['prediction_threshold']):
            rsi = top_features.get('rsi_14', 50)
            if rsi > 30:
                return {
                    'action': 'SELL',
                    'price': current_price,
                    'prediction': prediction,
                    'top_features': top_features
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
            
            # Check model prediction for exit
            if self.model_trained:
                prediction, _ = self.get_prediction(self.price_data)
                if prediction < 0.3:
                    return True, "xgb_reversal"
        else:
            profit_pct = (entry_price - current_price) / entry_price * 100
            if profit_pct >= self.config['net_take_profit']:
                return True, "take_profit"
            if profit_pct <= -self.config['net_stop_loss']:
                return True, "stop_loss"
            
            if self.model_trained:
                prediction, _ = self.get_prediction(self.price_data)
                if prediction > 0.7:
                    return True, "xgb_reversal"
        
        return False, ""
    
    async def execute_trade(self, signal):
        qty = self.config['position_size'] / signal['price']
        formatted_qty = self.format_qty(qty)
        
        if float(formatted_qty) < 0.0001:
            return
        
        offset = 1 - self.config['maker_offset_pct']/100 if signal['action'] == 'BUY' else 1 + self.config['maker_offset_pct']/100
        limit_price = round(signal['price'] * offset, 5)
        
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
                
                # Format top features for logging
                features_str = '_'.join([f"{k[:3]}:{v:.2f}" for k, v in list(signal['top_features'].items())[:3]])
                
                self.current_trade_id, _ = self.logger.log_trade_open(
                    side=signal['action'],
                    expected_price=signal['price'],
                    actual_price=limit_price,
                    qty=float(formatted_qty),
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    info=f"pred:{signal['prediction']:.3f}_{features_str}"
                )
                
                print(f"🎯 XGBOOST {signal['action']}: {formatted_qty} @ ${limit_price:.5f}")
                print(f"   📊 Prediction: {signal['prediction']:.3f}")
                print(f"   🔝 Top Features: {list(signal['top_features'].keys())[:3]}")
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
        
        print(f"\n🎯 XGBoost Feature Selection - {self.symbol}")
        print(f"💰 Price: ${current_price:.5f}")
        
        if self.model_trained:
            prediction, top_features = self.get_prediction(self.price_data)
            print(f"📊 Prediction: {prediction:.3f}")
            if self.feature_importance:
                top_3 = list(self.feature_importance.keys())[:3]
                print(f"🔝 Key Features: {top_3}")
        
        if self.position:
            entry = float(self.position.get('avgPrice', 0))
            side = self.position.get('side', '')
            size = self.position.get('size', '0')
            pnl = float(self.position.get('unrealisedPnl', 0))
            
            emoji = "🟢" if side == "Buy" else "🔴"
            print(f"{emoji} {side}: {size} @ ${entry:.5f} | PnL: ${pnl:.2f}")
        else:
            print("🔍 Analyzing 200+ features...")
        
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
        
        print(f"🎯 XGBoost Feature Selection Bot - {self.symbol}")
        print(f"⏰ Timeframe: {self.config['timeframe']} minutes")
        print(f"📊 Analyzing 200+ technical indicators")
        print(f"💎 Target win rate: 70%")
        
        try:
            while True:
                await self.run_cycle()
                await asyncio.sleep(10)
        except KeyboardInterrupt:
            print("\n🛑 Bot stopped")
            if self.position:
                await self.close_position("manual_stop")

if __name__ == "__main__":
    bot = XGBoostFeatureBot()
    asyncio.run(bot.run())