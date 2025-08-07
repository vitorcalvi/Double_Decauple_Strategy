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
    
    def log_trade_close(self, trade_id, expected_exit, actual_exit, reason, fees_entry=0.04, fees_exit=0.04):
        if trade_id not in self.open_trades:
            return None
            
        trade = self.open_trades[trade_id]
        duration = (datetime.now() - trade["entry_time"]).total_seconds()
        
        slippage = actual_exit - expected_exit if trade["side"] == "SELL" else expected_exit - actual_exit
        
        if trade["side"] == "BUY":
            gross_pnl = (actual_exit - trade["entry_price"]) * trade["qty"]
        else:
            gross_pnl = (trade["entry_price"] - actual_exit) * trade["qty"]
        
        # Fix: Fees are costs, not rebates
        entry_fee = trade["entry_price"] * trade["qty"] * fees_entry / 100
        exit_fee = actual_exit * trade["qty"] * fees_exit / 100
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
            "fees": round(entry_fee + exit_fee, 2),
            "reason": reason,
            "currency": self.currency
        }
        
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        
        del self.open_trades[trade_id]
        return log_entry

class LSTMXGBoostBot:
    def __init__(self):
        self.symbol = 'AVAXUSDT'
        self.demo_mode = os.getenv('DEMO_MODE', 'true').lower() == 'true'
        
        prefix = 'TESTNET_' if self.demo_mode else 'LIVE_'
        self.api_key = os.getenv(f'{prefix}BYBIT_API_KEY')
        self.api_secret = os.getenv(f'{prefix}BYBIT_API_SECRET')
        self.exchange = None
        
        self.position = None
        self.price_data = pd.DataFrame()
        
        # Account and instrument info
        self.account_balance = 0
        self.instrument_info = {}
        self.qty_step = 0.0001
        self.price_precision = 6
        
        self.config = {
            'timeframe': '30',
            'risk_pct': 1.0,  # Risk 1% of account per trade
            'max_position_value_pct': 10.0,  # Max 10% of account in one position
            'maker_offset_pct': 0.01,
            'net_take_profit': 0.9,
            'net_stop_loss': 0.45,
            'lstm_lookback': 60,
            'prediction_threshold': 0.6,
            'lookback': 200,
            'maker_fee': -0.01,  # Negative for rebate
            'taker_fee': 0.06,   # Positive for cost
            'slippage_base_pct': 0.02,  # Base slippage
            'slippage_vol_multiplier': 0.5  # Volatility multiplier for slippage
        }
        
        self.lstm_model = None
        self.xgb_model = None
        self.scaler = MinMaxScaler()
        self.models_trained = False
        
        self.logger = TradeLogger("LSTM_XGBOOST", self.symbol)
        self.current_trade_id = None
        self.last_volatility = 0
    
    def connect(self):
        try:
            self.exchange = HTTP(demo=self.demo_mode, api_key=self.api_key, api_secret=self.api_secret)
            return self.exchange.get_server_time().get('retCode') == 0
        except:
            return False
    
    async def fetch_account_info(self):
        """Fetch account balance and instrument info"""
        try:
            # Get account balance
            wallet = self.exchange.get_wallet_balance(accountType="UNIFIED", coin="USDT")
            if wallet.get('retCode') == 0:
                self.account_balance = float(wallet['result']['list'][0]['coin'][0]['walletBalance'])
            
            # Get instrument info for precision
            instruments = self.exchange.get_instruments_info(category="linear", symbol=self.symbol)
            if instruments.get('retCode') == 0:
                info = instruments['result']['list'][0]
                self.instrument_info = info
                self.qty_step = float(info['lotSizeFilter']['qtyStep'])
                price_filter = info['priceFilter']
                tick_size = float(price_filter['tickSize'])
                self.price_precision = len(str(tick_size).split('.')[-1]) if '.' in str(tick_size) else 0
                
            return True
        except:
            return False
    
    def calculate_slippage(self, side, volume_ratio=1.0):
        """Calculate dynamic slippage based on volatility and volume"""
        if len(self.price_data) < 20:
            return self.config['slippage_base_pct'] / 100
        
        # Calculate recent volatility
        returns = self.price_data['close'].pct_change().dropna()
        volatility = returns.rolling(20).std().iloc[-1] if len(returns) > 20 else 0.001
        self.last_volatility = volatility
        
        # Calculate volume impact
        avg_volume = self.price_data['volume'].rolling(20).mean().iloc[-1]
        current_volume = self.price_data['volume'].iloc[-1]
        volume_impact = max(0, (volume_ratio - current_volume/avg_volume) * 0.01)
        
        # Calculate total slippage
        slippage = self.config['slippage_base_pct'] / 100
        slippage += volatility * self.config['slippage_vol_multiplier']
        slippage += volume_impact
        
        # More slippage for taker orders
        return slippage * 1.5 if side == "market" else slippage
    
    def calculate_position_size(self, price, stop_loss_price):
        """Calculate position size based on risk management"""
        if self.account_balance <= 0:
            return 0
        
        # Calculate risk amount (1% of account)
        risk_amount = self.account_balance * (self.config['risk_pct'] / 100)
        
        # Calculate position size based on stop loss distance
        stop_distance = abs(price - stop_loss_price)
        if stop_distance == 0:
            return 0
        
        position_size = risk_amount / stop_distance
        
        # Apply max position value constraint
        max_position_value = self.account_balance * (self.config['max_position_value_pct'] / 100)
        max_position_size = max_position_value / price
        
        position_size = min(position_size, max_position_size)
        
        # Round to instrument precision
        position_size = np.floor(position_size / self.qty_step) * self.qty_step
        
        return position_size
    
    def format_qty(self, qty):
        """Format quantity to exchange precision"""
        return str(np.floor(qty / self.qty_step) * self.qty_step)
    
    def format_price(self, price):
        """Format price to exchange precision"""
        return str(round(price, self.price_precision))
    
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
        # Calculate stop loss price
        stop_loss_price = signal['price'] * (1 - self.config['net_stop_loss']/100) if signal['action'] == 'BUY' else signal['price'] * (1 + self.config['net_stop_loss']/100)
        
        # Calculate position size based on risk
        qty = self.calculate_position_size(signal['price'], stop_loss_price)
        
        if qty < self.qty_step:
            print(f"⚠️ Position size too small: {qty}")
            return
        
        formatted_qty = self.format_qty(qty)
        
        # Calculate slippage
        slippage = self.calculate_slippage("limit")
        
        # Apply maker offset and slippage
        offset = 1 - self.config['maker_offset_pct']/100 if signal['action'] == 'BUY' else 1 + self.config['maker_offset_pct']/100
        limit_price = signal['price'] * offset
        
        # Apply slippage to expected execution
        if signal['action'] == 'BUY':
            limit_price = limit_price * (1 + slippage)
        else:
            limit_price = limit_price * (1 - slippage)
        
        limit_price = float(self.format_price(limit_price))
        
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
                take_profit = limit_price * (1 + self.config['net_take_profit']/100) if signal['action'] == 'BUY' else limit_price * (1 - self.config['net_take_profit']/100)
                
                self.current_trade_id, _ = self.logger.log_trade_open(
                    side=signal['action'],
                    expected_price=signal['price'],
                    actual_price=limit_price,
                    qty=float(formatted_qty),
                    stop_loss=stop_loss_price,
                    take_profit=take_profit,
                    info=f"hybrid:{signal['hybrid_pred']:.3f}_lstm:{signal['lstm_pred']:.3f}_xgb:{signal['xgb_pred']:.3f}_vol:{self.last_volatility:.4f}"
                )
                
                position_value = float(formatted_qty) * limit_price
                risk_pct = (position_value / self.account_balance) * 100 if self.account_balance > 0 else 0
                
                print(f"🧠 HYBRID {signal['action']}: {formatted_qty} @ ${limit_price:.6f}")
                print(f"   📊 Predictions: H:{signal['hybrid_pred']:.3f} | L:{signal['lstm_pred']:.3f} | X:{signal['xgb_pred']:.3f}")
                print(f"   💰 Position: ${position_value:.2f} ({risk_pct:.1f}% of account)")
                print(f"   📈 Slippage: {slippage*100:.3f}% | Volatility: {self.last_volatility:.4f}")
        except Exception as e:
            print(f"❌ Trade failed: {e}")
    
    async def close_position(self, reason):
        if not self.position:
            return
        
        current_price = float(self.price_data['close'].iloc[-1])
        side = "Sell" if self.position.get('side') == "Buy" else "Buy"
        qty = float(self.position['size'])
        
        # Calculate market slippage
        slippage = self.calculate_slippage("market")
        
        # Apply slippage to expected execution
        if side == "Sell":
            actual_price = current_price * (1 - slippage)
        else:
            actual_price = current_price * (1 + slippage)
        
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
                    # Use taker fee for market orders
                    self.logger.log_trade_close(
                        trade_id=self.current_trade_id,
                        expected_exit=current_price,
                        actual_exit=actual_price,
                        reason=reason,
                        fees_entry=abs(self.config['maker_fee']) if self.config['maker_fee'] < 0 else self.config['maker_fee'],
                        fees_exit=self.config['taker_fee']
                    )
                    self.current_trade_id = None
                
                print(f"✅ Closed: {reason} | Slippage: {slippage*100:.3f}%")
        except:
            pass
    
    def show_status(self):
        if len(self.price_data) == 0:
            return
        
        current_price = float(self.price_data['close'].iloc[-1])
        
        print(f"\n🧠 LSTM-XGBoost Hybrid - {self.symbol}")
        print(f"💰 Price: ${current_price:.6f} | Balance: ${self.account_balance:.2f}")
        
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
            
            # Calculate fees
            position_value = float(size) * entry
            entry_fee = position_value * abs(self.config['maker_fee']) / 100
            exit_fee = position_value * self.config['taker_fee'] / 100
            net_pnl = pnl - entry_fee - exit_fee
            
            emoji = "🟢" if side == "Buy" else "🔴"
            print(f"{emoji} {side}: {size} @ ${entry:.6f}")
            print(f"   📊 Gross PnL: ${pnl:.2f} | Net PnL: ${net_pnl:.2f} (after fees)")
        else:
            print("🔍 Analyzing patterns... | Volatility: {:.4f}".format(self.last_volatility))
        
        print("-" * 50)
    
    async def run_cycle(self):
        if not await self.get_market_data():
            return
        
        # Update account info periodically
        if np.random.random() < 0.1:  # 10% chance each cycle
            await self.fetch_account_info()
        
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
        
        # Initial account setup
        if not await self.fetch_account_info():
            print("❌ Failed to fetch account info")
            return
        
        print(f"🧠 LSTM-XGBoost Hybrid Bot - {self.symbol}")
        print(f"💰 Account Balance: ${self.account_balance:.2f}")
        print(f"⚖️ Risk per trade: {self.config['risk_pct']}%")
        print(f"📏 Quantity Step: {self.qty_step}")
        print(f"⏰ Timeframe: {self.config['timeframe']} minutes")
        print(f"🎯 Target win rate: 66%")
        
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