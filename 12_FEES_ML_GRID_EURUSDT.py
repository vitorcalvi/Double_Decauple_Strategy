import os
import asyncio
import pandas as pd
import numpy as np
import json
from datetime import datetime, timezone
from pybit.unified_trading import HTTP
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
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

class MLGridBot:
    def __init__(self):
        self.symbol = 'EURUSDT'
        self.demo_mode = os.getenv('DEMO_MODE', 'true').lower() == 'true'
        
        prefix = 'TESTNET_' if self.demo_mode else 'LIVE_'
        self.api_key = os.getenv(f'{prefix}BYBIT_API_KEY')
        self.api_secret = os.getenv(f'{prefix}BYBIT_API_SECRET')
        self.exchange = None
        
        self.position = None
        self.price_data = pd.DataFrame()
        
        self.config = {
            'timeframe': '5',
            'base_grid_spacing': 0.5,  # 0.5% base spacing
            'grid_levels': 5,
            'position_size': 100,
            'maker_offset_pct': 0.01,
            'net_take_profit': 0.75,
            'net_stop_loss': 0.3,
            'ml_threshold': 0.65,
            'lookback': 100
        }
        
        self.ml_model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
        self.scaler = StandardScaler()
        self.model_trained = False
        self.grid_levels = []
        self.last_grid_level = None
        
        self.logger = TradeLogger("ML_GRID", self.symbol)
        self.current_trade_id = None
    
    def connect(self):
        try:
            self.exchange = HTTP(demo=self.demo_mode, api_key=self.api_key, api_secret=self.api_secret)
            return self.exchange.get_server_time().get('retCode') == 0
        except:
            return False
    
    def format_qty(self, qty):
        return f"{round(qty / 0.01) * 0.01:.2f}"
    
    def prepare_features(self, df):
        if len(df) < 30:
            return None
        
        features = []
        
        # Price features
        features.append(df['close'].pct_change().iloc[-1])
        features.append((df['high'].iloc[-1] - df['low'].iloc[-1]) / df['close'].iloc[-1])
        features.append((df['close'].iloc[-1] - df['close'].rolling(10).mean().iloc[-1]) / df['close'].iloc[-1])
        
        # Volume features
        features.append(df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1])
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi = 100 - (100 / (1 + gain / loss))
        features.append(rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50)
        
        # Volatility
        features.append(df['close'].rolling(20).std().iloc[-1] / df['close'].iloc[-1])
        
        return np.array(features).reshape(1, -1)
    
    def train_ml_model(self, df):
        if len(df) < 50:
            return
        
        X = []
        y = []
        
        for i in range(30, len(df) - 10):
            features = self.prepare_features(df.iloc[:i+1])
            if features is not None:
                X.append(features[0])
                # Target: optimal grid spacing based on future volatility
                future_volatility = df['close'].iloc[i:i+10].std() / df['close'].iloc[i]
                optimal_spacing = min(max(0.3, future_volatility * 100), 1.0)
                y.append(optimal_spacing)
        
        if len(X) > 20:
            X = np.array(X)
            y = np.array(y)
            
            self.scaler.fit(X)
            X_scaled = self.scaler.transform(X)
            
            self.ml_model.fit(X_scaled, y)
            self.model_trained = True
    
    def predict_grid_spacing(self, df):
        if not self.model_trained:
            return self.config['base_grid_spacing']
        
        features = self.prepare_features(df)
        if features is None:
            return self.config['base_grid_spacing']
        
        features_scaled = self.scaler.transform(features)
        predicted_spacing = self.ml_model.predict(features_scaled)[0]
        
        return min(max(0.3, predicted_spacing), 1.0)
    
    def update_grid_levels(self, current_price, optimal_spacing):
        self.grid_levels = []
        spacing_pct = optimal_spacing / 100
        
        for i in range(-self.config['grid_levels'], self.config['grid_levels'] + 1):
            if i != 0:
                level = current_price * (1 + i * spacing_pct)
                self.grid_levels.append({
                    'price': level,
                    'index': i,
                    'side': 'BUY' if i < 0 else 'SELL'
                })
        
        self.grid_levels.sort(key=lambda x: x['price'])
    
    def get_ml_confidence(self, df):
        if not self.model_trained:
            return 0.5
        
        features = self.prepare_features(df)
        if features is None:
            return 0.5
        
        # Use model's prediction confidence
        features_scaled = self.scaler.transform(features)
        predictions = []
        
        for estimator in self.ml_model.estimators_[:10]:
            predictions.append(estimator.predict(features_scaled)[0])
        
        std_dev = np.std(predictions)
        confidence = 1 - min(std_dev * 2, 1)
        
        return confidence
    
    def generate_signal(self, df):
        if len(df) < 50:
            return None
        
        current_price = float(df['close'].iloc[-1])
        
        # Train/update ML model periodically
        if not self.model_trained or np.random.random() < 0.1:
            self.train_ml_model(df)
        
        # Get ML confidence
        ml_confidence = self.get_ml_confidence(df)
        if ml_confidence < self.config['ml_threshold']:
            return None
        
        # Predict optimal grid spacing
        optimal_spacing = self.predict_grid_spacing(df)
        self.update_grid_levels(current_price, optimal_spacing)
        
        # Find nearest grid level
        for level in self.grid_levels:
            distance_pct = abs(current_price - level['price']) / level['price'] * 100
            if distance_pct < 0.1 and level != self.last_grid_level:
                self.last_grid_level = level
                return {
                    'action': level['side'],
                    'price': current_price,
                    'grid_price': level['price'],
                    'ml_confidence': ml_confidence,
                    'grid_spacing': optimal_spacing
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
                return True, "grid_target"
            if profit_pct <= -self.config['net_stop_loss']:
                return True, "stop_loss"
        else:
            profit_pct = (entry_price - current_price) / entry_price * 100
            if profit_pct >= self.config['net_take_profit']:
                return True, "grid_target"
            if profit_pct <= -self.config['net_stop_loss']:
                return True, "stop_loss"
        
        return False, ""
    
    async def execute_trade(self, signal):
        qty = self.config['position_size'] / signal['price']
        formatted_qty = self.format_qty(qty)
        
        if float(formatted_qty) < 0.01:
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
                
                self.current_trade_id, _ = self.logger.log_trade_open(
                    side=signal['action'],
                    expected_price=signal['price'],
                    actual_price=limit_price,
                    qty=float(formatted_qty),
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    info=f"ml_conf:{signal['ml_confidence']:.2f}_spacing:{signal['grid_spacing']:.2f}%"
                )
                
                print(f"🤖 ML GRID {signal['action']}: {formatted_qty} @ ${limit_price:.5f}")
                print(f"   📊 ML Confidence: {signal['ml_confidence']:.2f}")
                print(f"   📏 Grid Spacing: {signal['grid_spacing']:.2f}%")
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
                self.last_grid_level = None
        except:
            pass
    
    def show_status(self):
        if len(self.price_data) == 0:
            return
        
        current_price = float(self.price_data['close'].iloc[-1])
        
        print(f"\n🤖 ML Grid Trading - {self.symbol}")
        print(f"💰 Price: ${current_price:.5f}")
        
        if self.model_trained:
            ml_conf = self.get_ml_confidence(self.price_data)
            spacing = self.predict_grid_spacing(self.price_data)
            print(f"📊 ML Confidence: {ml_conf:.2f} | Grid Spacing: {spacing:.2f}%")
        
        if self.position:
            entry = float(self.position.get('avgPrice', 0))
            side = self.position.get('side', '')
            size = self.position.get('size', '0')
            pnl = float(self.position.get('unrealisedPnl', 0))
            
            emoji = "🟢" if side == "Buy" else "🔴"
            print(f"{emoji} {side}: {size} @ ${entry:.5f} | PnL: ${pnl:.2f}")
        else:
            print("🔍 Scanning grid levels with ML optimization...")
        
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
        
        print(f"🤖 ML-Optimized Grid Trading Bot - {self.symbol}")
        print(f"⏰ Timeframe: {self.config['timeframe']} minutes")
        print(f"🎯 Dynamic grid spacing with ML optimization")
        print(f"💎 Target win rate: 73%")
        
        try:
            while True:
                await self.run_cycle()
                await asyncio.sleep(5)
        except KeyboardInterrupt:
            print("\n🛑 Bot stopped")
            if self.position:
                await self.close_position("manual_stop")

if __name__ == "__main__":
    bot = MLGridBot()
    asyncio.run(bot.run())