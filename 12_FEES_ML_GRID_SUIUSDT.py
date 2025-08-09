import os
import asyncio
import pandas as pd
import numpy as np
import json
import time
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
        
        # Emergency stop tracking
        self.daily_pnl = 0
        self.consecutive_losses = 0
        self.max_daily_loss = 50
        
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
            "expected_price": round(expected_price, 5),
            "actual_price": round(actual_price, 5),
            "slippage": round(slippage, 5),
            "qty": round(qty, 6),
            "stop_loss": round(stop_loss, 5),
            "take_profit": round(take_profit, 5),
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
        
        # Proper slippage calculation
        slippage = (actual_exit - expected_exit if trade["side"] == "BUY"
                   else expected_exit - actual_exit)
        
        gross_pnl = ((actual_exit - trade["entry_price"]) * trade["qty"] if trade["side"] == "BUY"
                    else (trade["entry_price"] - actual_exit) * trade["qty"])
        
        # Proper fee/rebate calculation
        entry_rebate = (abs(fees_entry) / 100 * trade["entry_price"] * trade["qty"] if fees_entry < 0
                       else -fees_entry / 100 * trade["entry_price"] * trade["qty"])
        exit_rebate = (abs(fees_exit) / 100 * actual_exit * trade["qty"] if fees_exit < 0
                      else -fees_exit / 100 * actual_exit * trade["qty"])
        
        total_rebates = entry_rebate + exit_rebate
        net_pnl = gross_pnl + total_rebates
        
        log_entry = {
            "id": trade_id,
            "bot": self.bot_name,
            "symbol": self.symbol,
            "side": "LONG" if trade["side"] == "BUY" else "SHORT",
            "action": "CLOSE",
            "ts": datetime.now(timezone.utc).isoformat(),
            "duration_sec": int(duration),
            "entry_price": round(trade["entry_price"], 5),
            "expected_exit": round(expected_exit, 5),
            "actual_exit": round(actual_exit, 5),
            "slippage": round(slippage, 5),
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
        self.symbol = 'SUIUSDT'
        self.demo_mode = os.getenv('DEMO_MODE', 'true').lower() == 'true'
        
        prefix = 'TESTNET_' if self.demo_mode else 'LIVE_'
        self.api_key = os.getenv(f'{prefix}BYBIT_API_KEY')
        self.api_secret = os.getenv(f'{prefix}BYBIT_API_SECRET')
        self.exchange = None
        
        self.position = None
        self.price_data = pd.DataFrame()
        self.account_balance = 1000
        self.instrument_info = {}
        
        # Configuration
        self.config = {
            'timeframe': '5',
            'base_grid_spacing': 0.5,
            'grid_levels': 5,
            'risk_per_trade': 1.0,
            'maker_offset_pct': 0.01,
            'net_take_profit': 0.75,
            'net_stop_loss': 0.3,
            'ml_threshold': 0.60,
            'lookback': 100,
            'slippage_bps': 5
        }
        
        self.ml_model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
        self.scaler = StandardScaler()
        self.model_trained = False
        self.grid_levels = []
        self.last_grid_level = None
        
        # Trade cooldown
        self.last_trade_time = 0
        self.trade_cooldown = 30
        
        self.logger = TradeLogger("ML_GRID_FIXED", self.symbol)
        self.current_trade_id = None
    
    def connect(self):
        try:
            self.exchange = HTTP(demo=self.demo_mode, api_key=self.api_key, api_secret=self.api_secret)
            return self.exchange.get_server_time().get('retCode') == 0
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False
    
    async def get_account_balance(self):
        """Get account balance for position sizing"""
        try:
            wallet = self.exchange.get_wallet_balance(accountType="UNIFIED", coin="USDT")
            if wallet.get('retCode') == 0:
                balance_info = wallet['result']['list'][0]['coin'][0]
                self.account_balance = float(balance_info['walletBalance'])
                return True
        except Exception as e:
            print(f"❌ Balance error: {e}")
        return False
    
    async def get_instrument_info(self):
        """Get instrument precision for proper formatting"""
        try:
            instruments = self.exchange.get_instruments_info(category="linear", symbol=self.symbol)
            if instruments.get('retCode') == 0:
                info = instruments['result']['list'][0]
                self.instrument_info = {
                    'qty_step': float(info['lotSizeFilter']['qtyStep']),
                    'min_qty': float(info['lotSizeFilter']['minOrderQty']),
                    'price_precision': len(info['priceFilter']['tickSize'].split('.')[-1])
                }
                return True
        except Exception as e:
            print(f"❌ Instrument info error: {e}")
        return False
    
    def format_qty(self, qty):
        """Proper quantity formatting with instrument precision"""
        if not self.instrument_info:
            return "0.01"
        
        qty_step = self.instrument_info['qty_step']
        min_qty = self.instrument_info['min_qty']
        
        if qty < min_qty:
            return "0"
        
        rounded_qty = round(qty / qty_step) * qty_step
        
        if qty_step >= 1:
            return str(int(rounded_qty))
        else:
            decimals = len(str(qty_step).split('.')[-1])
            return f"{rounded_qty:.{decimals}f}"
    
    def calculate_position_size(self, price, stop_loss_price):
        """Risk-based position sizing"""
        if self.account_balance == 0:
            return 0
        
        risk_amount = self.account_balance * (self.config['risk_per_trade'] / 100)
        risk_per_unit = abs(price - stop_loss_price)
        
        if risk_per_unit == 0:
            return 0
        
        position_size = risk_amount / risk_per_unit
        return min(position_size, self.account_balance / price * 0.95)
    
    def apply_slippage(self, price, side, order_type="LIMIT"):
        """Realistic slippage modeling"""
        slippage_factor = self.config['slippage_bps'] / 10000
        
        if order_type == "MARKET":
            return price * (1 + slippage_factor) if side in ["BUY", "Buy"] else price * (1 - slippage_factor)
        else:
            # Limit orders get partial fill slippage
            return price * (1 + slippage_factor * 0.3) if side in ["BUY", "Buy"] else price * (1 - slippage_factor * 0.3)
    
    def prepare_features(self, df):
        if len(df) < 30:
            return None
        
        features = []
        
        # Price features
        features.append(df['close'].pct_change().iloc[-1] or 0)
        features.append((df['high'].iloc[-1] - df['low'].iloc[-1]) / df['close'].iloc[-1])
        
        sma_10 = df['close'].rolling(10).mean().iloc[-1]
        features.append((df['close'].iloc[-1] - sma_10) / df['close'].iloc[-1] if sma_10 > 0 else 0)
        
        # Volume features
        vol_sma = df['volume'].rolling(20).mean().iloc[-1]
        features.append(df['volume'].iloc[-1] / vol_sma if vol_sma > 0 else 1)
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi = 100 - (100 / (1 + gain / (loss + 1e-10)))
        features.append(rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50)
        
        # Volatility
        vol = df['close'].rolling(20).std().iloc[-1]
        features.append(vol / df['close'].iloc[-1] if vol > 0 else 0.01)
        
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
        
        features_scaled = self.scaler.transform(features)
        predictions = []
        
        for estimator in self.ml_model.estimators_[:10]:
            predictions.append(estimator.predict(features_scaled)[0])
        
        std_dev = np.std(predictions)
        confidence = max(0, 1 - min(std_dev * 2, 1))
        
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
        except Exception as e:
            print(f"❌ Market data error: {e}")
            return False
    
    async def check_position(self):
        """Enhanced position check with state cleanup"""
        try:
            positions = self.exchange.get_positions(category="linear", symbol=self.symbol)
            if positions.get('retCode') != 0:
                self.position = None
                return False
                
            pos_list = positions['result']['list']
            
            # Clear position if empty
            if not pos_list or float(pos_list[0].get('size', 0)) == 0:
                if self.position:  # Was set but now closed
                    print("✅ Position closed - clearing state")
                    self.position = None
                    self.pending_order = None  # Also clear pending
                return False
                
            # Valid position exists
            self.position = pos_list[0]
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
        side = self.position.get('side', '')
        
        if entry_price == 0:
            return False, ""
        
        profit_pct = ((current_price - entry_price) / entry_price * 100 if side == "Buy"
                     else (entry_price - current_price) / entry_price * 100)
        
        if profit_pct >= self.config['net_take_profit']:
            return True, "grid_target"
        if profit_pct <= -self.config['net_stop_loss']:
            return True, "stop_loss"
        
        return False, ""
    
    async def execute_trade(self, signal):
        # Check trade cooldown
        if time.time() - self.last_trade_time < self.trade_cooldown:
            remaining = self.trade_cooldown - (time.time() - self.last_trade_time)
            print(f"⏰ Trade cooldown: wait {remaining:.0f}s")
            return
        
        # Calculate stop loss price
        stop_loss_distance = self.config['net_stop_loss'] / 100
        stop_loss_price = (signal['price'] * (1 - stop_loss_distance) if signal['action'] == 'BUY'
                          else signal['price'] * (1 + stop_loss_distance))
        
        # Risk-based position sizing
        qty = self.calculate_position_size(signal['price'], stop_loss_price)
        formatted_qty = self.format_qty(qty)
        
        if formatted_qty == "0" or float(formatted_qty) == 0:
            print("❌ Position size too small or zero balance")
            return
        
        # Apply realistic slippage
        offset = 1 - self.config['maker_offset_pct']/100 if signal['action'] == 'BUY' else 1 + self.config['maker_offset_pct']/100
        limit_price = round(signal['price'] * offset, self.instrument_info.get('price_precision', 5))
        expected_fill_price = self.apply_slippage(limit_price, signal['action'], "LIMIT")
        
        try:
            order = self.exchange.place_order(
                category="linear",
                symbol=self.symbol,
                side="Buy" if signal['action'] == 'BUY' else "Sell",
                orderType="Limit",
                qty=formatted_qty,
                price=str(limit_price,
                timeInForce="PostOnly")
            )
            
            if order.get('retCode') == 0:
                self.last_trade_time = time.time()
                
                stop_loss = stop_loss_price
                take_profit = (limit_price * (1 + self.config['net_take_profit']/100) if signal['action'] == 'BUY'
                             else limit_price * (1 - self.config['net_take_profit']/100))
                
                self.current_trade_id, _ = self.logger.log_trade_open(
                    side=signal['action'],
                    expected_price=signal['price'],
                    actual_price=expected_fill_price,
                    qty=float(formatted_qty),
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    info=f"ml_conf:{signal['ml_confidence']:.2f}_spacing:{signal['grid_spacing']:.2f}%_risk:{self.config['risk_per_trade']}%"
                )
                
                print(f"🤖 ML GRID {signal['action']}: {formatted_qty} @ ${limit_price:.5f}")
                print(f"   📊 ML Confidence: {signal['ml_confidence']:.2f}")
                print(f"   📏 Grid Spacing: {signal['grid_spacing']:.2f}%")
                print(f"   💰 Risk: {self.config['risk_per_trade']}% of ${self.account_balance:.0f}")
        except Exception as e:
            print(f"❌ Trade failed: {e}")
    
    async def close_position(self, reason):
        if not self.position:
            return
        
        current_price = float(self.price_data['close'].iloc[-1])
        side = "Sell" if self.position.get('side') == "Buy" else "Buy"
        qty = float(self.position['size'])
        
        # Apply slippage to exit
        expected_exit = current_price
        actual_exit = self.apply_slippage(current_price, side, "MARKET")
        
        try:
            order = self.exchange.place_order(
                category="linear",
                symbol=self.symbol,
                side=side,
                orderType="Limit",
                qty=self.format_qty(qty,
                timeInForce="PostOnly"),
                price=str(round(current_price * (1.001 if side == "Sell" else 0.999), 5)),
                timeInForce="PostOnly",
                reduceOnly=True)
            
            if order.get('retCode') == 0:
                if self.current_trade_id:
                    self.logger.log_trade_close(
                        trade_id=self.current_trade_id,
                        expected_exit=expected_exit,
                        actual_exit=actual_exit,
                        reason=reason,
                        fees_entry=-0.04,  # Maker rebate
                        fees_exit=0.05     # Taker fee
                    )
                    self.current_trade_id = None
                
                print(f"✅ Closed: {reason}")
                self.last_grid_level = None
        except Exception as e:
            print(f"❌ Close failed: {e}")
    
    def show_status(self):
        if len(self.price_data) == 0:
            return
        
        current_price = float(self.price_data['close'].iloc[-1])
        
        print(f"\n🤖 FIXED ML Grid Trading - {self.symbol}")
        print(f"💰 Price: ${current_price:.5f}")
        print(f"💳 Balance: ${self.account_balance:.2f}")
        print(f"🎯 Risk per trade: {self.config['risk_per_trade']}%")
        
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
        
        # Initialize account info
        await self.get_account_balance()
        await self.get_instrument_info()
        
        if self.account_balance == 0:
            print("❌ No account balance found")
            return
        
        print(f"🤖 FIXED ML-Optimized Grid Trading Bot - {self.symbol}")
        print(f"⏰ Timeframe: {self.config['timeframe']} minutes")
        print(f"💰 Account Balance: ${self.account_balance:.2f}")
        print(f"🎯 Risk per trade: {self.config['risk_per_trade']}%")
        print(f"📏 Slippage modeling: {self.config['slippage_bps']} bps")
        print(f"✅ FIXES: Position sizing, fees, slippage")
        
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