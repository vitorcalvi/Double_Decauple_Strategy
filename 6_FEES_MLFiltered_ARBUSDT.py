import os
import asyncio
import pandas as pd
import numpy as np
import json
from datetime import datetime
from pybit.unified_trading import HTTP
from dotenv import load_dotenv
from collections import deque

load_dotenv()

class EnhancedMLScalpingBot:
    def __init__(self):
        self.symbol = 'ARBUSDT'
        self.demo_mode = os.getenv('DEMO_MODE', 'true').lower() == 'true'
        
        prefix = 'TESTNET_' if self.demo_mode else 'LIVE_'
        self.api_key = os.getenv(f'{prefix}BYBIT_API_KEY')
        self.api_secret = os.getenv(f'{prefix}BYBIT_API_SECRET')
        
        self.exchange = None
        self.position = None
        self.price_data = pd.DataFrame()
        self.trade_id = 0
        self.path_signatures = deque(maxlen=100)
        self.recent_trades = deque(maxlen=50)
        
        self.config = {
            'timeframe': '3',
            'rsi_period': 14,
            'ema_fast': 9,
            'ema_slow': 21,
            'bb_period': 20,
            'bb_std': 2,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'atr_period': 14,
            'path_lookback': 20,
            'ml_confidence_threshold': 0.70,
            'base_position_size': 100,
            'lookback': 100,
            'maker_offset_pct': 0.01,
            'base_take_profit_pct': 0.4,
            'base_stop_loss_pct': 0.3,
            'trailing_activation_pct': 0.2,
            'trailing_distance_pct': 0.15,
        }
        
        self.pattern_memory = {
            'winning': deque(maxlen=200),
            'losing': deque(maxlen=200)
        }
        
        self.daily_pnl = 0
        self.consecutive_losses = 0
        self.volatility_regime = 'normal'
        self.position_metadata = {}
        
        os.makedirs("logs", exist_ok=True)
        self.log_file = f"logs/ml_scalping_{datetime.now().strftime('%Y%m%d')}.log"
    
    def connect(self):
        try:
            self.exchange = HTTP(demo=self.demo_mode, api_key=self.api_key, api_secret=self.api_secret)
            return self.exchange.get_server_time().get('retCode') == 0
        except:
            return False
    
    def safe_float(self, value, default=0):
        try:
            if value is None or value == '' or value == 'None':
                return default
            return float(value)
        except:
            return default
    
    def calculate_atr(self, df):
        try:
            high = df['high']
            low = df['low']
            close = df['close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=self.config['atr_period']).mean()
            
            return atr.iloc[-1] if len(atr) > 0 and pd.notna(atr.iloc[-1]) else None
        except:
            return None
    
    def calculate_volatility(self, df):
        try:
            returns = df['close'].pct_change().dropna()
            if len(returns) < 20:
                return 0.01
            
            volatility = returns.rolling(window=20).std().iloc[-1]
            
            if pd.isna(volatility) or volatility <= 0:
                volatility = 0.01
            
            if volatility > 0.025:
                self.volatility_regime = 'high'
            elif volatility < 0.01:
                self.volatility_regime = 'low'
            else:
                self.volatility_regime = 'normal'
            
            return volatility
        except:
            return 0.01
    
    def calculate_indicators(self, df):
        if len(df) < self.config['lookback']:
            return None
        
        try:
            close = df['close']
            volume = df['volume']
            
            # RSI
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(window=self.config['rsi_period']).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=self.config['rsi_period']).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # EMAs
            ema_fast = close.ewm(span=self.config['ema_fast']).mean()
            ema_slow = close.ewm(span=self.config['ema_slow']).mean()
            
            # Bollinger Bands
            bb_middle = close.rolling(window=self.config['bb_period']).mean()
            bb_std = close.rolling(window=self.config['bb_period']).std()
            bb_upper = bb_middle + (bb_std * self.config['bb_std'])
            bb_lower = bb_middle - (bb_std * self.config['bb_std'])
            
            # MACD
            exp1 = close.ewm(span=self.config['macd_fast']).mean()
            exp2 = close.ewm(span=self.config['macd_slow']).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=self.config['macd_signal']).mean()
            macd_histogram = macd - signal
            
            # Volume
            volume_ma = volume.rolling(window=20).mean()
            volume_ratio = volume.iloc[-1] / volume_ma.iloc[-1] if volume_ma.iloc[-1] > 0 else 1
            
            # ATR and Volatility
            atr = self.calculate_atr(df)
            volatility = self.calculate_volatility(df)
            
            rsi_val = rsi.iloc[-1] if pd.notna(rsi.iloc[-1]) else 50
            bb_pos = (close.iloc[-1] - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1]) if bb_upper.iloc[-1] != bb_lower.iloc[-1] else 0.5
            
            return {
                'price': close.iloc[-1],
                'rsi': rsi_val,
                'ema_trend': ema_fast.iloc[-1] > ema_slow.iloc[-1],
                'bb_position': bb_pos if pd.notna(bb_pos) else 0.5,
                'macd_histogram': macd_histogram.iloc[-1] if pd.notna(macd_histogram.iloc[-1]) else 0,
                'volume_ratio': volume_ratio,
                'atr': atr,
                'volatility': volatility
            }
        except:
            return None
    
    def calculate_path_signature(self, prices):
        if len(prices) < self.config['path_lookback']:
            return None
        
        try:
            recent = prices.iloc[-self.config['path_lookback']:]
            returns = recent.pct_change().dropna()
            
            if len(returns) < 2:
                return None
            
            cumulative_return = (1 + returns).prod() - 1
            volatility = returns.std() if len(returns) > 1 else 0
            momentum = returns.mean()
            trend_strength = abs(returns.sum()) / returns.abs().sum() if returns.abs().sum() > 0 else 0
            
            return {
                'cumulative_return': cumulative_return,
                'volatility': volatility if pd.notna(volatility) else 0.01,
                'momentum': momentum if pd.notna(momentum) else 0,
                'trend_strength': trend_strength
            }
        except:
            return None
    
    def ml_filter_confidence(self, signature, indicators):
        if not signature or not indicators:
            return 0
        
        confidence = 0.5
        
        momentum = signature.get('momentum', 0)
        volatility = signature.get('volatility', 0.01)
        rsi = indicators.get('rsi', 50)
        ema_trend = indicators.get('ema_trend', False)
        macd_histogram = indicators.get('macd_histogram', 0)
        bb_position = indicators.get('bb_position', 0.5)
        volume_ratio = indicators.get('volume_ratio', 1)
        
        # Momentum and trend alignment
        if ema_trend:
            confidence += 0.1 if momentum > 0.0005 else -0.05
            confidence += 0.1 if macd_histogram > 0 else -0.05
        else:
            confidence += 0.1 if momentum < -0.0005 else -0.05
            confidence += 0.1 if macd_histogram < 0 else -0.05
        
        # Volatility scoring
        if 0.005 < volatility < 0.02:
            confidence += 0.15
        elif volatility > 0.025:
            confidence -= 0.1
        
        # RSI extremes
        if rsi < 30:
            confidence += 0.15 if not ema_trend else -0.1
        elif rsi > 70:
            confidence += 0.15 if ema_trend else -0.1
        
        # Bollinger Bands
        if bb_position < 0.2:
            confidence += 0.1 if not ema_trend else 0
        elif bb_position > 0.8:
            confidence += 0.1 if ema_trend else 0
        
        # Volume confirmation
        if volume_ratio > 1.5:
            confidence += 0.1
        
        # Volatility regime adjustment
        if self.volatility_regime == 'high':
            confidence *= 0.8
        elif self.volatility_regime == 'low':
            confidence *= 0.9
        
        return min(max(confidence, 0), 1)
    
    def calculate_dynamic_stops(self, indicators, side):
        atr = indicators.get('atr', None)
        price = indicators['price']
        
        if atr and atr > 0:
            atr_pct = (atr / price) * 100
            stop_loss_pct = max(self.config['base_stop_loss_pct'], min(atr_pct * 2, 0.5))
            take_profit_pct = max(self.config['base_take_profit_pct'], min(atr_pct * 3, 0.8))
        else:
            stop_loss_pct = self.config['base_stop_loss_pct']
            take_profit_pct = self.config['base_take_profit_pct']
        
        # Adjust for volatility regime
        if self.volatility_regime == 'high':
            stop_loss_pct *= 1.5
            take_profit_pct *= 1.3
        elif self.volatility_regime == 'low':
            stop_loss_pct *= 0.8
            take_profit_pct *= 0.9
        
        return stop_loss_pct, take_profit_pct
    
    def generate_signal(self, df):
        indicators = self.calculate_indicators(df)
        if not indicators:
            return None
        
        signature = self.calculate_path_signature(df['close'])
        if not signature:
            return None
        
        confidence = self.ml_filter_confidence(signature, indicators)
        
        if confidence < self.config['ml_confidence_threshold']:
            return None
        
        # Generate signal based on multiple confirmations
        buy_signals = 0
        sell_signals = 0
        
        ema_trend = indicators.get('ema_trend', False)
        rsi = indicators.get('rsi', 50)
        bb_position = indicators.get('bb_position', 0.5)
        macd_histogram = indicators.get('macd_histogram', 0)
        volume_ratio = indicators.get('volume_ratio', 1)
        
        if ema_trend:
            buy_signals += 1
        else:
            sell_signals += 1
        
        if rsi < 40:
            buy_signals += 1
        elif rsi > 60:
            sell_signals += 1
        
        if bb_position < 0.3:
            buy_signals += 1
        elif bb_position > 0.7:
            sell_signals += 1
        
        if macd_histogram > 0:
            buy_signals += 1
        else:
            sell_signals += 1
        
        if volume_ratio < 0.8:
            return None
        
        if buy_signals >= 3:
            stop_loss, take_profit = self.calculate_dynamic_stops(indicators, 'Buy')
            return {
                'action': 'BUY',
                'price': indicators['price'],
                'confidence': confidence,
                'rsi': rsi,
                'stop_loss_pct': stop_loss,
                'take_profit_pct': take_profit
            }
        elif sell_signals >= 3:
            stop_loss, take_profit = self.calculate_dynamic_stops(indicators, 'Sell')
            return {
                'action': 'SELL',
                'price': indicators['price'],
                'confidence': confidence,
                'rsi': rsi,
                'stop_loss_pct': stop_loss,
                'take_profit_pct': take_profit
            }
        
        return None
    
    def should_close(self):
        if not self.position:
            return False, ""
        
        try:
            current_price = float(self.price_data['close'].iloc[-1])
            entry_price = self.safe_float(self.position.get('avgPrice', 0))
            side = self.position.get('side', '')
            
            if entry_price == 0:
                return False, ""
        except:
            return False, ""
        
        # Get dynamic targets from metadata
        take_profit_pct = self.safe_float(self.position_metadata.get('takeProfit'), self.config['base_take_profit_pct'])
        stop_loss_pct = self.safe_float(self.position_metadata.get('stopLoss'), self.config['base_stop_loss_pct'])
        
        if side == "Buy":
            if current_price >= entry_price * (1 + take_profit_pct / 100):
                return True, "take_profit"
            if current_price <= entry_price * (1 - stop_loss_pct / 100):
                return True, "stop_loss"
            
            # Trailing stop
            pnl_pct = ((current_price - entry_price) / entry_price * 100)
            if pnl_pct >= self.config['trailing_activation_pct']:
                trailing_stop = entry_price * (1 + (pnl_pct - self.config['trailing_distance_pct']) / 100)
                if current_price <= trailing_stop:
                    return True, "trailing_stop"
        else:
            if current_price <= entry_price * (1 - take_profit_pct / 100):
                return True, "take_profit"
            if current_price >= entry_price * (1 + stop_loss_pct / 100):
                return True, "stop_loss"
            
            # Trailing stop
            pnl_pct = ((entry_price - current_price) / entry_price * 100)
            if pnl_pct >= self.config['trailing_activation_pct']:
                trailing_stop = entry_price * (1 - (pnl_pct - self.config['trailing_distance_pct']) / 100)
                if current_price >= trailing_stop:
                    return True, "trailing_stop"
        
        # Reversal detection
        indicators = self.calculate_indicators(self.price_data)
        if indicators:
            if side == "Buy" and indicators['rsi'] > 75 and not indicators['ema_trend']:
                return True, "reversal_signal"
            elif side == "Sell" and indicators['rsi'] < 25 and indicators['ema_trend']:
                return True, "reversal_signal"
        
        return False, ""
    
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
            
            data_list = klines.get('result', {}).get('list', [])
            if not data_list:
                return False
            
            df = pd.DataFrame(data_list, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
            ])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df.dropna(subset=['open', 'high', 'low', 'close'])
            
            if len(df) < 20:
                return False
            
            self.price_data = df.sort_values('timestamp').reset_index(drop=True)
            return True
        except:
            return False
    
    async def check_position(self):
        try:
            positions = self.exchange.get_positions(category="linear", symbol=self.symbol)
            if positions.get('retCode') == 0:
                pos_list = positions['result']['list']
                if pos_list:
                    size = self.safe_float(pos_list[0].get('size', 0))
                    self.position = pos_list[0] if size > 0 else None
                else:
                    self.position = None
        except:
            self.position = None
    
    async def execute_trade(self, signal):
        qty = self.config['base_position_size'] / signal['price']
        formatted_qty = str(int(round(qty))) if qty >= 1 else "0"
        
        if formatted_qty == "0":
            return
        
        # LIMIT order for entry
        limit_price = round(signal['price'] * (1 - self.config['maker_offset_pct']/100 if signal['action'] == 'BUY' else 1 + self.config['maker_offset_pct']/100), 4)
        
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
                self.trade_id += 1
                
                # Store dynamic targets in metadata
                self.position_metadata = {
                    'stopLoss': signal['stop_loss_pct'],
                    'takeProfit': signal['take_profit_pct']
                }
                
                print(f"🤖 {signal['action']}: {formatted_qty} @ ${limit_price:.4f}")
                print(f"   💎 ML Confidence: {signal['confidence']:.2f} | RSI: {signal['rsi']:.1f}")
                print(f"   📈 Volatility: {self.volatility_regime}")
                
                self.log_trade(signal['action'], limit_price, f"conf:{signal['confidence']:.2f}")
        except Exception as e:
            print(f"Trade failed: {e}")
    
    async def close_position(self, reason):
        if not self.position:
            return
        
        side = "Sell" if self.position.get('side') == "Buy" else "Buy"
        qty = self.safe_float(self.position.get('size', 0))
        
        if qty == 0:
            return
        
        # MARKET order for exit (immediate execution)
        try:
            order = self.exchange.place_order(
                category="linear",
                symbol=self.symbol,
                side=side,
                orderType="Market",
                qty=str(int(round(qty))),
                reduceOnly=True
            )
            
            if order.get('retCode') == 0:
                pnl = self.safe_float(self.position.get('unrealisedPnl', 0))
                self.daily_pnl += pnl
                
                if pnl < 0:
                    self.consecutive_losses += 1
                else:
                    self.consecutive_losses = 0
                
                self.recent_trades.append({'pnl': pnl, 'reason': reason})
                
                print(f"✅ Closed: {reason} | PnL: ${pnl:.2f}")
                print(f"   📊 Daily PnL: ${self.daily_pnl:.2f}")
                
                self.log_trade("CLOSE", 0, f"{reason}_PnL:${pnl:.2f}")
        except Exception as e:
            print(f"Close failed: {e}")
    
    def log_trade(self, action, price, info):
        with open(self.log_file, "a") as f:
            f.write(json.dumps({
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'action': action,
                'price': round(price, 6),
                'info': info
            }) + "\n")
    
    async def run_cycle(self):
        if not await self.get_market_data():
            return
        
        await self.check_position()
        
        if self.position:
            should_close, reason = self.should_close()
            if should_close:
                await self.close_position(reason)
        elif signal := self.generate_signal(self.price_data):
            await self.execute_trade(signal)
    
    async def run(self):
        if not self.connect():
            print("Failed to connect")
            return
        
        print(f"🚀 ML-Filtered Scalping Bot")
        print(f"⏰ Timeframe: 3 minutes")
        print(f"🎯 Dynamic TP/SL based on ATR and volatility")
        
        while True:
            try:
                await self.run_cycle()
                await asyncio.sleep(5)
            except KeyboardInterrupt:
                print("\n✋ Bot stopped")
                if self.position:
                    await self.close_position("manual_stop")
                print(f"📊 Final Daily PnL: ${self.daily_pnl:.2f}")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                await asyncio.sleep(10)

if __name__ == "__main__":
    bot = EnhancedMLScalpingBot()
    asyncio.run(bot.run())