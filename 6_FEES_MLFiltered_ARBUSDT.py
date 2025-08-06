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
    """Enhanced ML-Filtered Scalping Bot with fee calculations"""
    
    def __init__(self):
        self.symbol = 'ARBUSDT'
        self.demo_mode = os.getenv('DEMO_MODE', 'true').lower() == 'true'
        
        # API setup
        prefix = 'TESTNET_' if self.demo_mode else 'LIVE_'
        self.api_key = os.getenv(f'{prefix}BYBIT_API_KEY')
        self.api_secret = os.getenv(f'{prefix}BYBIT_API_SECRET')
        
        if not self.api_key or not self.api_secret:
            raise ValueError("Missing API credentials")
        
        self.exchange = None
        
        # Trading state
        self.position = None
        self.price_data = pd.DataFrame()
        self.trade_id = 0
        self.path_signatures = deque(maxlen=100)
        self.recent_trades = deque(maxlen=50)
        
        # Enhanced strategy parameters with Fee Calculations
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
            'volume_ma_period': 20,
            'atr_period': 14,
            'path_lookback': 20,
            'ml_confidence_threshold': 0.70,
            'base_position_size': 100,
            'max_position_size': 300,
            'lookback': 100,
            'maker_offset_pct': 0.01,
            # Fee structure
            'maker_fee_pct': -0.04,  # Negative = rebate
            # Dynamic risk parameters (base)
            'base_take_profit_pct': 0.4,
            'base_stop_loss_pct': 0.3,
            'trailing_activation_pct': 0.2,
            'trailing_distance_pct': 0.15,
            'max_daily_loss': 5.0,
            'max_consecutive_losses': 3,
            # Volatility adjustments
            'high_volatility_threshold': 0.025,
            'low_volatility_threshold': 0.01,
        }
        
        # ML pattern memory with decay
        self.pattern_memory = {
            'winning': deque(maxlen=200),
            'losing': deque(maxlen=200),
            'neutral': deque(maxlen=100)
        }
        
        # Performance tracking
        self.daily_pnl = 0
        self.consecutive_losses = 0
        self.session_start = datetime.now()
        self.volatility_regime = 'normal'
        self.position_open_time = None
        self.last_signal_signature = None
        self.position_metadata = {}
        
        os.makedirs("logs", exist_ok=True)
        self.log_file = f"logs/enhanced_ml_scalping_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    def connect(self):
        """Connect to exchange."""
        try:
            self.exchange = HTTP(demo=self.demo_mode, api_key=self.api_key, api_secret=self.api_secret)
            return self.exchange.get_server_time().get('retCode') == 0
        except:
            return False
    
    def safe_float(self, value, default=0):
        """Safely convert value to float."""
        try:
            if value is None or value == '' or value == 'None':
                return default
            return float(value)
        except:
            return default
    
    def format_qty(self, qty):
        """Format quantity for DOGE."""
        return str(int(round(qty))) if qty >= 1.0 else "0"
    
    def calculate_break_even(self, entry_price, side):
        """Calculate break-even price including fees."""
        fee_impact = 2 * abs(self.config['maker_fee_pct']) / 100
        
        # With rebate, break-even is better than entry
        if side == "Buy":
            return entry_price * (1 - fee_impact)
        else:
            return entry_price * (1 + fee_impact)
    
    def calculate_net_targets(self, entry_price, side, stop_loss_pct, take_profit_pct):
        """Calculate net TP/SL accounting for round-trip fees (dynamic)."""
        fee_adjustment = 2 * abs(self.config['maker_fee_pct']) / 100
        
        # Adjust dynamic targets for fees
        net_tp_pct = take_profit_pct + (fee_adjustment * 100)
        net_sl_pct = stop_loss_pct - (fee_adjustment * 100)
        
        if side == "Buy":
            net_tp = entry_price * (1 + net_tp_pct / 100)
            net_sl = entry_price * (1 - net_sl_pct / 100)
        else:
            net_tp = entry_price * (1 - net_tp_pct / 100)
            net_sl = entry_price * (1 + net_sl_pct / 100)
        
        return net_tp, net_sl, net_tp_pct, net_sl_pct
    
    def calculate_atr(self, df):
        """Calculate Average True Range for dynamic stops."""
        try:
            high = df['high']
            low = df['low']
            close = df['close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=self.config['atr_period']).mean()
            
            atr_value = atr.iloc[-1] if len(atr) > 0 else None
            return atr_value if pd.notna(atr_value) else None
        except:
            return None
    
    def calculate_volatility(self, df):
        """Calculate current market volatility."""
        try:
            returns = df['close'].pct_change().dropna()
            if len(returns) < 20:
                self.volatility_regime = 'normal'
                return 0.01
            
            volatility = returns.rolling(window=20).std().iloc[-1]
            
            if pd.isna(volatility) or volatility <= 0:
                volatility = 0.01
            
            # Classify volatility regime
            if volatility > self.config['high_volatility_threshold']:
                self.volatility_regime = 'high'
            elif volatility < self.config['low_volatility_threshold']:
                self.volatility_regime = 'low'
            else:
                self.volatility_regime = 'normal'
            
            return volatility
        except:
            self.volatility_regime = 'normal'
            return 0.01
    
    def calculate_indicators(self, df):
        """Calculate comprehensive indicators."""
        if len(df) < self.config['lookback']:
            return None
        
        try:
            close = df['close']
            high = df['high']
            low = df['low']
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
            
            # Volume analysis
            volume_ma = volume.rolling(window=self.config['volume_ma_period']).mean()
            volume_ratio = volume.iloc[-1] / volume_ma.iloc[-1] if volume_ma.iloc[-1] > 0 else 1
            
            # ATR and Volatility
            atr = self.calculate_atr(df)
            volatility = self.calculate_volatility(df)
            
            # Handle NaN values
            rsi_val = rsi.iloc[-1] if pd.notna(rsi.iloc[-1]) else 50
            bb_pos = (close.iloc[-1] - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1]) if bb_upper.iloc[-1] != bb_lower.iloc[-1] else 0.5
            if pd.isna(bb_pos):
                bb_pos = 0.5
            
            return {
                'price': close.iloc[-1],
                'rsi': rsi_val,
                'ema_fast': ema_fast.iloc[-1],
                'ema_slow': ema_slow.iloc[-1],
                'ema_trend': ema_fast.iloc[-1] > ema_slow.iloc[-1],
                'bb_upper': bb_upper.iloc[-1],
                'bb_lower': bb_lower.iloc[-1],
                'bb_position': bb_pos,
                'macd': macd.iloc[-1] if pd.notna(macd.iloc[-1]) else 0,
                'macd_signal': signal.iloc[-1] if pd.notna(signal.iloc[-1]) else 0,
                'macd_histogram': macd_histogram.iloc[-1] if pd.notna(macd_histogram.iloc[-1]) else 0,
                'volume_ratio': volume_ratio,
                'atr': atr,
                'volatility': volatility
            }
        except:
            return None
    
    def calculate_path_signature(self, prices):
        """Enhanced path signature with more metrics."""
        if len(prices) < self.config['path_lookback']:
            return None
        
        try:
            recent = prices.iloc[-self.config['path_lookback']:]
            returns = recent.pct_change().dropna()
            
            if len(returns) < 2:
                return None
            
            log_returns = np.log(recent / recent.shift(1)).dropna()
            
            # Enhanced metrics
            cumulative_return = (1 + returns).prod() - 1
            volatility = returns.std() if len(returns) > 1 else 0
            skewness = returns.skew() if len(returns) > 2 else 0
            kurtosis = returns.kurtosis() if len(returns) > 3 else 0
            direction_changes = np.sum(np.diff(np.sign(returns)) != 0) if len(returns) > 1 else 0
            momentum = returns.mean()
            max_drawdown = (recent / recent.cummax() - 1).min()
            trend_strength = abs(returns.sum()) / returns.abs().sum() if returns.abs().sum() > 0 else 0
            roughness = log_returns.std() / np.sqrt(len(log_returns)) if len(log_returns) > 0 else 0
            
            # Handle NaN values
            if pd.isna(volatility):
                volatility = 0.01
            if pd.isna(momentum):
                momentum = 0
            if pd.isna(max_drawdown):
                max_drawdown = 0
            
            return {
                'cumulative_return': cumulative_return,
                'volatility': volatility,
                'skewness': skewness,
                'kurtosis': kurtosis,
                'direction_changes': int(direction_changes),
                'momentum': momentum,
                'max_drawdown': max_drawdown,
                'roughness': roughness,
                'trend_strength': trend_strength
            }
        except:
            return None
    
    def ml_filter_confidence(self, signature, indicators):
        """Enhanced ML filter with pattern matching."""
        if not signature or not indicators:
            return 0
        
        confidence = 0.5  # Base confidence
        
        # Get values safely
        momentum = signature.get('momentum', 0)
        volatility = signature.get('volatility', 0.01)
        roughness = signature.get('roughness', 0.5)
        direction_changes = signature.get('direction_changes', 0)
        trend_strength = signature.get('trend_strength', 0)
        
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
        
        # Path smoothness
        confidence += 0.1 if roughness < 0.5 else -0.05
        confidence += 0.1 if direction_changes < 5 else -0.05
        
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
        
        # Trend strength bonus
        confidence += 0.1 if trend_strength > 0.6 else 0
        
        # Pattern memory influence
        if self.pattern_memory['winning']:
            similar_wins = self._count_similar_patterns(signature, self.pattern_memory['winning'])
            similar_losses = self._count_similar_patterns(signature, self.pattern_memory['losing'])
            
            if similar_wins > similar_losses:
                confidence += 0.15
            elif similar_losses > similar_wins:
                confidence -= 0.15
        
        # Volatility regime adjustment
        if self.volatility_regime == 'high':
            confidence *= 0.8
        elif self.volatility_regime == 'low':
            confidence *= 0.9
        
        return min(max(confidence, 0), 1)
    
    def _count_similar_patterns(self, current, pattern_list, threshold=0.8):
        """Count similar patterns in memory."""
        if not current or not pattern_list:
            return 0
            
        similar_count = 0
        for pattern in pattern_list:
            if not pattern:
                continue
                
            similarity = 0
            comparisons = 0
            
            for key in ['momentum', 'volatility', 'trend_strength']:
                if key in current and key in pattern:
                    curr_val = current.get(key, 0)
                    pat_val = pattern.get(key, 0)
                    
                    if curr_val is None or pat_val is None:
                        continue
                        
                    comparisons += 1
                    diff = abs(curr_val - pat_val)
                    max_val = max(abs(curr_val), abs(pat_val), 0.0001)
                    similarity += 1 - (diff / max_val)
            
            if comparisons > 0 and similarity / comparisons > threshold:
                similar_count += 1
        
        return similar_count
    
    def calculate_dynamic_stops(self, indicators, side):
        """Calculate dynamic stop loss and take profit based on ATR and volatility."""
        atr = indicators.get('atr', None)
        volatility = indicators.get('volatility', 0.01)
        price = indicators['price']
        
        # ATR-based stops
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
    
    def calculate_position_size(self, confidence, indicators):
        """Dynamic position sizing based on confidence and volatility."""
        base_size = self.config['base_position_size']
        
        # Confidence multiplier
        size_multiplier = 0.5 + (confidence * 1.5)
        
        # Volatility adjustment
        if self.volatility_regime == 'high':
            size_multiplier *= 0.7
        elif self.volatility_regime == 'low':
            size_multiplier *= 1.2
        
        # Kelly Criterion approximation
        if len(self.recent_trades) > 10:
            wins = [t for t in self.recent_trades if t.get('net_pnl', 0) > 0]
            total_trades = len(self.recent_trades)
            if total_trades > 0:
                win_rate = len(wins) / total_trades
                if win_rate > 0.5:
                    kelly_fraction = (win_rate - 0.5) * 2
                    size_multiplier *= (1 + kelly_fraction * 0.25)
        
        # Apply limits
        position_size = base_size * size_multiplier
        position_size = min(position_size, self.config['max_position_size'])
        
        # Reduce size after consecutive losses
        if self.consecutive_losses > 0:
            position_size *= (0.8 ** min(self.consecutive_losses, 3))
        
        return max(position_size, 50)
    
    def generate_signal(self, df):
        """Generate enhanced ML-filtered signal."""
        indicators = self.calculate_indicators(df)
        if not indicators:
            return None
        
        signature = self.calculate_path_signature(df['close'])
        if not signature:
            return None
        
        confidence = self.ml_filter_confidence(signature, indicators)
        
        # Store signature for pattern learning
        self.path_signatures.append({**signature, 'confidence': confidence, 'timestamp': datetime.now()})
        
        if confidence < self.config['ml_confidence_threshold']:
            return None
        
        # Check daily loss limit
        if self.daily_pnl <= -self.config['max_daily_loss']:
            print(f"⚠️ Daily loss limit reached: ${self.daily_pnl:.2f}")
            return None
        
        # Generate signal based on multiple confirmations
        buy_signals = 0
        sell_signals = 0
        
        # Get values safely
        ema_trend = indicators.get('ema_trend', False)
        rsi = indicators.get('rsi', 50)
        bb_position = indicators.get('bb_position', 0.5)
        macd_histogram = indicators.get('macd_histogram', 0)
        volume_ratio = indicators.get('volume_ratio', 1)
        
        # Trend following
        if ema_trend:
            buy_signals += 1
        else:
            sell_signals += 1
        
        # RSI
        if rsi < 40:
            buy_signals += 1
        elif rsi > 60:
            sell_signals += 1
        
        # Bollinger Bands
        if bb_position < 0.3:
            buy_signals += 1
        elif bb_position > 0.7:
            sell_signals += 1
        
        # MACD
        if macd_histogram > 0:
            buy_signals += 1
        else:
            sell_signals += 1
        
        # Volume confirmation required
        if volume_ratio < 0.8:
            return None
        
        # Generate signal if we have majority agreement
        if buy_signals >= 3:
            position_size = self.calculate_position_size(confidence, indicators)
            stop_loss, take_profit = self.calculate_dynamic_stops(indicators, 'Buy')
            
            return {
                'action': 'BUY',
                'price': indicators['price'],
                'confidence': confidence,
                'rsi': rsi,
                'position_size': position_size,
                'stop_loss_pct': stop_loss,
                'take_profit_pct': take_profit,
                'signature': signature
            }
        elif sell_signals >= 3:
            position_size = self.calculate_position_size(confidence, indicators)
            stop_loss, take_profit = self.calculate_dynamic_stops(indicators, 'Sell')
            
            return {
                'action': 'SELL',
                'price': indicators['price'],
                'confidence': confidence,
                'rsi': rsi,
                'position_size': position_size,
                'stop_loss_pct': stop_loss,
                'take_profit_pct': take_profit,
                'signature': signature
            }
        
        return None
    
    def should_close(self):
        """Enhanced position closing logic with NET targets and trailing stop."""
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
        
        # Get NET targets from position metadata
        net_tp_pct = self.safe_float(self.position_metadata.get('netTakeProfit'), self.config['base_take_profit_pct'] + 0.08)
        net_sl_pct = self.safe_float(self.position_metadata.get('netStopLoss'), self.config['base_stop_loss_pct'] - 0.08)
        
        # Calculate actual prices
        if side == "Buy":
            net_tp = entry_price * (1 + net_tp_pct / 100)
            net_sl = entry_price * (1 - net_sl_pct / 100)
            
            if current_price >= net_tp:
                return True, f"take_profit_net_{net_tp_pct:.2f}%"
            if current_price <= net_sl:
                return True, f"stop_loss_net_{net_sl_pct:.2f}%"
            
            # Trailing stop
            pnl_pct = ((current_price - entry_price) / entry_price * 100)
            if pnl_pct >= self.config['trailing_activation_pct']:
                trailing_stop_price = entry_price * (1 + (pnl_pct - self.config['trailing_distance_pct']) / 100)
                if current_price <= trailing_stop_price:
                    return True, "trailing_stop"
        else:
            net_tp = entry_price * (1 - net_tp_pct / 100)
            net_sl = entry_price * (1 + net_sl_pct / 100)
            
            if current_price <= net_tp:
                return True, f"take_profit_net_{net_tp_pct:.2f}%"
            if current_price >= net_sl:
                return True, f"stop_loss_net_{net_sl_pct:.2f}%"
            
            # Trailing stop
            pnl_pct = ((entry_price - current_price) / entry_price * 100)
            if pnl_pct >= self.config['trailing_activation_pct']:
                trailing_stop_price = entry_price * (1 - (pnl_pct - self.config['trailing_distance_pct']) / 100)
                if current_price >= trailing_stop_price:
                    return True, "trailing_stop"
        
        # Time-based exit
        if hasattr(self, 'position_open_time'):
            position_duration = (datetime.now() - self.position_open_time).seconds / 60
            if position_duration > 30 and abs(pnl_pct) < 0.1:
                return True, "time_exit"
        
        # Reversal detection
        indicators = self.calculate_indicators(self.price_data)
        if indicators:
            if side == "Buy":
                if indicators['rsi'] > 75 and not indicators['ema_trend']:
                    return True, "reversal_signal"
            else:
                if indicators['rsi'] < 25 and indicators['ema_trend']:
                    return True, "reversal_signal"
        
        return False, ""
    
    async def get_market_data(self):
        """Get market data."""
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
        """Check current position."""
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
        """Execute maker-only trade with dynamic sizing and NET targets."""
        qty = signal['position_size'] / signal['price']
        formatted_qty = self.format_qty(qty)
        
        if formatted_qty == "0":
            return
        
        # Calculate limit price
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
                self.position_open_time = datetime.now()
                
                # Calculate NET targets and break-even
                break_even = self.calculate_break_even(limit_price, signal['action'])
                net_tp, net_sl, net_tp_pct, net_sl_pct = self.calculate_net_targets(
                    limit_price, signal['action'], 
                    signal['stop_loss_pct'], signal['take_profit_pct']
                )
                
                # Store NET targets in position metadata
                self.position_metadata = {
                    'stopLoss': signal['stop_loss_pct'],
                    'takeProfit': signal['take_profit_pct'],
                    'netStopLoss': net_sl_pct,
                    'netTakeProfit': net_tp_pct,
                    'breakEven': break_even
                }
                
                self.log_trade(
                    signal['action'], 
                    limit_price, 
                    f"conf:{signal['confidence']:.2f}_BE:{break_even:.4f}_NetTP:{net_tp:.4f}({net_tp_pct:.2f}%)_NetSL:{net_sl:.4f}({net_sl_pct:.2f}%)"
                )
                
                print(f"🤖 ENHANCED {signal['action']}: {formatted_qty} DOGE @ ${limit_price:.4f}")
                print(f"   📊 Break-Even: ${break_even:.4f}")
                print(f"   🎯 Net TP: ${net_tp:.4f} ({net_tp_pct:.2f}%) | Net SL: ${net_sl:.4f} ({net_sl_pct:.2f}%)")
                print(f"   💎 ML Confidence: {signal['confidence']:.2f} | RSI: {signal['rsi']:.1f}")
                print(f"   📈 Position Size: ${signal['position_size']:.0f} | Volatility: {self.volatility_regime}")
        except Exception as e:
            print(f"Trade failed: {e}")
    
    async def close_position(self, reason):
        """Close position with maker order and NET PnL calculation."""
        if not self.position:
            return
        
        try:
            current_price = float(self.price_data['close'].iloc[-1])
            side = "Sell" if self.position.get('side') == "Buy" else "Buy"
            qty = self.safe_float(self.position.get('size', 0))
            entry_price = self.safe_float(self.position.get('avgPrice', 0))
            
            if qty == 0:
                return
            
            # Calculate limit price
            limit_price = round(current_price * (1 + self.config['maker_offset_pct']/100 if side == "Sell" else 1 - self.config['maker_offset_pct']/100), 4)
            
            order = self.exchange.place_order(
                category="linear",
                symbol=self.symbol,
                side=side,
                orderType="Limit",
                qty=self.format_qty(qty),
                price=str(limit_price),
                timeInForce="PostOnly",
                reduceOnly=True
            )
            
            if order.get('retCode') == 0:
                # Calculate NET PnL including fees
                gross_pnl = self.safe_float(self.position.get('unrealisedPnl', 0))
                fee_earned = (entry_price * qty + current_price * qty) * abs(self.config['maker_fee_pct']) / 100
                net_pnl = gross_pnl + fee_earned
                
                self.daily_pnl += net_pnl
                
                # Update consecutive losses based on NET PnL
                if net_pnl < 0:
                    self.consecutive_losses += 1
                else:
                    self.consecutive_losses = 0
                
                # Store trade result with NET PnL
                trade_result = {
                    'gross_pnl': gross_pnl,
                    'net_pnl': net_pnl,
                    'reason': reason,
                    'timestamp': datetime.now()
                }
                self.recent_trades.append(trade_result)
                
                # Update pattern memory based on NET PnL
                if hasattr(self, 'last_signal_signature'):
                    if net_pnl > 0:
                        self.pattern_memory['winning'].append(self.last_signal_signature)
                    elif net_pnl < 0:
                        self.pattern_memory['losing'].append(self.last_signal_signature)
                    else:
                        self.pattern_memory['neutral'].append(self.last_signal_signature)
                
                self.log_trade("CLOSE", limit_price, f"{reason}_GrossPnL:${gross_pnl:.2f}_NetPnL:${net_pnl:.2f}_Daily:${self.daily_pnl:.2f}")
                
                emoji = "🟢" if net_pnl > 0 else "🔴" if net_pnl < 0 else "⚪"
                print(f"{emoji} Closed: {reason}")
                print(f"   💵 Gross PnL: ${gross_pnl:.2f} | Net PnL: ${net_pnl:.2f}")
                print(f"   📊 Daily PnL: ${self.daily_pnl:.2f}")
        except Exception as e:
            print(f"Close failed: {e}")
    
    def log_trade(self, action, price, info):
        """Log trade."""
        with open(self.log_file, "a") as f:
            f.write(json.dumps({
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'id': self.trade_id,
                'action': action,
                'price': round(price, 6),
                'info': info
            }) + "\n")
    
    def show_status(self):
        """Show enhanced status."""
        if len(self.price_data) == 0:
            print("⏳ Waiting for market data...")
            return
        
        indicators = self.calculate_indicators(self.price_data)
        signature = self.calculate_path_signature(self.price_data['close'])
        
        if not indicators:
            return
        
        confidence = 0
        if signature:
            confidence = self.ml_filter_confidence(signature, indicators)
        
        print(f"\n🤖 Enhanced ML Scalping v2.0 - {self.symbol}")
        print(f"💰 Price: ${indicators['price']:.4f} | ATR: ${indicators.get('atr', 0):.4f}")
        print(f"📊 ML Confidence: {confidence:.2f} | RSI: {indicators['rsi']:.1f}")
        print(f"📈 Trend: {'Bullish' if indicators['ema_trend'] else 'Bearish'} | MACD: {indicators['macd_histogram']:.5f}")
        print(f"🌊 Volatility: {self.volatility_regime.upper()} ({indicators['volatility']:.4f})")
        
        if self.position:
            try:
                entry_price = self.safe_float(self.position.get('avgPrice', 0))
                side = self.position.get('side', '')
                size = self.position.get('size', '0')
                
                # Calculate current NET PnL
                gross_pnl = self.safe_float(self.position.get('unrealisedPnl', 0))
                fee_earned = (entry_price * float(size)) * abs(self.config['maker_fee_pct']) / 100
                net_pnl = gross_pnl + fee_earned
                
                # Get targets from metadata
                break_even = self.position_metadata.get('breakEven', entry_price)
                net_tp_pct = self.position_metadata.get('netTakeProfit', self.config['base_take_profit_pct'] + 0.08)
                net_sl_pct = self.position_metadata.get('netStopLoss', self.config['base_stop_loss_pct'] - 0.08)
                
                emoji = "🟢" if side == "Buy" else "🔴"
                print(f"{emoji} {side}: {size} DOGE @ ${entry_price:.4f}")
                print(f"   💵 Gross PnL: ${gross_pnl:.2f} | Net PnL: ${net_pnl:.2f}")
                print(f"   🎯 BE: ${break_even:.4f} | Net TP: +{net_tp_pct:.2f}% | Net SL: -{net_sl_pct:.2f}%")
            except:
                pass
        else:
            print(f"⏳ Waiting for ML confidence > {self.config['ml_confidence_threshold']:.2f}")
        
        print(f"💼 Daily PnL: ${self.daily_pnl:.2f} | Consecutive Losses: {self.consecutive_losses}")
        
        # Show recent pattern performance
        if len(self.recent_trades) > 0:
            wins = sum(1 for t in self.recent_trades if t.get('net_pnl', 0) > 0)
            total = len(self.recent_trades)
            print(f"📊 Recent Win Rate: {wins}/{total} ({wins/total*100:.1f}%)")
        
        print("-" * 70)
    
    async def run_cycle(self):
        """Main trading cycle."""
        if not await self.get_market_data():
            return
        
        await self.check_position()
        
        if self.position:
            should_close, reason = self.should_close()
            if should_close:
                await self.close_position(reason)
        elif signal := self.generate_signal(self.price_data):
            self.last_signal_signature = signal['signature']
            await self.execute_trade(signal)
        
        self.show_status()
    
    async def run(self):
        """Main bot loop."""
        if not self.connect():
            print("Failed to connect")
            return
        
        print(f"🚀 Enhanced ML-Filtered Scalping Bot v2.0")
        print(f"⏰ Timeframe: 3 minutes | Target Win Rate: 65%+")
        print(f"🎯 Dynamic TP/SL based on ATR and volatility")
        print(f"💎 Using MAKER-ONLY orders for {self.config['maker_fee_pct']}% fee rebate")
        print(f"🧠 Pattern learning with memory decay")
        print("-" * 70)
        
        while True:
            try:
                await self.run_cycle()
                await asyncio.sleep(5)
            except KeyboardInterrupt:
                print("\n✋ Bot stopped by user")
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