import os
import asyncio
import pandas as pd
import numpy as np
import json
from datetime import datetime, timezone
from pybit.unified_trading import HTTP
from dotenv import load_dotenv

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
            "qty": qty,
            "stop_loss": stop_loss,
            "take_profit": take_profit
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
        
        entry_fee_pct = abs(fees_entry) if fees_entry < 0 else fees_entry
        exit_fee_pct = abs(fees_exit) if fees_exit < 0 else fees_exit
        
        if fees_entry < 0:
            entry_rebate = trade["entry_price"] * trade["qty"] * entry_fee_pct / 100
        else:
            entry_rebate = -(trade["entry_price"] * trade["qty"] * entry_fee_pct / 100)
            
        if fees_exit < 0:
            exit_rebate = actual_exit * trade["qty"] * exit_fee_pct / 100
        else:
            exit_rebate = -(actual_exit * trade["qty"] * exit_fee_pct / 100)
        
        total_fee_impact = entry_rebate + exit_rebate
        net_pnl = gross_pnl + total_fee_impact
        
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
            "fee_impact": {
                "entry": round(entry_rebate, 2), 
                "exit": round(exit_rebate, 2), 
                "total": round(total_fee_impact, 2)
            },
            "net_pnl": round(net_pnl, 2),
            "reason": reason,
            "currency": self.currency
        }
        
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        
        del self.open_trades[trade_id]
        return log_entry

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
        self.pending_order = None
        self.daily_pnl = 0
        self.current_trade_id = None
        
        self.order_timeout = 180
        
        # FIXED PARAMETERS - Based on Research Analysis
        self.config = {
            'timeframe': '5',        # Increased from 3min to reduce noise
            'rsi_period': 14,
            'ema_fast': 9,
            'ema_slow': 21,
            'bb_period': 20,
            'bb_std': 2,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'ml_confidence_threshold': 0.75,
            'base_position_size': 100,
            'lookback': 100,
            'maker_offset_pct': 0.01,
            'base_take_profit_pct': 1.0,    # INCREASED: 0.4% → 1.0% (better target)
            'base_stop_loss_pct': 0.5,     # INCREASED: 0.3% → 0.5% (avoid noise)
        }
        
        self.volatility_regime = 'normal'
        
        self.logger = TradeLogger("ML_ARB", self.symbol)
    
    def connect(self):
        try:
            self.exchange = HTTP(demo=self.demo_mode, api_key=self.api_key, api_secret=self.api_secret)
            return self.exchange.get_server_time().get('retCode') == 0
        except Exception as e:
            print(f"Connection error: {e}")
            return False
    
    async def check_pending_orders(self):
        try:
            orders = self.exchange.get_open_orders(category="linear", symbol=self.symbol)
            if orders.get('retCode') != 0:
                return False
            
            order_list = orders['result']['list']
            if not order_list:
                self.pending_order = None
                return False
            
            order = order_list[0]
            age = datetime.now().timestamp() - int(order['createdTime']) / 1000
            
            if age > self.order_timeout:
                self.exchange.cancel_order(category="linear", symbol=self.symbol, orderId=order['orderId'])
                print(f"❌ Cancelled stale order (aged {age:.0f}s)")
                self.pending_order = None
                return False
            
            self.pending_order = order
            return True
        except Exception as e:
            print(f"Order check error: {e}")
            return False
    
    def calculate_indicators(self, df):
        if len(df) < self.config['lookback']:
            return None
        
        try:
            close = df['close']
            
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(window=self.config['rsi_period']).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=self.config['rsi_period']).mean()
            rs = gain / (loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))
            
            ema_fast = close.ewm(span=self.config['ema_fast']).mean()
            ema_slow = close.ewm(span=self.config['ema_slow']).mean()
            
            bb_middle = close.rolling(window=self.config['bb_period']).mean()
            bb_std = close.rolling(window=self.config['bb_period']).std()
            bb_upper = bb_middle + (bb_std * self.config['bb_std'])
            bb_lower = bb_middle - (bb_std * self.config['bb_std'])
            
            exp1 = close.ewm(span=self.config['macd_fast']).mean()
            exp2 = close.ewm(span=self.config['macd_slow']).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=self.config['macd_signal']).mean()
            macd_histogram = macd - signal
            
            bb_range = bb_upper.iloc[-1] - bb_lower.iloc[-1]
            bb_pos = (close.iloc[-1] - bb_lower.iloc[-1]) / bb_range if bb_range != 0 else 0.5
            
            returns = close.pct_change().dropna()
            volatility = returns.rolling(window=20).std().iloc[-1] if len(returns) >= 20 else 0.01
            
            if volatility > 0.025:
                self.volatility_regime = 'high'
            elif volatility < 0.01:
                self.volatility_regime = 'low'
            else:
                self.volatility_regime = 'normal'
            
            return {
                'price': close.iloc[-1],
                'rsi': rsi.iloc[-1] if pd.notna(rsi.iloc[-1]) else 50,
                'ema_trend': ema_fast.iloc[-1] > ema_slow.iloc[-1],
                'bb_position': bb_pos,
                'macd_histogram': macd_histogram.iloc[-1] if pd.notna(macd_histogram.iloc[-1]) else 0,
                'volatility': volatility
            }
        except Exception as e:
            print(f"Indicator calculation error: {e}")
            return None
    
    def ml_filter_confidence(self, indicators):
        if not indicators:
            return 0
        
        confidence = 0.5
        
        if indicators['ema_trend']:
            confidence += 0.1 if indicators['macd_histogram'] > 0 else -0.05
        else:
            confidence += 0.1 if indicators['macd_histogram'] < 0 else -0.05
        
        if indicators['rsi'] < 35:  # Slightly adjusted for better signals
            confidence += 0.15
        elif indicators['rsi'] > 65:  # Slightly adjusted for better signals
            confidence += 0.15
        elif 40 < indicators['rsi'] < 60:
            confidence += 0.05
        
        if indicators['bb_position'] < 0.2 or indicators['bb_position'] > 0.8:
            confidence += 0.15
        
        if self.volatility_regime == 'normal':
            confidence += 0.1
        elif self.volatility_regime == 'high':
            confidence *= 0.9
        
        return min(max(confidence, 0), 1)
    
    def generate_signal(self, df):
        indicators = self.calculate_indicators(df)
        if not indicators:
            return None
        
        confidence = self.ml_filter_confidence(indicators)
        
        if confidence < self.config['ml_confidence_threshold']:
            return None
        
        buy_score = 0
        sell_score = 0
        
        if indicators['ema_trend']:
            buy_score += 1
        else:
            sell_score += 1
        
        if indicators['rsi'] < 40:
            buy_score += 2
        elif indicators['rsi'] > 60:
            sell_score += 2
        
        if indicators['bb_position'] < 0.3:
            buy_score += 1
        elif indicators['bb_position'] > 0.7:
            sell_score += 1
        
        if indicators['macd_histogram'] > 0:
            buy_score += 1
        else:
            sell_score += 1
        
        if buy_score >= 3:
            return {
                'action': 'BUY',
                'price': indicators['price'],
                'confidence': confidence,
                'rsi': indicators['rsi']
            }
        elif sell_score >= 3:
            return {
                'action': 'SELL',
                'price': indicators['price'],
                'confidence': confidence,
                'rsi': indicators['rsi']
            }
        
        return None
    
    def should_close(self):
        if not self.position:
            return False, ""
        
        try:
            current_price = float(self.price_data['close'].iloc[-1])
            entry_price = float(self.position.get('avgPrice', 0))
            side = self.position.get('side', '')
            
            if entry_price == 0:
                return False, ""
            
            take_profit_pct = self.config['base_take_profit_pct']
            stop_loss_pct = self.config['base_stop_loss_pct']
            
            profit_pct = ((current_price - entry_price) / entry_price * 100) if side == "Buy" else ((entry_price - current_price) / entry_price * 100)
            
            if profit_pct >= take_profit_pct:
                return True, "take_profit"
            if profit_pct <= -stop_loss_pct:
                return True, "stop_loss"
            
            indicators = self.calculate_indicators(self.price_data)
            if indicators:
                if side == "Buy" and indicators['rsi'] > 75 and not indicators['ema_trend']:
                    return True, "reversal_signal"
                elif side == "Sell" and indicators['rsi'] < 25 and indicators['ema_trend']:
                    return True, "reversal_signal"
            
            return False, ""
        except Exception as e:
            print(f"Position check error: {e}")
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
            
            df = pd.DataFrame(data_list, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df.dropna(subset=['open', 'high', 'low', 'close'])
            
            if len(df) < 20:
                return False
            
            self.price_data = df.sort_values('timestamp').reset_index(drop=True)
            return True
        except Exception as e:
            print(f"Market data error: {e}")
            return False
    
    async def check_position(self):
        try:
            positions = self.exchange.get_positions(category="linear", symbol=self.symbol)
            if positions.get('retCode') == 0:
                pos_list = positions['result']['list']
                if pos_list:
                    size = float(pos_list[0].get('size', 0))
                    self.position = pos_list[0] if size > 0 else None
                else:
                    self.position = None
        except Exception as e:
            print(f"Position check error: {e}")
            self.position = None
    
    async def execute_trade(self, signal):
        if await self.check_pending_orders() or self.position:
            return
        
        qty = self.config['base_position_size'] / signal['price']
        formatted_qty = str(int(round(qty)))
        
        if int(formatted_qty) == 0:
            return
        
        offset = 1 - self.config['maker_offset_pct']/100 if signal['action'] == 'BUY' else 1 + self.config['maker_offset_pct']/100
        limit_price = round(signal['price'] * offset, 4)
        
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
                self.pending_order = order['result']
                
                stop_loss = limit_price * (1 - self.config['base_stop_loss_pct']/100) if signal['action'] == 'BUY' else limit_price * (1 + self.config['base_stop_loss_pct']/100)
                take_profit = limit_price * (1 + self.config['base_take_profit_pct']/100) if signal['action'] == 'BUY' else limit_price * (1 - self.config['base_take_profit_pct']/100)
                
                self.current_trade_id, _ = self.logger.log_trade_open(
                    side=signal['action'],
                    expected_price=signal['price'],
                    actual_price=limit_price,
                    qty=float(formatted_qty),
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    info=f"confidence:{signal['confidence']:.2f}_rsi:{signal['rsi']:.1f}"
                )
                
                print(f"🤖 {signal['action']}: {formatted_qty} @ ${limit_price:.4f}")
                print(f"   💎 Confidence: {signal['confidence']:.2f} | RSI: {signal['rsi']:.1f}")
                print(f"   📈 Volatility: {self.volatility_regime}")
                print(f"   🎯 TP: ${take_profit:.4f} | SL: ${stop_loss:.4f} | R:R = 1:2")
        except Exception as e:
            print(f"Trade failed: {e}")
    
    async def close_position(self, reason):
        if not self.position:
            return
        
        side = "Sell" if self.position.get('side') == "Buy" else "Buy"
        qty = float(self.position.get('size', 0))
        
        if qty == 0:
            return
        
        try:
            current_price = float(self.price_data['close'].iloc[-1])
            
            order = self.exchange.place_order(
                category="linear",
                symbol=self.symbol,
                side=side,
                orderType="Market",
                qty=str(int(round(qty))),
                reduceOnly=True
            )
            
            if order.get('retCode') == 0:
                if self.current_trade_id:
                    log_entry = self.logger.log_trade_close(
                        trade_id=self.current_trade_id,
                        expected_exit=current_price,
                        actual_exit=current_price,
                        reason=reason,
                        fees_entry=-0.04,
                        fees_exit=0.1
                    )
                    
                    if log_entry:
                        pnl = log_entry.get('net_pnl', 0)
                        self.daily_pnl += pnl
                        print(f"✅ Closed: {reason} | PnL: ${pnl:.2f}")
                        print(f"   📊 Daily PnL: ${self.daily_pnl:.2f}")
                    
                    self.current_trade_id = None
        except Exception as e:
            print(f"Close failed: {e}")
    
    def show_status(self):
        if len(self.price_data) == 0:
            return
        
        current_price = float(self.price_data['close'].iloc[-1])
        
        print(f"\n🤖 FIXED ML-Filtered Bot - {self.symbol}")
        print(f"💰 Price: ${current_price:.4f}")
        print(f"🔧 IMPROVEMENTS:")
        print(f"   • Take Profit: 0.4% → 1.0% (better targets)")
        print(f"   • Stop Loss: 0.3% → 0.5% (reduce noise)")
        print(f"   • Timeframe: 3min → 5min (less noise)")
        print(f"   • Risk:Reward: 1:2 ratio")
        
        if self.position:
            entry_price = float(self.position.get('avgPrice', 0))
            side = self.position.get('side', '')
            size = self.position.get('size', '0')
            
            pnl = float(self.position.get('unrealisedPnl', 0))
            
            emoji = "🟢" if side == "Buy" else "🔴"
            print(f"{emoji} {side}: {size} @ ${entry_price:.4f} | PnL: ${pnl:.2f}")
        elif self.pending_order:
            order_price = float(self.pending_order.get('price', 0))
            order_side = self.pending_order.get('side', '')
            print(f"⏳ Pending {order_side} @ ${order_price:.4f}")
        else:
            print("🔍 ML scanning for high-confidence signals...")
        
        print("-" * 50)
    
    async def run_cycle(self):
        if not await self.get_market_data():
            return
        
        await self.check_position()
        await self.check_pending_orders()
        
        if self.position:
            should_close, reason = self.should_close()
            if should_close:
                await self.close_position(reason)
        elif not self.pending_order:
            signal = self.generate_signal(self.price_data)
            if signal:
                await self.execute_trade(signal)
        
        self.show_status()
    
    async def run(self):
        if not self.connect():
            print("Failed to connect")
            return
        
        print(f"🚀 FIXED ML-Filtered Scalping Bot - {self.symbol}")
        print(f"⏰ Timeframe: {self.config['timeframe']} minutes")
        print(f"🎯 ML Threshold: {self.config['ml_confidence_threshold']:.2f}")
        print(f"💰 TP: {self.config['base_take_profit_pct']}% | SL: {self.config['base_stop_loss_pct']}%")
        print(f"🔧 Fixed parameters for optimal 1:2 Risk:Reward ratio")
        
        while True:
            try:
                await self.run_cycle()
                await asyncio.sleep(8)
            except KeyboardInterrupt:
                print("\n✋ Bot stopped")
                try:
                    self.exchange.cancel_all_orders(category="linear", symbol=self.symbol)
                except:
                    pass
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