import os
import asyncio
import pandas as pd
import json
from datetime import datetime, timezone
from pybit.unified_trading import HTTP
from dotenv import load_dotenv

load_dotenv()

# Integrated Trade Logger
class TradeLogger:
    def __init__(self, bot_name, symbol):
        self.bot_name = bot_name
        self.symbol = symbol
        self.currency = "USDT"
        self.open_trades = {}
        self.trade_id = 1000
        
        # Ensure logs directory exists
        os.makedirs("logs", exist_ok=True)
        self.log_file = f"logs/{bot_name}_{symbol}.log"
        
    def generate_trade_id(self):
        self.trade_id += 1
        return self.trade_id
    
    def log_trade_open(self, side, expected_price, actual_price, qty, stop_loss, take_profit, info=""):
        """Log position opening with slippage"""
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
        """Log position closing with slippage and PnL calculation including rebates"""
        if trade_id not in self.open_trades:
            return None
            
        trade = self.open_trades[trade_id]
        duration = (datetime.now() - trade["entry_time"]).total_seconds()
        
        slippage = actual_exit - expected_exit if trade["side"] == "SELL" else expected_exit - actual_exit
        
        # Calculate gross PnL
        if trade["side"] == "BUY":
            gross_pnl = (actual_exit - trade["entry_price"]) * trade["qty"]
        else:
            gross_pnl = (trade["entry_price"] - actual_exit) * trade["qty"]
        
        # Calculate fee rebates (negative fees = rebate)
        entry_fee_pct = abs(fees_entry) if fees_entry < 0 else -fees_entry
        exit_fee_pct = abs(fees_exit) if fees_exit < 0 else -fees_exit
        
        entry_rebate = trade["entry_price"] * trade["qty"] * entry_fee_pct / 100
        exit_rebate = actual_exit * trade["qty"] * exit_fee_pct / 100
        
        # Calculate net PnL
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
            "entry_price": round(trade["entry_price"], 4),
            "expected_exit": round(expected_exit, 4),
            "actual_exit": round(actual_exit, 4),
            "slippage": round(slippage, 4),
            "qty": round(trade["qty"], 6),
            "gross_pnl": round(gross_pnl, 2),
            "fee_rebates": {"entry": round(entry_rebate, 2), "exit": round(exit_rebate, 2), "total": round(total_rebates, 2)},
            "net_pnl": round(net_pnl, 2),
            "reason": reason,
            "currency": self.currency
        }
        
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        
        del self.open_trades[trade_id]
        return log_entry

class MACDVWAPBot:
    def __init__(self):
        self.symbol = 'BTCUSDT'
        self.demo_mode = os.getenv('DEMO_MODE', 'true').lower() == 'true'
        
        # API connection
        prefix = 'TESTNET_' if self.demo_mode else 'LIVE_'
        self.api_key = os.getenv(f'{prefix}BYBIT_API_KEY')
        self.api_secret = os.getenv(f'{prefix}BYBIT_API_SECRET')
        self.exchange = None
        
        # Trading state
        self.position = None
        self.price_data = pd.DataFrame()
        
        # MACD + VWAP Strategy Config with Fee Calculations
        self.config = {
            'macd_fast': 8,
            'macd_slow': 21,
            'macd_signal': 5,
            'rsi_period': 9,
            'ema_period': 13,
            'rsi_oversold': 35,
            'rsi_overbought': 65,
            'position_size': 100,
            'maker_offset_pct': 0.01,
            # Fee structure
            'maker_fee_pct': -0.04,  # Negative = rebate
            # Gross TP/SL
            'gross_take_profit': 0.35,
            'gross_stop_loss': 0.25,
            # Net TP/SL (adjusted for 2x maker rebate)
            'net_take_profit': 0.43,  # 0.35 + 0.08 rebate
            'net_stop_loss': 0.17,    # 0.25 - 0.08 rebate
        }
        
        # Quantity formatting
        self.qty_step, self.min_qty = '0.01', 0.01  # BNB uses 0.01 precision
        
        # Trade logging
        self.logger = TradeLogger("MACD_VWAP", self.symbol)
        self.current_trade_id = None
    
    def connect(self):
        """Connect to exchange API"""
        try:
            self.exchange = HTTP(demo=self.demo_mode, api_key=self.api_key, api_secret=self.api_secret)
            return self.exchange.get_server_time().get('retCode') == 0
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False
    
    def format_qty(self, qty):
        """Format quantity according to exchange requirements"""
        if qty < self.min_qty:
            return "0"
        
        decimals = len(str(self.qty_step).split('.')[1]) if '.' in str(self.qty_step) else 0
        return f"{round(qty, decimals):.{decimals}f}"
    
    def calculate_break_even(self, entry_price, side):
        """Calculate break-even price including fee rebates"""
        fee_impact = 2 * abs(self.config['maker_fee_pct']) / 100
        
        # With rebate, break-even is better than entry
        if side == "Buy":
            return entry_price * (1 - fee_impact)
        else:
            return entry_price * (1 + fee_impact)
    
    def calculate_net_targets(self, entry_price, side):
        """Calculate net TP/SL accounting for round-trip fee rebates"""
        if side == "Buy":
            net_tp = entry_price * (1 + self.config['net_take_profit'] / 100)
            net_sl = entry_price * (1 - self.config['net_stop_loss'] / 100)
        else:
            net_tp = entry_price * (1 - self.config['net_take_profit'] / 100)
            net_sl = entry_price * (1 + self.config['net_stop_loss'] / 100)
        
        return net_tp, net_sl
    
    def calculate_vwap(self, df):
        """Calculate Volume Weighted Average Price"""
        if len(df) < 20:
            return None
        
        recent_data = df.tail(min(1440, len(df)))  # Last 24 hours max
        typical_price = (recent_data['high'] + recent_data['low'] + recent_data['close']) / 3
        volume = recent_data['volume']
        
        vwap = (typical_price * volume).cumsum() / volume.cumsum()
        return vwap.iloc[-1] if not vwap.empty else None
    
    def calculate_indicators(self, df):
        """Calculate all technical indicators for signal generation"""
        if len(df) < 50:
            return None
        
        close = df['close']
        
        # MACD
        ema_fast = close.ewm(span=self.config['macd_fast']).mean()
        ema_slow = close.ewm(span=self.config['macd_slow']).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.config['macd_signal']).mean()
        histogram = macd_line - signal_line
        
        # VWAP
        vwap = self.calculate_vwap(df)
        if not vwap:
            return None
        
        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(window=self.config['rsi_period']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.config['rsi_period']).mean()
        rs = gain / (loss + 1e-10)  # Avoid division by zero
        rsi = 100 - (100 / (1 + rs))
        
        # EMA trend filter
        ema = close.ewm(span=self.config['ema_period']).mean()
        
        return {
            'histogram': histogram.iloc[-1],
            'histogram_prev': histogram.iloc[-2] if len(histogram) > 1 else histogram.iloc[-1],
            'vwap': vwap,
            'rsi': rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50,
            'rsi_prev': rsi.iloc[-2] if len(rsi) > 1 and not pd.isna(rsi.iloc[-2]) else 50,
            'ema': ema.iloc[-1],
            'price': close.iloc[-1]
        }
    
    def detect_signals(self, indicators):
        """Detect histogram flip, RSI cross, and VWAP alignment"""
        # Histogram flip
        hist_bullish = indicators['histogram_prev'] <= 0 and indicators['histogram'] > 0
        hist_bearish = indicators['histogram_prev'] >= 0 and indicators['histogram'] < 0
        
        # RSI cross
        rsi_bullish = indicators['rsi_prev'] <= self.config['rsi_oversold'] and indicators['rsi'] > self.config['rsi_oversold']
        rsi_bearish = indicators['rsi_prev'] >= self.config['rsi_overbought'] and indicators['rsi'] < self.config['rsi_overbought']
        
        # VWAP alignment
        vwap_bullish = indicators['price'] > indicators['vwap'] and indicators['ema'] > indicators['vwap']
        vwap_bearish = indicators['price'] < indicators['vwap'] and indicators['ema'] < indicators['vwap']
        
        return {
            'hist_bullish': hist_bullish,
            'hist_bearish': hist_bearish,
            'rsi_bullish': rsi_bullish,
            'rsi_bearish': rsi_bearish,
            'vwap_bullish': vwap_bullish,
            'vwap_bearish': vwap_bearish
        }
    
    def generate_signal(self, df):
        """Generate trading signal based on MACD, RSI and VWAP alignment"""
        indicators = self.calculate_indicators(df)
        if not indicators:
            return None
        
        signals = self.detect_signals(indicators)
        
        # Bullish signal: All three align
        if signals['hist_bullish'] and signals['rsi_bullish'] and signals['vwap_bullish']:
            return {
                'action': 'BUY',
                'price': indicators['price'],
                'macd_hist': indicators['histogram'],
                'rsi': indicators['rsi'],
                'vwap': indicators['vwap']
            }
        
        # Bearish signal: All three align
        if signals['hist_bearish'] and signals['rsi_bearish'] and signals['vwap_bearish']:
            return {
                'action': 'SELL',
                'price': indicators['price'],
                'macd_hist': indicators['histogram'],
                'rsi': indicators['rsi'],
                'vwap': indicators['vwap']
            }
        
        return None
    
    async def get_market_data(self):
        """Retrieve market data from exchange"""
        try:
            klines = self.exchange.get_kline(
                category="linear",
                symbol=self.symbol,
                interval="1",
                limit=100
            )
            
            if klines.get('retCode') != 0:
                return False
            
            df = pd.DataFrame(klines['result']['list'], columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
            ])
            
            # Convert data types
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
            
            # Sort by time
            self.price_data = df.sort_values('timestamp').reset_index(drop=True)
            return True
        except Exception as e:
            print(f"❌ Market data error: {e}")
            return False
    
    async def check_position(self):
        """Check current position status"""
        try:
            positions = self.exchange.get_positions(category="linear", symbol=self.symbol)
            if positions.get('retCode') == 0:
                pos_list = positions['result']['list']
                self.position = pos_list[0] if pos_list and float(pos_list[0]['size']) > 0 else None
        except Exception as e:
            print(f"❌ Position check error: {e}")
            pass
    
    def should_close(self):
        """Determine if position should be closed based on net targets"""
        if not self.position:
            return False, ""
        
        current_price = float(self.price_data['close'].iloc[-1])
        entry_price = float(self.position.get('avgPrice', 0))
        side = self.position.get('side', '')
        
        if entry_price == 0:
            return False, ""
        
        # Calculate NET targets
        net_tp, net_sl = self.calculate_net_targets(entry_price, side)
        
        # Check against NET targets
        if side == "Buy":
            if current_price >= net_tp:
                return True, f"take_profit_net_{self.config['net_take_profit']}%"
            if current_price <= net_sl:
                return True, f"stop_loss_net_{self.config['net_stop_loss']}%"
        else:
            if current_price <= net_tp:
                return True, f"take_profit_net_{self.config['net_take_profit']}%"
            if current_price >= net_sl:
                return True, f"stop_loss_net_{self.config['net_stop_loss']}%"
        
        # Check for signal reversal
        indicators = self.calculate_indicators(self.price_data)
        if indicators:
            signals = self.detect_signals(indicators)
            if side == "Buy" and signals['hist_bearish']:
                return True, "macd_reversal"
            elif side == "Sell" and signals['hist_bullish']:
                return True, "macd_reversal"
        
        return False, ""
    
    async def execute_trade(self, signal):
        """Execute maker-only trade with rebate benefits"""
        current_price = signal['price']
        qty = self.config['position_size'] / current_price
        formatted_qty = self.format_qty(qty)
        
        if formatted_qty == "0":
            return
        
        # Calculate limit price with maker offset
        offset_mult = 1 - self.config['maker_offset_pct']/100 if signal['action'] == 'BUY' else 1 + self.config['maker_offset_pct']/100
        limit_price = round(current_price * offset_mult, 4)
        
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
                # Calculate targets for logging
                break_even = self.calculate_break_even(limit_price, signal['action'])
                net_tp, net_sl = self.calculate_net_targets(limit_price, signal['action'])
                
                # Log the trade
                self.current_trade_id, _ = self.logger.log_trade_open(
                    side=signal['action'],
                    expected_price=signal['price'],
                    actual_price=limit_price,
                    qty=float(formatted_qty),
                    stop_loss=net_sl,
                    take_profit=net_tp,
                    info=f"HIST:{signal['macd_hist']:.6f}_RSI:{signal['rsi']:.1f}_BE:{break_even:.4f}"
                )
                
                print(f"✅ MAKER {signal['action']}: {formatted_qty} BNB @ ${limit_price:.4f}")
                print(f"   📊 Break-Even: ${break_even:.4f} | Net TP: ${net_tp:.4f} | Net SL: ${net_sl:.4f}")
                print(f"   📈 MACD Hist:{signal['macd_hist']:.6f} | RSI:{signal['rsi']:.1f} | VWAP:${signal['vwap']:.4f}")
        except Exception as e:
            print(f"❌ Trade failed: {e}")
    
    async def close_position(self, reason):
        """Close position with maker order for rebate benefits"""
        if not self.position:
            return
        
        current_price = float(self.price_data['close'].iloc[-1])
        side = "Sell" if self.position.get('side') == "Buy" else "Buy"
        qty = float(self.position['size'])
        
        # Calculate limit price with offset
        offset_mult = 1 + self.config['maker_offset_pct']/100 if side == "Sell" else 1 - self.config['maker_offset_pct']/100
        limit_price = round(current_price * offset_mult, 4)
        
        try:
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
                # Log the trade close with rebates
                if self.current_trade_id:
                    self.logger.log_trade_close(
                        trade_id=self.current_trade_id,
                        expected_exit=current_price,
                        actual_exit=limit_price,
                        reason=reason,
                        fees_entry=self.config['maker_fee_pct'],
                        fees_exit=self.config['maker_fee_pct']
                    )
                    self.current_trade_id = None
                
                # Calculate NET PnL including fee rebates
                entry_price = float(self.position.get('avgPrice', 0))
                gross_pnl = float(self.position.get('unrealisedPnl', 0))
                fee_earned = (entry_price * qty + current_price * qty) * abs(self.config['maker_fee_pct']) / 100
                net_pnl = gross_pnl + fee_earned
                
                print(f"✅ Closed: {reason} | Gross PnL: ${gross_pnl:.2f} | Net PnL: ${net_pnl:.2f}")
                
                # Clear position after successful close
                self.position = None
                
        except Exception as e:
            print(f"❌ Close failed: {e}")
    
    def show_status(self):
        """Show current status and indicator values"""
        if len(self.price_data) == 0:
            return
        
        price = float(self.price_data['close'].iloc[-1])
        indicators = self.calculate_indicators(self.price_data)
        
        print(f"\n⚡ MACD + VWAP BOT - {self.symbol}")
        print(f"💰 Price: ${price:.4f}")
        
        if indicators:
            print(f"📊 MACD Hist: {indicators['histogram']:.6f} | RSI: {indicators['rsi']:.1f}")
            print(f"📈 VWAP: ${indicators['vwap']:.4f} | EMA: ${indicators['ema']:.4f}")
        
        if self.position:
            entry_price = float(self.position.get('avgPrice', 0))
            side = self.position.get('side', '')
            size = self.position.get('size', '0')
            
            # Calculate current NET PnL with rebates
            gross_pnl = float(self.position.get('unrealisedPnl', 0))
            fee_earned = (entry_price * float(size)) * abs(self.config['maker_fee_pct']) / 100
            net_pnl = gross_pnl + fee_earned
            
            # Calculate break-even and targets
            break_even = self.calculate_break_even(entry_price, side)
            net_tp, net_sl = self.calculate_net_targets(entry_price, side)
            
            emoji = "🟢" if side == "Buy" else "🔴"
            print(f"{emoji} {side}: {size} BNB @ ${entry_price:.4f}")
            print(f"   💵 Gross PnL: ${gross_pnl:.2f} | Net PnL: ${net_pnl:.2f}")
            print(f"   🎯 BE: ${break_even:.4f} | TP: ${net_tp:.4f} | SL: ${net_sl:.4f}")
        else:
            print("⚡ Scanning for MACD + VWAP signals...")
        
        print("-" * 60)
    
    async def run_cycle(self):
        """Run one trading cycle"""
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
        """Main bot loop"""
        if not self.connect():
            print("❌ Failed to connect to exchange")
            return
        
        print(f"✅ Connected! Starting MACD + VWAP bot for {self.symbol}")
        print("📊 Strategy: MACD histogram flip + RSI cross + VWAP alignment")
        print(f"🎯 Net TP: {self.config['net_take_profit']}% | Net SL: {self.config['net_stop_loss']}%")
        print(f"💎 Using MAKER-ONLY orders for {abs(self.config['maker_fee_pct'])}% fee rebate")
        
        try:
            while True:
                await self.run_cycle()
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Bot stopped by user")
            if self.position:
                await self.close_position("manual_stop")
        except Exception as e:
            print(f"⚠️ Runtime error: {e}")
            if self.position:
                await self.close_position("error_stop")
            await asyncio.sleep(5)

if __name__ == "__main__":
    bot = MACDVWAPBot()
    asyncio.run(bot.run())