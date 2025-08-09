import os
import asyncio
import pandas as pd
import numpy as np
import json
import time
from datetime import datetime, timezone
from pybit.unified_trading import HTTP
from dotenv import load_dotenv

load_dotenv()

class TradeLogger:
    """Handles trade logging with fee calculations"""
    
    def __init__(self, bot_name, symbol):
        self.bot_name = bot_name
        self.symbol = symbol
        self.currency = "USDT"
        self.open_trades = {}
        self.trade_id = 1000
        self.daily_pnl = 0
        self.consecutive_losses = 0
        self.max_daily_loss = 50
        
        os.makedirs("logs", exist_ok=True)
        self.log_file = f"logs/8_FEES_VWAP_RSI_DIV_AVAXUSDT.log"
    
    def generate_trade_id(self):
        self.trade_id += 1
        return self.trade_id
    
    def log_trade_open(self, side, expected_price, actual_price, qty, stop_loss, take_profit, info=""):
        trade_id = self.generate_trade_id()
        
        log_entry = {
            "id": trade_id,
            "bot": self.bot_name,
            "symbol": self.symbol,
            "side": "LONG" if side == "BUY" else "SHORT",
            "action": "OPEN",
            "ts": datetime.now(timezone.utc).isoformat(),
            "expected_price": round(expected_price, 4),
            "actual_price": round(actual_price, 4),
            "slippage": 0,
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
        
        if trade["side"] == "BUY":
            gross_pnl = (actual_exit - trade["entry_price"]) * trade["qty"]
        else:
            gross_pnl = (trade["entry_price"] - actual_exit) * trade["qty"]
        
        entry_rebate = trade["entry_price"] * trade["qty"] * abs(fees_entry) / 100
        exit_rebate = actual_exit * trade["qty"] * abs(fees_exit) / 100
        total_rebates = entry_rebate + exit_rebate
        net_pnl = gross_pnl + total_rebates
        
        self.daily_pnl += net_pnl
        if net_pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
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
            "slippage": 0,
            "qty": round(trade["qty"], 6),
            "gross_pnl": round(gross_pnl, 2),
            "fee_rebates": {
                "entry": round(entry_rebate, 2),
                "exit": round(exit_rebate, 2),
                "total": round(total_rebates, 2)
            },
            "net_pnl": round(net_pnl, 2),
            "reason": reason,
            "currency": self.currency
        }
        
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        
        del self.open_trades[trade_id]
        return log_entry


class VWAPRSIDivergenceBot:
    """VWAP + RSI Divergence Trading Bot with proper instrument info"""
    
    def __init__(self):
        # Exchange settings
        self.symbol = 'AVAXUSDT'
        self.demo_mode = os.getenv('DEMO_MODE', 'true').lower() == 'true'
        
        prefix = 'TESTNET_' if self.demo_mode else 'LIVE_'
        self.api_key = os.getenv(f'{prefix}BYBIT_API_KEY')
        self.api_secret = os.getenv(f'{prefix}BYBIT_API_SECRET')
        self.exchange = None
        
        # Trading state
        self.position = None
        self.pending_order = None
        self.price_data = pd.DataFrame()
        self.account_balance = 1000
        
        # Instrument info (will be fetched from exchange)
        self.tick_size = 0.001  # Default, will be updated
        self.qty_step = 0.1      # Default, will be updated
        self.min_qty = 0.1       # Default, will be updated
        self.min_notional = 5    # Minimum order value in USDT
        
        # Configuration
        self.config = {
            'timeframe': '5',
            'rsi_period': 9,
            'divergence_lookback': 4,
            'ema_period': 50,
            'risk_per_trade': 2.0,
            'maker_offset_pct': 0.01,
            'maker_fee_pct': -0.04,
            'net_take_profit': 0.70,
            'net_stop_loss': 0.35,
            'slippage_basis_points': 3,
        }
        
        # Trade management
        self.last_trade_time = 0
        self.trade_cooldown = 30
        self.last_order_time = 0
        self.order_timeout = 60
        
        # Logging
        self.logger = TradeLogger("VWAP_RSI_DIV_FIXED", self.symbol)
        self.current_trade_id = None
        
        # Debug mode
        self.debug_mode = True
        self.signal_attempts = 0
    
    def connect(self):
        """Establish connection to exchange"""
        try:
            self.exchange = HTTP(
                demo=self.demo_mode,
                api_key=self.api_key,
                api_secret=self.api_secret
            )
            result = self.exchange.get_server_time()
            connected = result.get('retCode') == 0
            if connected:
                print(f"✅ Connected to {'TESTNET' if self.demo_mode else 'LIVE'} Bybit")
            return connected
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False
    
    async def get_instrument_info(self):
        """Fetch instrument specifications from exchange"""
        try:
            result = self.exchange.get_instruments_info(
                category="linear",
                symbol=self.symbol
            )
            
            if result.get('retCode') == 0:
                info = result['result']['list'][0]
                
                # Get price filter
                price_filter = info.get('priceFilter', {})
                self.tick_size = float(price_filter.get('tickSize', 0.001))
                
                # Get lot size filter
                lot_filter = info.get('lotSizeFilter', {})
                self.qty_step = float(lot_filter.get('qtyStep', 0.1))
                self.min_qty = float(lot_filter.get('minOrderQty', 0.1))
                self.max_qty = float(lot_filter.get('maxOrderQty', 10000))
                
                # Get min notional
                self.min_notional = float(lot_filter.get('minNotionalValue', 5))
                
                print(f"📋 Instrument Info for {self.symbol}:")
                print(f"   • Tick Size: {self.tick_size}")
                print(f"   • Qty Step: {self.qty_step}")
                print(f"   • Min Qty: {self.min_qty}")
                print(f"   • Min Notional: ${self.min_notional}")
                
                return True
                
        except Exception as e:
            print(f"⚠️ Could not fetch instrument info: {e}")
            print(f"   Using defaults: qty_step={self.qty_step}, min_qty={self.min_qty}")
        
        return False
    
    def format_price(self, price):
        """Format price according to tick size"""
        if self.tick_size == 0:
            return price
        return round(round(price / self.tick_size) * self.tick_size, 8)
    
    def format_qty(self, qty):
        """Format quantity according to exchange requirements"""
        if qty < self.min_qty:
            return None
        
        # Round to qty_step
        rounded_qty = round(qty / self.qty_step) * self.qty_step
        
        # Ensure it meets minimum
        if rounded_qty < self.min_qty:
            rounded_qty = self.min_qty
        
        # Format with appropriate decimals
        if self.qty_step >= 1:
            return str(int(rounded_qty))
        else:
            decimals = len(str(self.qty_step).split('.')[-1])
            return f"{rounded_qty:.{decimals}f}"
    
    async def get_account_balance(self):
        """Fetch current account balance"""
        try:
            wallet = self.exchange.get_wallet_balance(accountType="UNIFIED")
            if wallet.get('retCode') == 0:
                for coin in wallet['result']['list'][0]['coin']:
                    if coin['coin'] == 'USDT':
                        self.account_balance = float(coin['availableToWithdraw'])
                        return True
        except Exception as e:
            print(f"⚠️ Using fallback balance: ${self.account_balance}")
        return False
    
    def calculate_position_size(self, price, stop_loss_price):
        """Calculate position size based on risk management"""
        if self.account_balance <= 0 or stop_loss_price == price:
            return 0
        
        risk_amount = self.account_balance * (self.config['risk_per_trade'] / 100)
        stop_distance = abs(price - stop_loss_price) / price
        
        if stop_distance == 0:
            return 0
        
        # Position size in base currency
        position_size = risk_amount / (price * stop_distance)
        
        # Check minimum notional
        if position_size * price < self.min_notional:
            position_size = self.min_notional / price
        
        # Cap at 10% of account
        max_position = (self.account_balance * 0.1) / price
        
        return min(position_size, max_position)
    
    def calculate_vwap(self, df):
        """Calculate Volume Weighted Average Price"""
        if len(df) < 20:
            return None
        
        try:
            recent = df.tail(min(100, len(df)))
            typical_price = (recent['high'] + recent['low'] + recent['close']) / 3
            cumulative_tpv = (typical_price * recent['volume']).cumsum()
            cumulative_volume = recent['volume'].cumsum()
            
            if cumulative_volume.iloc[-1] == 0:
                return None
            
            vwap = cumulative_tpv / cumulative_volume
            return float(vwap.iloc[-1])
        except Exception:
            return None
    
    def calculate_rsi(self, prices):
        """Calculate RSI with proper handling"""
        if len(prices) < self.config['rsi_period'] + 1:
            return pd.Series([50] * len(prices))
        
        try:
            delta = prices.diff()
            gain = delta.where(delta > 0, 0).rolling(window=self.config['rsi_period']).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=self.config['rsi_period']).mean()
            
            rs = gain / (loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))
            
            rsi = rsi.fillna(50)
            return rsi
        except Exception:
            return pd.Series([50] * len(prices))
    
    def generate_signal(self, df):
        """Generate trading signal"""
        if len(df) < 50:
            return None
        
        current_price = float(df['close'].iloc[-1])
        vwap = self.calculate_vwap(df)
        rsi = self.calculate_rsi(df['close']).iloc[-1]
        
        self.signal_attempts += 1
        
        # Debug mode: simplified signals
        if self.debug_mode:
            if self.signal_attempts % 10 == 0:
                print(f"📊 Check #{self.signal_attempts}: Price=${current_price:.4f}, RSI={rsi:.1f}")
            
            # Relaxed conditions for testing
            if rsi < 40 and vwap and current_price < vwap:
                print(f"🟢 BUY Signal: RSI={rsi:.1f}, Price < VWAP")
                return {
                    'action': 'BUY',
                    'price': current_price,
                    'vwap': vwap,
                    'rsi': rsi,
                    'divergence_strength': 5.0
                }
            elif rsi > 60 and vwap and current_price > vwap:
                print(f"🔴 SELL Signal: RSI={rsi:.1f}, Price > VWAP")
                return {
                    'action': 'SELL',
                    'price': current_price,
                    'vwap': vwap,
                    'rsi': rsi,
                    'divergence_strength': 5.0
                }
        
        return None
    
    async def get_market_data(self):
        """Fetch market data from exchange"""
        try:
            klines = self.exchange.get_kline(
                category="linear",
                symbol=self.symbol,
                interval=self.config['timeframe'],
                limit=100
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
        """Check current position status"""
        try:
            positions = self.exchange.get_positions(category="linear", symbol=self.symbol)
            if positions.get('retCode') != 0:
                return False
            
            pos_list = positions['result']['list']
            
            if pos_list and float(pos_list[0].get('size', 0)) > 0:
                self.position = pos_list[0]
                return True
            else:
                if self.position:
                    print("✅ Position closed - clearing state")
                self.position = None
                return False
                
        except Exception as e:
            print(f"❌ Position check error: {e}")
            return False
    
    async def check_pending_orders(self):
        """Check and manage pending orders"""
        try:
            orders = self.exchange.get_open_orders(category="linear", symbol=self.symbol)
            if orders.get('retCode') != 0:
                return False
            
            order_list = orders['result']['list']
            
            if order_list:
                self.pending_order = order_list[0]
                
                created_time = int(order_list[0].get('createdTime', 0))
                if created_time and (time.time() * 1000 - created_time) > self.order_timeout * 1000:
                    print("🗑️ Cancelling stale order")
                    self.exchange.cancel_order(
                        category="linear",
                        symbol=self.symbol,
                        orderId=order_list[0]['orderId']
                    )
                    self.pending_order = None
            else:
                if self.pending_order:
                    print("📝 Order filled or cancelled")
                self.pending_order = None
                
        except Exception as e:
            print(f"❌ Order check error: {e}")
            return False
    
    def should_close(self):
        """Determine if position should be closed"""
        if not self.position:
            return False, ""
        
        current_price = float(self.price_data['close'].iloc[-1])
        entry_price = float(self.position.get('avgPrice', 0))
        side = self.position.get('side', '')
        
        if entry_price == 0:
            return False, ""
        
        if side == "Buy":
            profit_pct = ((current_price - entry_price) / entry_price) * 100
        else:
            profit_pct = ((entry_price - current_price) / entry_price) * 100
        
        if profit_pct >= self.config['net_take_profit']:
            return True, "take_profit"
        if profit_pct <= -self.config['net_stop_loss']:
            return True, "stop_loss"
        
        rsi = self.calculate_rsi(self.price_data['close']).iloc[-1]
        if (side == "Buy" and rsi > 70) or (side == "Sell" and rsi < 30):
            return True, f"rsi_extreme_{rsi:.1f}"
        
        return False, ""
    
    async def execute_trade(self, signal):
        """Execute trade based on signal"""
        if time.time() - self.last_trade_time < self.trade_cooldown:
            return
        
        if self.pending_order:
            return
        
        await self.get_account_balance()
        
        if self.account_balance < 10:
            print(f"❌ Insufficient balance: ${self.account_balance:.2f}")
            return
        
        # Calculate position size
        stop_loss_pct = self.config['net_stop_loss'] / 100
        if signal['action'] == 'BUY':
            stop_loss_price = signal['price'] * (1 - stop_loss_pct)
        else:
            stop_loss_price = signal['price'] * (1 + stop_loss_pct)
        
        qty = self.calculate_position_size(signal['price'], stop_loss_price)
        formatted_qty = self.format_qty(qty)
        
        if not formatted_qty:
            print(f"❌ Position size too small: {qty:.6f}")
            return
        
        # Calculate limit price
        if signal['action'] == 'BUY':
            limit_price = self.format_price(signal['price'] * (1 - self.config['maker_offset_pct']/100))
        else:
            limit_price = self.format_price(signal['price'] * (1 + self.config['maker_offset_pct']/100))
        
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
                self.last_trade_time = time.time()
                self.last_order_time = time.time()
                self.pending_order = order['result']
                
                if signal['action'] == 'BUY':
                    take_profit = limit_price * (1 + self.config['net_take_profit']/100)
                    stop_loss = limit_price * (1 - self.config['net_stop_loss']/100)
                else:
                    take_profit = limit_price * (1 - self.config['net_take_profit']/100)
                    stop_loss = limit_price * (1 + self.config['net_stop_loss']/100)
                
                self.current_trade_id, _ = self.logger.log_trade_open(
                    side=signal['action'],
                    expected_price=signal['price'],
                    actual_price=limit_price,
                    qty=float(formatted_qty),
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    info=f"vwap:{signal.get('vwap', 0):.4f}_rsi:{signal['rsi']:.1f}"
                )
                
                print(f"✅ {signal['action']} ORDER: {formatted_qty} @ ${limit_price:.4f}")
                print(f"   🎯 TP: ${take_profit:.4f} | SL: ${stop_loss:.4f}")
                
        except Exception as e:
            print(f"❌ Trade failed: {e}")
    
    async def close_position(self, reason):
        """Close current position"""
        if not self.position:
            return
        
        current_price = float(self.price_data['close'].iloc[-1])
        side = "Sell" if self.position.get('side') == "Buy" else "Buy"
        qty = self.position.get('size', '0')
        
        if side == "Sell":
            limit_price = self.format_price(current_price * (1 + self.config['maker_offset_pct']/100))
        else:
            limit_price = self.format_price(current_price * (1 - self.config['maker_offset_pct']/100))
        
        try:
            order = self.exchange.place_order(
                category="linear",
                symbol=self.symbol,
                side=side,
                orderType="Limit",
                qty=str(qty),
                price=str(limit_price),
                timeInForce="PostOnly",
                reduceOnly=True
            )
            
            if order.get('retCode') == 0:
                if self.current_trade_id:
                    self.logger.log_trade_close(
                        trade_id=self.current_trade_id,
                        expected_exit=current_price,
                        actual_exit=limit_price,
                        reason=reason
                    )
                    self.current_trade_id = None
                
                print(f"✅ Position closed: {reason} @ ${limit_price:.4f}")
                self.position = None
                
        except Exception as e:
            print(f"❌ Close failed: {e}")
    
    def show_status(self):
        """Display current bot status"""
        if len(self.price_data) == 0:
            return
        
        current_price = float(self.price_data['close'].iloc[-1])
        vwap = self.calculate_vwap(self.price_data)
        rsi = self.calculate_rsi(self.price_data['close']).iloc[-1]
        
        print(f"\n{'='*60}")
        print(f"📈 VWAP+RSI Bot - {self.symbol} | Mode: {'TESTNET' if self.demo_mode else 'LIVE'}")
        print(f"💰 Price: ${current_price:.4f} | Balance: ${self.account_balance:.2f}")
        
        if vwap:
            position = "Above" if current_price > vwap else "Below"
            print(f"📊 VWAP: ${vwap:.4f} ({position}) | RSI: {rsi:.1f}")
        
        if self.position:
            entry = float(self.position.get('avgPrice', 0))
            side = self.position.get('side', '')
            size = self.position.get('size', '0')
            
            if side == "Buy":
                pnl_pct = ((current_price - entry) / entry * 100) if entry > 0 else 0
            else:
                pnl_pct = ((entry - current_price) / entry * 100) if entry > 0 else 0
            
            emoji = "🟢" if side == "Buy" else "🔴"
            print(f"{emoji} Position: {side} {size} @ ${entry:.4f} | PnL: {pnl_pct:+.2f}%")
        
        if self.pending_order:
            print(f"⏳ Pending order active")
        
        print(f"{'='*60}")
    
    async def run_cycle(self):
        """Main trading cycle"""
        if self.logger.daily_pnl < -self.logger.max_daily_loss:
            print(f"🔴 EMERGENCY STOP: Daily loss ${abs(self.logger.daily_pnl):.2f}")
            if self.position:
                await self.close_position("emergency_stop")
            return
        
        if not await self.get_market_data():
            return
        
        await self.check_pending_orders()
        await self.check_position()
        
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
        """Main bot loop"""
        if not self.connect():
            print("❌ Failed to connect to exchange")
            return
        
        # Fetch instrument info
        await self.get_instrument_info()
        
        print(f"\n🚀 Starting VWAP + RSI Divergence Bot")
        print(f"📊 Symbol: {self.symbol}")
        print(f"⚙️ Mode: {'DEBUG' if self.debug_mode else 'NORMAL'}")
        print(f"💎 Risk: {self.config['risk_per_trade']}% per trade")
        print(f"🎯 TP: {self.config['net_take_profit']}% | SL: {self.config['net_stop_loss']}%")
        
        try:
            while True:
                await self.run_cycle()
                await asyncio.sleep(10)
                
        except KeyboardInterrupt:
            print("\n🛑 Bot stopped by user")
            if self.position:
                await self.close_position("manual_stop")
                
        except Exception as e:
            print(f"⚠️ Runtime error: {e}")
            await asyncio.sleep(5)


if __name__ == "__main__":
    bot = VWAPRSIDivergenceBot()
    asyncio.run(bot.run())