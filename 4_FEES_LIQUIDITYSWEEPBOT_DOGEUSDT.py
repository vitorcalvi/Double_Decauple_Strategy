import os
import asyncio
import pandas as pd
import json
from datetime import datetime, timezone
from collections import deque
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

class LiquiditySweepBot:
    """Smart-Money Liquidity Sweep Strategy"""
    
    def __init__(self):
        self.symbol = 'DOGEUSDT'
        self.demo_mode = os.getenv('DEMO_MODE', 'true').lower() == 'true'
        
        # API setup
        prefix = 'TESTNET_' if self.demo_mode else 'LIVE_'
        self.api_key = os.getenv(f'{prefix}BYBIT_API_KEY')
        self.api_secret = os.getenv(f'{prefix}BYBIT_API_SECRET')
        self.exchange = None
        
        # Trading state
        self.position = None
        self.price_data = pd.DataFrame()
        
        # Strategy parameters with Fee Calculations
        self.config = {
            'timeframe': '5',
            'liquidity_lookback': 50,
            'order_block_lookback': 20,
            'sweep_threshold': 0.15,
            'retracement_ratio': 0.5,
            'position_size': 100,
            'lookback': 100,
            'maker_offset_pct': 0.01,
            'maker_fee_pct': -0.04,  # Negative = rebate
            'gross_take_profit': 1.5,
            'gross_stop_loss': 0.5,
            'net_take_profit': 1.58,  # 1.5 + 0.08 rebate
            'net_stop_loss': 0.42,    # 0.5 - 0.08 rebate
        }
        
        # Liquidity tracking
        self.liquidity_pools = {'highs': deque(maxlen=10), 'lows': deque(maxlen=10)}
        self.order_blocks = []
        
        # Trade logging
        self.logger = TradeLogger("LIQUIDITY_SWEEP", self.symbol)
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
        """Format quantity for DOGE"""
        return str(int(round(qty))) if qty >= 1.0 else "0"
    
    def calculate_break_even(self, entry_price, side):
        """Calculate break-even price including fee rebates"""
        fee_impact = 2 * abs(self.config['maker_fee_pct']) / 100
        multiplier = 1 - fee_impact if side == "Buy" else 1 + fee_impact
        return entry_price * multiplier
    
    def calculate_net_targets(self, entry_price, side):
        """Calculate net TP/SL accounting for round-trip fee rebates"""
        if side == "Buy":
            net_tp = entry_price * (1 + self.config['net_take_profit'] / 100)
            net_sl = entry_price * (1 - self.config['net_stop_loss'] / 100)
        else:
            net_tp = entry_price * (1 - self.config['net_take_profit'] / 100)
            net_sl = entry_price * (1 + self.config['net_stop_loss'] / 100)
        return net_tp, net_sl
    
    def identify_liquidity_pools(self, df):
        """Identify liquidity pools (areas of interest)"""
        if len(df) < self.config['liquidity_lookback']:
            return
        
        window = 5
        highs = df['high'].rolling(window=window, center=True).max()
        lows = df['low'].rolling(window=window, center=True).min()
        
        self.liquidity_pools['highs'].clear()
        self.liquidity_pools['lows'].clear()
        
        # Find significant highs and lows
        for i in range(len(df) - 10, max(0, len(df) - self.config['liquidity_lookback']), -1):
            # Check significant high
            if df['high'].iloc[i] == highs.iloc[i]:
                is_significant = (
                    all(df['high'].iloc[max(0, i-3):i] < df['high'].iloc[i]) and
                    all(df['high'].iloc[i+1:min(len(df), i+4)] < df['high'].iloc[i])
                )
                if is_significant:
                    self.liquidity_pools['highs'].append({
                        'price': df['high'].iloc[i],
                        'index': i,
                        'volume': df['volume'].iloc[i]
                    })
            
            # Check significant low
            if df['low'].iloc[i] == lows.iloc[i]:
                is_significant = (
                    all(df['low'].iloc[max(0, i-3):i] > df['low'].iloc[i]) and
                    all(df['low'].iloc[i+1:min(len(df), i+4)] > df['low'].iloc[i])
                )
                if is_significant:
                    self.liquidity_pools['lows'].append({
                        'price': df['low'].iloc[i],
                        'index': i,
                        'volume': df['volume'].iloc[i]
                    })
    
    def identify_order_blocks(self, df):
        """Identify order blocks (institutional activity)"""
        if len(df) < self.config['order_block_lookback']:
            return []
        
        blocks = []
        
        for i in range(len(df) - 3, max(0, len(df) - self.config['order_block_lookback']), -1):
            # Bullish order block
            if (df['close'].iloc[i] < df['open'].iloc[i] and
                df['close'].iloc[i+1] > df['open'].iloc[i+1] and
                (df['close'].iloc[i+1] - df['open'].iloc[i+1]) > 2 * abs(df['close'].iloc[i] - df['open'].iloc[i])):
                
                blocks.append({
                    'type': 'bullish',
                    'high': df['high'].iloc[i],
                    'low': df['low'].iloc[i],
                    'index': i
                })
            
            # Bearish order block
            elif (df['close'].iloc[i] > df['open'].iloc[i] and
                  df['close'].iloc[i+1] < df['open'].iloc[i+1] and
                  abs(df['close'].iloc[i+1] - df['open'].iloc[i+1]) > 2 * (df['close'].iloc[i] - df['open'].iloc[i])):
                
                blocks.append({
                    'type': 'bearish',
                    'high': df['high'].iloc[i],
                    'low': df['low'].iloc[i],
                    'index': i
                })
        
        self.order_blocks = blocks[-5:] if blocks else []
    
    def detect_liquidity_sweep(self, df):
        """Detect liquidity sweep (smart money activity)"""
        if len(df) < 3:
            return None
        
        current_high = df['high'].iloc[-1]
        current_low = df['low'].iloc[-1]
        current_close = df['close'].iloc[-1]
        current_volume = df['volume'].iloc[-1]
        avg_volume = df['volume'].iloc[-20:].mean()
        
        # Check sweep above liquidity
        for pool in self.liquidity_pools['highs']:
            sweep_level = pool['price'] * (1 + self.config['sweep_threshold'] / 100)
            
            if current_high > sweep_level and current_close < pool['price']:
                return {
                    'type': 'bearish_sweep',
                    'swept_level': pool['price'],
                    'volume_ratio': current_volume / avg_volume if avg_volume > 0 else 1
                }
        
        # Check sweep below liquidity
        for pool in self.liquidity_pools['lows']:
            sweep_level = pool['price'] * (1 - self.config['sweep_threshold'] / 100)
            
            if current_low < sweep_level and current_close > pool['price']:
                return {
                    'type': 'bullish_sweep',
                    'swept_level': pool['price'],
                    'volume_ratio': current_volume / avg_volume if avg_volume > 0 else 1
                }
        
        return None
    
    def check_order_block_confluence(self, sweep_type, current_price):
        """Check for order block confluence"""
        if not self.order_blocks:
            return False
        
        for block in self.order_blocks:
            if ((sweep_type == 'bullish_sweep' and block['type'] == 'bullish') or
                (sweep_type == 'bearish_sweep' and block['type'] == 'bearish')):
                if block['low'] <= current_price <= block['high']:
                    return True
        
        return False
    
    def generate_signal(self, df):
        """Generate trading signal based on liquidity sweep and order blocks"""
        if len(df) < self.config['lookback']:
            return None
        
        self.identify_liquidity_pools(df)
        self.identify_order_blocks(df)
        
        sweep = self.detect_liquidity_sweep(df)
        if not sweep:
            return None
        
        current_price = df['close'].iloc[-1]
        has_confluence = self.check_order_block_confluence(sweep['type'], current_price)
        
        action = 'BUY' if sweep['type'] == 'bullish_sweep' else 'SELL' if sweep['type'] == 'bearish_sweep' else None
        
        if action:
            return {
                'action': action,
                'price': current_price,
                'swept_level': sweep['swept_level'],
                'volume_ratio': sweep['volume_ratio'],
                'confluence': has_confluence
            }
        
        return None
    
    async def get_market_data(self):
        """Retrieve market data from exchange"""
        try:
            klines = self.exchange.get_kline(
                category="linear",
                symbol=self.symbol,
                interval=self.config['timeframe'],
                limit=self.config['lookback']
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
                return True, f"take_profit_1.5RR_net_{self.config['net_take_profit']}%"
            if current_price <= net_sl:
                return True, f"stop_loss_net_{self.config['net_stop_loss']}%"
            
            # Check for next liquidity pool
            for pool in self.liquidity_pools['highs']:
                if current_price >= pool['price'] * 0.995:
                    return True, "next_liquidity_pool"
        else:
            if current_price <= net_tp:
                return True, f"take_profit_1.5RR_net_{self.config['net_take_profit']}%"
            if current_price >= net_sl:
                return True, f"stop_loss_net_{self.config['net_stop_loss']}%"
            
            # Check for next liquidity pool
            for pool in self.liquidity_pools['lows']:
                if current_price <= pool['price'] * 1.005:
                    return True, "next_liquidity_pool"
        
        return False, ""
    
    async def execute_trade(self, signal):
        """Execute maker-only trade with rebate benefits"""
        qty = self.config['position_size'] / signal['price']
        formatted_qty = self.format_qty(qty)
        
        if formatted_qty == "0":
            return
        
        # Calculate limit price
        offset_mult = 1 - self.config['maker_offset_pct']/100 if signal['action'] == 'BUY' else 1 + self.config['maker_offset_pct']/100
        limit_price = round(signal['price'] * offset_mult, 4)
        
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
                    info=f"swept:{signal['swept_level']:.4f}_BE:{break_even:.4f}_conf:{signal['confluence']}"
                )
                
                confluence_str = "WITH_OB" if signal['confluence'] else "NO_OB"
                
                print(f"🎯 MAKER {signal['action']}: {formatted_qty} DOGE @ ${limit_price:.4f}")
                print(f"   📊 Break-Even: ${break_even:.4f} | Net TP: ${net_tp:.4f} | Net SL: ${net_sl:.4f}")
                print(f"   💎 Liquidity Swept: ${signal['swept_level']:.4f} | Volume: {signal['volume_ratio']:.1f}x")
                print(f"   📦 Order Block Confluence: {'✅' if signal['confluence'] else '❌'}")
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
                
                print(f"💰 Closed: {reason} | Gross PnL: ${gross_pnl:.2f} | Net PnL: ${net_pnl:.2f}")
                
                # Clear position after successful close
                self.position = None
                
        except Exception as e:
            print(f"❌ Close failed: {e}")
    
    def show_status(self):
        """Show current status and key market levels"""
        if len(self.price_data) == 0:
            return
        
        current_price = float(self.price_data['close'].iloc[-1])
        
        print(f"\n💎 Smart-Money Liquidity Sweep - {self.symbol}")
        print(f"💰 Price: ${current_price:.4f}")
        
        if self.liquidity_pools['highs']:
            top_resistance = max(self.liquidity_pools['highs'], key=lambda x: x['price'])
            print(f"🔴 Next Resistance: ${top_resistance['price']:.4f}")
        
        if self.liquidity_pools['lows']:
            bottom_support = min(self.liquidity_pools['lows'], key=lambda x: x['price'])
            print(f"🟢 Next Support: ${bottom_support['price']:.4f}")
        
        if self.order_blocks:
            bullish = sum(1 for b in self.order_blocks if b['type'] == 'bullish')
            bearish = sum(1 for b in self.order_blocks if b['type'] == 'bearish')
            print(f"📦 Order Blocks: {bullish} Bullish | {bearish} Bearish")
        
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
            print(f"{emoji} {side}: {size} DOGE @ ${entry_price:.4f}")
            print(f"   💵 Gross PnL: ${gross_pnl:.2f} | Net PnL: ${net_pnl:.2f}")
            print(f"   🎯 BE: ${break_even:.4f} | TP: ${net_tp:.4f} | SL: ${net_sl:.4f}")
        else:
            print("🔍 Scanning for liquidity sweeps...")
        
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
        
        print(f"💎 Smart-Money Liquidity Sweep Bot")
        print(f"⏰ Timeframe: {self.config['timeframe']} minutes")
        print(f"🎯 Net TP: 1.5 RR ({self.config['net_take_profit']}%) | Net SL: {self.config['net_stop_loss']}%")
        print(f"💎 Using MAKER-ONLY orders for {abs(self.config['maker_fee_pct'])}% fee rebate")
        
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
            if self.position:
                await self.close_position("error_stop")
            await asyncio.sleep(5)

if __name__ == "__main__":
    bot = LiquiditySweepBot()
    asyncio.run(bot.run())