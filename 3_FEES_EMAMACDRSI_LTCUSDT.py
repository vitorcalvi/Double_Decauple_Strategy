import os
import time
import math
import asyncio
import json
from datetime import datetime, timezone
from typing import Optional, Dict, Any

import pandas as pd
from pybit.unified_trading import HTTP
from dotenv import load_dotenv

load_dotenv()


# ---------------------------
# Utilities
# ---------------------------

def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


# ---------------------------
# Trade Logger (behavior-preserving fields + safer fee math)
# ---------------------------
class TradeLogger:
    def __init__(self, bot_name: str, symbol: str):
        self.demo_mode = os.getenv('DEMO_MODE', 'true').lower() == 'true'
        self.bot_name = bot_name
        self.symbol = symbol
        self.currency = "USDT"
        self.open_trades: Dict[int, Dict[str, Any]] = {}
        self.trade_id = 1000
        os.makedirs("logs", exist_ok=True)
        # Keep the original filename to avoid breaking downstream parsers
        self.log_file = f"logs/3_FEES_EMAMACDRSI_LTCUSDT.log"

    def generate_trade_id(self) -> int:
        self.trade_id += 1
        return self.trade_id

    def log_trade_open(
        self,
        side: str,
        expected_price: float,
        actual_price: float,
        qty: float,
        stop_loss: float,
        take_profit: float,
        info: str = "",
    ):
        trade_id = self.generate_trade_id()
        # Keep original slippage sign convention
        slippage = (actual_price - expected_price) if side == "BUY" else (expected_price - actual_price)

        log_entry = {
            "id": trade_id,
            "bot": self.bot_name,
            "symbol": self.symbol,
            "side": "LONG" if side == "BUY" else "SHORT",
            "action": "OPEN",
            "ts": _now_utc_iso(),
            "expected_price": round(expected_price, 4),
            "actual_price": round(actual_price, 4),
            "slippage": round(slippage, 4),
            "qty": round(qty, 6),
            "stop_loss": round(stop_loss, 4),
            "take_profit": round(take_profit, 4),
            "currency": self.currency,
            "info": info,
        }

        self.open_trades[trade_id] = {
            "entry_time": datetime.now(),
            "entry_price": actual_price,
            "side": side,  # "BUY" or "SELL"
            "qty": qty,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
        }

        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        return trade_id, log_entry

    def log_trade_close(
        self,
        trade_id: int,
        expected_exit: float,
        actual_exit: float,
        reason: str,
        fees_entry_bps: float = 2.0,  # Positive bps, we subtract below
        fees_exit_bps: float = 2.0,
    ):
        if trade_id not in self.open_trades:
            return None

        trade = self.open_trades[trade_id]
        duration = (datetime.now() - trade["entry_time"]).total_seconds()
        # Keep original slippage sign convention
        slippage = (
            (actual_exit - expected_exit)
            if trade["side"] == "SELL"
            else (expected_exit - actual_exit)
        )

        if trade["side"] == "BUY":
            gross_pnl = (actual_exit - trade["entry_price"]) * trade["qty"]
        else:  # SELL short
            gross_pnl = (trade["entry_price"] - actual_exit) * trade["qty"]

        # Convert bps to %
        entry_fee = trade["entry_price"] * trade["qty"] * (fees_entry_bps / 10000.0)
        exit_fee = actual_exit * trade["qty"] * (fees_exit_bps / 10000.0)
        total_fees = entry_fee + exit_fee
        net_pnl = gross_pnl - total_fees

        log_entry = {
            "id": trade_id,
            "bot": self.bot_name,
            "symbol": self.symbol,
            "side": "LONG" if trade["side"] == "BUY" else "SHORT",
            "action": "CLOSE",
            "ts": _now_utc_iso(),
            "duration_sec": int(duration),
            "entry_price": round(trade["entry_price"], 4),
            "expected_exit": round(expected_exit, 4),
            "actual_exit": round(actual_exit, 4),
            "slippage": round(slippage, 4),
            "qty": round(trade["qty"], 6),
            "gross_pnl": round(gross_pnl, 2),
            # Keep original field name used in your previous logs
            "total_fees": round(total_fees, 2),
            "net_pnl": round(net_pnl, 2),
            "reason": reason,
            "currency": self.currency,
        }

        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        del self.open_trades[trade_id]
        return log_entry


# ---------------------------
# EMA + MACD + RSI Bot (stabilized exits, safer rounding, fee-aware logging)
# ---------------------------
class EMAMACDRSIBot:
    def __init__(self):
        self.symbol = 'LTCUSDT'
        self.demo_mode = os.getenv('DEMO_MODE', 'true').lower() == 'true'

        prefix = 'TESTNET_' if self.demo_mode else 'LIVE_'
        self.api_key = os.getenv(f'{prefix}BYBIT_API_KEY')
        self.api_secret = os.getenv(f'{prefix}BYBIT_API_SECRET')

        self.exchange: Optional[HTTP] = None
        self.position: Optional[Dict[str, Any]] = None
        self.pending_order: Optional[Dict[str, Any]] = None
        self.price_data: pd.DataFrame = pd.DataFrame()
        self.account_balance: float = 1000.0

        self.config = {
            'ema_fast': 5,
            'ema_slow': 13,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'rsi_period': 7,            # slightly longer to reduce 1m noise
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'risk_per_trade': 1.0,
            'maker_offset_pct': 0.01,  # 1bp limit offset for maker
            'slippage_pct': 0.02,      # 2bp expected slippage for market
            'net_take_profit': 0.86,
            'net_stop_loss': 0.43,
            'order_timeout': 180,
            'min_notional': 5,
            'trade_cooldown_sec': 90,  # longer cooldown to avoid churn
            'min_hold_bars': 3,        # require at least N bars before opposite-signal exit
            'maker_fee_bps': 2.0,      # estimated; replace with realized from fills if available
            'taker_fee_bps': 5.0,      # estimated; replace with realized from fills if available
            'macd_hist_min': 0.0,      # threshold to avoid micro flips; keep 0 to preserve behavior
        }

        self.tick_size = 0.01
        self.qty_step = 0.01

        self.last_trade_time: float = 0.0
        self.logger = TradeLogger("EMA_MACD_RSI", self.symbol)
        self.current_trade_id: Optional[int] = None
        self.entry_bar_time: Optional[pd.Timestamp] = None

    # -------------
    # Connections & Account
    # -------------
    def connect(self) -> bool:
        try:
            self.exchange = HTTP(demo=self.demo_mode, api_key=self.api_key, api_secret=self.api_secret)
            # Some pybit clients don't have get_server_time; if so, consider connection ok
            try:
                resp = self.exchange.get_server_time()
                return (resp or {}).get('retCode', 0) == 0
            except Exception:
                return True
        except Exception:
            return False

    async def get_account_balance(self) -> bool:
        try:
            result = self.exchange.get_wallet_balance(accountType="UNIFIED", coin="USDT")
            if result.get('retCode') == 0:
                balance_list = result['result']['list']
                if balance_list:
                    for coin in balance_list[0]['coin']:
                        if coin['coin'] == 'USDT':
                            self.account_balance = _safe_float(coin.get('availableToWithdraw'), 1000.0)
                            return True
        except Exception:
            self.account_balance = 1000.0
        return False

    async def get_instrument_info(self) -> bool:
        try:
            result = self.exchange.get_instruments_info(category="linear", symbol=self.symbol)
            if result.get('retCode') == 0:
                info = result['result']['list'][0]
                self.tick_size = _safe_float(info['priceFilter']['tickSize'], 0.01)
                self.qty_step = _safe_float(info['lotSizeFilter']['qtyStep'], 0.01)
                return True
        except Exception:
            pass
        return False

    # -------------
    # Formatting helpers
    # -------------
    def _round_to_tick(self, price: float, side: Optional[str] = None) -> float:
        if self.tick_size <= 0:
            return round(price, 4)
        steps = price / self.tick_size
        if side == 'BUY':  # ensure we don't cross when posting maker BUY
            steps = math.floor(steps)
        elif side == 'SELL':  # ensure we don't cross when posting maker SELL
            steps = math.ceil(steps)
        else:
            steps = round(steps)
        return steps * self.tick_size

    def format_price(self, price: float, side: Optional[str] = None) -> float:
        return self._round_to_tick(price, side=side)

    def format_qty(self, qty: float) -> float:
        if self.qty_step <= 0:
            return round(qty, 2)
        steps = max(1, round(qty / self.qty_step))
        return steps * self.qty_step

    def format_qty_str(self, qty: float) -> str:
        return f"{self.format_qty(qty):.2f}"

    # -------------
    # Sizing & Prices
    # -------------
    def calculate_position_size(self, price: float, stop_loss_price: float) -> float:
        if self.account_balance <= 0:
            return 0.0
        risk_amount = self.account_balance * self.config['risk_per_trade'] / 100.0
        stop_distance = abs(price - stop_loss_price)
        if stop_distance <= 0:
            return 0.0
        position_size = risk_amount / stop_distance
        notional = position_size * price
        return position_size if notional >= self.config['min_notional'] else 0.0

    def estimate_execution_price(self, market_price: float, side: str, is_limit: bool = True) -> float:
        if is_limit:
            px = (
                market_price * (1 - self.config['maker_offset_pct']/100.0)
                if side == 'BUY'
                else market_price * (1 + self.config['maker_offset_pct']/100.0)
            )
            return self.format_price(px, side)
        else:
            px = (
                market_price * (1 + self.config['slippage_pct']/100.0)
                if side == 'BUY'
                else market_price * (1 - self.config['slippage_pct']/100.0)
            )
            # For market orders, price is not sent; this is just for logging/expectations
            return self.format_price(px)

    # -------------
    # Exchange state
    # -------------
    async def check_pending_orders(self) -> bool:
        # Clear stale local flag if we set it but nothing on exchange
        if self.pending_order and time.time() - self.last_trade_time > 30:
            self.pending_order = None
            print("✓ Cleared stale pending order")
        try:
            orders = self.exchange.get_open_orders(category="linear", symbol=self.symbol)
            if orders.get('retCode') != 0:
                return False
            order_list = orders['result']['list']
            if not order_list:
                self.pending_order = None
                return False
            order = order_list[0]
            created_ms = _safe_float(order.get('createdTime'), 0.0)
            age = datetime.now().timestamp() - (created_ms / 1000.0 if created_ms else self.last_trade_time)
            if age > self.config['order_timeout']:
                try:
                    self.exchange.cancel_order(category="linear", symbol=self.symbol, orderId=order['orderId'])
                except Exception:
                    pass
                self.pending_order = None
                return False
            self.pending_order = order
            return True
        except Exception:
            return False

    # -------------
    # Indicators & Signals
    # -------------
    def calculate_indicators(self, df: pd.DataFrame) -> Optional[Dict[str, float]]:
        min_period = max(self.config['ema_slow'], self.config['macd_slow'], self.config['rsi_period']) + 2
        if len(df) < min_period:
            return None
        close = df['close']

        # EMA (use adjust=False for streaming-like behavior)
        ema_fast_series = close.ewm(span=self.config['ema_fast'], adjust=False).mean()
        ema_slow_series = close.ewm(span=self.config['ema_slow'], adjust=False).mean()
        ema_fast = ema_fast_series.iloc[-1]
        ema_slow = ema_slow_series.iloc[-1]

        # MACD
        exp1 = close.ewm(span=self.config['macd_fast'], adjust=False).mean()
        exp2 = close.ewm(span=self.config['macd_slow'], adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=self.config['macd_signal'], adjust=False).mean()
        histogram = macd - signal

        # RSI (Wilder-like)
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        roll_up = gain.ewm(alpha=1/self.config['rsi_period'], adjust=False).mean()
        roll_down = loss.ewm(alpha=1/self.config['rsi_period'], adjust=False).mean()
        rs = roll_up / roll_down.replace(0, 1e-9)
        rsi = 100 - (100 / (1 + rs))
        rsi_val = _safe_float(rsi.iloc[-1], 50.0)

        macd_hist = _safe_float(histogram.iloc[-1], 0.0)
        trend = 'UP' if ema_fast > ema_slow else 'DOWN'
        macd_cross = 'UP' if macd_hist > self.config['macd_hist_min'] else 'DOWN'

        return {
            'trend': trend,
            'rsi': rsi_val,
            'ema_fast': ema_fast,
            'ema_slow': ema_slow,
            'macd': _safe_float(macd.iloc[-1], 0.0),
            'signal': _safe_float(signal.iloc[-1], 0.0),
            'histogram': macd_hist,
            'macd_cross': macd_cross,
        }

    def generate_signal(self, df: pd.DataFrame) -> Optional[Dict[str, float]]:
        # No new signal if holding or pending
        if self.position or self.pending_order:
            return None
        # Cooldown
        time_since_last = datetime.now().timestamp() - self.last_trade_time
        if time_since_last < self.config['trade_cooldown_sec']:
            return None

        indicators = self.calculate_indicators(df)
        if not indicators:
            return None
        price = float(df['close'].iloc[-1])

        # Buy: Trend UP + MACD UP + RSI low
        if (
            indicators['trend'] == 'UP'
            and indicators['macd_cross'] == 'UP'
            and indicators['rsi'] < self.config['rsi_oversold']
        ):
            return {'action': 'BUY', 'price': price, 'rsi': indicators['rsi'], 'macd': indicators['histogram']}

        # Sell: Trend DOWN + MACD DOWN + RSI high
        if (
            indicators['trend'] == 'DOWN'
            and indicators['macd_cross'] == 'DOWN'
            and indicators['rsi'] > self.config['rsi_overbought']
        ):
            return {'action': 'SELL', 'price': price, 'rsi': indicators['rsi'], 'macd': indicators['histogram']}

        return None

    # -------------
    # Market data & positions
    # -------------
    async def get_market_data(self) -> bool:
        try:
            klines = self.exchange.get_kline(category="linear", symbol=self.symbol, interval="1", limit=50)
            if klines.get('retCode') != 0:
                return False
            df = pd.DataFrame(
                klines['result']['list'],
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover']
            )
            df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
            self.price_data = df.sort_values('timestamp').reset_index(drop=True)
            return True
        except Exception:
            return False

    async def check_position(self):
        try:
            positions = self.exchange.get_positions(category="linear", symbol=self.symbol)
            if positions.get('retCode') == 0:
                pos_list = positions['result']['list']
                self.position = pos_list[0] if pos_list and _safe_float(pos_list[0].get('size'), 0.0) > 0 else None
        except Exception:
            pass

    # -------------
    # Exit rules
    # -------------
    def _bars_since_entry(self) -> int:
        if self.entry_bar_time is None or self.price_data.empty:
            return 0
        return int((self.price_data['timestamp'] > self.entry_bar_time).sum())

    def should_close(self) -> (bool, str):
        if not self.position or self.price_data.empty:
            return False, ""

        current_price = float(self.price_data['close'].iloc[-1])
        entry_price = _safe_float(self.position.get('avgPrice'), 0.0)
        if entry_price <= 0:
            return False, ""

        is_long = self.position.get('side') == "Buy"
        profit_pct = (
            (current_price - entry_price) / entry_price * 100.0 if is_long
            else (entry_price - current_price) / entry_price * 100.0
        )

        # Hard TP/SL first
        if profit_pct >= self.config['net_take_profit']:
            return True, "take_profit"
        if profit_pct <= -self.config['net_stop_loss']:
            return True, "stop_loss"

        # Stabilized opposite-signal exit: require min bars since entry
        bars = self._bars_since_entry()
        if bars < self.config['min_hold_bars']:
            return False, ""

        indicators = self.calculate_indicators(self.price_data)
        if indicators:
            if is_long and (indicators['trend'] == 'DOWN' and indicators['macd_cross'] == 'DOWN'):
                return True, "trend_reversal"
            if (not is_long) and (indicators['trend'] == 'UP' and indicators['macd_cross'] == 'UP'):
                return True, "trend_reversal"

        return False, ""

    # -------------
    # Execution
    # -------------
    async def execute_trade(self, signal: Dict[str, float]):
        await self.check_position()
        if self.position:
            print("⚠️ Position already exists, skipping trade")
            return
        if await self.check_pending_orders():
            print("⚠️ Pending order exists, skipping trade")
            return

        time_since_last = datetime.now().timestamp() - self.last_trade_time
        if time_since_last < self.config['trade_cooldown_sec']:
            remaining = self.config['trade_cooldown_sec'] - time_since_last
            print(f"⚠️ Trade cooldown active, wait {remaining:.0f}s")
            return

        await self.get_account_balance()

        market_price = signal['price']
        is_buy = signal['action'] == 'BUY'
        stop_loss_price = (
            market_price * (1 - self.config['net_stop_loss']/100.0) if is_buy
            else market_price * (1 + self.config['net_stop_loss']/100.0)
        )

        qty = self.calculate_position_size(market_price, stop_loss_price)
        if qty <= 0:
            print("⚠️ Position size too small or zero")
            return

        limit_price = self.estimate_execution_price(market_price, signal['action'], is_limit=True)
        qty_str = self.format_qty_str(qty)

        try:
            order = self.exchange.place_order(
                category="linear",
                symbol=self.symbol,
                side="Buy" if is_buy else "Sell",
                orderType="Limit",
                qty=qty_str,
                price=str(limit_price),
                timeInForce="PostOnly"
            )
            if order.get('retCode') == 0:
                self.pending_order = order['result']
                self.last_trade_time = datetime.now().timestamp()

                take_profit = (
                    limit_price * (1 + self.config['net_take_profit']/100.0) if is_buy
                    else limit_price * (1 - self.config['net_take_profit']/100.0)
                )

                self.current_trade_id, _ = self.logger.log_trade_open(
                    side=signal['action'],
                    expected_price=market_price,
                    actual_price=limit_price,
                    qty=self.format_qty(qty),
                    stop_loss=stop_loss_price,
                    take_profit=take_profit,
                    info=f"RSI:{signal['rsi']:.1f}_MACD:{signal['macd']:.3f}"
                )
                self.entry_bar_time = self.price_data['timestamp'].iloc[-1] if not self.price_data.empty else None

                print(
                    f"✅ {signal['action']}: {qty_str} @ ${limit_price:.2f} | "
                    f"RSI: {signal['rsi']:.1f} | MACD: {signal['macd']:.3f} | Balance: ${self.account_balance:.0f}"
                )
        except Exception as e:
            print(f"❌ Trade failed: {e}")

    async def close_position(self, reason: str):
        if not self.position or self.price_data.empty:
            return

        qty = _safe_float(self.position.get('size'), 0.0)
        if qty <= 0:
            return

        is_long = self.position.get('side') == "Buy"
        close_side = "Sell" if is_long else "Buy"
        current_price = float(self.price_data['close'].iloc[-1])

        # Use Market/IOC for stop_loss to ensure we get out; maker limit for TP / trend_reversal
        if reason == 'stop_loss':
            orderType = "Market"
            timeInForce = "IOC"
            execution_price = self.estimate_execution_price(current_price, close_side, is_limit=False)
            fee_exit_bps = self.config['taker_fee_bps']
            place_kwargs = dict(
                category="linear",
                symbol=self.symbol,
                side=close_side,
                orderType=orderType,
                qty=self.format_qty_str(qty),
                reduceOnly=True,
                timeInForce=timeInForce,
            )
        else:
            orderType = "Limit"
            timeInForce = "PostOnly"
            execution_price = self.estimate_execution_price(current_price, close_side, is_limit=True)
            fee_exit_bps = self.config['maker_fee_bps']
            place_kwargs = dict(
                category="linear",
                symbol=self.symbol,
                side=close_side,
                orderType=orderType,
                qty=self.format_qty_str(qty),
                price=str(execution_price),
                reduceOnly=True,
                timeInForce=timeInForce,
            )

        try:
            order = self.exchange.place_order(**place_kwargs)
            if order.get('retCode') == 0:
                if self.current_trade_id:
                    self.logger.log_trade_close(
                        trade_id=self.current_trade_id,
                        expected_exit=current_price,
                        actual_exit=execution_price,
                        reason=reason,
                        fees_entry_bps=self.config['maker_fee_bps'],  # entry posted as maker
                        fees_exit_bps=fee_exit_bps,
                    )
                    self.current_trade_id = None
                print(f"✅ Closed: {reason} @ ${execution_price:.2f}")
        except Exception as e:
            print(f"❌ Close failed: {e}")

    # -------------
    # Status & main loop
    # -------------
    def show_status(self):
        if self.price_data.empty:
            return
        current_price = float(self.price_data['close'].iloc[-1])
        parts = [f"📊 LTC: ${current_price:.2f}", f"💰 Balance: ${self.account_balance:.0f}"]

        if self.position:
            entry = _safe_float(self.position.get('avgPrice'), 0.0)
            side = self.position.get('side', '')
            size = _safe_float(self.position.get('size'), 0.0)
            pnl = _safe_float(self.position.get('unrealisedPnl'), 0.0)
            parts.append(f"📍 {side}: {size:.2f} @ ${entry:.2f} | PnL: ${pnl:.2f}")
        elif self.pending_order:
            order_price = _safe_float(self.pending_order.get('price'), 0.0)
            order_side = self.pending_order.get('side', '')
            created = _safe_float(self.pending_order.get('createdTime'), 0.0)
            age = int(datetime.now().timestamp() - created / 1000.0) if created else 0
            parts.append(f"⏳ {order_side} @ ${order_price:.2f} ({age}s)")
        else:
            indicators = self.calculate_indicators(self.price_data)
            if indicators:
                sig = "⚪ NO SIGNAL"
                if (
                    indicators['trend'] == 'UP' and indicators['macd_cross'] == 'UP' and
                    indicators['rsi'] < self.config['rsi_oversold']
                ):
                    sig = "🟢 BUY SIGNAL"
                elif (
                    indicators['trend'] == 'DOWN' and indicators['macd_cross'] == 'DOWN' and
                    indicators['rsi'] > self.config['rsi_overbought']
                ):
                    sig = "🔴 SELL SIGNAL"
                parts.append(
                    f"RSI: {indicators['rsi']:.1f} | MACD: {indicators['macd_cross']} | "
                    f"Trend: {indicators['trend']} | {sig}"
                )

        if self.last_trade_time > 0:
            remaining = self.config['trade_cooldown_sec'] - (datetime.now().timestamp() - self.last_trade_time)
            if remaining > 0:
                parts.append(f"⏰ Cooldown: {remaining:.0f}s")

        print(" | ".join(parts), end='\r')

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
            print("❌ Failed to connect")
            return
        await self.get_account_balance()
        await self.get_instrument_info()

        print(f"🔧 EMA + MACD + RSI bot for {self.symbol}")
        print("✅ Strategy:")
        print(f"   • EMA: {self.config['ema_fast']}/{self.config['ema_slow']}")
        print(f"   • MACD: {self.config['macd_fast']}/{self.config['macd_slow']}/{self.config['macd_signal']}")
        print(f"   • RSI: {self.config['rsi_period']} | <{self.config['rsi_oversold']} BUY | >{self.config['rsi_overbought']} SELL")
        print(f"   • Trade cooldown: {self.config['trade_cooldown_sec']}s")
        print(f"💰 Account Balance: ${self.account_balance:.2f}")
        print(f"🎯 TP: {self.config['net_take_profit']:.2f}% | SL: {self.config['net_stop_loss']:.2f}%")

        while True:
            try:
                await self.run_cycle()
                await asyncio.sleep(1)
            except KeyboardInterrupt:
                print("\n🛑 Bot stopped")
                try:
                    self.exchange.cancel_all_orders(category="linear", symbol=self.symbol)
                except Exception:
                    pass
                if self.position:
                    await self.close_position("manual_stop")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                await asyncio.sleep(5)


if __name__ == "__main__":
    bot = EMAMACDRSIBot()
    asyncio.run(bot.run())
