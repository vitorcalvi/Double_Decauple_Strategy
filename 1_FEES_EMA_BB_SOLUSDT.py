#!/usr/bin/env python3
"""
EMA + BB Trend-Following Bot (Bybit TESTNET, linear futures) with Terminal Monitor

Fixes & updates:
  • Strong-trend override lets signals through even when vol filters fail (no more '–' blocking steady grind-ups).
  • Breakout levels computed from prior 20 bars (exclude current) and smaller breakout buffer (0.02%).
  • Adds trend continuation entry (above EMA fast, momentum rising, RSI>=60).
  • Tighter price debounce (0.1%).
  • Monitor shows signal REASON.

.env required:
  TESTNET_BYBIT_API_KEY=xxx
  TESTNET_BYBIT_API_SECRET=xxx
  DEMO_MODE=true
"""

import os
import asyncio
import json
import time
import itertools
import shutil
from datetime import datetime, timezone

import pandas as pd
from dotenv import load_dotenv
from pybit.unified_trading import HTTP

load_dotenv(override=True)


class TradeLogger:
    def __init__(self, bot_name, symbol):
        self.bot_name = bot_name
        self.symbol = symbol
        self.currency = "USDT"
        self.open_trades = {}
        self.trade_id = 1000
        self.daily_pnl = 0.0
        self.consecutive_losses = 0
        self.max_daily_loss = 50.0  # USD
        os.makedirs("logs", exist_ok=True)
        self.log_file = f"logs/{bot_name}_{symbol}.log"

    def generate_trade_id(self):
        self.trade_id += 1
        return self.trade_id

    def _write(self, payload: dict):
        with open(self.log_file, "a") as f:
            f.write(json.dumps(payload) + "\n")

    def log_trade_open(self, side, expected_price, actual_price, qty, stop_loss, take_profit, info=""):
        trade_id = self.generate_trade_id()
        now = datetime.now(timezone.utc).isoformat()
        entry = {
            "id": trade_id,
            "bot": self.bot_name,
            "symbol": self.symbol,
            "side": "LONG" if side == "BUY" else "SHORT",
            "action": "OPEN",
            "ts": now,
            "expected_price": round(float(expected_price), 6),
            "actual_price": round(float(actual_price), 6),
            "slippage": 0.0,  # PostOnly assumed
            "qty": float(qty),
            "stop_loss": round(float(stop_loss), 6),
            "take_profit": round(float(take_profit), 6),
            "currency": self.currency,
            "info": info,
        }
        self.open_trades[trade_id] = {
            "entry_time": datetime.now(),
            "entry_price": float(actual_price),
            "side": "BUY" if side == "BUY" else "SELL",
            "qty": float(qty),
            "stop_loss": float(stop_loss),
            "take_profit": float(take_profit),
        }
        self._write(entry)
        return trade_id, entry

    def log_trade_close(self, trade_id, expected_exit, actual_exit, reason):
        if trade_id not in self.open_trades:
            return None

        tr = self.open_trades[trade_id]
        duration = (datetime.now() - tr["entry_time"]).total_seconds()
        entry_price = float(tr["entry_price"])
        qty = float(tr["qty"])
        if tr["side"] == "BUY":
            gross_pnl = (float(actual_exit) - entry_price) * qty
        else:
            gross_pnl = (entry_price - float(actual_exit)) * qty

        # Illustrative maker rebates (0.01% each side on notional)
        entry_rebate = entry_price * qty * 0.0001
        exit_rebate  = float(actual_exit) * qty * 0.0001
        net_pnl = gross_pnl + entry_rebate + exit_rebate

        self.daily_pnl += net_pnl
        self.consecutive_losses = self.consecutive_losses + 1 if net_pnl < 0 else 0

        payload = {
            "id": trade_id,
            "bot": self.bot_name,
            "symbol": self.symbol,
            "side": "LONG" if tr["side"] == "BUY" else "SHORT",
            "action": "CLOSE",
            "ts": datetime.now(timezone.utc).isoformat(),
            "duration_sec": int(duration),
            "entry_price": round(entry_price, 6),
            "expected_exit": round(float(expected_exit), 6),
            "actual_exit": round(float(actual_exit), 6),
            "slippage": 0.0,
            "qty": qty,
            "gross_pnl": round(gross_pnl, 4),
            "rebates_earned": round(entry_rebate + exit_rebate, 4),
            "net_pnl": round(net_pnl, 4),
            "reason": reason,
            "currency": self.currency,
        }
        self._write(payload)
        del self.open_trades[trade_id]
        return payload


class EMABBTrendBot:
    def __init__(self):
        self.LIVE_TRADING = False
        self.demo_mode = os.getenv("DEMO_MODE", "true").lower() == "true"

        self.account_balance_fallback = 1000.0
        self.pending_order = None
        self.open_orders_count = 0
        self.last_trade_time = 0.0
        self.trade_cooldown = 30  # seconds

        self.symbol = "SOLUSDT"
        prefix = "TESTNET_" if self.demo_mode else "LIVE_"
        self.api_key = os.getenv(f"{prefix}BYBIT_API_KEY", "")
        self.api_secret = os.getenv(f"{prefix}BYBIT_API_SECRET", "")
        self.exchange: HTTP | None = None

        self.position = None
        self.price_data = pd.DataFrame()
        self.last_signal = None
        self.preview_signal = None
        self.last_signal_time = None

        self.order_timeout = 180  # seconds

        self.config = {
            "timeframe": "5",
            "ema_fast": 9,
            "ema_slow": 21,
            "ema_trend": 50,
            "macd_fast": 5,
            "macd_slow": 13,
            "macd_signal": 9,
            "rsi_period": 9,
            "bb_period": 20,
            "bb_std": 2,
            "risk_per_trade": 0.01,
            "max_position_pct": 0.05,
            "lookback": 100,
            "maker_offset_pct": 0.08,  # 0.08%
            "stop_loss_pct": 0.50,     # %
            "take_profit_pct": 1.00,   # %

            # Breakout sensitivity (relative to prior 20-bar high/low)
            "breakout_buffer": 0.0002,  # 0.02%

            # Filters (relaxed + override)
            "vol_std_min": 0.0015,  # only check lower bound
            "vol_sma_mult": 0.60,   # volume > 0.6 × 20-SMA(volume)

            # Debounce to avoid duplicate signals on tiny moves
            "price_debounce": 0.001,  # 0.1%
        }

        # instrument steps — loaded from exchange
        self.min_qty = 0.1
        self.qty_step = 0.1
        self.price_step = 0.01

        # terminal monitor
        self.monitor_enabled = True
        self.monitor_full_refresh = True
        self.monitor_interval = 5
        self._spinner = itertools.cycle("|/-\\")
        self._last_monitor_t = 0

        self.logger = TradeLogger("EMA_BB_TREND", self.symbol)
        self.current_trade_id = None

        self._last_ind = None
        self._last_filters = (False, False, False)

    # ---------- helpers ----------
    @staticmethod
    def _round_down_to_step(x: float, step: float) -> float:
        return (int(x / step) * step) if step > 0 else x

    def format_price(self, price: float) -> str:
        p = self._round_down_to_step(float(price), self.price_step)
        return f"{p:.10f}".rstrip("0").rstrip(".")

    def format_qty(self, qty: float) -> str:
        q = max(self.min_qty, self._round_down_to_step(float(qty), self.qty_step))
        return f"{q:.8f}".rstrip("0").rstrip(".")

    # ---------- connect & specs ----------
    def connect(self) -> bool:
        try:
            self.exchange = HTTP(demo=self.demo_mode, api_key=self.api_key, api_secret=self.api_secret)
            ok = self.exchange.get_server_time().get("retCode") == 0
            if not ok:
                return False
            info = self.exchange.get_instruments_info(category="linear", symbol=self.symbol)
            lst = info.get("result", {}).get("list", [])
            if lst:
                lot = lst[0].get("lotSizeFilter", {})
                pf = lst[0].get("priceFilter", {})
                self.min_qty = float(lot.get("minOrderQty", self.min_qty))
                self.qty_step = float(lot.get("qtyStep", self.qty_step))
                self.price_step = float(pf.get("tickSize", self.price_step))
            return True
        except Exception as e:
            print(f"Connect error: {e}")
            return False

    # ---------- data ----------
    async def get_market_data(self) -> bool:
        try:
            kl = self.exchange.get_kline(
                category="linear",
                symbol=self.symbol,
                interval=self.config["timeframe"],
                limit=max(self.config["lookback"], 100),
            )
            if kl.get("retCode") != 0:
                return False
            df = pd.DataFrame(
                kl["result"]["list"],
                columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"],
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
            for c in ["open", "high", "low", "close", "volume"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.dropna(subset=["open", "high", "low", "close"]).sort_values("timestamp").reset_index(drop=True)
            self.price_data = df
            return len(self.price_data) >= self.config["lookback"]
        except Exception:
            return False

    # ---------- account/position ----------
    def _get_available_usdt(self) -> float:
        try:
            acct = self.exchange.get_wallet_balance(accountType="UNIFIED", coin="USDT")
            if acct.get("retCode") == 0:
                lst = acct["result"]["list"]
                if lst:
                    for coin in lst[0]["coin"]:
                        if coin["coin"] == "USDT":
                            return float(coin["availableToWithdraw"])
        except Exception:
            pass
        return self.account_balance_fallback

    async def check_position(self):
        try:
            ps = self.exchange.get_positions(category="linear", symbol=self.symbol)
            if ps.get("retCode") == 0:
                lst = ps["result"]["list"]
                if lst and float(lst[0].get("size", 0) or 0) > 0:
                    self.position = lst[0]
                else:
                    self.position = None
        except Exception:
            self.position = None

    # ---------- indicators & filters ----------
    def enhanced_filters(self, df: pd.DataFrame):
        if len(df) < 50:
            return False, False, False
        ema20 = df["close"].ewm(span=20).mean()
        trend_up = df["close"].iloc[-1] > ema20.iloc[-1]

        rets = df["close"].pct_change().dropna()
        vol_ok = True
        if len(rets) >= 20:
            v = rets.rolling(20).std().iloc[-1]
            vol_ok = v > self.config["vol_std_min"]  # only lower bound; allow grind-ups

        vol_avg = df["volume"].rolling(20).mean()
        vol_pass = True
        if len(df) >= 20:
            vol_pass = df["volume"].iloc[-1] > vol_avg.iloc[-1] * self.config["vol_sma_mult"]

        return trend_up, vol_ok, vol_pass

    def calculate_indicators(self, df: pd.DataFrame):
        if len(df) < self.config["lookback"]:
            return None
        try:
            close = df["close"]
            high = df["high"]
            low  = df["low"]

            ema_fast = close.ewm(span=self.config["ema_fast"]).mean()
            ema_slow = close.ewm(span=self.config["ema_slow"]).mean()
            ema_trend= close.ewm(span=self.config["ema_trend"]).mean()

            exp1 = close.ewm(span=self.config["macd_fast"]).mean()
            exp2 = close.ewm(span=self.config["macd_slow"]).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=self.config["macd_signal"]).mean()
            hist = macd - signal

            delta = close.diff()
            gain = delta.clip(lower=0).rolling(self.config["rsi_period"]).mean()
            loss = (-delta.clip(upper=0)).rolling(self.config["rsi_period"]).mean()
            if pd.isna(loss.iloc[-1]) or loss.iloc[-1] == 0:
                rsi_val = 50.0
            else:
                rs = gain.iloc[-1] / loss.iloc[-1]
                rsi_val = 100 - (100 / (1 + rs))

            # Prior 20 bars (exclude current) for breakout levels
            window_ok = len(high) >= 21
            if window_ok:
                recent_high = high.iloc[-21:-1].max()
                recent_low  = low.iloc[-21:-1].min()
            else:
                recent_high = float(high.max())
                recent_low  = float(low.min())

            return {
                "price": float(close.iloc[-1]),
                "ema_fast": float(ema_fast.iloc[-1]),
                "ema_slow": float(ema_slow.iloc[-1]),
                "ema_trend": float(ema_trend.iloc[-1]),
                "trend_bullish": ema_fast.iloc[-1] > ema_slow.iloc[-1] and close.iloc[-1] > ema_trend.iloc[-1],
                "trend_bearish": ema_fast.iloc[-1] < ema_slow.iloc[-1] and close.iloc[-1] < ema_trend.iloc[-1],
                "hist": float(hist.iloc[-1]),
                "hist_prev": float(hist.iloc[-2]) if len(hist) > 1 else 0.0,
                "rsi": float(rsi_val),
                "recent_high": float(recent_high),
                "recent_low": float(recent_low),
            }
        except Exception as e:
            print(f"Indicator error: {e}")
            return None

    def _in_pullback_band_long(self, price, ema_fast, ema_slow):
        lo, hi = sorted([ema_slow, ema_fast])
        return lo <= price <= hi

    def _in_pullback_band_short(self, price, ema_fast, ema_slow):
        lo, hi = sorted([ema_fast, ema_slow])
        return lo <= price <= hi

    def generate_signal(self, df: pd.DataFrame):
        ind = self.calculate_indicators(df)
        self._last_ind = ind
        if not ind:
            return None

        trend_up, vol_normal, vol_ok = self.enhanced_filters(df)
        self._last_filters = (trend_up, vol_normal, vol_ok)

        # Strong-trend override: allow if bullish momentum is clearly building
        strong_bull = ind["trend_bullish"] and ind["rsi"] >= 60 and ind["hist"] > ind["hist_prev"]
        strong_bear = ind["trend_bearish"] and ind["rsi"] <= 40 and ind["hist"] < ind["hist_prev"]

        if not (vol_normal and vol_ok) and not (strong_bull or strong_bear):
            return None

        # Debounce duplicate signals on tiny price changes
        if self.last_signal:
            price_change = abs(ind["price"] - self.last_signal["price"]) / max(self.last_signal["price"], 1e-9)
            if price_change < self.config["price_debounce"]:
                return None

        p   = ind["price"]
        ef  = ind["ema_fast"]
        es  = ind["ema_slow"]
        et  = ind["ema_trend"]
        rsi = ind["rsi"]
        h, hp = ind["hist"], ind["hist_prev"]
        bof = self.config["breakout_buffer"]

        # --- Primary: Trend Pullback Entries ---
        if ind["trend_bullish"] and trend_up:
            if self._in_pullback_band_long(p, ef, es) and (h > hp) and (rsi >= 45):
                return {"action": "BUY", "price": p, "rsi": rsi, "reason": "trend_pullback"}

        if ind["trend_bearish"] and (not trend_up):
            if self._in_pullback_band_short(p, ef, es) and (h < hp) and (rsi <= 55):
                return {"action": "SELL", "price": p, "rsi": rsi, "reason": "trend_pullback"}

        # --- Secondary: Trend Continuation (works in grind-ups/downs) ---
        if ind["trend_bullish"] and p > ef and (h > hp) and rsi >= 60:
            return {"action": "BUY", "price": p, "rsi": rsi, "reason": "trend_continuation"}

        if ind["trend_bearish"] and p < ef and (h < hp) and rsi <= 40:
            return {"action": "SELL", "price": p, "rsi": rsi, "reason": "trend_continuation"}

        # --- Tertiary: Breakout continuation ---
        if ind["trend_bullish"] and p > ind["recent_high"] * (1 + bof) and (h >= hp) and (rsi >= 50):
            return {"action": "BUY", "price": p, "rsi": rsi, "reason": "trend_breakout"}

        if ind["trend_bearish"] and p < ind["recent_low"] * (1 - bof) and (h <= hp) and (rsi <= 50):
            return {"action": "SELL", "price": p, "rsi": rsi, "reason": "trend_breakout"}

        return None

    # ---------- orders ----------
    async def check_pending_orders(self):
        if self.pending_order and (time.time() - self.last_trade_time > self.order_timeout):
            try:
                self.exchange.cancel_order(
                    category="linear", symbol=self.symbol, orderId=self.pending_order.get("orderId")
                )
            except Exception:
                pass
            self.pending_order = None

        try:
            oo = self.exchange.get_open_orders(category="linear", symbol=self.symbol)
            if oo.get("retCode") != 0:
                self.pending_order = None
                self.open_orders_count = 0
                return False
            lst = oo["result"]["list"] or []
            self.open_orders_count = len(lst)
            if not lst:
                self.pending_order = None
                return False
            self.pending_order = lst[0]
            created_ms = int(self.pending_order.get("createdTime", "0") or "0")
            if created_ms:
                age = time.time() - created_ms / 1000.0
                if age > self.order_timeout:
                    self.exchange.cancel_order(
                        category="linear", symbol=self.symbol, orderId=self.pending_order.get("orderId")
                    )
                    self.pending_order = None
                    self.open_orders_count = 0
                    return False
            return True
        except Exception:
            self.pending_order = None
            self.open_orders_count = 0
            return False

    def calculate_position_size(self, price: float, stop_loss_price: float) -> float:
        balance = self._get_available_usdt()
        risk_amount = balance * self.config["risk_per_trade"]
        price_diff = abs(price - stop_loss_price)
        if price_diff <= 0:
            return max(self.min_qty, self.qty_step)
        qty = risk_amount / price_diff
        max_value = balance * self.config["max_position_pct"]
        max_qty_by_value = max_value / max(price, 1e-9)
        return max(self.min_qty, min(qty, max_qty_by_value))

    def _apply_maker_offset(self, side: str, ref_price: float) -> float:
        off = self.config["maker_offset_pct"] / 100.0
        if side == "Buy":
            return ref_price * (1.0 - off)
        return ref_price * (1.0 + off)

    async def execute_trade(self, signal: dict):
        if time.time() - self.last_trade_time < self.trade_cooldown:
            return
        if await self.check_pending_orders():
            return
        if self.position:
            return

        # Percent-based SL for sizing
        if signal["action"] == "BUY":
            stop_loss_price = signal["price"] * (1 - self.config["stop_loss_pct"] / 100.0)
        else:
            stop_loss_price = signal["price"] * (1 + self.config["stop_loss_pct"] / 100.0)

        qty_raw = self.calculate_position_size(signal["price"], stop_loss_price)
        qty = self.format_qty(qty_raw)

        limit = self._apply_maker_offset("Buy" if signal["action"] == "BUY" else "Sell", signal["price"])
        limit = self.format_price(limit)

        params = {
            "category": "linear",
            "symbol": self.symbol,
            "side": "Buy" if signal["action"] == "BUY" else "Sell",
            "orderType": "Limit",
            "qty": qty,
            "price": limit,
            "timeInForce": "PostOnly",
        }

        try:
            resp = self.exchange.place_order(**params)
            if resp.get("retCode") == 0:
                self.last_trade_time = time.time()
                self.last_signal = signal
                self.last_signal_time = datetime.now()
                self.pending_order = resp.get("result", {})
                tp = signal["price"] * (1 + self.config["take_profit_pct"] / 100.0) if signal["action"] == "BUY" else signal["price"] * (1 - self.config["take_profit_pct"] / 100.0)
                self.current_trade_id, _ = self.logger.log_trade_open(
                    side=signal["action"],
                    expected_price=signal["price"],
                    actual_price=float(limit),
                    qty=float(qty),
                    stop_loss=stop_loss_price,
                    take_profit=tp,
                    info=f"RSI:{signal['rsi']:.1f}_{signal['reason']}_TREND",
                )
                print(f"✅ {signal['action']} {qty} @ {limit} | RSI {signal['rsi']:.1f} | {signal['reason']}")
            else:
                print(f"❌ place_order retCode={resp.get('retCode')} msg={resp.get('retMsg')}")
        except Exception as e:
            print(f"❌ Trade failed: {e}")

    def should_close(self):
        if not self.position or self.price_data.empty:
            return False, ""
        try:
            current_price = float(self.price_data["close"].iloc[-1])
            entry_price = float(self.position.get("avgPrice", 0) or 0)
            side = self.position.get("side", "")
            if entry_price <= 0:
                return False, ""

            profit_pct = (
                (current_price - entry_price) / entry_price * 100.0
                if side == "Buy"
                else (entry_price - current_price) / entry_price * 100.0
            )
            if profit_pct >= self.config["take_profit_pct"]:
                return True, "take_profit"
            if profit_pct <= -self.config["stop_loss_pct"]:
                return True, "stop_loss"

            # Time-based churn protection (1h)
            if time.time() - self.last_trade_time > 3600:
                return True, "timeout"

            return False, ""
        except Exception:
            return False, ""

    async def close_position(self, reason: str):
        if not self.position or self.price_data.empty:
            return
        side = "Sell" if self.position.get("side") == "Buy" else "Buy"
        qty = self.format_qty(float(self.position.get("size", "0") or 0))
        ref_price = float(self.price_data["close"].iloc[-1])
        price = self.format_price(ref_price)
        try:
            resp = self.exchange.place_order(
                category="linear",
                symbol=self.symbol,
                side=side,
                orderType="Limit",
                qty=qty,
                price=price,
                timeInForce="PostOnly",
                reduceOnly=True,
            )
            if resp.get("retCode") == 0:
                if self.current_trade_id:
                    self.logger.log_trade_close(
                        trade_id=self.current_trade_id,
                        expected_exit=ref_price,
                        actual_exit=float(price),
                        reason=reason,
                    )
                    self.current_trade_id = None
                print(f"💰 Closed: {reason}")
                self.position = None
                self.last_signal = None
            else:
                print(f"❌ close_order retCode={resp.get('retCode')} msg={resp.get('retMsg')}")
        except Exception as e:
            print(f"❌ Close failed: {e}")

    # ---------- terminal monitor ----------
    @staticmethod
    def _check(x): return "✓" if x else "–"

    def _clear_screen(self):
        if self.monitor_full_refresh:
            print("\x1b[2J\x1b[H", end="")

    def _print_monitor(self):
        if not self.monitor_enabled:
            return
        now = time.time()
        if now - self._last_monitor_t < self.monitor_interval:
            return
        self._last_monitor_t = now

        cols = shutil.get_terminal_size((100, 20)).columns
        spin = next(self._spinner)
        ind = self._last_ind or {}
        trend_up, vol_norm, vol_vol = self._last_filters

        price = ind.get("price", float("nan"))
        rsi = ind.get("rsi", float("nan"))
        hist = ind.get("hist", float("nan"))

        pos_str = "FLAT"
        if self.position:
            side = self.position.get("side")
            size = self.position.get("size")
            avg = float(self.position.get("avgPrice", 0) or 0)
            if not pd.isna(price) and avg > 0:
                pnl_pct = ((price - avg) / avg * 100.0) if side == "Buy" else ((avg - price) / avg * 100.0)
                pos_str = f"{side} {size}@{avg:.4f} | PnL {pnl_pct:+.2f}%"
            else:
                pos_str = f"{side} {size}@{avg}"

        po = self.pending_order
        if po:
            created_ms = int(po.get("createdTime", "0") or "0")
            age = f"{int(time.time() - created_ms/1000):>3}s" if created_ms else "--"
            ord_str = f"OPEN ORDERS: {self.open_orders_count}  (age {age})"
        else:
            ord_str = "OPEN ORDERS: 0"

        cooldown = max(0, int(self.trade_cooldown - (time.time() - self.last_trade_time)))
        if self.preview_signal:
            preview = f"{self.preview_signal['action']}@{self.preview_signal['price']:.4f} [{self.preview_signal.get('reason','')}]"
        else:
            preview = "—"
        last_sig = f"{self.last_signal['action']} {self.last_signal['price']:.4f} [{self.last_signal.get('reason','')}]" if self.last_signal else "—"

        self._clear_screen()
        print(f"{spin} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  {self.symbol}  |  TF {self.config['timeframe']}m  |  DEMO:{self.demo_mode}  |  MakerOff {self.config['maker_offset_pct']}%".ljust(cols))
        print("-" * cols)
        print(f"Price {price:.4f}  |  RSI {rsi:5.1f}  |  MACD hist {hist: .5f}  |  Trend↑ {self._check(trend_up)}  Vol OK {self._check(vol_norm)}  Vol† {self._check(vol_vol)}".ljust(cols))
        print(f"{pos_str}".ljust(cols))
        print(f"{ord_str}  |  Cooldown: {cooldown:>2}s  |  Daily PnL: {self.logger.daily_pnl:+.2f} USDT  |  ConsLoss: {self.logger.consecutive_losses}".ljust(cols))
        print(f"Signal(preview): {preview}   |   Last placed: {last_sig}".ljust(cols))
        print("-" * cols)

    # ---------- main loop ----------
    async def run_cycle(self):
        if self.logger.daily_pnl < -self.logger.max_daily_loss:
            print(f"🔴 EMERGENCY STOP: daily PnL {self.logger.daily_pnl:.2f} < -{self.logger.max_daily_loss}")
            if self.position:
                await self.close_position("emergency_stop")
            self._print_monitor()
            return

        if not await self.get_market_data():
            self._print_monitor()
            return

        await self.check_position()
        await self.check_pending_orders()

        self.preview_signal = self.generate_signal(self.price_data)

        if self.position:
            should, reason = self.should_close()
            if should:
                await self.close_position(reason)
        elif not self.pending_order and self.preview_signal:
            await self.execute_trade(self.preview_signal)

        self._print_monitor()

    async def run(self):
        if not self.connect():
            print("❌ Failed to connect to Bybit")
            return

        print(f"🔧 EMA+BB Trend-Following Bot for {self.symbol}")
        print(f"📊 Mode: {'DEMO' if self.demo_mode else 'LIVE'} | Trading: {'ON' if self.LIVE_TRADING else 'OFF'} (signals still run)")
        print(f"✅ Using tickSize={self.price_step}, qtyStep={self.qty_step}, minQty={self.min_qty}")
        print(f"🎯 TP {self.config['take_profit_pct']}% | SL {self.config['stop_loss_pct']}% | Maker offset {self.config['maker_offset_pct']}%")
        print(f"⏰ Cooldown {self.trade_cooldown}s | Order timeout {self.order_timeout}s")

        try:
            while True:
                await self.run_cycle()
                await asyncio.sleep(5)
        except KeyboardInterrupt:
            print("\n🛑 Bot stopped by user")
            try:
                self.exchange.cancel_all_orders(category="linear", symbol=self.symbol)
            except Exception:
                pass
            if self.position:
                await self.close_position("manual_stop")


if __name__ == "__main__":
    bot = EMABBTrendBot()
    asyncio.run(bot.run())
