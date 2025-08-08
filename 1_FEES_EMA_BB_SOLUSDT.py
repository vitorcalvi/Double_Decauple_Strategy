#!/usr/bin/env python3
"""
EMA + BB Fixed Bot (Bybit TESTNET, linear futures) with Terminal Monitor

- Signals run in demo/paper too (no LIVE_TRADING gate).
- Uses Bybit instrument specs (tickSize, qtyStep, minOrderQty) for proper rounding.
- Fractional qty allowed; no hard skip at qty < 1.
- Safer PostOnly pricing (default 0.08% maker offset).
- Compact terminal dashboard refreshed every loop, showing:
  price, RSI, MACD hist, filters ✓/✗, position, open orders, cooldown, daily PnL, and next step.
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

        # emergency stop tracking (daily)
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
        slippage = 0.0  # PostOnly (assume maker fill when it happens)

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
            "slippage": round(float(slippage), 6),
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

        # Maker rebate illustration (0.01% both sides)
        entry_rebate = entry_price * qty * 0.0001
        exit_rebate = float(actual_exit) * qty * 0.0001
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


class EMABBFixedBot:
    def __init__(self):
        self.LIVE_TRADING = False  # demo/paper
        self.demo_mode = os.getenv("DEMO_MODE", "true").lower() == "true"

        self.account_balance_fallback = 1000.0
        self.pending_order = None
        self.open_orders_count = 0
        self.last_trade_time = 0.0
        self.trade_cooldown = 30  # seconds

        self.daily_pnl = 0.0  # mirrored from logger

        self.symbol = "SOLUSDT"
        prefix = "TESTNET_" if self.demo_mode else "LIVE_"
        self.api_key = os.getenv(f"{prefix}BYBIT_API_KEY", "")
        self.api_secret = os.getenv(f"{prefix}BYBIT_API_SECRET", "")
        self.exchange: HTTP | None = None

        self.position = None
        self.price_data = pd.DataFrame()
        self.last_signal = None           # last placed signal (after order placed)
        self.preview_signal = None        # signal generated this cycle (before place)
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
            "base_slippage": 0.0,      # PostOnly
            "stop_loss_pct": 0.50,     # %
            "take_profit_pct": 1.00,   # %
        }

        # instrument steps — loaded from exchange
        self.min_qty = 0.001
        self.qty_step = 0.001
        self.price_step = 0.01

        # terminal monitor settings
        self.monitor_enabled = True
        self.monitor_full_refresh = True   # clear screen & redraw
        self.monitor_interval = 5          # seconds (same as loop)
        self._spinner = itertools.cycle("|/-\\")
        self._last_monitor_t = 0

        self.logger = TradeLogger("EMA_BB_FIXED", self.symbol)
        self.current_trade_id = None

        # cache for monitor
        self._last_ind = None
        self._last_filters = (False, False, False)

    # ---------- helpers (exchange steps) ----------
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
            vol_ok = 0.005 < v < 0.03

        vol_avg = df["volume"].rolling(20).mean()
        vol_pass = True
        if len(df) >= 20:
            vol_pass = df["volume"].iloc[-1] > vol_avg.iloc[-1] * 0.8

        return trend_up, vol_ok, vol_pass

    def calculate_indicators(self, df: pd.DataFrame):
        if len(df) < self.config["lookback"]:
            return None
        try:
            close = df["close"]
            ema_fast = close.ewm(span=self.config["ema_fast"]).mean()
            ema_slow = close.ewm(span=self.config["ema_slow"]).mean()
            ema_trend = close.ewm(span=self.config["ema_trend"]).mean()

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

            bb_mid = close.rolling(self.config["bb_period"]).mean()
            bb_sd = close.rolling(self.config["bb_period"]).std()
            bb_upper = bb_mid + self.config["bb_std"] * bb_sd
            bb_lower = bb_mid - self.config["bb_std"] * bb_sd

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
                "bb_upper": float(bb_upper.iloc[-1]),
                "bb_lower": float(bb_lower.iloc[-1]),
                "bb_middle": float(bb_mid.iloc[-1]),
            }
        except Exception as e:
            print(f"Indicator error: {e}")
            return None

    def generate_signal(self, df: pd.DataFrame):
        ind = self.calculate_indicators(df)
        self._last_ind = ind  # cache for monitor
        if not ind:
            return None

        trend_up, vol_normal, vol_ok = self.enhanced_filters(df)
        self._last_filters = (trend_up, vol_normal, vol_ok)
        if not (vol_normal and vol_ok):
            return None

        if self.last_signal:
            price_change = abs(ind["price"] - self.last_signal["price"]) / self.last_signal["price"]
            if price_change < 0.002:  # debounce 0.2%
                return None

        # BUY
        if (
            ind["trend_bullish"]
            and trend_up
            and ind["rsi"] < 35
            and ind["hist"] > 0 >= ind["hist_prev"]
            and ind["price"] > ind["bb_lower"]
        ):
            return {"action": "BUY", "price": ind["price"], "rsi": ind["rsi"], "reason": "oversold_reversal"}

        # SELL
        if (
            ind["trend_bearish"]
            and (not trend_up)
            and ind["rsi"] > 65
            and ind["hist"] < 0 <= ind["hist_prev"]
            and ind["price"] < ind["bb_upper"]
        ):
            return {"action": "SELL", "price": ind["price"], "rsi": ind["rsi"], "reason": "overbought_reversal"}

        return None

    # ---------- orders ----------
    async def check_pending_orders(self):
        # local timeout
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
        # cooldown
        if time.time() - self.last_trade_time < self.trade_cooldown:
            return
        # no overlap
        if await self.check_pending_orders():
            return
        if self.position:
            return

        # SL first for sizing
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
                    info=f"RSI:{signal['rsi']:.1f}_{signal['reason']}_FIXED",
                )
                print(f"✅ {signal['action']} {qty} @ {limit} | RSI {signal['rsi']:.1f}")
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
                    self.daily_pnl = self.logger.daily_pnl
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
        # clear terminal if full refresh is desired
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

        # position summary
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

        # pending order summary
        po = self.pending_order
        if po:
            created_ms = int(po.get("createdTime", "0") or "0")
            age = f"{int(time.time() - created_ms/1000):>3}s" if created_ms else "--"
            ord_str = f"OPEN ORDERS: {self.open_orders_count}  (age {age})"
        else:
            ord_str = "OPEN ORDERS: 0"

        cooldown = max(0, int(self.trade_cooldown - (time.time() - self.last_trade_time)))
        preview = f"{self.preview_signal['action']}@{self.preview_signal['price']:.4f}" if self.preview_signal else "—"
        last_sig = f"{self.last_signal['action']} {self.last_signal['price']:.4f}" if self.last_signal else "—"

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
        # emergency daily stop
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

        # compute preview signal for monitor
        self.preview_signal = self.generate_signal(self.price_data)

        if self.position:
            should, reason = self.should_close()
            if should:
                await self.close_position(reason)
        elif not self.pending_order and self.preview_signal:
            await self.execute_trade(self.preview_signal)

        # finally print monitor
        self._print_monitor()

    async def run(self):
        if not self.connect():
            print("❌ Failed to connect to Bybit")
            return

        print(f"🔧 EMA+BB Bot for {self.symbol}")
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
    bot = EMABBFixedBot()
    asyncio.run(bot.run())
