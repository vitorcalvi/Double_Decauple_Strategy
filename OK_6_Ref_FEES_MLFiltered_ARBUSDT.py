# BOT REFERENCE

import os
import time
import asyncio
import json
from datetime import datetime, timezone

import pandas as pd
from pybit.unified_trading import HTTP
from dotenv import load_dotenv

load_dotenv()

# =========================
# Compact JSONL TradeLogger
# =========================
def _iso_now():
    return datetime.now(timezone.utc).isoformat()

def _bps(delta, base):
    try:
        return (float(delta) / float(base)) * 1e4
    except Exception:
        return None

class TradeLogger:
    """
    Emits Compact JSONL:

    OPEN:
      {"t":"O","id","bot","sym","sd":"L|S","ccy","px","exp","slip","slip_bps",
       "qty","sl","tp","risk","rr","bal","tags","ts"}

    CLOSE:
      {"t":"C","ref","bot","sym","ccy","px","exp","slip","slip_bps","dur_s","qty",
       "gross","fees":{"in_bps","out_bps","entry","exit","total"},"net","R",
       "exit","pnl_day","streak_loss","tags","extra","ts"}
    """
    def __init__(self, bot_name: str, symbol: str, log_file: str = None):
        self.bot = bot_name
        self.sym = symbol
        self.ccy = "USDT"
        self.id_seq = 1000
        self.open = {}  # id -> cached info
        self.pnl_day = 0.0
        self.streak_loss = 0
        os.makedirs("logs", exist_ok=True)
        self.log_file = log_file or f"logs/6_FEES_MLFiltered_ARBUSDT.log"

    def _next_id(self):
        self.id_seq += 1
        return self.id_seq

    def _write(self, obj: dict):
        with open(self.log_file, "a") as f:
            f.write(json.dumps(obj, separators=(",", ":")) + "\n")

    # --------- OPEN (after fill) ----------
    def log_open(
        self,
        side_long_short: str,   # "L" or "S"
        expected_px: float,     # intended limit price
        actual_px: float,       # avg fill price from position
        qty: float,
        stop_loss_px: float,
        take_profit_px: float,
        balance_usd: float | None,
        tags: str = "",
    ):
        tid = self._next_id()
        slip = float(actual_px) - float(expected_px)
        slip_bps = _bps(slip, expected_px)

        stop_move = abs(float(actual_px) - float(stop_loss_px))
        tp_move   = abs(float(take_profit_px) - float(actual_px))
        risk_usd  = stop_move * float(qty) if stop_move > 0 else 0.0
        rr_plan   = (tp_move / stop_move) if stop_move > 0 else None

        rec = {
            "t":"O",
            "id":tid,
            "bot":self.bot,
            "sym":self.sym,
            "sd":side_long_short,
            "ccy":self.ccy,
            "px":round(float(actual_px),6),
            "exp":round(float(expected_px),6),
            "slip":round(float(slip),6),
            "slip_bps":round(float(slip_bps),2) if slip_bps is not None else None,
            "qty":round(float(qty),6),
            "sl":round(float(stop_loss_px),6),
            "tp":round(float(take_profit_px),6),
            "risk":round(float(risk_usd),4) if risk_usd else 0.0,
            "rr":round(float(rr_plan),4) if rr_plan is not None else None,
            "bal":round(float(balance_usd),2) if balance_usd is not None else None,
            "tags":tags,
            "ts":_iso_now(),
        }
        self._write(rec)

        self.open[tid] = {
            "ts_open": datetime.now(timezone.utc),
            "entry_px": float(actual_px),
            "exp_entry": float(expected_px),
            "sd": side_long_short,
            "qty": float(qty),
            "risk_usd": float(risk_usd),
            "tags": tags,
        }
        return tid

    # --------- CLOSE (our own exit) ----------
    def log_close(
        self,
        trade_id: int,
        expected_exit: float,
        actual_exit: float,
        exit_reason: str,
        in_bps: float,     # e.g. -4.0
        out_bps: float,    # e.g. -4.0
        extra: dict | None = None,
    ):
        st = self.open.get(trade_id)
        if not st:
            return None

        dur_s = max(0, int((datetime.now(timezone.utc) - st["ts_open"]).total_seconds()))
        slip = float(actual_exit) - float(expected_exit)
        slip_bps = _bps(slip, expected_exit)

        qty = st["qty"]
        sd = st["sd"]
        entry_px = st["entry_px"]

        gross = (actual_exit - entry_px) * qty if sd == "L" else (entry_px - actual_exit) * qty

        fee_entry = (float(in_bps) / 1e4) * entry_px * qty if in_bps is not None else 0.0
        fee_exit  = (float(out_bps)/ 1e4) * actual_exit * qty if out_bps is not None else 0.0
        fee_total = fee_entry + fee_exit
        net = gross - fee_total

        R = (net / st["risk_usd"]) if st["risk_usd"] > 0 else None

        self.pnl_day += net
        if net < 0: self.streak_loss += 1
        else: self.streak_loss = 0

        rec = {
            "t":"C",
            "ref":trade_id,
            "bot":self.bot,
            "sym":self.sym,
            "ccy":self.ccy,
            "px":round(float(actual_exit),6),
            "exp":round(float(expected_exit),6),
            "slip":round(float(slip),6),
            "slip_bps":round(float(slip_bps),2) if slip_bps is not None else None,
            "dur_s":dur_s,
            "qty":round(float(qty),6),
            "gross":round(float(gross),4),
            "fees":{
                "in_bps":round(float(in_bps),2) if in_bps is not None else None,
                "out_bps":round(float(out_bps),2) if out_bps is not None else None,
                "entry":round(float(fee_entry),4) if in_bps is not None else None,
                "exit":round(float(fee_exit),4) if out_bps is not None else None,
                "total":round(float(fee_total),4) if (in_bps is not None and out_bps is not None) else None,
            },
            "net":round(float(net),4),
            "R":round(float(R),4) if R is not None else None,
            "exit":exit_reason,
            "pnl_day":round(float(self.pnl_day),4),
            "streak_loss":int(self.streak_loss),
            "tags":st["tags"],
            "extra": extra or {},
            "ts":_iso_now(),
        }
        self._write(rec)
        del self.open[trade_id]
        return rec

    # --------- CLOSE (unknown details; e.g., exchange stop filled) ----------
    def log_close_unknown(self, trade_id: int, reason: str = "Unknown (not logged)", extra: dict | None = None):
        st = self.open.get(trade_id)
        if not st:
            return None
        dur_s = max(0, int((datetime.now(timezone.utc) - st["ts_open"]).total_seconds()))
        rec = {
            "t":"C",
            "ref":trade_id,
            "bot":self.bot,
            "sym":self.sym,
            "ccy":self.ccy,
            "px":None,
            "exp":None,
            "slip":None,
            "slip_bps":None,
            "dur_s":dur_s,
            "qty":round(float(st["qty"]),6),
            "gross":None,
            "fees":{"in_bps":None,"out_bps":None,"entry":None,"exit":None,"total":None},
            "net":None,
            "R":None,
            "exit":reason,
            "pnl_day":round(float(self.pnl_day),4),  # unchanged
            "streak_loss":int(self.streak_loss),     # unchanged
            "tags":st["tags"],
            "extra": extra or {"note":"position closed externally (e.g., protective stop)"},
            "ts":_iso_now(),
        }
        self._write(rec)
        del self.open[trade_id]
        return rec


# =========================
# Enhanced ML Scalping Bot
# =========================
class EnhancedMLScalpingBot:
    def __init__(self):
        self.symbol = "ARBUSDT"
        self.demo_mode = os.getenv("DEMO_MODE", "true").lower() == "true"

        prefix = "TESTNET_" if self.demo_mode else "LIVE_"
        self.api_key = os.getenv(f"{prefix}BYBIT_API_KEY")
        self.api_secret = os.getenv(f"{prefix}BYBIT_API_SECRET")
        self.exchange = None

        self.position = None
        self.prev_had_position = False
        self.price_data = pd.DataFrame()
        self.pending_order = None
        self.pending_entry_ctx = None  # store signal/limit info until fill
        self.daily_pnl = 0.0
        self.current_trade_id = None
        self.max_daily_loss = 100.0
        self.account_balance = 0.0

        self.last_trade_time = 0.0
        self.trade_cooldown = 30
        self.max_position_checks = 3
        self.order_timeout = 180

        self.config = {
            "timeframe": "5",
            "rsi_period": 14,
            "ema_fast": 9,
            "ema_slow": 21,
            "bb_period": 20,
            "bb_std": 2,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "ml_confidence_threshold": 0.70,
            "risk_per_trade_pct": 2.0,
            "lookback": 100,
            "maker_offset_pct": 0.01,   # 0.01% maker offset
            "maker_fee_pct": -0.04,     # -0.04% (rebate) → -4.0 bps
            "base_take_profit_pct": 1.0,
            "base_stop_loss_pct": 0.5,
            "expected_slippage_pct": 0.02,

            # Protective Stop (optional)
            "enable_protective_stop": True,
            "ps_trigger_by": "LastPrice",  # or "MarkPrice"
        }

        self.volatility_regime = "normal"
        self.logger = TradeLogger("ML_ARB_FIXED", self.symbol)
        self.ps_set_for_trade = set()  # trade_ids with protective stop set

    # ------------- Connectivity -------------
    def connect(self):
        try:
            self.exchange = HTTP(demo=self.demo_mode, api_key=self.api_key, api_secret=self.api_secret)
            return self.exchange.get_server_time().get("retCode") == 0
        except Exception as e:
            print(f"Connection error: {e}")
            return False

    # ------------- Balance -------------
    async def get_account_balance(self):
        try:
            wallet = self.exchange.get_wallet_balance(accountType="UNIFIED", coin="USDT")
            if wallet.get("retCode") == 0:
                lst = wallet["result"]["list"]
                if lst:
                    for c in lst[0]["coin"]:
                        if c["coin"] == "USDT":
                            # Fix: Handle empty string or invalid values
                            available = c.get("availableToWithdraw", "")
                            if available and available != "":
                                self.account_balance = float(available)
                            else:
                                # Use fallback if empty
                                self.account_balance = 1000.0
                            return True
        except Exception as e:
            print(f"❌ Balance error: {e}")
        self.account_balance = 1000.0
        return True

    # ------------- Sizing / formatting -------------
    def calculate_position_size(self, price, stop_loss_price):
        if self.account_balance <= 0:
            return 0
        risk_amount = self.account_balance * (self.config["risk_per_trade_pct"] / 100.0)
        price_diff = abs(float(price) - float(stop_loss_price))
        if price_diff == 0:
            return 0
        slippage_factor = 1 + (self.config["expected_slippage_pct"] / 100.0)
        adjusted_risk = risk_amount / slippage_factor
        qty = adjusted_risk / price_diff
        return max(qty, 1)  # at least 1 ARB

    def format_qty(self, qty):
        return str(int(round(qty)))

    def apply_slippage(self, price, side, order_type="market"):
        if order_type == "limit":
            return price
        s = self.config["expected_slippage_pct"] / 100.0
        return price * (1 + s) if side in ("BUY", "Buy") else price * (1 - s)

    def can_execute_trade(self):
        now = time.time()
        if now - self.last_trade_time < self.trade_cooldown:
            remaining = self.trade_cooldown - (now - self.last_trade_time)
            print(f"⏰ Trade cooldown active: {remaining:.1f}s remaining")
            return False
        return True

    # ------------- Exchange helpers -------------
    async def check_pending_orders(self):
        try:
            orders = self.exchange.get_open_orders(category="linear", symbol=self.symbol)
            if orders.get("retCode") != 0:
                return False
            lst = orders["result"]["list"]
            if not lst:
                self.pending_order = None
                return False
            order = lst[0]
            age = datetime.now().timestamp() - int(order["createdTime"]) / 1000
            if age > self.order_timeout:
                self.exchange.cancel_order(category="linear", symbol=self.symbol, orderId=order["orderId"])
                print(f"❌ Cancelled stale order (aged {age:.0f}s)")
                self.pending_order = None
                return False
            self.pending_order = order
            return True
        except Exception as e:
            print(f"Order check error: {e}")
            return False

    async def ensure_protective_stop(self):
        """Set protective StopMarket after we detect a filled position and we have the trade_id."""
        if not self.config.get("enable_protective_stop", False):
            return
        if not self.position or not self.current_trade_id:
            return
        if self.current_trade_id in self.ps_set_for_trade:
            return

        side = self.position.get("side", "")
        size = float(self.position.get("size", 0))
        if size <= 0:
            return

        # We stored SL in pending_entry_ctx at order time; recompute from actual avg if needed
        avg = float(self.position.get("avgPrice", 0))
        if avg <= 0:
            return

        if side == "Buy":
            stop_loss_px = avg * (1 - self.config["base_stop_loss_pct"]/100.0)
        else:
            stop_loss_px = avg * (1 + self.config["base_stop_loss_pct"]/100.0)

        try:
            # Bybit v5 set trading stop on position
            resp = self.exchange.set_trading_stop(
                category="linear",
                symbol=self.symbol,
                stopLoss=str(round(stop_loss_px, 4)),
                slTriggerBy=self.config.get("ps_trigger_by", "LastPrice")
            )
            if resp.get("retCode") == 0:
                self.ps_set_for_trade.add(self.current_trade_id)
                print(f"🛡️ Protective Stop set @ {stop_loss_px:.4f} ({self.config.get('ps_trigger_by','LastPrice')})")
            else:
                print(f"⚠️ Protective Stop rejected: {resp.get('retMsg')}")
        except Exception as e:
            print(f"⚠️ Protective Stop error: {e}")

    # ------------- Indicators / signal -------------
    def calculate_indicators(self, df):
        if len(df) < self.config["lookback"]:
            return None
        try:
            close = df["close"]
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(window=self.config["rsi_period"]).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=self.config["rsi_period"]).mean()
            rs = gain / (loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))

            ema_fast = close.ewm(span=self.config["ema_fast"]).mean()
            ema_slow = close.ewm(span=self.config["ema_slow"]).mean()

            bb_middle = close.rolling(window=self.config["bb_period"]).mean()
            bb_std = close.rolling(window=self.config["bb_period"]).std()
            bb_upper = bb_middle + (bb_std * self.config["bb_std"])
            bb_lower = bb_middle - (bb_std * self.config["bb_std"])

            exp1 = close.ewm(span=self.config["macd_fast"]).mean()
            exp2 = close.ewm(span=self.config["macd_slow"]).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=self.config["macd_signal"]).mean()
            macd_hist = macd - signal

            bb_range = bb_upper.iloc[-1] - bb_lower.iloc[-1]
            bb_pos = (close.iloc[-1] - bb_lower.iloc[-1]) / bb_range if bb_range != 0 else 0.5

            returns = close.pct_change().dropna()
            vol = returns.rolling(window=20).std().iloc[-1] if len(returns) >= 20 else 0.01

            self.volatility_regime = "high" if vol > 0.025 else ("low" if vol < 0.01 else "normal")

            return {
                "price": float(close.iloc[-1]),
                "rsi": float(rsi.iloc[-1]) if pd.notna(rsi.iloc[-1]) else 50.0,
                "ema_trend": bool(ema_fast.iloc[-1] > ema_slow.iloc[-1]),
                "bb_position": float(bb_pos),
                "macd_histogram": float(macd_hist.iloc[-1]) if pd.notna(macd_hist.iloc[-1]) else 0.0,
                "volatility": float(vol),
            }
        except Exception as e:
            print(f"Indicator calculation error: {e}")
            return None

    def ml_filter_confidence(self, ind):
        if not ind: return 0.0
        c = 0.5
        c += (0.1 if ind["macd_histogram"] > 0 else -0.05) if ind["ema_trend"] else (0.1 if ind["macd_histogram"] < 0 else -0.05)
        if ind["rsi"] < 35: c += 0.15
        elif ind["rsi"] > 65: c += 0.15
        elif 40 < ind["rsi"] < 60: c += 0.05
        if ind["bb_position"] < 0.2 or ind["bb_position"] > 0.8: c += 0.15
        if self.volatility_regime == "normal": c += 0.1
        elif self.volatility_regime == "high": c *= 0.9
        return min(max(c, 0.0), 1.0)

    def generate_signal(self, df):
        ind = self.calculate_indicators(df)
        if not ind: return None
        conf = self.ml_filter_confidence(ind)
        if conf < self.config["ml_confidence_threshold"]:
            return None

        buy_score = 0; sell_score = 0
        if ind["ema_trend"]: buy_score += 1
        else: sell_score += 1
        if ind["rsi"] < 40: buy_score += 2
        elif ind["rsi"] > 60: sell_score += 2
        if ind["bb_position"] < 0.3: buy_score += 1
        elif ind["bb_position"] > 0.7: sell_score += 1
        if ind["macd_histogram"] > 0: buy_score += 1
        else: sell_score += 1

        if buy_score >= 3:
            return {"action":"BUY","price":ind["price"],"confidence":conf,"rsi":ind["rsi"]}
        if sell_score >= 3:
            return {"action":"SELL","price":ind["price"],"confidence":conf,"rsi":ind["rsi"]}
        return None

    # ------------- Position state -------------
    async def check_position(self):
        """Detect fills and closures; log O/C accordingly."""
        prev = self.position
        try:
            res = self.exchange.get_positions(category="linear", symbol=self.symbol)
            if res.get("retCode") == 0:
                lst = res["result"]["list"]
                self.position = None
                if lst:
                    for p in lst:
                        size = float(p.get("size", 0))
                        if size > 0:
                            self.position = p
                            break
            else:
                self.position = None
        except Exception as e:
            print(f"❌ Position check error: {e}")
            self.position = None

        # Detect new fill → log OPEN & set protective stop
        if self.position and not prev:
            side = self.position.get("side")  # "Buy"/"Sell"
            avg = float(self.position.get("avgPrice", 0))
            size = float(self.position.get("size", 0))
            if self.pending_entry_ctx and avg > 0 and size > 0:
                sd = "L" if side == "Buy" else "S"
                # Build TP/SL from actual avg to keep RR accurate
                if sd == "L":
                    tp = avg * (1 + self.config["base_take_profit_pct"]/100.0)
                    sl = avg * (1 - self.config["base_stop_loss_pct"]/100.0)
                else:
                    tp = avg * (1 - self.config["base_take_profit_pct"]/100.0)
                    sl = avg * (1 + self.config["base_stop_loss_pct"]/100.0)

                self.current_trade_id = self.logger.log_open(
                    side_long_short=sd,
                    expected_px=self.pending_entry_ctx["limit_price"],
                    actual_px=avg,
                    qty=size,
                    stop_loss_px=sl,
                    take_profit_px=tp,
                    balance_usd=self.account_balance,
                    tags=self.pending_entry_ctx["tags"]
                )
                print(f"✅ Position FILLED: {side} {size} @ {avg:.4f} (tid {self.current_trade_id})")
                await self.ensure_protective_stop()

        # Detect closure outside our code → log UNKNOWN close
        if (prev is not None) and (self.position is None) and self.current_trade_id:
            # If we didn't log a close ourselves, record an unknown close
            self.logger.log_close_unknown(
                self.current_trade_id,
                reason="Unknown (not logged)"
            )
            print("ℹ️ Position closed externally (e.g., protective stop) — logged UNKNOWN close")
            self.current_trade_id = None
            self.pending_entry_ctx = None
            self.ps_set_for_trade.clear()

        self.prev_had_position = self.position is not None

    def should_close(self):
        if not self.position or len(self.price_data) == 0:
            return False, ""
        try:
            current_price = float(self.price_data["close"].iloc[-1])
            entry_price = float(self.position.get("avgPrice", 0))
            side = self.position.get("side", "")
            if entry_price == 0:
                return False, ""

            profit_pct = ((current_price - entry_price) / entry_price * 100.0) if side == "Buy" \
                         else ((entry_price - current_price) / entry_price * 100.0)

            if profit_pct >= self.config["base_take_profit_pct"]:
                return True, f"take_profit_{profit_pct:.2f}%"
            if profit_pct <= -self.config["base_stop_loss_pct"]:
                return True, f"stop_loss_{profit_pct:.2f}%"
            if profit_pct <= -1.0:
                return True, f"safety_stop_{profit_pct:.2f}%"

            ind = self.calculate_indicators(self.price_data)
            if ind:
                if side == "Buy" and ind["rsi"] > 75 and not ind["ema_trend"]:
                    return True, "reversal_signal"
                if side == "Sell" and ind["rsi"] < 25 and ind["ema_trend"]:
                    return True, "reversal_signal"
            return False, ""
        except Exception as e:
            print(f"❌ Position check error: {e}")
            return False, ""

    # ------------- Market data -------------
    async def get_market_data(self):
        try:
            kl = self.exchange.get_kline(category="linear", symbol=self.symbol, interval=self.config["timeframe"], limit=self.config["lookback"])
            if kl.get("retCode") != 0:
                return False
            data_list = kl.get("result", {}).get("list", [])
            if not data_list:
                return False
            df = pd.DataFrame(data_list, columns=["timestamp","open","high","low","close","volume","turnover"])
            df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
            for col in ["open","high","low","close","volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.dropna(subset=["open","high","low","close"])
            if len(df) < 20:
                return False
            self.price_data = df.sort_values("timestamp").reset_index(drop=True)
            return True
        except Exception as e:
            print(f"Market data error: {e}")
            return False

    # ------------- Execution -------------
    async def execute_trade(self, signal):
        if not self.can_execute_trade():
            return

        # No double-position
        for _ in range(self.max_position_checks):
            await self.check_position()
            if self.position:
                print("⚠️ Existing position — blocking new trade")
                return
            await asyncio.sleep(1)

        # No concurrent pending
        if await self.check_pending_orders():
            print("⚠️ Pending order exists — blocking new trade")
            return

        await self.get_account_balance()

        # Draft sizing from signal price (we'll log with actual avg on fill)
        if signal["action"] == "BUY":
            stop_px = signal["price"] * (1 - self.config["base_stop_loss_pct"]/100.0)
        else:
            stop_px = signal["price"] * (1 + self.config["base_stop_loss_pct"]/100.0)

        qty = self.calculate_position_size(signal["price"], stop_px)
        fqty = self.format_qty(qty)
        if int(fqty) <= 0:
            print(f"❌ Position size too small: {qty}")
            return

        offset = (1 - self.config["maker_offset_pct"]/100.0) if signal["action"] == "BUY" else (1 + self.config["maker_offset_pct"]/100.0)
        limit_price = round(signal["price"] * offset, 4)
        expected_fill_price = self.apply_slippage(limit_price, signal["action"], "limit")

        try:
            order = self.exchange.place_order(
                category="linear",
                symbol=self.symbol,
                side="Buy" if signal["action"] == "BUY" else "Sell",
                orderType="Limit",
                qty=fqty,
                price=str(limit_price),
                timeInForce="PostOnly"
            )
            if order.get("retCode") == 0:
                self.last_trade_time = time.time()
                self.pending_order = order["result"]
                self.pending_entry_ctx = {
                    "limit_price": limit_price,
                    "expected_fill": expected_fill_price,
                    "side": signal["action"],
                    "tags": f"conf:{signal['confidence']:.2f}|rsi:{signal['rsi']:.1f}|risk%:{self.config['risk_per_trade_pct']:.2f}"
                }
                risk_amount = self.account_balance * (self.config["risk_per_trade_pct"]/100.0)
                print(f"✅ Placed {signal['action']} {fqty} @ {limit_price:.4f} (maker)")
                print(f"   💰 Risk: ${risk_amount:.2f} on balance ${self.account_balance:.2f}")
            else:
                print(f"❌ Order rejected: {order.get('retMsg')}")
        except Exception as e:
            print(f"❌ Trade failed: {e}")

    async def close_position(self, reason):
        if not self.position:
            return
        current_price = float(self.price_data["close"].iloc[-1])
        side_close = "Sell" if self.position.get("side") == "Buy" else "Buy"
        qty = float(self.position.get("size", 0))
        if qty <= 0:
            print("⚠️ No quantity to close")
            return

        offset_mult = (1 + self.config["maker_offset_pct"]/100.0) if side_close == "Sell" else (1 - self.config["maker_offset_pct"]/100.0)
        limit_price = round(current_price * offset_mult, 4)
        expected_fill_price = self.apply_slippage(limit_price, side_close, "limit")

        try:
            od = self.exchange.place_order(
                category="linear",
                symbol=self.symbol,
                side=side_close,
                orderType="Limit",
                qty=self.format_qty(qty),
                price=str(limit_price),
                timeInForce="PostOnly",
                reduceOnly=True
            )
            if od.get("retCode") == 0:
                # Log close immediately (limit at maker); we use current market as 'exp' and our limit as 'px'
                if self.current_trade_id:
                    self.logger.log_close(
                        trade_id=self.current_trade_id,
                        expected_exit=current_price,
                        actual_exit=expected_fill_price,
                        exit_reason=reason,
                        in_bps=self.config["maker_fee_pct"]*100.0,   # % → bps
                        out_bps=self.config["maker_fee_pct"]*100.0,
                        extra={"mode":"maker","cooldown":self.trade_cooldown}
                    )
                    self.current_trade_id = None
                    self.pending_entry_ctx = None
                    self.ps_set_for_trade.clear()
                print(f"✅ Close placed ({side_close}) @ {limit_price:.4f} → reason: {reason}")
                self.position = None
            else:
                print(f"❌ Close rejected: {od.get('retMsg')}")
        except Exception as e:
            print(f"❌ Close failed: {e}")

    # ------------- UI/status & loop -------------
    def show_status(self):
        if len(self.price_data) == 0: return
        px = float(self.price_data["close"].iloc[-1])
        print(f"\n🤖 FIXED ML-Filtered Bot - {self.symbol}")
        print(f"💰 Price: ${px:.4f} | Balance: ${self.account_balance:.2f}")
        print(f"⚡ Risk/trade: {self.config['risk_per_trade_pct']}% | Slippage exp: {self.config['expected_slippage_pct']}%")
        print(f"🛡️ Cooldown: {self.trade_cooldown}s | Protective Stop: {self.config.get('enable_protective_stop', False)}")

        if self.position:
            entry = float(self.position.get("avgPrice", 0))
            side = self.position.get("side", "")
            size = self.position.get("size", "0")
            if entry > 0:
                pnl_pct = ((px - entry)/entry*100.0) if side == "Buy" else ((entry - px)/entry*100.0)
                emoji = "🟢" if side == "Buy" else "🔴"
                print(f"{emoji} {side}: {size} @ {entry:.4f} | P&L: {pnl_pct:+.3f}%")
        elif self.pending_order:
            op = float(self.pending_order.get("price", 0))
            sd = self.pending_order.get("side", "")
            print(f"⏳ Pending {sd} @ {op:.4f}")
        else:
            print("🔍 ML scanning for high-confidence signals...")

        if self.last_trade_time > 0:
            dt = time.time() - self.last_trade_time
            if dt < self.trade_cooldown:
                print(f"⏰ Cooldown: {self.trade_cooldown - dt:.1f}s remaining")
        print("-"*50)

    async def run_cycle(self):

        if not await self.get_market_data():
            return

        await self.check_position()
        await self.check_pending_orders()

        if self.position:
            await self.ensure_protective_stop()
            should_close, reason = self.should_close()
            if should_close:
                await self.close_position(reason)
        elif not self.pending_order:
            sig = self.generate_signal(self.price_data)
            if sig:
                await self.execute_trade(sig)

        self.show_status()

    async def run(self):
        if not self.connect():
            print("Failed to connect")
            return

        print(f"🤖 FIXED ML-Filtered Scalping Bot - {self.symbol}")
        print("✅ FIXES APPLIED:")
        print(f"   • Compact JSONL (O/C) with risk, RR, slip_bps, fees, R, pnl_day")
        print(f"   • OPEN logged on actual fill (from position avgPrice)")
        print(f"   • Protective Stop (optional): {self.config.get('enable_protective_stop', False)} via set_trading_stop")
        print(f"   • Maker-only entries/exits; correct fee bps logging")
        print(f"   • Risk-based sizing, slippage modeling, dynamic balance")
        print(f"🎯 ML Threshold: {self.config['ml_confidence_threshold']:.2f}")
        print(f"💰 TP: {self.config['base_take_profit_pct']}% | SL: {self.config['base_stop_loss_pct']}%")

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
