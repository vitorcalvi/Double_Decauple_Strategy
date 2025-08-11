#!/usr/bin/env python3
"""
Unified Bot Framework with ML Grid Strategy — Bybit v5 (pybit)
-------------------------------------------------------------
Streamlined version combining the Unified Framework with ML Grid Trading
"""

from __future__ import annotations
import os, json, time, math, asyncio, typing as T
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pybit.unified_trading import HTTP
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

load_dotenv(override=True)

# ---------- utils ----------
def _iso(): 
    return datetime.now(timezone.utc).isoformat()

def _float(v, d=0.0):
    try:
        s = str(v).strip()
        if s == "" or s.lower() == "none": 
            return d
        return float(s)
    except: 
        return d

def _bps(delta, base):
    try:
        base = float(base)
        return None if base == 0 else float(delta)/base*1e4
    except: 
        return None

def _tags(tags: T.Union[None, str, dict]) -> dict:
    if tags is None: 
        return {}
    if isinstance(tags, dict): 
        return tags
    if isinstance(tags, str):
        out = {}
        for kv in tags.split("|"):
            if not kv: 
                continue
            if ":" in kv:
                k, v = kv.split(":", 1)
                k = k.strip()
                v = v.strip()
                try: 
                    out[k] = float(v)
                except:
                    lv = v.lower()
                    out[k] = True if lv == "true" else False if lv == "false" else v
            else:
                out[kv.strip()] = True
        return out
    return {"_raw": str(tags)}

# ---------- logger ----------
class TradeLogger:
    def __init__(self, bot_name: str, symbol: str, tf: str):
        self.ver, self.cls = 1, "TradeLogger.v1"
        self.sess = f"s{int(time.time()*1000):x}"
        self.env = "testnet" if os.getenv("DEMO_MODE", "true").lower() == "true" else "live"
        self.bot, self.sym, self.tf, self.ccy = bot_name, symbol, tf, "USDT"
        self.id_seq, self.pnl_day, self.streak_loss = 1000, 0.0, 0
        self.open: dict[int, dict] = {}
        os.makedirs("logs", exist_ok=True)
        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        self.log_file = f"logs/{self.bot}_{self.sym}_{today}.jsonl"

    def _id(self): 
        self.id_seq += 1
        return self.id_seq
    
    def _write(self, obj): 
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, separators=(",", ":")) + "\n")

    def _bars(self, dur_s: int) -> T.Optional[int]:
        try:
            tf_min = int(self.tf)
            return max(1, (dur_s + tf_min*60 - 1)//(tf_min*60))
        except: 
            return None

    def open_trade(self, side_LS: str, exp_px: float, act_px: float, qty: float, 
                   sl: float, tp: float, bal: float, tags=None) -> int:
        tid = self._id()
        slip = _bps(act_px - exp_px, exp_px)
        risk = abs(act_px - sl) * qty if sl and qty else 0.0
        rr = (abs(tp - act_px) / abs(act_px - sl)) if sl and tp and sl != act_px else None
        
        rec = {
            "ver": self.ver, "cls": self.cls, "sess": self.sess, "env": self.env,
            "bot": self.bot, "sym": self.sym, "tf": self.tf,
            "t": "O", "id": tid, "sd": side_LS, "ccy": self.ccy,
            "px": round(act_px, 6), "exp": round(exp_px, 6),
            "slip_bps": round(slip, 2) if slip is not None else None,
            "qty": round(qty, 6), "sl": round(_float(sl), 6), "tp": round(_float(tp), 6),
            "risk_usd": round(risk, 4), "rr_plan": round(rr, 4) if rr else None,
            "bal": round(_float(bal, 1000), 2), "tags": _tags(tags), "ts": _iso(),
        }
        self._write(rec)
        self.open[tid] = {
            "tso": datetime.now(timezone.utc), 
            "entry": act_px, 
            "sd": side_LS, 
            "qty": qty, 
            "risk": risk, 
            "tags": _tags(tags)
        }
        return tid

    def close_trade(self, tid: int, exp_exit: float, act_exit: float, reason: str, 
                    in_bps: float, out_bps: float, extra=None) -> dict:
        st = self.open.get(tid)
        if not st: 
            return {}
        
        dur = int((datetime.now(timezone.utc) - st["tso"]).total_seconds())
        edge = _bps(act_exit - exp_exit, exp_exit)
        qty, sd, entry = st["qty"], st["sd"], st["entry"]
        gross = (act_exit - entry) * qty if sd == "L" else (entry - act_exit) * qty
        fee_in = abs(_float(in_bps))/1e4 * entry * qty if in_bps is not None else 0.0
        fee_out = abs(_float(out_bps))/1e4 * act_exit * qty if out_bps is not None else 0.0
        fees = fee_in + fee_out
        net = gross - fees
        R = (net/st["risk"]) if st["risk"] > 0 else None
        self.pnl_day += net
        self.streak_loss = self.streak_loss + 1 if net < 0 else 0
        
        rec = {
            "ver": self.ver, "cls": self.cls, "sess": self.sess, "env": self.env,
            "bot": self.bot, "sym": self.sym, "tf": self.tf,
            "t": "C", "ref": tid, "ccy": self.ccy,
            "px": round(act_exit, 6), "ref_px": round(exp_exit, 6),
            "edge_bps": round(edge, 2) if edge is not None else None,
            "dur_s": dur, "bars_held": self._bars(dur),
            "qty": round(qty, 6), "gross": round(gross, 4),
            "fees_in_bps": round(_float(in_bps), 2) if in_bps is not None else None,
            "fees_out_bps": round(_float(out_bps), 2) if out_bps is not None else None,
            "fees_total": round(fees, 4), "net": round(net, 4), 
            "R": round(R, 4) if R is not None else None,
            "exit": reason, "pnl_day": round(self.pnl_day, 4), 
            "streak_loss": int(self.streak_loss),
            "tags": st["tags"], "extra": extra or {}, "ts": _iso(),
        }
        self._write(rec)
        self.open.pop(tid, None)
        return rec

    def close_unknown(self, tid: int, reason="external_close", extra=None) -> dict:
        st = self.open.get(tid)
        if not st: 
            return {}
        
        dur = int((datetime.now(timezone.utc) - st["tso"]).total_seconds())
        rec = {
            "ver": self.ver, "cls": self.cls, "sess": self.sess, "env": self.env,
            "bot": self.bot, "sym": self.sym, "tf": self.tf,
            "t": "C", "ref": tid, "ccy": self.ccy, 
            "px": None, "ref_px": None, "edge_bps": None,
            "dur_s": dur, "bars_held": self._bars(dur), "qty": round(st["qty"], 6),
            "gross": None, "fees_in_bps": None, "fees_out_bps": None, 
            "fees_total": None, "net": None, "R": None,
            "exit": reason, "pnl_day": round(self.pnl_day, 4),
            "streak_loss": int(self.streak_loss), "tags": st["tags"], 
            "extra": extra or {}, "ts": _iso(),
        }
        self._write(rec)
        self.open.pop(tid, None)
        return rec

# ---------- exchange ----------
class ExchangeClient:
    def __init__(self, symbol: str, demo: bool, api_key: str, api_secret: str):
        self.symbol, self.demo = symbol, demo
        self.http = HTTP(demo=demo, api_key=api_key, api_secret=api_secret)
        self.spec = {}

    def ping(self) -> bool:
        try: 
            return self.http.get_server_time().get("retCode") == 0
        except: 
            return False

    def fetch_specs(self):
        try:
            r = self.http.get_instruments_info(category="linear", symbol=self.symbol)
            if r.get("retCode") != 0: 
                return
            lst = r.get("result", {}).get("list", [])
            if not lst: 
                return
            it = lst[0]
            pf = it.get("priceFilter", {})
            lf = it.get("lotSizeFilter", {})
            self.spec = {
                "tickSize": _float(pf.get("tickSize", 0.01), 0.01),
                "qtyStep": _float(lf.get("qtyStep", 0.001), 0.001),
                "minOrderQty": _float(lf.get("minOrderQty", 0.0), 0.0),
                "minNotional": _float(lf.get("minOrderAmt", 0.0), 0.0),
            }
        except: 
            self.spec = {}

    def get_klines(self, tf: str, limit: int=100) -> pd.DataFrame:
        try:
            r = self.http.get_kline(category="linear", symbol=self.symbol, 
                                    interval=tf, limit=limit)
            if r.get("retCode") != 0: 
                return pd.DataFrame()
            
            df = pd.DataFrame(r["result"]["list"], 
                            columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
            df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms", utc=True)
            for c in ["open", "high", "low", "close", "volume"]: 
                df[c] = pd.to_numeric(df[c], errors="coerce")
            return df.sort_values("timestamp").reset_index(drop=True)
        except:
            return pd.DataFrame()

    def get_balance_usdt(self) -> float:
        try:
            r = self.http.get_wallet_balance(accountType="UNIFIED", coin="USDT")
            if r.get("retCode") == 0:
                for lst in r.get("result", {}).get("list", []):
                    for c in lst.get("coin", []):
                        if c.get("coin") == "USDT":
                            for field in ['availableToWithdraw', 'walletBalance', 'equity', 'availableBalance', 'balance']:
                                v = c.get(field)
                                if v and str(v).strip():
                                    return _float(v, 1000.0)
        except: 
            pass
        return 1000.0

    def get_open_orders(self) -> list:
        try:
            r = self.http.get_open_orders(category="linear", symbol=self.symbol, limit=50)
            if r.get("retCode") != 0: 
                return []
            return r.get("result", {}).get("list", [])
        except: 
            return []

    def cancel_order(self, order_id: str):
        try: 
            self.http.cancel_order(category="linear", symbol=self.symbol, orderId=order_id)
        except: 
            pass

    def get_position(self) -> dict:
        try:
            r = self.http.get_positions(category="linear", symbol=self.symbol)
            if r.get("retCode") != 0: 
                return {}
            lst = r.get("result", {}).get("list", [])
            if not lst: 
                return {}
            pos = lst[0]
            if _float(pos.get("size", 0)) == 0: 
                return {}
            return pos
        except: 
            return {}

    def place_limit(self, side: str, qty: str, price: str, reduce_only: bool=False) -> dict:
        return self.http.place_order(
            category="linear", symbol=self.symbol, side=side, orderType="Limit",
            qty=qty, price=price, timeInForce="PostOnly", reduceOnly=reduce_only, positionIdx=0
        )

# ---------- risk ----------
@dataclass
class SizingSpec:
    tick_size: float = 0.01
    qty_step: float = 0.001
    min_notional: float = 5.0
    min_order_qty: float = 0.0

class RiskManager:
    def __init__(self, risk_pct: float, spec: SizingSpec):
        self.risk_pct = float(risk_pct)
        self.spec = spec

    @staticmethod
    def _round_step(x: float, step: float) -> float:
        if step <= 0: 
            return float(x)
        return math.floor(float(x)/step + 1e-12) * step

    def _decimals(self, step: float) -> int:
        s = f"{step:.12f}".rstrip("0").rstrip(".")
        return len(s.split(".")[1]) if "." in s else 0

    def fmt_qty(self, q: float) -> str:
        q2 = self._round_step(q, self.spec.qty_step)
        dec = self._decimals(self.spec.qty_step)
        return f"{q2:.{dec}f}"

    def fmt_price(self, p: float) -> str:
        p2 = self._round_step(p, self.spec.tick_size)
        dec = self._decimals(self.spec.tick_size)
        return f"{p2:.{dec}f}"

    def position_size(self, balance_usd: float, price: float, stop_loss: float) -> float:
        diff = abs(price - stop_loss)
        if balance_usd <= 0 or diff <= 0: 
            return 0.0
        
        qty = (balance_usd * self.risk_pct / 100.0) / diff
        
        # Apply minimum constraints
        min_qty_notional = self.spec.min_notional / max(price, 1e-9) if self.spec.min_notional > 0 else 0.0
        if self.spec.min_order_qty > 0: 
            qty = max(qty, self.spec.min_order_qty)
        if min_qty_notional > 0: 
            qty = max(qty, min_qty_notional)
        
        # Cap to max position (10% of balance or $1000)
        max_position_value = min(balance_usd * 0.1, 1000)
        max_qty = max_position_value / price
        return min(qty, max_qty)

# ---------- strategy base ----------
@dataclass
class Signal:
    action: str      # "BUY" / "SELL"
    ref_price: float
    sl: float
    tp: float
    tags: dict

class IStrategy:
    name = "BASE"
    def __init__(self, cfg: dict): 
        self.cfg = cfg
        self.state = {}
    
    def update(self, df: pd.DataFrame) -> None: 
        raise NotImplementedError
    
    def entry(self, df: pd.DataFrame) -> T.Optional[Signal]: 
        raise NotImplementedError
    
    def exit_reason(self, df: pd.DataFrame, position: dict) -> T.Optional[str]: 
        raise NotImplementedError

# ---------- ML Grid Strategy ----------
class MLGridStrategy(IStrategy):
    name = "ML_GRID"
    
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.ml_model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
        self.scaler = StandardScaler()
        self.model_trained = False
        self.grid_levels = []
        self.last_grid_level = None
        
    def _prepare_features(self, df: pd.DataFrame) -> T.Optional[np.ndarray]:
        if len(df) < 30:
            return None
        
        features = []
        
        # Price features
        features.append(df['close'].pct_change().iloc[-1] or 0)
        features.append((df['high'].iloc[-1] - df['low'].iloc[-1]) / df['close'].iloc[-1])
        
        # SMA feature
        sma_10 = df['close'].rolling(10).mean().iloc[-1]
        features.append((df['close'].iloc[-1] - sma_10) / df['close'].iloc[-1] if sma_10 > 0 else 0)
        
        # Volume feature
        vol_sma = df['volume'].rolling(20).mean().iloc[-1]
        features.append(df['volume'].iloc[-1] / vol_sma if vol_sma > 0 else 1)
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi = 100 - (100 / (1 + gain / (loss + 1e-10)))
        features.append(rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50)
        
        # Volatility
        vol = df['close'].rolling(20).std().iloc[-1]
        features.append(vol / df['close'].iloc[-1] if vol > 0 else 0.01)
        
        return np.array(features).reshape(1, -1)
    
    def _train_model(self, df: pd.DataFrame):
        if len(df) < 50:
            return
        
        X, y = [], []
        
        for i in range(30, min(len(df) - 10, 100)):  # Limit training samples
            features = self._prepare_features(df.iloc[:i+1])
            if features is not None:
                X.append(features[0])
                # Target: optimal grid spacing based on future volatility
                future_volatility = df['close'].iloc[i:i+10].std() / df['close'].iloc[i]
                optimal_spacing = min(max(0.3, future_volatility * 100), 1.0)
                y.append(optimal_spacing)
        
        if len(X) > 20:
            X = np.array(X)
            y = np.array(y)
            
            self.scaler.fit(X)
            X_scaled = self.scaler.transform(X)
            self.ml_model.fit(X_scaled, y)
            self.model_trained = True
    
    def _predict_grid_spacing(self, df: pd.DataFrame) -> float:
        if not self.model_trained:
            return _float(self.cfg.get("base_grid_spacing", 0.5), 0.5)
        
        features = self._prepare_features(df)
        if features is None:
            return _float(self.cfg.get("base_grid_spacing", 0.5), 0.5)
        
        features_scaled = self.scaler.transform(features)
        predicted_spacing = self.ml_model.predict(features_scaled)[0]
        
        return min(max(0.3, predicted_spacing), 1.0)
    
    def _get_ml_confidence(self, df: pd.DataFrame) -> float:
        if not self.model_trained:
            return 0.5
        
        features = self._prepare_features(df)
        if features is None:
            return 0.5
        
        features_scaled = self.scaler.transform(features)
        predictions = []
        
        for estimator in self.ml_model.estimators_[:10]:
            predictions.append(estimator.predict(features_scaled)[0])
        
        std_dev = np.std(predictions)
        confidence = max(0, 1 - min(std_dev * 2, 1))
        
        return confidence
    
    def _update_grid_levels(self, current_price: float, optimal_spacing: float):
        self.grid_levels = []
        spacing_pct = optimal_spacing / 100
        grid_count = int(_float(self.cfg.get("grid_levels", 5), 5))
        
        for i in range(-grid_count, grid_count + 1):
            if i != 0:
                level = current_price * (1 + i * spacing_pct)
                self.grid_levels.append({
                    'price': level,
                    'index': i,
                    'side': 'BUY' if i < 0 else 'SELL'
                })
        
        self.grid_levels.sort(key=lambda x: x['price'])
    
    def update(self, df: pd.DataFrame) -> None:
        self.state.clear()
        if df.empty:
            return
        
        # Train model periodically
        if not self.model_trained or np.random.random() < 0.05:
            self._train_model(df)
        
        if self.model_trained:
            current_price = float(df['close'].iloc[-1])
            optimal_spacing = self._predict_grid_spacing(df)
            ml_confidence = self._get_ml_confidence(df)
            
            self._update_grid_levels(current_price, optimal_spacing)
            
            self.state = {
                "ml_confidence": ml_confidence,
                "grid_spacing": optimal_spacing,
                "current_price": current_price
            }
    
    def entry(self, df: pd.DataFrame) -> T.Optional[Signal]:
        if not self.state or df.empty:
            return None
        
        ml_confidence = self.state.get("ml_confidence", 0)
        ml_threshold = _float(self.cfg.get("ml_threshold", 0.45), 0.45)
        
        if ml_confidence < ml_threshold:
            return None
        
        current_price = float(df['close'].iloc[-1])
        
        # Find nearest grid level
        for level in self.grid_levels:
            distance_pct = abs(current_price - level['price']) / level['price'] * 100
            if distance_pct < 0.50 and level != self.last_grid_level:
                self.last_grid_level = level
                
                # Calculate stop loss and take profit
                sl_pct = _float(self.cfg.get("net_stop_loss", 0.3), 0.3) / 100
                tp_pct = _float(self.cfg.get("net_take_profit", 0.75), 0.75) / 100
                
                if level['side'] == 'BUY':
                    sl = current_price * (1 - sl_pct)
                    tp = current_price * (1 + tp_pct)
                else:
                    sl = current_price * (1 + sl_pct)
                    tp = current_price * (1 - tp_pct)
                
                tags = {
                    "ml_conf": round(ml_confidence, 2),
                    "grid_spacing": round(self.state["grid_spacing"], 2),
                    "grid_index": level['index']
                }
                
                return Signal(level['side'], current_price, sl, tp, tags)
        
        return None
    
    def exit_reason(self, df: pd.DataFrame, pos: dict) -> T.Optional[str]:
        if not pos:
            return None
        
        p = _float(df["close"].iloc[-1])
        e = _float(pos.get("avgPrice", 0))
        side = pos.get("side", "")
        
        if e <= 0:
            return None
        
        tp = _float(self.cfg.get("net_take_profit", 0.75), 0.75)
        sl = _float(self.cfg.get("net_stop_loss", 0.3), 0.3)
        
        pnl = ((p - e) / e * 100 if side == "Buy" else (e - p) / e * 100)
        
        if pnl >= tp:
            return "grid_target"
        if pnl <= -sl:
            return "stop_loss"
        
        # Exit if ML confidence drops
        if self.state.get("ml_confidence", 0) < 0.3:
            return "low_ml_confidence"
        
        return None

# ---------- strategy registry ----------
STRATEGY_REGISTRY: dict[str, T.Type[IStrategy]] = {
    "ML_GRID": MLGridStrategy,
}

# ---------- engine ----------
class TraderEngine:
    def __init__(self):
        self.symbol = os.getenv("SYMBOL", "SUIUSDT")
        self.demo = os.getenv("DEMO_MODE", "true").lower() == "true"
        pref = "TESTNET_" if self.demo else "LIVE_"
        api_key = os.getenv(f"{pref}BYBIT_API_KEY", "")
        api_sec = os.getenv(f"{pref}BYBIT_API_SECRET", "")
        self.client = ExchangeClient(self.symbol, self.demo, api_key, api_sec)

        tf = os.getenv("TIMEFRAME", "1")
        self.cfg = {
            "timeframe": tf,
            "risk_pct": _float(os.getenv("RISK_PCT", 1.0), 1.0),
            "maker_fee_bps": _float(os.getenv("MAKER_FEE_BPS", -4), -4),
            "min_notional": _float(os.getenv("MIN_NOTIONAL", 5), 5.0),
            "cooldown_sec": int(_float(os.getenv("COOLDOWN_SEC", 10), 10)),
            "min_order_interval_sec": int(_float(os.getenv("MIN_ORDER_INTERVAL_SEC", 10), 10)),
            "maker_offset_bps": os.getenv("MAKER_OFFSET_BPS", "").strip(),
            "maker_offset_pct": os.getenv("MAKER_OFFSET_PCT", "0.01").strip(),
            # ML Grid specific
            "base_grid_spacing": _float(os.getenv("BASE_GRID_SPACING", 0.5), 0.5),
            "grid_levels": int(_float(os.getenv("GRID_LEVELS", 5), 5)),
            "ml_threshold": _float(os.getenv("ML_THRESHOLD", 0.45), 0.45),
            "net_take_profit": _float(os.getenv("NET_TAKE_PROFIT", 0.75), 0.75),
            "net_stop_loss": _float(os.getenv("NET_STOP_LOSS", 0.3), 0.3),
        }
        
        sname = os.getenv("STRATEGY", "ML_GRID")
        StrategyClass = STRATEGY_REGISTRY.get(sname, MLGridStrategy)
        self.strategy = StrategyClass(self.cfg)

        self.price_data = pd.DataFrame()
        self.account_balance = 1000.0
        self.position = {}
        self.pending_order = False
        self.active_order_id = None
        self.last_order_time = None
        self.last_trade_ts = 0.0
        self.sizing = SizingSpec()
        self.risk = RiskManager(self.cfg["risk_pct"], self.sizing)
        self.logger = TradeLogger(self.strategy.name, self.symbol, tf=self.cfg["timeframe"])
        self.current_trade_id = None

    def _hydrate_specs(self):
        self.client.fetch_specs()
        if self.client.spec:
            self.sizing.tick_size = self.client.spec.get("tickSize", self.sizing.tick_size)
            self.sizing.qty_step = self.client.spec.get("qtyStep", self.sizing.qty_step)
            self.sizing.min_order_qty = self.client.spec.get("minOrderQty", 0.0)
            min_notional = self.client.spec.get("minNotional", 0.0) or 0.0
            self.sizing.min_notional = max(self.cfg["min_notional"], min_notional)
        self.risk = RiskManager(self.cfg["risk_pct"], self.sizing)

    def _limit_with_offset(self, market_px: float, side: str) -> float:
        if self.cfg.get("maker_offset_bps"):
            off = float(self.cfg["maker_offset_bps"]) / 1e4
        else:
            off = float(self.cfg.get("maker_offset_pct") or 0.01) / 100.0
        mult = (1 - off) if side.lower().startswith("buy") else (1 + off)
        return market_px * mult

    def _gc_stale_orders(self):
        """Garbage collect stale orders"""
        active_found = False
        for o in self.client.get_open_orders():
            try:
                created = int(o.get("createdTime", "0"))
                age = (datetime.now(timezone.utc) - datetime.fromtimestamp(created/1000, tz=timezone.utc)).total_seconds()
            except:
                age = 0
            
            if age > 60:
                self.client.cancel_order(o["orderId"])
            else:
                self.pending_order = True
                self.active_order_id = o.get("orderId")
                active_found = True
                break
        
        if not active_found:
            self.pending_order = False
            self.active_order_id = None

    def _check_position_change(self, new_pos: dict):
        """Check if position closed externally"""
        if not new_pos and self.position:
            if self.current_trade_id:
                self.logger.close_unknown(self.current_trade_id)
                self.current_trade_id = None
            # Reset grid level on position close
            if hasattr(self.strategy, 'last_grid_level'):
                self.strategy.last_grid_level = None

    async def run_cycle(self):
        # Fetch market data
        df = self.client.get_klines(self.cfg["timeframe"], limit=100)
        if df.empty:
            return
        
        self.price_data = df
        self.strategy.update(df)
        
        # Update balance if no position
        if not self.position:
            self.account_balance = self.client.get_balance_usdt()
        
        # Clean up stale orders
        self._gc_stale_orders()
        
        # Check position
        new_pos = self.client.get_position()
        self._check_position_change(new_pos)
        self.position = new_pos
        
        now = datetime.now(timezone.utc)
        
        # Process exits first
        if self.position and not self.pending_order:
            reason = self.strategy.exit_reason(self.price_data, self.position)
            if reason:
                await self._close(reason)
                return
        
        # Process entries
        if not self.position and not self.pending_order:
            if self.last_order_time and (now - self.last_order_time).total_seconds() < self.cfg["min_order_interval_sec"]:
                return
            sig = self.strategy.entry(self.price_data)
            if sig:
                await self._open(sig)

    async def _open(self, sig: Signal):
        # Check cooldown
        if time.time() - self.last_trade_ts < self.cfg["cooldown_sec"]:
            return
        
        self.pending_order = True
        self.last_order_time = datetime.now(timezone.utc)

        # Calculate position size
        qty = self.risk.position_size(self.account_balance, sig.ref_price, sig.sl)
        qty_str = self.risk.fmt_qty(qty)
        
        if _float(qty_str) <= 0:
            self.pending_order = False
            return
        
        if _float(qty_str) * sig.ref_price < self.sizing.min_notional:
            self.pending_order = False
            return

        # Calculate limit price
        limit = self._limit_with_offset(sig.ref_price, "Buy" if sig.action == "BUY" else "Sell")
        price_str = self.risk.fmt_price(limit)

        try:
            r = self.client.place_limit(
                side="Buy" if sig.action == "BUY" else "Sell",
                qty=qty_str,
                price=price_str,
                reduce_only=False
            )
            
            if r.get("retCode") == 0:
                self.last_trade_ts = time.time()
                self.active_order_id = r["result"]["orderId"]
                tags = {**sig.tags, "risk_pct": self.cfg["risk_pct"]}
                
                self.current_trade_id = self.logger.open_trade(
                    "L" if sig.action == "BUY" else "S",
                    exp_px=sig.ref_price,
                    act_px=_float(price_str, sig.ref_price),
                    qty=_float(qty_str),
                    sl=sig.sl,
                    tp=sig.tp,
                    bal=self.account_balance,
                    tags=tags
                )
                print(f"✅ OPEN {sig.action}: {qty_str} @ {price_str}")
                if hasattr(self.strategy, 'state') and self.strategy.state:
                    print(f"   📊 ML Confidence: {self.strategy.state.get('ml_confidence', 0):.2f}")
                    print(f"   📏 Grid Spacing: {self.strategy.state.get('grid_spacing', 0):.2f}%")
            else:
                print(f"❌ Open failed: {r.get('retMsg')}")
                self.pending_order = False
        except Exception as e:
            print(f"❌ Open error: {e}")
            self.pending_order = False

    async def _close(self, reason: str):
        if not self.position:
            return
        
        self.pending_order = True
        price = _float(self.price_data['close'].iloc[-1])
        side = "Sell" if self.position.get("side") == "Buy" else "Buy"
        qty = _float(self.position.get("size", 0))
        
        if qty <= 0:
            self.pending_order = False
            return
        
        limit = self._limit_with_offset(price, side)
        price_str = self.risk.fmt_price(limit)
        qty_str = self.risk.fmt_qty(qty)
        
        try:
            r = self.client.place_limit(
                side=side,
                qty=qty_str,
                price=price_str,
                reduce_only=True
            )
            
            if r.get("retCode") == 0:
                print(f"✅ CLOSE {side}: {qty_str} @ {price_str} ({reason})")
                if self.current_trade_id:
                    self.logger.close_trade(
                        tid=self.current_trade_id,
                        exp_exit=price,
                        act_exit=_float(price_str, price),
                        reason=reason,
                        in_bps=self.cfg["maker_fee_bps"],
                        out_bps=self.cfg["maker_fee_bps"],
                        extra={"mode": "maker", "cooldown": self.cfg["cooldown_sec"]}
                    )
                    self.current_trade_id = None
                self.pending_order = False
            else:
                print(f"❌ Close failed: {r.get('retMsg')}")
                self.pending_order = False
        except Exception as e:
            print(f"❌ Close error: {e}")
            self.pending_order = False

    def show_status(self):
        """Display current status"""
        if self.price_data.empty:
            return
        
        current_price = float(self.price_data['close'].iloc[-1])
        
        print(f"\n{'='*60}")
        print(f"🤖 {self.strategy.name} | {self.symbol} | tf={self.cfg['timeframe']}m")
        print(f"💰 Price: ${current_price:.5f} | Balance: ${self.account_balance:.2f}")
        print(f"🎯 Risk: {self.cfg['risk_pct']}% | Cooldown: {self.cfg['cooldown_sec']}s")
        
        if hasattr(self.strategy, 'state') and self.strategy.state:
            state = self.strategy.state
            if 'ml_confidence' in state:
                print(f"📊 ML Confidence: {state['ml_confidence']:.2f} | Grid Spacing: {state.get('grid_spacing', 0):.2f}%")
        
        if self.position:
            entry = float(self.position.get('avgPrice', 0))
            side = self.position.get('side', '')
            size = self.position.get('size', '0')
            pnl = float(self.position.get('unrealisedPnl', 0))
            
            emoji = "🟢" if side == "Buy" else "🔴"
            print(f"{emoji} Position: {side} {size} @ ${entry:.5f} | PnL: ${pnl:.2f}")
        else:
            print("🔍 No position - scanning for signals...")
        
        print(f"{'='*60}")

    async def run(self):
        if not self.client.ping():
            print("❌ Connection failed")
            return
        
        self._hydrate_specs()
        print(f"📊 {self.strategy.name} | {self.logger.sess} | {self.logger.env} | {self.symbol} tf={self.cfg['timeframe']}")
        print(f"✅ Streamlined version initialized")
        
        try:
            cycle_count = 0
            while True:
                await self.run_cycle()
                cycle_count += 1
                
                # Show status every 12 cycles (1 minute at 5s intervals)
                if cycle_count % 12 == 0:
                    self.show_status()
                
                await asyncio.sleep(5)
        except KeyboardInterrupt:
            print("\n🛑 Stopping...")
            if self.position:
                await self._close("manual_stop")

if __name__ == "__main__":
    asyncio.run(TraderEngine().run())