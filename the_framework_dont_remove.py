#!/usr/bin/env python3
"""
Streamlined Trading Bot Framework - Bybit v5
Essential components only for LLM analysis and real trading
"""
import os
import json
import time
import math
import asyncio
from datetime import datetime, timezone
import pandas as pd
from dotenv import load_dotenv
from pybit.unified_trading import HTTP

load_dotenv(override=True)

# -------------------- Config --------------------
class Config:
    def __init__(self):
        self.symbol = os.getenv("SYMBOL", "SUIUSDT")
        self.demo = os.getenv("DEMO_MODE", "true").lower() == "true"
        self.timeframe = os.getenv("TIMEFRAME", "1")
        self.risk_pct = float(os.getenv("RISK_PCT", 1.0))
        self.maker_offset_bps = float(os.getenv("MAKER_OFFSET_BPS", 10))
        self.max_position_pct = float(os.getenv("MAX_POSITION_PCT", 0.10))
        self.max_trades_hour = int(os.getenv("MAX_TRADES_PER_HOUR", 12))
        self.max_daily_loss = float(os.getenv("MAX_DAILY_LOSS_USD", 150))
        self.cooldown_sec = int(os.getenv("COOLDOWN_SEC", 10))
        
        prefix = "TESTNET_" if self.demo else "LIVE_"
        self.api_key = os.getenv(f"{prefix}BYBIT_API_KEY", "")
        self.api_secret = os.getenv(f"{prefix}BYBIT_API_SECRET", "")

# -------------------- Exchange Client --------------------
class Exchange:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.http = HTTP(demo=cfg.demo, api_key=cfg.api_key, api_secret=cfg.api_secret)
        self.specs = {"tick": 0.01, "qty_step": 0.001, "min_qty": 0.001, "min_notional": 5.0}
        
    def fetch_specs(self):
        try:
            r = self.http.get_instruments_info(category="linear", symbol=self.cfg.symbol)
            if r.get("retCode") == 0:
                info = r["result"]["list"][0]
                self.specs["tick"] = float(info["priceFilter"]["tickSize"])
                self.specs["qty_step"] = float(info["lotSizeFilter"]["qtyStep"])
                self.specs["min_qty"] = float(info["lotSizeFilter"]["minOrderQty"])
                self.specs["min_notional"] = float(info["lotSizeFilter"].get("minOrderAmt", 5))
        except:
            pass
    
    def get_klines(self, limit=100):
        try:
            r = self.http.get_kline(category="linear", symbol=self.cfg.symbol, 
                                   interval=self.cfg.timeframe, limit=limit)
            if r.get("retCode") != 0:
                return pd.DataFrame()
            df = pd.DataFrame(r["result"]["list"], 
                            columns=["timestamp","open","high","low","close","volume","turnover"])
            df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms", utc=True)
            for col in ["open","high","low","close","volume"]:
                df[col] = pd.to_numeric(df[col])
            return df.sort_values("timestamp").reset_index(drop=True)
        except:
            return pd.DataFrame()
    
    def get_balance(self):
        try:
            r = self.http.get_wallet_balance(accountType="UNIFIED", coin="USDT")
            if r.get("retCode") == 0:
                coin = r["result"]["list"][0]["coin"][0]
                return float(coin.get("availableBalance", 1000))
        except:
            return 1000.0
    
    def get_position(self):
        try:
            r = self.http.get_positions(category="linear", symbol=self.cfg.symbol)
            if r.get("retCode") == 0 and r["result"]["list"]:
                pos = r["result"]["list"][0]
                if float(pos.get("size", 0)) > 0:
                    return pos
        except:
            pass
        return None
    
    def place_order(self, side, qty, price, reduce_only=False):
        qty_str = self.fmt_qty(qty)
        price_str = self.fmt_price(price)
        return self.http.place_order(
            category="linear", symbol=self.cfg.symbol, side=side,
            orderType="Limit", qty=qty_str, price=price_str,
            timeInForce="PostOnly", reduceOnly=reduce_only, positionIdx=0
        )
    
    def cancel_order(self, order_id):
        try:
            self.http.cancel_order(category="linear", symbol=self.cfg.symbol, orderId=order_id)
        except:
            pass
    
    def get_open_orders(self):
        try:
            r = self.http.get_open_orders(category="linear", symbol=self.cfg.symbol)
            return r.get("result", {}).get("list", []) if r.get("retCode") == 0 else []
        except:
            return []
    
    def fmt_qty(self, qty):
        step = self.specs["qty_step"]
        qty = math.floor(qty / step) * step
        decimals = len(str(step).split(".")[1]) if "." in str(step) else 0
        return f"{qty:.{decimals}f}"
    
    def fmt_price(self, price):
        tick = self.specs["tick"]
        price = math.floor(price / tick) * tick
        decimals = len(str(tick).split(".")[1]) if "." in str(tick) else 0
        return f"{price:.{decimals}f}"

# -------------------- Logger --------------------
class LogTrader:
    """
    JSONL trade logger for LLM analysis.
    Records: O=Open, F=Post-fill, C=Close
    """
    def __init__(self, bot_name: str, symbol: str, tf: str):
        self.ver, self.cls = 1, "LogTrader.v1"
        self.sess = f"s{int(time.time()*1000):x}"
        self.env = "testnet" if os.getenv("DEMO_MODE", "true").lower() == "true" else "live"
        self.bot, self.sym, self.tf, self.ccy = bot_name, symbol, tf, "USDT"
        self.id_seq, self.pnl_day, self.streak_loss = 1000, 0.0, 0
        self.open = {}
        os.makedirs("logs", exist_ok=True)
        today = datetime.now().strftime("%Y%m%d")
        self.log_file = f"logs/{self.bot}_{self.sym}_{today}.jsonl"

    def _id(self):
        self.id_seq += 1
        return self.id_seq

    def _write(self, obj):
        with open(self.log_file, "a") as f:
            f.write(json.dumps(obj, separators=(",", ":")) + "\n")

    def _bars(self, dur_s):
        try:
            tf_min = int(self.tf)
            return max(1, (dur_s + tf_min*60 - 1)//(tf_min*60))
        except:
            return None

    def _bps(self, delta, base):
        try:
            return None if base == 0 else float(delta) / base * 1e4
        except:
            return None

    def open_trade(self, side, exp_px, act_px, qty, sl=None, tp=None, bal=1000, tags=None):
        tid = self._id()
        slip = self._bps(act_px - exp_px, exp_px)
        risk = abs(act_px - sl) * qty if sl and qty else 0.0
        rr = (abs(float(tp or 0) - act_px) / abs(act_px - sl)) if (sl and tp and sl != act_px) else None
        rec = {
            "ver": self.ver, "cls": self.cls, "sess": self.sess, "env": self.env,
            "bot": self.bot, "sym": self.sym, "tf": self.tf, "ccy": self.ccy,
            "t": "O", "id": tid, "sd": side,
            "px": round(act_px, 6), "exp": round(exp_px, 6),
            "slip_bps": round(slip, 2) if slip is not None else None,
            "qty": round(qty, 6), "sl": sl, "tp": tp,
            "risk_usd": round(risk, 6), "rr_plan": round(rr, 4) if rr else None,
            "bal": round(bal, 2), "tags": tags or {}, "ts": datetime.now(timezone.utc).isoformat(),
        }
        self._write(rec)
        self.open[tid] = {
            "tso": datetime.now(timezone.utc),
            "entry": act_px, "sd": side, "qty": qty, "risk": risk, "tags": tags or {}
        }
        return tid

    def post_fill(self, tid, phase, venue_order, venue_position=None, fills_summary=None, placed_px=None):
        vo = venue_order or {}
        avg_px = float(vo.get("avgPrice", 0)) if vo.get("avgPrice") else None
        slip_bps = self._bps(avg_px - placed_px, placed_px) if (avg_px and placed_px) else None
        
        rec = {
            "ver": self.ver, "cls": self.cls, "sess": self.sess, "env": self.env,
            "bot": self.bot, "sym": self.sym, "tf": self.tf,
            "t": "F", "phase": phase, "ref": tid,
            "venue_order": {
                "orderId": vo.get("orderId"),
                "orderStatus": vo.get("orderStatus"),
                "side": vo.get("side"),
                "price": float(vo.get("price", 0)) if vo.get("price") else None,
                "avgPrice": avg_px,
                "qty": float(vo.get("qty", 0)) if vo.get("qty") else None,
            },
            "fills": fills_summary or {},
            "slip_bps_placed": round(slip_bps, 2) if slip_bps else None,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        self._write(rec)

    def close_trade(self, tid, exp_exit, act_exit, reason, in_bps=None, out_bps=None, extra=None):
        st = self.open.get(tid)
        if not st:
            return {}
        dur = int((datetime.now(timezone.utc) - st["tso"]).total_seconds())
        edge = self._bps(act_exit - exp_exit, exp_exit)
        qty, sd, entry = st["qty"], st["sd"], st["entry"]
        gross = (act_exit - entry) * qty if sd == "L" else (entry - act_exit) * qty
        fee_in = abs(float(in_bps or 0))/1e4 * entry * qty if in_bps else 0.0
        fee_out = abs(float(out_bps or 0))/1e4 * act_exit * qty if out_bps else 0.0
        fees = fee_in + fee_out
        net = gross - fees
        R = (net/st["risk"]) if st["risk"] > 0 else None
        self.pnl_day += net
        self.streak_loss = self.streak_loss + 1 if net < 0 else 0
        rec = {
            "ver": self.ver, "cls": self.cls, "sess": self.sess, "env": self.env,
            "bot": self.bot, "sym": self.sym, "tf": self.tf,
            "t": "C", "ref": tid,
            "px": round(act_exit, 6), "ref_px": round(exp_exit, 6),
            "edge_bps": round(edge, 2) if edge else None,
            "dur_s": dur, "bars_held": self._bars(dur),
            "qty": round(qty, 6), "gross": round(gross, 6), "net": round(net, 6),
            "R": round(R, 6) if R else None,
            "exit": reason, "pnl_day": round(self.pnl_day, 6),
            "streak_loss": int(self.streak_loss),
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        self._write(rec)
        self.open.pop(tid, None)
        return rec

# -------------------- Strategy Base --------------------
class Strategy:
    def __init__(self):
        self.state = {}
    
    def update(self, df):
        """Update internal state with latest data"""
        if not df.empty:
            self.state["price"] = float(df["close"].iloc[-1])
    
    def signal(self, df):
        """Return BUY/SELL signal or None"""
        return None
    
    def should_exit(self, df, position):
        """Return exit reason or None"""
        return None

# -------------------- Risk Manager --------------------
class RiskManager:
    def __init__(self, cfg: Config, specs: dict):
        self.cfg = cfg
        self.specs = specs
    
    def calculate_size(self, balance, price, stop_loss):
        """Calculate position size based on risk"""
        if not stop_loss or abs(price - stop_loss) < 0.00001:
            return 0
        
        risk_amount = balance * (self.cfg.risk_pct / 100)
        qty = risk_amount / abs(price - stop_loss)
        
        # Apply limits
        max_position = balance * self.cfg.max_position_pct
        qty = min(qty, max_position / price)
        
        # Exchange minimums
        min_notional = self.specs["min_notional"] / price
        qty = max(qty, max(self.specs["min_qty"], min_notional))
        
        return qty if qty * price >= self.specs["min_notional"] else 0

# -------------------- Main Engine --------------------
class TradingEngine:
    def __init__(self):
        self.cfg = Config()
        self.exchange = Exchange(self.cfg)
        self.logger = LogTrader("BOT", self.cfg.symbol, self.cfg.timeframe)
        self.strategy = Strategy()
        self.risk = RiskManager(self.cfg, self.exchange.specs)
        
        self.position = None
        self.balance = 1000.0
        self.last_bar_time = None
        self.last_trade_time = 0
        self.trades_hour = []
        self.active_order = None
        self.current_tid = None  # Track current trade ID
    
    def check_limits(self):
        """Check trading limits"""
        now = time.time()
        self.trades_hour = [t for t in self.trades_hour if now - t < 3600]
        
        if len(self.trades_hour) >= self.cfg.max_trades_hour:
            return False
        if self.logger.pnl_day <= -self.cfg.max_daily_loss:
            return False
        if self.logger.streak_loss >= 5:
            return False
        return True
    
    def new_bar(self, df):
        """Check if new bar is available"""
        if df.empty:
            return False
        current = pd.to_datetime(df["timestamp"].iloc[-1])
        if self.last_bar_time is None or current > self.last_bar_time:
            self.last_bar_time = current
            return True
        return False
    
    async def run_cycle(self):
        """Main trading cycle"""
        # Get market data
        df = self.exchange.get_klines()
        if df.empty:
            return
        
        price = float(df["close"].iloc[-1])
        self.strategy.update(df)
        
        # Update position
        self.position = self.exchange.get_position()
        if not self.position:
            self.balance = self.exchange.get_balance()
        
        # Check if order was filled
        open_orders = self.exchange.get_open_orders()
        order_ids = [o["orderId"] for o in open_orders]
        
        if self.active_order and self.active_order not in order_ids:
            # Order was filled or cancelled - log post-fill
            if self.current_tid:
                self.logger.post_fill(
                    self.current_tid, 
                    "entry" if not self.position else "close",
                    {"orderId": self.active_order, "orderStatus": "Filled"},
                    self.position
                )
            self.active_order = None
        
        # Exit logic
        if self.position and not self.active_order:
            reason = self.strategy.should_exit(df, self.position)
            if reason and self.current_tid:
                side = "Sell" if self.position["side"] == "Buy" else "Buy"
                qty = float(self.position["size"])
                offset = self.cfg.maker_offset_bps / 10000
                exit_price = price * (1 - offset if side == "Buy" else 1 + offset)
                
                r = self.exchange.place_order(side, qty, exit_price, reduce_only=True)
                if r.get("retCode") == 0:
                    self.active_order = r["result"]["orderId"]
                    # Pre-log the close intent
                    entry_price = float(self.position["avgPrice"])
                    self.logger.close_trade(
                        self.current_tid,
                        exp_exit=price,
                        act_exit=exit_price,  # Will be updated on fill
                        reason=reason,
                        in_bps=-4,  # Maker fee
                        out_bps=-4
                    )
                    self.current_tid = None
        
        # Entry logic
        if not self.position and not self.active_order and self.new_bar(df):
            if not self.check_limits():
                return
            if time.time() - self.last_trade_time < self.cfg.cooldown_sec:
                return
            
            signal = self.strategy.signal(df)
            if signal:
                # Calculate position size
                stop_loss = signal.get("sl", price * 0.99)
                qty = self.risk.calculate_size(self.balance, price, stop_loss)
                if qty <= 0:
                    return
                
                # Place order with offset
                side = "Buy" if signal["action"] == "BUY" else "Sell"
                offset = self.cfg.maker_offset_bps / 10000
                entry_price = price * (1 - offset if side == "Buy" else 1 + offset)
                
                r = self.exchange.place_order(side, qty, entry_price)
                if r.get("retCode") == 0:
                    self.active_order = r["result"]["orderId"]
                    self.last_trade_time = time.time()
                    self.trades_hour.append(self.last_trade_time)
                    
                    # Log the trade open
                    self.current_tid = self.logger.open_trade(
                        side="L" if side == "Buy" else "S",
                        exp_px=price,
                        act_px=entry_price,
                        qty=qty,
                        sl=stop_loss,
                        tp=signal.get("tp"),
                        bal=self.balance,
                        tags={"strategy": self.strategy.__class__.__name__}
                    )
                    print(f"✅ {side} {qty:.4f} @ {entry_price:.5f} [TID: {self.current_tid}]")
    
    def show_status(self):
        """Display current status"""
        print(f"\n{'='*50}")
        print(f"🤖 {self.cfg.symbol} | TF: {self.cfg.timeframe}m | Session: {self.logger.sess}")
        print(f"💰 Balance: ${self.balance:.2f} | Daily PnL: ${self.logger.pnl_day:.2f}")
        print(f"📊 Streak: {self.logger.streak_loss} losses")
        if self.position:
            side = self.position["side"]
            size = self.position["size"]
            entry = float(self.position["avgPrice"])
            pnl = float(self.position.get("unrealisedPnl", 0))
            print(f"📈 {side} {size} @ {entry:.5f} | PnL: ${pnl:.2f}")
        print(f"{'='*50}")
    
    async def run(self):
        """Main loop"""
        print(f"🚀 Starting {self.cfg.symbol} bot...")
        self.exchange.fetch_specs()
        
        cycle = 0
        while True:
            try:
                await self.run_cycle()
                cycle += 1
                if cycle % 12 == 0:
                    self.show_status()
                await asyncio.sleep(5)
            except KeyboardInterrupt:
                print("\n🛑 Stopped")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                await asyncio.sleep(10)

if __name__ == "__main__":
    asyncio.run(TradingEngine().run())