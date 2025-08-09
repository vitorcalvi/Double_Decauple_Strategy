#!/usr/bin/env python3
"""
Dynamic Grid Bot — ETHUSDT (Bybit v5, PyBit unified)
----------------------------------------------------
Fixes in this version:
• Robust auth check to stop the 401 spam and explain the root cause
• Graceful "market‑data only" fallback if private endpoints aren't available
• Maker‑only on BOTH entry and exit (timeInForce=PostOnly) for rebates
• Safer quantity formatting (ETH min qty 0.001)
• Clear startup checklist + detailed error messages
• Small hygiene: exponential backoff on transient errors, cleaner logs

Environment variables (for TESTNET by default):
  DEMO_MODE=true
  TESTNET_BYBIT_API_KEY=xxxxx
  TESTNET_BYBIT_API_SECRET=xxxxx
(For mainnet set DEMO_MODE=false and use LIVE_BYBIT_API_KEY / LIVE_BYBIT_API_SECRET)

Common 401 causes you’ll now see surfaced at startup:
  - Wrong key vs environment (live key on testnet or vice‑versa)
  - Key lacks "Contract Trade" permissions
  - Unified account not enabled / wrong account type
  - IP allowlist mismatch (if enabled in Bybit)
"""

import os
import asyncio
import json
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pybit.unified_trading import HTTP

load_dotenv(override=True)

# ------------------------ Utils ------------------------ #

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _round_step(value: float, step: float, ndigits: int) -> float:
    if step <= 0:
        return value
    return round(round(value / step) * step, ndigits)


# --------------------- Trade Logger -------------------- #

class TradeLogger:
    def __init__(self, bot_name: str, symbol: str):
        self.bot_name = bot_name
        self.symbol = symbol
        self.currency = "USDT"
        self.open_trades = {}
        self.trade_id = 1000

        # Emergency stop tracking
        self.daily_pnl = 0.0
        self.consecutive_losses = 0
        self.max_daily_loss = 50.0

        os.makedirs("logs", exist_ok=True)
        self.log_file = "logs/7_FEES_DYNAMIC_GRID_ETHUSDT.log"

    def _write(self, obj: dict) -> None:
        with open(self.log_file, "a") as f:
            f.write(json.dumps(obj) + "\n")

    def generate_trade_id(self) -> int:
        self.trade_id += 1
        return self.trade_id

    def log_trade_open(
        self,
        side: str,  # 'BUY' or 'SELL'
        expected_price: float,
        actual_price: float,
        qty: float,
        stop_loss: float,
        take_profit: float,
        info: str = "",
    ):
        trade_id = self.generate_trade_id()
        slippage = (actual_price - expected_price) if side == "BUY" else (expected_price - actual_price)

        log_entry = {
            "id": trade_id,
            "bot": self.bot_name,
            "symbol": self.symbol,
            "side": "LONG" if side == "BUY" else "SHORT",
            "action": "OPEN",
            "ts": _now_iso(),
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
            "side": side,
            "qty": qty,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
        }

        self._write(log_entry)
        return trade_id, log_entry

    def log_trade_close(
        self,
        trade_id: int,
        expected_exit: float,
        actual_exit: float,
        reason: str,
        fees_entry: float = -0.04,  # maker rebate (%)
        fees_exit: float = -0.04,   # maker rebate (%)
    ):
        if trade_id not in self.open_trades:
            return None

        trade = self.open_trades[trade_id]
        duration = (datetime.now() - trade["entry_time"]).total_seconds()
        slippage = (
            actual_exit - expected_exit if trade["side"] == "SELL" else expected_exit - actual_exit
        )

        # Gross PnL (linear contract sized in coin qty * price delta)
        if trade["side"] == "BUY":
            gross_pnl = (actual_exit - trade["entry_price"]) * trade["qty"]
        else:
            gross_pnl = (trade["entry_price"] - actual_exit) * trade["qty"]

        # Proper maker rebates (negative fee = rebate)
        entry_rebate = trade["entry_price"] * trade["qty"] * abs(fees_entry) / 100
        exit_rebate = actual_exit * trade["qty"] * abs(fees_exit) / 100
        total_rebates = entry_rebate + exit_rebate
        net_pnl = gross_pnl + total_rebates

        log_entry = {
            "id": trade_id,
            "bot": self.bot_name,
            "symbol": self.symbol,
            "side": "LONG" if trade["side"] == "BUY" else "SHORT",
            "action": "CLOSE",
            "ts": _now_iso(),
            "duration_sec": int(duration),
            "entry_price": round(trade["entry_price"], 4),
            "expected_exit": round(expected_exit, 4),
            "actual_exit": round(actual_exit, 4),
            "slippage": round(slippage, 4),
            "qty": round(trade["qty"], 6),
            "gross_pnl": round(gross_pnl, 2),
            "fee_rebates": {
                "entry": round(entry_rebate, 2),
                "exit": round(exit_rebate, 2),
                "total": round(total_rebates, 2),
            },
            "net_pnl": round(net_pnl, 2),
            "reason": reason,
            "currency": self.currency,
        }

        self._write(log_entry)
        del self.open_trades[trade_id]
        return log_entry


# -------------------- Dynamic Grid Bot -------------------- #

class DynamicGridBot:
    def __init__(self):
        self.symbol = "ETHUSDT"
        self.demo_mode = os.getenv("DEMO_MODE", "true").lower() == "true"

        prefix = "TESTNET_" if self.demo_mode else "LIVE_"
        self.api_key = os.getenv(f"{prefix}BYBIT_API_KEY")
        self.api_secret = os.getenv(f"{prefix}BYBIT_API_SECRET")
        self.exchange: HTTP | None = None

        # Private API availability gate (auth ok?)
        self.private_ok = False

        # State
        self.position = None
        self.price_data = pd.DataFrame()
        self.account_balance = 1000.0  # fallback balance in market‑data‑only mode

        # --- Config ---
        self.config = {
            "grid_levels": 10,
            "grid_spacing_pct": 0.5,       # base spacing before volatility scaling
            "risk_per_trade": 2.0,         # % of balance
            "maker_offset_pct": 0.01,      # post‑only offset to stay off the touch
            "maker_fee_pct": -0.04,        # maker rebate (%)
            "net_take_profit": 1.2,        # % move from entry
            "net_stop_loss": 0.6,          # % move from entry
            "atr_period": 14,
            "volatility_threshold": 0.015,
            "slippage_basis_points": 2,   # model 2 bps slippage on fills
            "recv_window": 20000,
            "min_qty": 0.001,             # ETH min order size
            "qty_decimals": 3,
        }

        self.grid_levels = []
        self.current_grid_index = -1
        self.last_update_time = None

        # Trade cooldown
        self.last_trade_time = 0.0
        self.trade_cooldown = 30

        self.logger = TradeLogger("DYNAMIC_GRID_FIXED", self.symbol)
        self.current_trade_id = None

    # ---------------- Connection / Auth ---------------- #

    def connect(self) -> bool:
        # Early checks for env presence
        missing = []
        if not self.api_key:
            missing.append("API key")
        if not self.api_secret:
            missing.append("API secret")
        if missing:
            print(
                "\n🔒 Private endpoints disabled: missing "
                + ", ".join(missing)
                + ". Running in MARKET‑DATA ONLY mode.\n"
                "   Set env vars: DEMO_MODE=true, TESTNET_BYBIT_API_KEY, TESTNET_BYBIT_API_SECRET\n"
                "   Or DEMO_MODE=false with LIVE_* keys.\n"
            )
            # We'll still create a public client so klines work
        try:
            self.exchange = HTTP(
                demo=self.demo_mode,
                api_key=self.api_key,
                api_secret=self.api_secret,
                recv_window=self.config["recv_window"],
            )
            # Basic connectivity check
            ok = self.exchange.get_server_time().get("retCode") == 0
            if not ok:
                print("❌ Could not reach Bybit server time endpoint")
                return False

            # If credentials present, verify a private call to prevent 401 spam later
            if self.api_key and self.api_secret:
                try:
                    res = self.exchange.get_wallet_balance(accountType="UNIFIED")
                    if res.get("retCode") == 0:
                        self.private_ok = True
                        print("🔑 Auth OK: private endpoints enabled")
                    else:
                        print(
                            f"🔒 Private endpoints disabled (retCode={res.get('retCode')}): "
                            f"{res.get('retMsg', 'Unknown error')}\n"
                            "   Checklist: correct env vs testnet/mainnet, Contract permissions, Unified account, IP allowlist."
                        )
                        self.private_ok = False
                except Exception as e:
                    msg = str(e)
                    print(
                        "🔒 Private endpoints disabled due to error:\n"
                        f"   {msg}\n"
                        "   Checklist: correct env vs testnet/mainnet, Contract permissions, Unified account, IP allowlist."
                    )
                    self.private_ok = False

            return True
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False

    # ------------------- Helpers ------------------- #

    def format_qty(self, qty: float) -> str:
        step = self.config["min_qty"]
        if qty < step:
            return "0"
        rounded = _round_step(qty, step, self.config["qty_decimals"])
        return f"{rounded:.{self.config['qty_decimals']}f}"

    async def get_account_balance(self) -> bool:
        if not self.private_ok:
            return False
        try:
            wallet = self.exchange.get_wallet_balance(accountType="UNIFIED")
            if wallet.get("retCode") == 0:
                for coin in wallet["result"]["list"][0]["coin"]:
                    if coin["coin"] == "USDT":
                        self.account_balance = float(coin["availableToWithdraw"])
                        return True
            else:
                print(
                    f"❌ Balance check failed (retCode={wallet.get('retCode')}): {wallet.get('retMsg')}"
                )
        except Exception as e:
            print(f"❌ Balance check error: {e}")
        return False

    def calculate_position_size(self, price: float, stop_loss_price: float) -> float:
        if self.account_balance <= 0:
            return 0.0
        risk_amount = self.account_balance * (self.config["risk_per_trade"] / 100.0)
        stop_distance = abs(price - stop_loss_price)
        if stop_distance == 0:
            return 0.0
        # Cap at 10% of balance notionally
        position_value_usdt = min(risk_amount / stop_distance * price, self.account_balance * 0.10)
        return position_value_usdt / price

    def apply_slippage(self, expected_price: float, side: str) -> float:
        s = self.config["slippage_basis_points"] / 10000.0
        return expected_price * (1 + s) if side.upper() == "BUY" else expected_price * (1 - s)

    def calculate_atr(self, df: pd.DataFrame) -> float | None:
        if len(df) < self.config["atr_period"]:
            return None
        high, low, close = df["high"], df["low"], df["close"]
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=self.config["atr_period"]).mean()
        return None if pd.isna(atr.iloc[-1]) else float(atr.iloc[-1])

    def update_grid_levels(self, current_price: float, atr: float | None) -> None:
        if not atr:
            atr = current_price * 0.01
        volatility_factor = min(max(atr / current_price, 0.005), 0.03)
        adjusted_spacing = (self.config["grid_spacing_pct"] / 100.0) * (1 + volatility_factor * 10)

        levels = []
        for i in range(-self.config["grid_levels"], self.config["grid_levels"] + 1):
            if i == 0:
                continue
            level_price = current_price * (1 + i * adjusted_spacing)
            levels.append({
                "price": level_price,
                "index": i,
                "side": "BUY" if i < 0 else "SELL",
            })
        self.grid_levels = sorted(levels, key=lambda x: x["price"])
        self.last_update_time = datetime.now()

    def find_nearest_grid(self, current_price: float):
        if not self.grid_levels:
            return None
        for i, level in enumerate(self.grid_levels):
            if abs(current_price - level["price"]) / level["price"] < 0.002:  # within 0.2%
                return i, level
        return None

    def generate_signal(self, df: pd.DataFrame):
        if len(df) < 20:
            return None
        current_price = float(df["close"].iloc[-1])
        atr = self.calculate_atr(df)

        if not self.grid_levels or not self.last_update_time:
            self.update_grid_levels(current_price, atr)
            return None

        if (datetime.now() - self.last_update_time).total_seconds() > 300:
            self.update_grid_levels(current_price, atr)

        match = self.find_nearest_grid(current_price)
        if not match:
            return None
        grid_index, grid_level = match
        if grid_index == self.current_grid_index:
            return None

        ema_short = df["close"].ewm(span=9).mean().iloc[-1]
        trend = "UP" if current_price > float(ema_short) else "DOWN"

        if grid_level["side"] == "BUY" and trend == "UP":
            self.current_grid_index = grid_index
            return {
                "action": "BUY",
                "price": current_price,
                "grid_level": grid_level["price"],
                "grid_index": grid_level["index"],
            }
        if grid_level["side"] == "SELL" and trend == "DOWN":
            self.current_grid_index = grid_index
            return {
                "action": "SELL",
                "price": current_price,
                "grid_level": grid_level["price"],
                "grid_index": grid_level["index"],
            }
        return None

    # ------------------- API Calls ------------------- #

    async def get_market_data(self) -> bool:
        try:
            klines = self.exchange.get_kline(
                category="linear", symbol=self.symbol, interval="15", limit=50
            )
            if klines.get("retCode") != 0:
                return False
            df = pd.DataFrame(
                klines["result"]["list"],
                columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"],
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col])
            self.price_data = df.sort_values("timestamp").reset_index(drop=True)
            return True
        except Exception as e:
            print(f"❌ Market data error: {e}")
            return False

    async def check_position(self):
        """Enhanced position check with state cleanup"""
        try:
            positions = self.exchange.get_positions(category="linear", symbol=self.symbol)
            if positions.get('retCode') != 0:
                self.position = None
                return False
                
            pos_list = positions['result']['list']
            
            # Clear position if empty
            if not pos_list or float(pos_list[0].get('size', 0)) == 0:
                if self.position:  # Was set but now closed
                    print("✅ Position closed - clearing state")
                    self.position = None
                    self.pending_order = None  # Also clear pending
                return False
                
            # Valid position exists
            self.position = pos_list[0]
            return True
            
        except Exception as e:
            print(f"❌ Position check error: {e}")
            self.position = None
            self.pending_order = None
            return False


    def should_close(self) -> tuple[bool, str]:
        if not self.position or len(self.price_data) == 0:
            return False, ""
        current_price = float(self.price_data["close"].iloc[-1])
        entry_price = float(self.position.get("avgPrice", 0) or 0)
        side = self.position.get("side", "")
        if entry_price <= 0:
            return False, ""
        profit_pct = (
            (current_price - entry_price) / entry_price * 100 if side == "Buy" else (entry_price - current_price) / entry_price * 100
        )
        if profit_pct >= self.config["net_take_profit"]:
            return True, "grid_target_reached"
        if profit_pct <= -self.config["net_stop_loss"]:
            return True, "stop_loss"
        return False, ""

    async def execute_trade(self, signal: dict) -> None:
        if not self.private_ok:
            print("🔒 Private endpoints unavailable — skip trade")
            return

        # Cooldown
        dt = time.time() - self.last_trade_time
        if dt < self.trade_cooldown:
            print(f"⏰ Trade cooldown: wait {self.trade_cooldown - dt:.0f}s")
            return

        if not await self.get_account_balance():
            print("❌ Could not get account balance — skip trade")
            return
        if self.account_balance < 10:
            print(f"❌ Insufficient balance: ${self.account_balance:.2f}")
            return

        stop_loss_pct = self.config["net_stop_loss"] / 100.0
        stop_loss_price = (
            signal["price"] * (1 - stop_loss_pct) if signal["action"] == "BUY" else signal["price"] * (1 + stop_loss_pct)
        )

        qty = self.calculate_position_size(signal["price"], stop_loss_price)
        formatted_qty = self.format_qty(qty)
        if formatted_qty == "0":
            print(f"❌ Position size too small: {qty:.6f}")
            return

        # Maker‑only limit with small offset from touch
        offset_mult = (
            1 - self.config["maker_offset_pct"] / 100.0 if signal["action"] == "BUY" else 1 + self.config["maker_offset_pct"] / 100.0
        )
        limit_price = round(signal["price"] * offset_mult, 2)
        expected_exec = self.apply_slippage(limit_price, signal["action"])  # model slippage

        try:
            order = self.exchange.place_order(
                category="linear",
                symbol=self.symbol,
                side="Buy" if signal["action"] == "BUY" else "Sell",
                orderType="Limit",
                qty=formatted_qty,
                price=str(limit_price),
                timeInForce="PostOnly",   # <— ensure maker rebate
                reduceOnly=False,
                positionIdx=0,            # one‑way mode
            )
            if order.get("retCode") == 0:
                self.last_trade_time = time.time()

                net_tp = (
                    expected_exec * (1 + self.config["net_take_profit"] / 100.0)
                    if signal["action"] == "BUY"
                    else expected_exec * (1 - self.config["net_take_profit"] / 100.0)
                )
                net_sl = (
                    expected_exec * (1 - self.config["net_stop_loss"] / 100.0)
                    if signal["action"] == "BUY"
                    else expected_exec * (1 + self.config["net_stop_loss"] / 100.0)
                )

                position_value = float(formatted_qty) * expected_exec
                risk_pct = (position_value * (self.config["net_stop_loss"] / 100.0) / self.account_balance) * 100.0

                self.current_trade_id, _ = self.logger.log_trade_open(
                    side=signal["action"],
                    expected_price=limit_price,
                    actual_price=expected_exec,
                    qty=float(formatted_qty),
                    stop_loss=net_sl,
                    take_profit=net_tp,
                    info=(
                        f"grid_level:{signal['grid_level']:.2f}_"
                        f"risk:{risk_pct:.1f}%_bal:{self.account_balance:.2f}"
                    ),
                )

                print(f"📊 GRID {signal['action']}: {formatted_qty} @ ${limit_price:.2f}")
                print(f"   💰 Position Value: ${position_value:.2f} ({risk_pct:.1f}% of balance)")
                print(f"   🎯 Expected Execution: ${expected_exec:.2f} (with slippage)")
                print(f"   💎 TP: ${net_tp:.2f} | SL: ${net_sl:.2f}")
            else:
                print(
                    f"❌ Order rejected (retCode={order.get('retCode')}): {order.get('retMsg')}"
                )
        except Exception as e:
            print(f"❌ Trade failed: {e}")

    async def close_position(self, reason: str) -> None:
        if not (self.private_ok and self.position):
            return
        current_price = float(self.price_data["close"].iloc[-1])
        side = "Sell" if self.position.get("side") == "Buy" else "Buy"
        qty = float(self.position.get("size", 0))
        if qty <= 0:
            return

        offset_mult = 1 + self.config["maker_offset_pct"] / 100.0 if side == "Sell" else 1 - self.config["maker_offset_pct"] / 100.0
        limit_price = round(current_price * offset_mult, 2)
        expected_exit = self.apply_slippage(limit_price, side)

        try:
            order = self.exchange.place_order(
                category="linear",
                symbol=self.symbol,
                side=side,
                orderType="Limit",
                price=str(limit_price),
                qty=f"{qty:.3f}",
                timeInForce="PostOnly",   # <— ensure maker rebate on exit
                reduceOnly=True,
                positionIdx=0,
            )
            if order.get("retCode") == 0:
                if self.current_trade_id:
                    self.logger.log_trade_close(
                        trade_id=self.current_trade_id,
                        expected_exit=limit_price,
                        actual_exit=expected_exit,
                        reason=reason,
                        fees_entry=self.config["maker_fee_pct"],
                        fees_exit=self.config["maker_fee_pct"],
                    )
                    self.current_trade_id = None
                print(f"✅ Closed: {reason} @ ${expected_exit:.2f}")
                self.position = None
            else:
                print(
                    f"❌ Close rejected (retCode={order.get('retCode')}): {order.get('retMsg')}"
                )
        except Exception as e:
            print(f"❌ Close failed: {e}")

    # ------------------- Console UI ------------------- #

    def show_status(self) -> None:
        if len(self.price_data) == 0:
            return
        current_price = float(self.price_data["close"].iloc[-1])

        print(f"\n📊 FIXED Dynamic Grid Bot - {self.symbol}")
        print(f"💰 Price: ${current_price:.2f} | Balance: ${self.account_balance:.2f}")
        print("🔧 FIXES APPLIED:")
        print(f"   • Position Sizing: Risk-based ({self.config['risk_per_trade']}% per trade)")
        print("   • Fee Calculations: Proper maker rebates")
        print(f"   • Slippage Modeling: {self.config['slippage_basis_points']} basis points")
        if not self.private_ok:
            print("   • Mode: MARKET‑DATA ONLY (fix auth to trade)")

        if self.grid_levels:
            buy_grids = [g for g in self.grid_levels if g["side"] == "BUY"]
            sell_grids = [g for g in self.grid_levels if g["side"] == "SELL"]
            if buy_grids:
                print(f"🟢 Next Buy Grid: ${buy_grids[-1]['price']:.2f}")
            if sell_grids:
                print(f"🔴 Next Sell Grid: ${sell_grids[0]['price']:.2f}")

        if self.private_ok and self.position:
            entry_price = float(self.position.get("avgPrice", 0) or 0)
            side = self.position.get("side", "")
            size = float(self.position.get("size", 0) or 0)
            pnl = float(self.position.get("unrealisedPnl", 0) or 0)
            position_value = size * current_price
            risk_pct = (position_value / self.account_balance) * 100 if self.account_balance > 0 else 0
            emoji = "🟢" if side == "Buy" else "🔴"
            print(f"{emoji} {side}: {size:.3f} ETH @ ${entry_price:.2f} | PnL: ${pnl:.2f}")
            print(f"   📊 Position: ${position_value:.2f} ({risk_pct:.1f}% of balance)")
        else:
            print("⚡ Waiting for optimal grid signals...")
        print("-" * 50)

    # ------------------- Main Loop ------------------- #

    async def run_cycle(self) -> None:
        

        if not await self.get_market_data():
            return

        # Private branch
        if self.private_ok:
            await self.check_position()
            if self.position:
                should_close, reason = self.should_close()
                if should_close:
                    await self.close_position(reason)
            else:
                signal = self.generate_signal(self.price_data)
                if signal:
                    await self.execute_trade(signal)
        else:
            # Market‑data only: still compute grids/signals to show UI hints
            signal = self.generate_signal(self.price_data)
            if signal:
                pass  # no trading without auth

        self.show_status()

    async def run(self) -> None:
        if not self.connect():
            print("❌ Failed to connect")
            return

        print(f"📊 FIXED Dynamic Grid Trading Bot - {self.symbol}")
        print("🔧 CRITICAL FIXES:")
        print(
            f"   ✅ Position Sizing: balance‑based with {self.config['risk_per_trade']}% risk per trade"
        )
        print("   ✅ Fee Calculations: proper maker rebate handling")
        print(
            f"   ✅ Slippage Modeling: {self.config['slippage_basis_points']} bps expected slippage"
        )
        print("   ✅ Instrument Precision: ETH 0.001 minimum quantity")
        print(
            f"💎 Using MAKER‑ONLY orders (PostOnly) for {abs(self.config['maker_fee_pct'])}% fee rebate"
        )
        if not self.private_ok:
            print(
                "⚠️ Trading disabled until auth is fixed. See checklist above. "
                "Meanwhile the bot will keep showing live grids and market data."
            )

        try:
            backoff = 2
            while True:
                try:
                    await self.run_cycle()
                    await asyncio.sleep(2)
                    backoff = 2  # reset after a clean cycle
                except Exception as e:
                    print(f"⚠️ Runtime error (will retry): {e}")
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 30)
        except KeyboardInterrupt:
            print("\n🛑 Bot stopped")
            if self.private_ok and self.position:
                await self.close_position("manual_stop")


if __name__ == "__main__":
    bot = DynamicGridBot()
    asyncio.run(bot.run())
