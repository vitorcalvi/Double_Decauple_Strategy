#!/usr/bin/env python3
"""
Streamlined Dynamic Grid Bot — ETHUSDT (Bybit v5)
Clean, simplified version with identical functionality
"""

import os
import asyncio
import json
import math
import time
from datetime import datetime, timezone
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pybit.unified_trading import HTTP

load_dotenv(override=True)


# ==================== UTILITIES ====================

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def round_to_step(value: float, step: float, decimals: int) -> float:
    if step <= 0:
        return value
    return round(round(value / step) * step, decimals)


# ==================== TRADE LOGGER ====================

class TradeLogger:
    def __init__(self, bot_name: str, symbol: str):
        self.bot_name = bot_name
        self.symbol = symbol
        self.currency = "USDT"
        self.open_trades = {}
        self.trade_id = 1000
        self.daily_pnl = 0.0
        self.consecutive_losses = 0
        self.max_daily_loss = 50.0
        
        os.makedirs("logs", exist_ok=True)
        self.log_file = f"logs/{bot_name}_{symbol}.log"

    def _write_log(self, entry: Dict[str, Any]) -> None:
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def _next_trade_id(self) -> int:
        self.trade_id += 1
        return self.trade_id

    def log_trade_open(self, side: str, expected_price: float, actual_price: float,
                      qty: float, stop_loss: float, take_profit: float, info: str = ""):
        trade_id = self._next_trade_id()
        slippage = (actual_price - expected_price) if side == "BUY" else (expected_price - actual_price)

        log_entry = {
            "id": trade_id,
            "bot": self.bot_name,
            "symbol": self.symbol,
            "side": "LONG" if side == "BUY" else "SHORT",
            "action": "OPEN",
            "ts": now_iso(),
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
        }

        self._write_log(log_entry)
        return trade_id, log_entry

    def log_trade_close(self, trade_id: int, expected_exit: float, actual_exit: float,
                       reason: str, fees_entry: float = -0.04, fees_exit: float = -0.04):
        if trade_id not in self.open_trades:
            return None

        trade = self.open_trades[trade_id]
        duration = (datetime.now() - trade["entry_time"]).total_seconds()
        slippage = (actual_exit - expected_exit) if trade["side"] == "SELL" else (expected_exit - actual_exit)

        # Calculate PnL and fees
        gross_pnl = ((actual_exit - trade["entry_price"]) if trade["side"] == "BUY" 
                     else (trade["entry_price"] - actual_exit)) * trade["qty"]
        
        entry_rebate = trade["entry_price"] * trade["qty"] * abs(fees_entry) / 100
        exit_rebate = actual_exit * trade["qty"] * abs(fees_exit) / 100
        total_rebates = entry_rebate + exit_rebate
        net_pnl = gross_pnl + total_rebates

        self.daily_pnl += net_pnl
        self.consecutive_losses = self.consecutive_losses + 1 if net_pnl < 0 else 0

        log_entry = {
            "id": trade_id,
            "bot": self.bot_name,
            "symbol": self.symbol,
            "side": "LONG" if trade["side"] == "BUY" else "SHORT",
            "action": "CLOSE",
            "ts": now_iso(),
            "duration_sec": int(duration),
            "entry_price": round(trade["entry_price"], 4),
            "expected_exit": round(expected_exit, 4),
            "actual_exit": round(actual_exit, 4),
            "slippage": round(slippage, 4),
            "qty": round(trade["qty"], 6),
            "gross_pnl": round(gross_pnl, 2),
            "fee_rebates": {"entry": round(entry_rebate, 2), "exit": round(exit_rebate, 2), 
                           "total": round(total_rebates, 2)},
            "net_pnl": round(net_pnl, 2),
            "reason": reason,
            "currency": self.currency,
        }

        self._write_log(log_entry)
        del self.open_trades[trade_id]
        return log_entry


# ==================== DYNAMIC GRID BOT ====================

class DynamicGridBot:
    def __init__(self):
        # Basic setup
        self.symbol = "ETHUSDT"
        self.demo_mode = os.getenv("DEMO_MODE", "true").lower() == "true"
        
        # API credentials
        prefix = "TESTNET_" if self.demo_mode else "LIVE_"
        self.api_key = os.getenv(f"{prefix}BYBIT_API_KEY")
        self.api_secret = os.getenv(f"{prefix}BYBIT_API_SECRET")
        
        # Exchange and auth
        self.exchange: Optional[HTTP] = None
        self.private_ok = False
        
        # Trading state
        self.position = None
        self.price_data = pd.DataFrame()
        self.account_balance = 1000.0
        self.current_trade_id = None
        self.last_trade_time = 0.0
        
        # Grid state
        self.grid_levels = []
        self.current_grid_index = -1
        self.last_update_time = None
        
        # Configuration
        self.config = {
            "grid_levels": 10,
            "grid_spacing_pct": 0.5,
            "risk_per_trade": 2.0,
            "maker_offset_pct": 0.01,
            "maker_fee_pct": -0.04,
            "net_take_profit": 1.2,
            "net_stop_loss": 0.6,
            "atr_period": 14,
            "volatility_threshold": 0.015,
            "slippage_basis_points": 2,
            "min_qty": 0.001,
            "qty_decimals": 3,
            "trade_cooldown": 30,
        }
        
        self.logger = TradeLogger("DYNAMIC_GRID_FIXED", self.symbol)

    # ==================== CONNECTION & AUTH ====================

    def connect(self) -> bool:
        if not self._validate_credentials():
            return False
            
        try:
            self.exchange = HTTP(
                demo=self.demo_mode,
                api_key=self.api_key,
                api_secret=self.api_secret,
                recv_window=20000,
            )
            
            # Test connectivity
            if self.exchange.get_server_time().get("retCode") != 0:
                print("❌ Could not reach Bybit server")
                return False
                
            print(f"✅ Connected to {'Testnet' if self.demo_mode else 'Live'} Bybit")
            
            # Test authentication
            self.private_ok = self._test_authentication()
            return True
            
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False

    def _validate_credentials(self) -> bool:
        if not self.api_key or not self.api_secret:
            print("❌ API credentials missing")
            prefix = "TESTNET_" if self.demo_mode else "LIVE_"
            print(f"   Required: {prefix}BYBIT_API_KEY")
            print(f"   Required: {prefix}BYBIT_API_SECRET")
            return False
        return True

    def _test_authentication(self) -> bool:
        try:
            wallet = self.exchange.get_wallet_balance(accountType="UNIFIED")
            
            if wallet.get('retCode') == 0:
                print("🔑 Auth OK: private endpoints enabled")
                print("✅ Trading ENABLED")
                return True
            elif wallet.get('retCode') == 401:
                print("❌ API Authentication failed!")
                print("   1. Check API key and secret in .env")
                print("   2. Verify testnet vs mainnet keys")
                print("   3. Enable Contract Trade permissions")
                return False
            else:
                print(f"🔒 Auth error (retCode={wallet.get('retCode')}): {wallet.get('retMsg')}")
                return False
                
        except Exception as e:
            print(f"❌ Authentication test failed: {e}")
            return False

    # ==================== ACCOUNT & BALANCE ====================

    async def get_account_balance(self) -> bool:
        if not self.private_ok:
            self.account_balance = 1000.0
            return True
            
        try:
            wallet = self.exchange.get_wallet_balance(accountType="UNIFIED", coin="USDT")
            
            if wallet.get("retCode") == 0:
                balance_list = wallet.get("result", {}).get("list", [])
                if balance_list:
                    for coin_data in balance_list[0].get("coin", []):
                        if coin_data.get("coin") == "USDT":
                            balance_fields = ["availableToWithdraw", "walletBalance", "equity"]
                            
                            for field in balance_fields:
                                balance_str = coin_data.get(field, "")
                                if balance_str and str(balance_str).strip() not in ["", "0", "None"]:
                                    try:
                                        balance = float(balance_str)
                                        if balance > 0:
                                            self.account_balance = balance
                                            return True
                                    except (ValueError, TypeError):
                                        continue
                            break
                            
        except Exception as e:
            print(f"❌ Balance check error: {e}")
            
        # Fallback for testnet
        self.account_balance = 1000.0
        return True

    async def check_position(self) -> bool:
        if not self.private_ok:
            return False
            
        try:
            positions = self.exchange.get_positions(category="linear", symbol=self.symbol)
            
            if positions.get("retCode") == 0:
                pos_list = positions["result"]["list"]
                for pos in pos_list:
                    if pos.get("symbol") == self.symbol and float(pos.get("size", 0)) > 0:
                        self.position = pos
                        return True
                self.position = None
                return False
            else:
                print(f"❌ Position check failed: {positions.get('retMsg')}")
                self.position = None
                return False
                
        except Exception as e:
            print(f"❌ Position check error: {e}")
            self.position = None
            return False

    # ==================== MARKET DATA ====================

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
                df[col] = pd.to_numeric(df[col], errors="coerce")
                
            self.price_data = df.sort_values("timestamp").reset_index(drop=True)
            return len(self.price_data) >= 20
            
        except Exception:
            return False

    # ==================== GRID CALCULATIONS ====================

    def calculate_atr(self, df: pd.DataFrame) -> Optional[float]:
        if len(df) < self.config["atr_period"]:
            return None
            
        high, low, close = df["high"], df["low"], df["close"]
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=self.config["atr_period"]).mean()
        
        return float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else None

    def update_grid_levels(self, current_price: float, atr: Optional[float]) -> None:
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
            if abs(current_price - level["price"]) / level["price"] < 0.005:
                return i, level
        return None

    def generate_signal(self, df: pd.DataFrame):
        if len(df) < 20:
            return None
            
        current_price = float(df["close"].iloc[-1])
        atr = self.calculate_atr(df)

        # Update grids if needed
        if (not self.grid_levels or not self.last_update_time or 
            (datetime.now() - self.last_update_time).total_seconds() > 300):
            self.update_grid_levels(current_price, atr)
            return None

        # Find matching grid
        match = self.find_nearest_grid(current_price)
        if not match or match[0] == self.current_grid_index:
            return None
            
        grid_index, grid_level = match
        
        # Check trend alignment
        ema_short = df["close"].ewm(span=9).mean().iloc[-1]
        trend = "UP" if current_price > float(ema_short) else "DOWN"

        # Generate signal based on grid side and trend
        if ((grid_level["side"] == "BUY" and trend == "UP") or 
            (grid_level["side"] == "SELL" and trend == "DOWN")):
            self.current_grid_index = grid_index
            return {
                "action": grid_level["side"],
                "price": current_price,
                "grid_level": grid_level["price"],
                "grid_index": grid_level["index"],
            }
        return None

    # ==================== TRADING LOGIC ====================

    def format_qty(self, qty: float) -> str:
        if qty < self.config["min_qty"]:
            return "0"
        rounded = round_to_step(qty, self.config["min_qty"], self.config["qty_decimals"])
        return f"{rounded:.{self.config['qty_decimals']}f}"

    def calculate_position_size(self, price: float, stop_loss_price: float) -> float:
        if self.account_balance <= 0:
            return 0.0
            
        risk_amount = self.account_balance * (self.config["risk_per_trade"] / 100.0)
        price_diff = abs(price - stop_loss_price)
        
        if price_diff == 0:
            return self.config["min_qty"]
            
        # Calculate raw quantity and limit position value
        raw_qty = risk_amount / price_diff
        max_position_value = min(self.account_balance * 0.05, 1000.0)
        max_qty_by_value = max_position_value / price
        
        return max(min(raw_qty, max_qty_by_value), self.config["min_qty"])

    def apply_slippage(self, price: float, side: str) -> float:
        slippage = self.config["slippage_basis_points"] / 10000.0
        return price * (1 + slippage) if side == "BUY" else price * (1 - slippage)

    async def execute_trade(self, signal: Dict[str, Any]) -> None:
        if not self.private_ok:
            print("🔒 Trading disabled - authentication required")
            return

        # Check cooldown
        if time.time() - self.last_trade_time < self.config["trade_cooldown"]:
            return

        # Don't open new positions if we already have one (grid strategy)
        if self.position:
            return

        if not await self.get_account_balance() or self.account_balance < 10:
            return

        # Calculate position sizing
        stop_loss_pct = self.config["net_stop_loss"] / 100.0
        stop_loss_price = (signal["price"] * (1 - stop_loss_pct) if signal["action"] == "BUY"
                          else signal["price"] * (1 + stop_loss_pct))

        qty = self.calculate_position_size(signal["price"], stop_loss_price)
        formatted_qty = self.format_qty(qty)
        
        if formatted_qty == "0":
            return

        # Validate order value
        order_value = float(formatted_qty) * signal["price"]
        if order_value < 5.0 or order_value > 10000.0:
            return

        # Calculate limit price
        offset_mult = (1 - self.config["maker_offset_pct"] / 100.0 if signal["action"] == "BUY"
                      else 1 + self.config["maker_offset_pct"] / 100.0)
        limit_price = round(signal["price"] * offset_mult, 2)
        expected_exec = self.apply_slippage(limit_price, signal["action"])

        try:
            order = self.exchange.place_order(
                category="linear",
                symbol=self.symbol,
                side="Buy" if signal["action"] == "BUY" else "Sell",
                orderType="Limit",
                qty=formatted_qty,
                price=str(limit_price),
                timeInForce="PostOnly",
                reduceOnly=False,
                positionIdx=0,
            )
            
            if order.get("retCode") == 0:
                self.last_trade_time = time.time()
                
                # Calculate TP/SL - FIXED
                if signal["action"] == "BUY":
                    net_tp = expected_exec * (1 + self.config["net_take_profit"] / 100.0)
                    net_sl = expected_exec * (1 - self.config["net_stop_loss"] / 100.0)
                else:  # SELL
                    net_tp = expected_exec * (1 - self.config["net_take_profit"] / 100.0)
                    net_sl = expected_exec * (1 + self.config["net_stop_loss"] / 100.0)

                self.current_trade_id, _ = self.logger.log_trade_open(
                    side=signal["action"],
                    expected_price=limit_price,
                    actual_price=expected_exec,
                    qty=float(formatted_qty),
                    stop_loss=net_sl,
                    take_profit=net_tp,
                    info=f"grid_level:{signal['grid_level']:.2f}_bal:{self.account_balance:.2f}"
                )

                print(f"✅ GRID {signal['action']}: {formatted_qty} ETH @ ${limit_price:.2f}")
            else:
                print(f"❌ Order rejected: {order.get('retMsg')}")
                
        except Exception as e:
            print(f"❌ Trade failed: {e}")

    def should_close(self):
        if not self.position:
            return False, ""
            
        current_price = float(self.price_data["close"].iloc[-1])
        entry_price = float(self.position.get("avgPrice", 0))
        side = self.position.get("side", "")
        
        if not entry_price:
            return False, ""
            
        profit_pct = ((current_price - entry_price) / entry_price * 100 if side == "Buy"
                     else (entry_price - current_price) / entry_price * 100)

        if profit_pct >= self.config["net_take_profit"]:
            return True, "take_profit"
        if profit_pct <= -self.config["net_stop_loss"]:
            return True, "stop_loss"
            
        return False, ""

    async def close_position(self, reason: str) -> None:
        if not (self.private_ok and self.position):
            return
            
        current_price = float(self.price_data["close"].iloc[-1])
        side = "Sell" if self.position.get("side") == "Buy" else "Buy"
        qty = float(self.position.get("size", 0))
        
        if qty <= 0:
            return

        offset_mult = (1 + self.config["maker_offset_pct"] / 100.0 if side == "Sell"
                      else 1 - self.config["maker_offset_pct"] / 100.0)
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
                timeInForce="PostOnly",
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
                print(f"❌ Close rejected: {order.get('retMsg')}")
                
        except Exception as e:
            print(f"❌ Close failed: {e}")

    # ==================== STATUS & MAIN LOOP ====================

    def show_status(self) -> None:
        if len(self.price_data) == 0:
            return
            
        current_price = float(self.price_data["close"].iloc[-1])
        
        print(f"\n📊 Dynamic Grid Bot - {self.symbol}")
        print(f"💰 Price: ${current_price:.2f} | Balance: ${self.account_balance:.2f}")
        
        if not self.private_ok:
            print("🔒 Market-data only mode - fix auth to enable trading")
        
        # Show grid levels
        if self.grid_levels:
            buy_grids = [g for g in self.grid_levels if g["side"] == "BUY"]
            sell_grids = [g for g in self.grid_levels if g["side"] == "SELL"]
            if buy_grids:
                print(f"🟢 Next Buy Grid: ${buy_grids[-1]['price']:.2f}")
            if sell_grids:
                print(f"🔴 Next Sell Grid: ${sell_grids[0]['price']:.2f}")

        # Show position
        if self.private_ok and self.position:
            entry_price = float(self.position.get("avgPrice", 0) or 0)
            side = self.position.get("side", "")
            size = float(self.position.get("size", 0) or 0)
            pnl = float(self.position.get("unrealisedPnl", 0) or 0)
            
            emoji = "🟢" if side == "Buy" else "🔴"
            print(f"{emoji} {side}: {size:.3f} ETH @ ${entry_price:.2f} | PnL: ${pnl:.2f}")
        elif not self.position and self.private_ok:
            print("⚡ Waiting for optimal grid signals...")
            
        print("-" * 50)

    async def run_cycle(self) -> None:
        # Emergency stop
        if self.logger.daily_pnl < -self.logger.max_daily_loss:
            print(f"🔴 EMERGENCY STOP: Daily loss ${abs(self.logger.daily_pnl):.2f}")
            if self.private_ok and self.position:
                await self.close_position("emergency_stop")
            return

        if not await self.get_market_data():
            return

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
            # Show what would trade in market-data mode
            signal = self.generate_signal(self.price_data)
            if signal:
                print(f"🔒 Would trade {signal['action']} @ ${signal['price']:.2f} - Fix auth to enable")

        self.show_status()

    async def run(self) -> None:
        if not self.connect():
            print("❌ Failed to connect")
            return

        print(f"📊 Dynamic Grid Trading Bot - {self.symbol}")
        print("🔧 Features:")
        print(f"   ✅ Risk-based position sizing ({self.config['risk_per_trade']}% per trade)")
        print("   ✅ Maker-only orders for rebates")
        print(f"   ✅ Grid spacing: {self.config['grid_spacing_pct']}%")
        print(f"   ✅ Trade cooldown: {self.config['trade_cooldown']}s")
        
        if not self.private_ok:
            print("⚠️ Trading disabled - market data only mode")

        try:
            while True:
                await self.run_cycle()
                await asyncio.sleep(5)
        except KeyboardInterrupt:
            print("\n🛑 Bot stopped by user")
            if self.position:
                await self.close_position("manual_stop")


# ==================== MAIN ====================

if __name__ == "__main__":
    bot = DynamicGridBot()
    asyncio.run(bot.run())