#2_FEES_EMA_RSI_BNBUSDT.py

import os
import asyncio
import pandas as pd
import json
import time
import math
from datetime import datetime, timezone
from pybit.unified_trading import HTTP
from dotenv import load_dotenv

load_dotenv()

# ---------- helpers ----------
def _round_step(x: float, step: float) -> float:
    if step <= 0:
        return float(x)
    return math.floor(float(x) / step + 1e-12) * step

def _now_iso():
    return datetime.now(timezone.utc).isoformat()


class TradeLogger:
    def __init__(self, bot_name, symbol):
        self.demo_mode = os.getenv('DEMO_MODE', 'true').lower() == 'true'
        self.bot_name = bot_name
        self.symbol = symbol
        self.currency = "USDT"
        self.open_trades = {}
        self.trade_id = 1000

        # Trading limits
        self.daily_pnl = 0.0
        self.consecutive_losses = 0
        self.max_daily_loss = 50.0

        os.makedirs("logs", exist_ok=True)
        self.log_file = f"logs/2_FEES_EMA_RSI_BNBUSDT.log"

    def generate_trade_id(self):
        self.trade_id += 1
        return self.trade_id

    def log_trade_open(self, side, expected_price, actual_price, qty, stop_loss, take_profit, info=""):
        trade_id = self.generate_trade_id()
        slippage = 0.0  # PostOnly = targeting zero

        log_entry = {
            "id": trade_id,
            "bot": self.bot_name,
            "symbol": self.symbol,
            "side": "LONG" if side == "BUY" else "SHORT",
            "action": "OPEN",
            "ts": _now_iso(),
            "expected_price": round(expected_price, 4),
            "actual_price": round(actual_price, 4),
            "slippage": round(slippage, 6),
            "qty": float(qty),
            "stop_loss": round(stop_loss, 4),
            "take_profit": round(take_profit, 4),
            "currency": self.currency,
            "info": info
        }

        self.open_trades[trade_id] = {
            "entry_time": datetime.now(),
            "entry_price": float(actual_price),
            "side": "BUY" if side == "BUY" else "SELL",
            "qty": float(qty),
            "stop_loss": float(stop_loss),
            "take_profit": float(take_profit)
        }

        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        return trade_id, log_entry

    def log_trade_close(self, trade_id, expected_exit, actual_exit, reason,
                        fees_entry_bps=0.01, fees_exit_bps=0.01):
        """fees_*_bps positive = cost, negative = rebate (VIP maker)."""
        if trade_id not in self.open_trades:
            return None

        t = self.open_trades[trade_id]
        duration = (datetime.now() - t["entry_time"]).total_seconds()

        # Side-aware gross PnL (linear)
        if t["side"] == "BUY":
            gross_pnl = (float(actual_exit) - t["entry_price"]) * t["qty"]
        else:
            gross_pnl = (t["entry_price"] - float(actual_exit)) * t["qty"]

        # Maker-only: use maker bps for both legs (we only submit PostOnly)
        entry_fee = t["entry_price"] * t["qty"] * (fees_entry_bps / 100.0)
        exit_fee  = float(actual_exit) * t["qty"] * (fees_exit_bps / 100.0)
        total_fees = entry_fee + exit_fee

        net_pnl = gross_pnl - total_fees

        self.daily_pnl += net_pnl
        self.consecutive_losses = self.consecutive_losses + 1 if net_pnl < 0 else 0

        log_entry = {
            "id": trade_id,
            "bot": self.bot_name,
            "symbol": self.symbol,
            "side": "LONG" if t["side"] == "BUY" else "SHORT",
            "action": "CLOSE",
            "ts": _now_iso(),
            "duration_sec": int(duration),
            "entry_price": round(t["entry_price"], 4),
            "expected_exit": round(expected_exit, 4),
            "actual_exit": round(float(actual_exit), 4),
            "qty": float(t["qty"]),
            "gross_pnl": round(gross_pnl, 2),
            "fees": {
                "entry_bps": fees_entry_bps,
                "exit_bps": fees_exit_bps,
                "entry": round(entry_fee, 2),
                "exit": round(exit_fee, 2),
                "total": round(total_fees, 2)
            },
            "net_pnl": round(net_pnl, 2),
            "reason": reason,
            "currency": self.currency,
            "daily_pnl": round(self.daily_pnl, 2),
            "consecutive_losses": self.consecutive_losses
        }

        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        del self.open_trades[trade_id]
        return log_entry


class EMARSIBot:
    def __init__(self):
        self.symbol = 'BNBUSDT'
        self.demo_mode = os.getenv('DEMO_MODE', 'true').lower() == 'true'

        prefix = 'TESTNET_' if self.demo_mode else 'LIVE_'
        self.api_key = os.getenv(f'{prefix}BYBIT_API_KEY')
        self.api_secret = os.getenv(f'{prefix}BYBIT_API_SECRET')

        self.exchange = None
        self.position = None
        self.price_data = pd.DataFrame()
        self.account_balance = 0.0

        # Strategy + execution
        self.config = {
            'ema_fast': 5,
            'ema_slow': 13,
            'rsi_period': 5,
            'rsi_oversold': 35,
            'rsi_overbought': 65,
            'risk_per_trade': 1.0,      # % of balance
            'maker_offset_pct': 0.02,   # entry limit offset
            'maker_bps': 0.01,          # +cost (0.01%) or rebate if negative
            'net_take_profit': 0.86,    # %
            'net_stop_loss': 0.43,      # %
            'order_timeout': 30,
            'min_notional': 5,
            'limit_order_retries': 3,
            # maker-only mode: ALL orders are Limit + PostOnly (no market)
            'maker_only_mode': True,
        }

        self.tick_size = 0.01
        self.qty_step = 0.01
        self.qty_decimals = 2
        self.price_decimals = 2

        self.last_trade_time = 0
        self.trade_cooldown = 30

        self.logger = TradeLogger("EMA_RSI_FIXED", self.symbol)
        self.current_trade_id = None

        # Track our resting exit orders and soft stop levels when maker-only
        self.exit_orders = {'tp': None, 'sl': None}  # {'tp': {'id':..., 'px':...}, 'sl': {'id':..., 'px':...}}
        self.soft_stop = None  # price

    def connect(self):
        try:
            self.exchange = HTTP(demo=self.demo_mode, api_key=self.api_key, api_secret=self.api_secret)
            return self.exchange.get_server_time().get('retCode') == 0
        except Exception:
            return False

    async def get_instrument_info(self):
        try:
            result = self.exchange.get_instruments_info(category="linear", symbol=self.symbol)
            if result.get('retCode') == 0:
                info = result['result']['list'][0]
                self.tick_size = float(info['priceFilter']['tickSize'])
                self.qty_step = float(info['lotSizeFilter']['qtyStep'])
                # infer decimals for formatting
                self.price_decimals = max(0, str(self.tick_size)[::-1].find('.'))
                self.qty_decimals = max(0, str(self.qty_step)[::-1].find('.'))
                return True
        except Exception:
            pass
        return False

    def format_price(self, price):
        px = _round_step(price, self.tick_size)
        return float(f"{px:.{self.price_decimals}f}")

    def format_qty(self, qty):
        q = _round_step(qty, self.qty_step)
        return float(f"{q:.{self.qty_decimals}f}")

    async def get_account_balance(self):
        try:
            r = self.exchange.get_wallet_balance(accountType="UNIFIED", coin="USDT")
            if r.get('retCode') != 0:
                print(f"❌ Failed to get wallet balance: {r.get('retMsg')}")
                return False
            lst = r.get('result', {}).get('list', [])
            if not lst:
                print("❌ No wallet data returned")
                return False
            for coin in lst[0].get('coin', []):
                if coin.get('coin') == 'USDT':
                    for k in ['availableToWithdraw', 'walletBalance', 'equity', 'availableBalance', 'balance']:
                        v = coin.get(k)
                        try:
                            bal = float(v)
                            if bal >= 0:
                                self.account_balance = bal
                                return True
                        except Exception:
                            continue
            print("❌ USDT not found in wallet")
            return False
        except Exception as e:
            print(f"❌ Balance error: {e}")
            return False

    async def get_market_data(self):
        try:
            kl = self.exchange.get_kline(category="linear", symbol=self.symbol, interval="1", limit=50)
            if kl.get('retCode') != 0:
                return False
            df = pd.DataFrame(kl['result']['list'],
                              columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
            df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
            self.price_data = df.sort_values('timestamp').reset_index(drop=True)
            return True
        except Exception:
            return False

    def calculate_indicators(self, df):
        if len(df) < max(self.config['ema_slow'], self.config['rsi_period']) + 2:
            return None
        close = df['close']
        ema_fast = close.ewm(span=self.config['ema_fast']).mean().iloc[-1]
        ema_slow = close.ewm(span=self.config['ema_slow']).mean().iloc[-1]
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(self.config['rsi_period']).mean()
        loss = (-delta.clip(upper=0)).rolling(self.config['rsi_period']).mean()
        rs = gain / (loss.replace(0, pd.NA))
        rsi = 100 - (100 / (1 + rs))
        if pd.isna(rsi.iloc[-1]):
            rsi_val = 50.0
        else:
            rsi_val = float(rsi.iloc[-1])

        return {
            'trend': 'UP' if ema_fast > ema_slow else 'DOWN',
            'rsi': rsi_val,
            'ema_fast': float(ema_fast),
            'ema_slow': float(ema_slow)
        }

    def generate_signal(self, df):
        if self.position:
            return None
        if time.time() - self.last_trade_time < self.trade_cooldown:
            return None

        ind = self.calculate_indicators(df)
        if not ind:
            return None

        price = float(df['close'].iloc[-1])

        # Primary RSI extremes
        if ind['rsi'] < 30:
            return {'action': 'BUY', 'price': price, 'rsi': ind['rsi']}
        if ind['rsi'] > 70:
            return {'action': 'SELL', 'price': price, 'rsi': ind['rsi']}

        # Secondary: pullback-with-trend
        if ind['trend'] == 'UP' and ind['rsi'] < 45:
            return {'action': 'BUY', 'price': price, 'rsi': ind['rsi']}
        if ind['trend'] == 'DOWN' and ind['rsi'] > 55:
            return {'action': 'SELL', 'price': price, 'rsi': ind['rsi']}

        return None

    def calculate_position_size(self, price, stop_loss_price):
        if self.account_balance <= 0:
            print("❌ No valid account balance for position sizing")
            return 0.0
        risk_amount = self.account_balance * self.config['risk_per_trade'] / 100.0
        stop_distance = abs(float(price) - float(stop_loss_price))
        if stop_distance <= 0:
            return 0.0
        qty = risk_amount / stop_distance
        notional = qty * float(price)
        if notional < self.config['min_notional']:
            return 0.0
        return qty

    async def check_position(self):
        try:
            r = self.exchange.get_positions(category="linear", symbol=self.symbol)
            if r.get('retCode') != 0:
                self.position = None
                return False
            cur = None
            for p in r['result']['list']:
                if float(p.get('size', 0)) > 0:
                    cur = p
                    break
            if cur is None:
                self.position = None
                return False
            self.position = cur
            return True
        except Exception as e:
            print(f"❌ Position check error: {e}")
            self.position = None
            return False

    async def get_open_orders(self):
        """Return all open orders for symbol."""
        try:
            o = self.exchange.get_open_orders(category="linear", symbol=self.symbol)
            if o.get('retCode') != 0:
                return []
            return o.get('result', {}).get('list', [])
        except Exception:
            return []

    async def has_pending_entry_orders(self):
        """Ignore reduce-only (TP/SL) when deciding if an entry is pending."""
        orders = await self.get_open_orders()
        for od in orders:
            # Some fields may be strings
            ro = od.get('reduceOnly')
            if isinstance(ro, str):
                ro = (ro.lower() == 'true')
            if not ro:
                return True
        return False

    async def execute_limit_order(self, side, qty, price, is_reduce=False, rest=False):
        """
        Maker PostOnly limit.
        - If rest=False: wait for fill (poll); cancel if not filled within timeout.
        - If rest=True: just place and return orderId (do NOT cancel).
        """
        q = self.format_qty(qty)
        limit_price = self.format_price(price)

        params = {
            "category": "linear",
            "symbol": self.symbol,
            "side": side,
            "orderType": "Limit",
            "qty": f"{q}",
            "price": f"{limit_price}",
            "timeInForce": "PostOnly"
        }
        if is_reduce:
            params["reduceOnly"] = True

        try:
            res = self.exchange.place_order(**params)
            if res.get('retCode') != 0:
                return None
            oid = res['result']['orderId']
            if rest:
                return oid  # resting maker order left on book

            # else, wait for fill
            t0 = time.time()
            while time.time() - t0 < self.config['order_timeout']:
                await asyncio.sleep(1)
                oo = self.exchange.get_open_orders(category="linear", symbol=self.symbol, orderId=oid)
                if oo['retCode'] == 0 and not oo['result']['list']:
                    return float(limit_price)  # assumed filled at our limit
            # cancel if not filled
            try:
                self.exchange.cancel_order(category="linear", symbol=self.symbol, orderId=oid)
            except Exception:
                pass
            return None
        except Exception:
            return None

    async def place_tp_order(self, is_long, qty, entry_price):
        tp = entry_price * (1 + self.config['net_take_profit']/100.0) if is_long \
             else entry_price * (1 - self.config['net_take_profit']/100.0)
        tp = self.format_price(tp)
        side = "Sell" if is_long else "Buy"
        oid = await self.execute_limit_order(side, qty, tp, is_reduce=True, rest=True)
        if oid:
            self.exit_orders['tp'] = {'id': oid, 'px': tp}
        return tp, oid

    async def place_soft_sl_order(self, is_long, qty, ref_price):
        """
        Maker-only 'soft stop': place a reduce-only PostOnly limit that will fill on a bounce.
        For longs: place a Sell limit slightly ABOVE current price (maker). For shorts: Buy limit slightly BELOW.
        """
        offset = self.config['maker_offset_pct'] / 100.0
        if is_long:
            limit_px = self.format_price(ref_price * (1 + offset))  # above market to remain maker
            side = "Sell"
        else:
            limit_px = self.format_price(ref_price * (1 - offset))  # below market to remain maker
            side = "Buy"
        oid = await self.execute_limit_order(side, qty, limit_px, is_reduce=True, rest=True)
        if oid:
            self.exit_orders['sl'] = {'id': oid, 'px': limit_px}
        return limit_px, oid

    async def cancel_order_safe(self, order_id):
        try:
            self.exchange.cancel_order(category="linear", symbol=self.symbol, orderId=order_id)
        except Exception:
            pass

    async def clear_exit_orders(self):
        for key in ['tp', 'sl']:
            od = self.exit_orders.get(key)
            if od and od.get('id'):
                await self.cancel_order_safe(od['id'])
        self.exit_orders = {'tp': None, 'sl': None}

    def should_close(self):
        """No market exits in maker-only mode; keep baseline emergency rule for non-maker mode only."""
        return False, ""

    async def execute_trade(self, signal):
        await self.check_position()
        if self.position:
            return
        if await self.has_pending_entry_orders():
            return
        if not await self.get_account_balance():
            print("❌ Cannot execute trade - failed to get account balance")
            return
        if self.account_balance <= 0:
            print("❌ Cannot execute trade - account balance is 0 or invalid")
            return

        market_price = float(signal['price'])
        is_buy = (signal['action'] == 'BUY')
        stop_price = market_price * (1 - self.config['net_stop_loss']/100.0) if is_buy \
                    else market_price * (1 + self.config['net_stop_loss']/100.0)
        qty = self.calculate_position_size(market_price, stop_price)
        if qty < self.qty_step:
            print(f"⚠️ Position size too small: {qty}")
            return

        # Entry as maker: offset away from touch
        entry_base = market_price * (1 - self.config['maker_offset_pct']/100.0) if is_buy \
                     else market_price * (1 + self.config['maker_offset_pct']/100.0)
        actual_price = await self.execute_limit_order("Buy" if is_buy else "Sell", qty, entry_base, is_reduce=False, rest=False)
        if actual_price is None:
            return

        self.last_trade_time = time.time()

        take_profit = actual_price * (1 + self.config['net_take_profit']/100.0) if is_buy \
                      else actual_price * (1 - self.config['net_take_profit']/100.0)

        self.current_trade_id, _ = self.logger.log_trade_open(
            side=signal['action'],
            expected_price=market_price,
            actual_price=actual_price,
            qty=self.format_qty(qty),
            stop_loss=stop_price,
            take_profit=take_profit,
            info=f"RSI:{signal['rsi']:.1f}|Bal:{self.account_balance:.2f}"
        )

        # Maker-only exits: place TP now; store soft stop level
        await self.clear_exit_orders()
        tp_px, tp_oid = await self.place_tp_order(is_buy, self.format_qty(qty), actual_price)
        self.soft_stop = self.format_price(stop_price)
        print(f"✅ Entry {signal['action']} @ {actual_price:.{self.price_decimals}f} | TP {tp_px} (PostOnly reduce-only) | Soft SL {self.soft_stop}")

    async def check_and_manage_soft_stop(self):
        """If price breaches soft stop, replace TP with soft-stop exit (maker-only)."""
        if not self.position or self.soft_stop is None:
            return
        last_px = float(self.price_data['close'].iloc[-1]) if len(self.price_data) else None
        if last_px is None:
            return
        is_long = (self.position.get('side') == "Buy")
        breach = (last_px <= self.soft_stop) if is_long else (last_px >= self.soft_stop)
        if not breach:
            return

        # Cancel TP and place soft SL maker exit near current price with maker offset
        if self.exit_orders['tp']:
            await self.cancel_order_safe(self.exit_orders['tp']['id'])
            self.exit_orders['tp'] = None

        qty = float(self.position.get('size', 0))
        if qty <= 0:
            return

        # Place maker reduce-only soft SL near current price
        sl_px, sl_oid = await self.place_soft_sl_order(is_long, self.format_qty(qty), last_px)
        print(f"⚠️ Soft stop breached. Placed maker exit at {sl_px} (reduce-only PostOnly)")

    async def detect_exit_and_log(self):
        """
        Poll for position close (size -> 0). If closed and we have a current_trade_id,
        log the exit with last known price and reason inferred by which order remains.
        """
        had_pos = await self.check_position()
        if had_pos and float(self.position.get('size', 0)) > 0:
            return  # still open
        # Position closed
        if self.current_trade_id:
            px = float(self.price_data['close'].iloc[-1]) if len(self.price_data) else 0.0
            reason = "tp_or_soft_sl_hit"
            self.logger.log_trade_close(
                trade_id=self.current_trade_id,
                expected_exit=px,
                actual_exit=px,
                reason=reason,
                fees_entry_bps=self.config['maker_bps'],
                fees_exit_bps=self.config['maker_bps']
            )
            self.current_trade_id = None
        await self.clear_exit_orders()
        self.soft_stop = None

    def show_status(self):
        if len(self.price_data) == 0:
            return
        px = float(self.price_data['close'].iloc[-1])
        parts = [f"📊 BNB: ${px:.{self.price_decimals}f}", f"💰 Balance: ${self.account_balance:.0f}"]
        if self.position:
            entry = float(self.position.get('avgPrice', 0))
            side = self.position.get('side', '')
            size = float(self.position.get('size', 0))
            parts.append(f"📍 {side}: {size:.{self.qty_decimals}f} @ ${entry:.{self.price_decimals}f}")
        else:
            ind = self.calculate_indicators(self.price_data)
            if ind:
                if ind['trend'] == 'UP' and ind['rsi'] < self.config['rsi_oversold']:
                    sig = "🟢 BUY SIGNAL"
                elif ind['trend'] == 'DOWN' and ind['rsi'] > self.config['rsi_overbought']:
                    sig = "🔴 SELL SIGNAL"
                else:
                    sig = "⚪ NO SIGNAL"
                parts.append(f"RSI:{ind['rsi']:.1f} | Trend:{ind['trend']} | {sig}")
        if self.last_trade_time > 0:
            remain = self.trade_cooldown - (time.time() - self.last_trade_time)
            if remain > 0:
                parts.append(f"⏰ Cooldown:{remain:.0f}s")
        print(" | ".join(parts), end="\r")

    async def run_cycle(self):
        if not await self.get_market_data():
            return

        # Manage maker-only soft stop and detect exits
        await self.check_position()
        await self.check_and_manage_soft_stop()
        await self.detect_exit_and_log()

        if not self.position:
            # Only look for entries if no non-reduce-only pending orders
            if not await self.has_pending_entry_orders():
                sig = self.generate_signal(self.price_data)
                if sig:
                    await self.execute_trade(sig)

        self.show_status()

    async def run(self):
        if not self.connect():
            print("❌ Failed to connect to exchange")
            return
        if not await self.get_instrument_info():
            print("⚠️ Using default instrument info")
        if not await self.get_account_balance():
            print("❌ Failed to get account balance - cannot continue")
            return
        if self.account_balance <= 0:
            print("❌ Account balance is 0 - cannot trade")
            return

        print(f"🔧 EMA + RSI Bot for {self.symbol} (Maker-only mode: all orders are Limit + PostOnly)")
        print(f"💰 Balance: ${self.account_balance:.2f}")
        print(f"🎯 TP {self.config['net_take_profit']:.2f}% | Soft SL {self.config['net_stop_loss']:.2f}%")
        print(f"💸 Maker fee bps: {self.config['maker_bps']}")

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
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                await asyncio.sleep(5)


if __name__ == "__main__":
    bot = EMARSIBot()
    asyncio.run(bot.run())
