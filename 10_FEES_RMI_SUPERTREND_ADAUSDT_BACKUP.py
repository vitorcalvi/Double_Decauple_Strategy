import os
import asyncio
import pandas as pd
import numpy as np
import json
import time
from datetime import datetime, timezone
from pybit.unified_trading import HTTP
from dotenv import load_dotenv

load_dotenv()

class TradeLogger:
    def __init__(self, bot_name, symbol):
        self.LIVE_TRADING = False  # Enable actual trading
        self.account_balance = 1000.0  # Default balance
        self.pending_order = False
        self.last_trade_time = 0
        self.trade_cooldown = 30  # 30 seconds between trades
        self.bot_name = bot_name
        self.symbol = symbol
        self.currency = "USDT"
        self.open_trades = {}
        self.trade_id = 1000
        
        os.makedirs("logs", exist_ok=True)
        self.log_file = f"logs/10_FEES_RMI_SUPERTREND_ADAUSDT.log"
        
    def generate_trade_id(self):
        self.trade_id += 1
        return self.trade_id
    
    def log_trade_open(self, side, expected_price, actual_price, qty, stop_loss, take_profit, info=""):
        trade_id = self.generate_trade_id()
        slippage = 0  # PostOnly = zero slippage
        
        log_entry = {
            "id": trade_id,
            "bot": self.bot_name,
            "symbol": self.symbol,
            "side": "LONG" if side == "BUY" else "SHORT",
            "action": "OPEN",
            "ts": datetime.now(timezone.utc).isoformat(),
            "expected_price": round(expected_price, 6),
            "actual_price": round(actual_price, 6),
            "slippage": round(slippage, 6),
            "qty": round(qty, 6),
            "stop_loss": round(stop_loss, 6),
            "take_profit": round(take_profit, 6),
            "currency": self.currency,
            "info": info
        }
        
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        self.open_trades[trade_id] = log_entry
        return trade_id, log_entry

class RMISupertrendBot:
    def __init__(self):
        self.LIVE_TRADING = False  # Enable actual trading
        self.account_balance = 1000.0  # Default balance
        self.pending_order = False
        self.last_trade_time = 0
        self.trade_cooldown = 30  # 30 seconds between trades
        self.config = {
            'symbol': 'ADAUSDT',
            'interval': '15',
            'rmi_period': 14,
            'momentum_period': 4,
            'atr_period': 10,
            'atr_multiplier': 3,
            'risk_percent': 2.0,
            'stop_loss_pct': 0.45,
            'take_profit_pct': 2.0,
            'maker_fee': -0.01,  # Rebate
            'taker_fee': 0.055,
            'maker_offset_pct': 0.02,
            'net_stop_loss': 0.505,
            'net_take_profit': 1.945,
            'max_position_size': 5000,  # ADA position limit
            'min_trade_interval': 300,
            'limit_order_timeout': 30,
            'limit_order_retries': 3
        }
        
        self.symbol = self.config['symbol']
        self.logger = TradeLogger("RMI_SUPERTREND_FIXED", self.symbol)
        
        prefix = "TESTNET_" if not self.LIVE_TRADING else "LIVE_"
        api_key = os.getenv(f"{prefix}BYBIT_API_KEY", "")
        api_secret = os.getenv(f"{prefix}BYBIT_API_SECRET", "")
    
        self.exchange = HTTP(
        testnet=(not self.LIVE_TRADING),  # Use testnet if not live
        api_key=api_key,
        api_secret=api_secret
        )
        self.position = None
        self.current_trade_id = None
        self.account_balance = 10000
        self.price_data = pd.DataFrame()
        self.last_trade_time = 0
        
    def format_qty(qty, min_qty=None, step=None, precision=None):
    """Return a properly formatted quantity **rounded up** to meet exchange minimums.
    - Rounds up to the nearest step if provided.
    - Ensures qty >= min_qty if provided.
    - Applies precision at the end if given.
    Never returns "0" if a positive qty was requested; returns 0 only if qty <= 0.
    """
    try:
        q = float(qty)
    except Exception:
        return format_qty(0 if qty<=0 else qty, min_qty=min_qty if "min_qty" in locals() else None, step=qtyStep if "qtyStep" in locals() else None, precision=None)
    if q <= 0:
        return format_qty(0 if qty<=0 else qty, min_qty=min_qty if "min_qty" in locals() else None, step=qtyStep if "qtyStep" in locals() else None, precision=None)
    if step:
        # round UP to step
        steps = int((q + 1e-15) // step)
        if steps * step < q:
            steps += 1
        q = steps * step
    if min_qty:
        q = max(q, float(min_qty))
    if precision is not None:
        fmt = "{:0." + str(int(precision)) + "f}"
        out = fmt.format(q)
    else:
        out = str(q)
    # avoid printing tiny scientific notation to strings like '0.0'
    if float(out) <= 0:
        # if rounding pushed to zero, bump to min_qty or one step
        if min_qty:
            out = str(min_qty if precision is None else ("{:0." + str(int(precision)) + "f}").format(float(min_qty)))
        elif step:
            out = str(step if precision is None else ("{:0." + str(int(precision)) + "f}").format(float(step)))
        else:
            out = "0"
    return out

# --- Fill policy improvements ---
MAKER_OFFSETS_PCT = [0.03, 0.06, 0.10]  # percent offsets for PostOnly ladder
MAKER_TIMEOUT_SEC = 25  # wait per step before escalating
USE_IOC_FALLBACK = True

def place_with_maker_ladder(place_limit_func, place_ioc_func, symbol, side, qty, ref_price):
    """Try PostOnly at progressively better prices; fall back to IOC/market if allowed.
    - place_limit_func(symbol, side, qty, price, post_only=True) must raise or return False on reject.
    - place_ioc_func(symbol, side, qty) is used only if USE_IOC_FALLBACK.
    """
    last_err = None
    for off in MAKER_OFFSETS_PCT:
        try:
            px = ref_price * (1 - off/100.0) if side.lower() == "buy" else ref_price * (1 + off/100.0)
            ok = place_limit_func(symbol, side, qty, px, post_only=True)
            if ok:
                return True
        except Exception as e:
            last_err = e
    if USE_IOC_FALLBACK and place_ioc_func:
        try:
            return place_ioc_func(symbol, side, qty)
        except Exception as e:
            last_err = e
    if last_err:
        raise last_err
    return False
