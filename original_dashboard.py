#!/usr/bin/env python3
"""
IRIS Dashboard – Bybit Positions with Actual Fees & Breakeven
- Single-file Flask app
- IRIS-styled UI (pure CSS)
- Order history with maker/taker tags
- Fee-aware breakeven (assumes Market close by default)
"""
import os
import time
import threading
from datetime import datetime, timezone
from flask import Flask, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv
from pybit.unified_trading import HTTP

load_dotenv(override=True)

# ---------- Config helpers ----------
def _bool_env(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return str(val).strip().lower() in ("1", "true", "yes", "on")

def _safe_float(val, default=0.0):
    if val in [None, '', 'null']:
        return default
    try:
        return float(val)
    except Exception:
        try:
            return float(str(val))
        except Exception:
            return default

def _now_hms():
    return datetime.now().strftime('%H:%M:%S')

def _now_iso_utc():
    return datetime.now(timezone.utc).isoformat()

# ---------- App ----------
app = Flask(__name__)
CORS(app)

class BybitDataProvider:
    def __init__(self, demo_mode=True, maker_rate=0.0002, taker_rate=0.00055):
        self.lock = threading.Lock()
        self.demo_mode = demo_mode
        self.maker_rate = maker_rate
        self.taker_rate = taker_rate
        prefix = "TESTNET_" if demo_mode else "LIVE_"
        self.exchange = HTTP(
            demo=demo_mode,
            api_key=os.getenv(f"{prefix}BYBIT_API_KEY", ""),
            api_secret=os.getenv(f"{prefix}BYBIT_API_SECRET", "")
        )
        self.data = {
            "positions": [],
            "account": {},
            "total_pnl": 0.0,
            "position_count": 0,
            "last_update": None,
            "demo_mode": self.demo_mode
        }
        # light cache for recent orders by symbol
        self._orders_cache = {}  # symbol -> {"ts": float, "rows": list}

    # ---- API calls ----
    def _get_positions(self):
        return self.exchange.get_positions(category="linear", settleCoin="USDT")

    def _get_wallet(self):
        return self.exchange.get_wallet_balance(accountType="UNIFIED", coin="USDT")

    def _get_order_history(self, symbol=None, limit=50):
        # B
