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
        
    # PATCHED: Fixed quantity formatting
    def format_qty(self, qty):
        min_qty = 1.0
        qty_step = 1.0
        min_notional = 5.0
        
        # Get current price
        try:
            current_price = float(self.last_price) if hasattr(self, 'last_price') else 1.0
        except:
            current_price = 1.0
        
        # Round to step
        import math
        qty = math.floor(qty / qty_step) * qty_step
        
        # Ensure minimum
        if qty < min_qty:
            qty = min_qty
            
        # Check notional
        if qty * current_price < min_notional:
            qty = math.ceil(min_notional / current_price / qty_step) * qty_step
            
        return qty
