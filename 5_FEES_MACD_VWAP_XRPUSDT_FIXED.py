import os
import asyncio
import pandas as pd
import json
from datetime import datetime, timezone
from pybit.unified_trading import HTTP
from dotenv import load_dotenv

load_dotenv()

class UnifiedLogger:
    def __init__(self, bot_name, symbol):
        self.bot_name = bot_name
        self.symbol = symbol
        self.currency = "USDT"
        self.open_trades = {}
        self.trade_counter = 1000
        
        # Emergency stop tracking
        self.daily_pnl = 0
        self.consecutive_losses = 0
        self.max_daily_loss = 50
        
    def generate_trade_id(self):
        self.trade_counter += 1
        return self.trade_counter
    
    def log_trade_open(self, side, expected_price, actual_price, qty, stop_loss, take_profit, info=""):
        trade_id = self.generate_trade_id()
        slippage = actual_price - expected_price if side == "BUY" else expected_price - actual_price
        
        log_entry = {
            "id": trade_id,
            "bot": self.bot_name,
            "symbol": self.symbol,
            "side": "LONG" if side == "BUY" else "SHORT",
            "action": "OPEN",
            "ts": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "expected_price": round(expected_price, 4),
            "actual_price": round(actual_price, 4),
            "slippage": round(slippage, 4),
            "qty": round(qty, 6),
            "stop_loss": round(stop_loss, 4),
            "take_profit": round(take_profit, 4),
            "currency": self.currency,
            "info": info
        }
        
        self.open_trades[trade_id] = {
            "entry_time": datetime.now(),
            "entry_price": actual_price,
            "side": side,
            "qty": qty,
            "stop_loss": stop_loss,
            "take_profit": take_profit
        }
        
        return trade_id, log_entry

    def log_trade_close(self, trade_id, expected_exit, actual_exit, reason, fees_entry=-0.04, fees_exit=-0.04):
        if trade_id not in self.open_trades:
            return None
            
        trade = self.open_trades[trade_id]
        duration = (datetime.now() - trade["entry_time"]).total_seconds()
        slippage = actual_exit - expected_exit if trade["side"] == "SELL" else expected_exit - actual_exit
        
        gross_pnl = ((actual_exit - trade["entry_price"]) * trade["qty"] 
                    if trade["side"] == "BUY" 
                    else (trade["entry_price"] - actual_exit) * trade["qty"])
        
        entry_rebate = trade["entry_price"] * trade["qty"] * abs(fees_entry) / 100
        exit_rebate = actual_exit * trade["qty"] * abs(fees_exit) / 100
        total_rebates = entry_rebate + exit_rebate
        net_pnl = gross_pnl + total_rebates
        
        # Update daily PnL
        self.daily_pnl += net_pnl
        self.consecutive_losses = self.consecutive_losses + 1 if net_pnl < 0 else 0
        
        log_entry = {
            "id": trade_id,
            "bot": self.bot_name,
            "symbol": self.symbol,
            "side": "LONG" if trade["side"] == "BUY" else "SHORT",
            "action": "CLOSE",
            "ts": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "duration_sec": int(duration),
            "entry_price": round(trade["entry_price"], 4),
            "expected_exit": round(expected_exit, 4),
            "actual_exit": round(actual_exit, 4),
            "slippage": round(slippage, 4),
            "qty": round(trade["qty"], 6),
            "gross_pnl": round(gross_pnl, 2),
            "rebates": {"entry": round(entry_rebate, 4), "exit": round(exit_rebate, 4), "total": round(total_rebates, 4)},
            "net_pnl": round(net_pnl, 2),
            "reason": reason,
            "currency": self.currency
        }
        
        del self.open_trades[trade_id]
        return log_entry
    
    def write_log(self, log_entry, log_file):
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

class EMARSIBot:
    def __init__(self):
        self.symbol = 'XRPUSDT'
        self.demo_mode = os.getenv('DEMO_MODE', 'true').lower() == 'true'
        
        prefix = 'TESTNET_' if self.demo_mode else 'LIVE_'
        self.api_key = os.getenv(f'{prefix}BYBIT_API_KEY')
        self.api_secret = os.getenv(f'{prefix}BYBIT_API_SECRET')
        
        self.exchange = None
        self.position = None
        self.pending_order = None
        self.price_data = pd.DataFrame()
        self.ema_divergence = 0
        self.account_balance = 1000
        
        # Configuration
        self.config = {
            'ema_fast': 9,
            'ema_slow': 21,
            'rsi_period': 7,
            'rsi_long_threshold': 30,
            'rsi_short_threshold': 70,
            'risk_per_trade_pct': 2.0,
            'maker_offset_pct': 0.01,
            'stop_loss': 0.35,
            'take_profit': 0.50,
            'trail_divergence': 0.15,
            'order_timeout': 180,
            'expected_slippage_pct': 0.02,
        }
        
        # Trade cooldown
        self.last_trade_time = 0
        self.trade_cooldown = 30
        
        os.makedirs("logs", exist_ok=True)
        self.log_file = f"logs/5_FEES_MACD_VWAP_XRPUSDT.log"
        self.unified_logger = UnifiedLogger("STRATEGY5_EMA_RSI_FIXED", self.symbol)
        self.current_trade_id = None
        self.entry_price = None
        self.trailing_stop = None
        self.last_ema_state = None
    
    def connect(self):
        try:
            self.exchange = HTTP(demo=self.demo_mode, api_key=self.api_key, api_secret=self.api_secret)
            return self.exchange.get_server_time().get('retCode') == 0
        except:
            return False
    
    async def get_account_balance(self):
        try:
            wallet = self.exchange.get_wallet_balance(accountType="UNIFIED", coin="USDT")
            if wallet.get('retCode') == 0:
                balance_list = wallet['result']['list']
                if balance_list:
                    for coin in balance_list[0]['coin']:
                        if coin['coin'] == 'USDT':
                            balance_str = coin['availableToWithdraw']
                            if balance_str and balance_str.strip():
                                self.account_balance = float(balance_str)
                                return True
        except:
            pass
        
        self.account_balance = 1000.0
        return True
    
    def calculate_position_size(self, price, stop_loss_price):
        if self.account_balance <= 0:
            return 0
        
        risk_amount = self.account_balance * (self.config['risk_per_trade_pct'] / 100)
        price_diff = abs(price - stop_loss_price)
        
        if price_diff == 0:
            return 0
        
        slippage_factor = 1 + (self.config['expected_slippage_pct'] / 100)
        adjusted_risk = risk_amount / slippage_factor
        
        raw_qty = adjusted_risk / price_diff
        max_position_value = self.account_balance * 0.1
        max_qty = max_position_value / price
        
        final_qty = min(raw_qty, max_qty)
        return max(final_qty, 0.1)
    
    # PATCHED: Fixed quantity formatting
    def format_qty(self, qty):
        min_qty = 0.1
        qty_step = 0.1
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
