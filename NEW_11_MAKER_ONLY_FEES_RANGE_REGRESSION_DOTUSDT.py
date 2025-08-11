import os, json, time, asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from pybit.unified_trading import HTTP
from dotenv import load_dotenv

load_dotenv()

# Utilities
_iso = lambda: datetime.now(timezone.utc).isoformat()
_bps = lambda d,b: (float(d)/float(b))*1e4 if b else None
_float = lambda v,d=0: float(v) if v and str(v).strip() else d

def _tags(tags):
    if not tags: return {}
    if isinstance(tags, dict): return tags
    out = {}
    if isinstance(tags, str):
        for kv in tags.split("|"):
            if not kv: continue
            if ":" in kv:
                k,v = kv.split(":",1)
                k,v = k.strip(), v.strip()
                try: out[k] = float(v)
                except: out[k] = True if v.lower()=="true" else False if v.lower()=="false" else v
            else: out[kv.strip()] = True
    return out

class TradeLogger:
    def __init__(self, bot_name, symbol, log_file=None, tf=None):
        self.ver = 1
        self.cls = "TradeLogger.v1"
        self.sess = f"s{int(time.time()*1000):x}"
        self.env = "testnet" if os.getenv("DEMO_MODE","true").lower()=="true" else "live"
        self.bot = bot_name
        self.sym = symbol
        self.tf = tf
        self.ccy = "USDT"
        self.id_seq = 1000
        self.open = {}
        self.pnl_day = 0.0
        self.streak_loss = 0
        os.makedirs("logs", exist_ok=True)
        self.log_file = log_file or f"logs/{self.bot}_{self.sym}_{datetime.utcnow():%Y%m%d}.jsonl"

    def _id(self): 
        self.id_seq += 1
        return self.id_seq
    
    def _w(self, o): 
        with open(self.log_file, "a") as f:
            f.write(json.dumps(o, separators=(",",":"), ensure_ascii=False) + "\n")
    
    def _bars(self, dur_s):
        if not self.tf: return None
        try: return max(1, (dur_s + int(self.tf)*60 - 1) // (int(self.tf)*60))
        except: return None

    def log_open(self, side_long_short, expected_px, actual_px, qty, stop_loss_px, take_profit_px, balance_usd, tags=None):
        tid = self._id()
        actual_px = _float(actual_px)
        expected_px = _float(expected_px)
        stop_loss_px = _float(stop_loss_px)
        take_profit_px = _float(take_profit_px)
        qty = _float(qty)
        
        slip_bps = _bps(actual_px - expected_px, expected_px)
        stop_move = abs(actual_px - stop_loss_px)
        tp_move = abs(take_profit_px - actual_px)
        risk = stop_move * qty if stop_move > 0 else 0.0
        rr = (tp_move / stop_move) if stop_move > 0 else None
        tg = _tags(tags)
        
        rec = {
            "ver": self.ver, "cls": self.cls, "sess": self.sess, "env": self.env,
            "bot": self.bot, "sym": self.sym, "tf": self.tf,
            "t": "O", "id": tid, "sd": side_long_short, "ccy": self.ccy,
            "px": round(actual_px, 6), "exp": round(expected_px, 6),
            "slip_bps": round(slip_bps, 2) if slip_bps is not None else None,
            "qty": round(qty, 6), "sl": round(stop_loss_px, 6), "tp": round(take_profit_px, 6),
            "risk_usd": round(risk, 4), "rr_plan": round(rr, 4) if rr else None,
            "bal": round(_float(balance_usd, 1000), 2), "tags": tg, "ts": _iso()
        }
        self._w(rec)
        self.open[tid] = {
            "tso": datetime.now(timezone.utc), "entry": actual_px, 
            "sd": side_long_short, "qty": qty, "risk": risk, "tags": tg
        }
        return tid

    def log_close(self, trade_id, expected_exit, actual_exit, exit_reason, in_bps, out_bps, extra=None):
        st = self.open.get(trade_id)
        if not st: return None
        
        actual_exit = _float(actual_exit)
        expected_exit = _float(expected_exit)
        dur = max(0, int((datetime.now(timezone.utc) - st["tso"]).total_seconds()))
        edge_bps = _bps(actual_exit - expected_exit, expected_exit)
        
        qty = st["qty"]
        sd = st["sd"]
        entry = st["entry"]
        gross = (actual_exit - entry) * qty if sd == "L" else (entry - actual_exit) * qty
        
        fe_in = (abs(_float(in_bps))/1e4 * entry * qty) if in_bps else 0.0
        fe_out = (abs(_float(out_bps))/1e4 * actual_exit * qty) if out_bps else 0.0
        fees = fe_in + fe_out
        net = gross - fees
        R = (net / st["risk"]) if st["risk"] > 0 else None
        
        self.pnl_day += net
        self.streak_loss = self.streak_loss + 1 if net < 0 else 0
        
        rec = {
            "ver": self.ver, "cls": self.cls, "sess": self.sess, "env": self.env,
            "bot": self.bot, "sym": self.sym, "tf": self.tf,
            "t": "C", "ref": trade_id, "ccy": self.ccy,
            "px": round(actual_exit, 6), "ref_px": round(expected_exit, 6),
            "edge_bps": round(edge_bps, 2) if edge_bps else None,
            "dur_s": dur, "bars_held": self._bars(dur),
            "qty": round(qty, 6), "gross": round(gross, 4),
            "fees_in_bps": round(_float(in_bps), 2) if in_bps else None,
            "fees_out_bps": round(_float(out_bps), 2) if out_bps else None,
            "fees_total": round(fees, 4),
            "net": round(net, 4), "R": round(R, 4) if R else None,
            "exit": exit_reason, "pnl_day": round(self.pnl_day, 4),
            "streak_loss": int(self.streak_loss),
            "tags": st["tags"], "extra": extra or {}, "ts": _iso()
        }
        self._w(rec)
        del self.open[trade_id]
        return rec

    def log_close_unknown(self, trade_id, reason="unknown", extra=None):
        st = self.open.get(trade_id)
        if not st: return None
        
        dur = max(0, int((datetime.now(timezone.utc) - st["tso"]).total_seconds()))
        rec = {
            "ver": self.ver, "cls": self.cls, "sess": self.sess, "env": self.env,
            "bot": self.bot, "sym": self.sym, "tf": self.tf,
            "t": "C", "ref": trade_id, "ccy": self.ccy,
            "px": None, "ref_px": None, "edge_bps": None,
            "dur_s": dur, "bars_held": self._bars(dur),
            "qty": round(st["qty"], 6),
            "gross": None, "fees_in_bps": None, "fees_out_bps": None, "fees_total": None,
            "net": None, "R": None, "exit": reason,
            "pnl_day": round(self.pnl_day, 4), "streak_loss": int(self.streak_loss),
            "tags": st["tags"], "extra": extra or {"note": "external close"}, "ts": _iso()
        }
        self._w(rec)
        del self.open[trade_id]
        return rec

class RangeBalancingBot:
    def __init__(self):
        # Core config
        self.symbol = os.getenv('SYMBOL', 'DOTUSDT')
        self.demo_mode = os.getenv('DEMO_MODE', 'true').lower() == 'true'
        prefix = 'TESTNET_' if self.demo_mode else 'LIVE_'
        self.api_key = os.getenv(f'{prefix}BYBIT_API_KEY')
        self.api_secret = os.getenv(f'{prefix}BYBIT_API_SECRET')
        
        # State
        self.exchange = None
        self.position = None
        self.price_data = pd.DataFrame()
        self.account_balance = 1000.0
        self.pending_order = False
        self.active_order_id = None
        self.current_trade_id = None
        self.regression_channel = None
        self.last_channel_update = None
        
        # Timing
        self.last_order_time = None
        self.last_trade_time = 0
        self.min_order_interval = 30
        self.trade_cooldown = 30
        
        # Strategy params
        self.cfg = {
            'timeframe': '5', 'regression_period': 50, 'bb_period': 20, 'bb_std': 2.0,
            'channel_width_pct': 1.5, 'risk_pct': 2.0, 'maker_offset_pct': 0.01,
            'maker_fee_pct': -0.04, 'net_take_profit': 0.6, 'net_stop_loss': 0.3,
            'min_notional': 5, 'qty_precision': 1
        }
        
        self.logger = TradeLogger("RANGE_REGRESSION", self.symbol, tf=self.cfg['timeframe'])
    
    # === Core Methods ===
    def connect(self):
        try:
            self.exchange = HTTP(demo=self.demo_mode, api_key=self.api_key, api_secret=self.api_secret)
            return self.exchange.get_server_time().get('retCode') == 0
        except: 
            return False
    
    def fmt_qty(self, qty):
        prec = self.cfg['qty_precision']
        step = 10**(-prec)
        return f"{round(qty/step)*step:.{prec}f}"
    
    async def get_balance(self):
        try:
            wallet = self.exchange.get_wallet_balance(accountType="UNIFIED", coin="USDT")
            if wallet.get("retCode") == 0:
                for lst in wallet.get("result", {}).get("list", []):
                    for c in lst.get("coin", []):
                        if c.get("coin") == "USDT":
                            val = c.get("availableToWithdraw", "")
                            self.account_balance = _float(val, 1000.0)
                            return True
        except Exception as e:
            print(f"❌ Balance error: {e}")
        self.account_balance = 1000.0
        return False
    
    # === Market Data ===
    async def get_data(self):
        try:
            klines = self.exchange.get_kline(
                category="linear", symbol=self.symbol, 
                interval=self.cfg['timeframe'], limit=100
            )
            if klines.get('retCode') != 0: 
                return False
            
            df = pd.DataFrame(klines['result']['list'], 
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
            
            self.price_data = df.sort_values('timestamp').reset_index(drop=True)
            return True
        except: 
            return False
    
    # === Position Management ===
    async def check_position(self):
        try:
            positions = self.exchange.get_positions(category="linear", symbol=self.symbol)
            if positions.get('retCode') != 0:
                self.position = None
                return False
            
            pos_list = positions.get('result', {}).get('list', [])
            if not pos_list or _float(pos_list[0].get('size', '0')) == 0:
                # Position closed
                if self.position and self.current_trade_id:
                    self.logger.log_close_unknown(self.current_trade_id, "external_close")
                    self.current_trade_id = None
                self.position = None
                self.pending_order = False  # Clear pending on position close
                self.active_order_id = None
                return False
            
            self.position = pos_list[0]
            return True
        except:
            self.position = None
            self.pending_order = False
            self.active_order_id = None
            return False
    
    async def check_orders(self):
        try:
            orders = self.exchange.get_open_orders(category="linear", symbol=self.symbol, limit=50)
            if orders.get('retCode') != 0: 
                return False
            
            open_orders = orders.get('result', {}).get('list', [])
            if not open_orders:
                self.pending_order = False
                self.active_order_id = None
                return False
            
            # Check order age and status
            for order in open_orders:
                created = order.get('createdTime', '0')
                age = (datetime.now(timezone.utc) - 
                      datetime.fromtimestamp(int(created)/1000, tz=timezone.utc)).total_seconds()
                
                if age > 60:  # Cancel stale orders
                    try:
                        self.exchange.cancel_order(
                            category="linear", symbol=self.symbol, 
                            orderId=order['orderId']
                        )
                    except: 
                        pass
                else:
                    self.pending_order = True
                    self.active_order_id = order['orderId']
                    return True
            
            self.pending_order = False
            self.active_order_id = None
            return False
        except: 
            return False
    
    # === Strategy Calculations ===
    def calc_regression(self, prices):
        if len(prices) < self.cfg['regression_period']: 
            return None
        
        recent = prices.tail(self.cfg['regression_period'])
        x = np.arange(len(recent))
        y = recent.values
        slope, intercept = np.polyfit(x, y, 1)
        regression = slope * x + intercept
        std = np.std(y - regression)
        width = std * self.cfg['channel_width_pct']
        
        return {
            'upper': regression[-1] + width,
            'lower': regression[-1] - width,
            'midline': regression[-1],
            'slope': slope,
            'angle': np.degrees(np.arctan(slope))
        }
    
    def calc_bb(self, prices):
        if len(prices) < self.cfg['bb_period']: 
            return None
        
        sma = prices.rolling(window=self.cfg['bb_period']).mean().iloc[-1]
        std = prices.rolling(window=self.cfg['bb_period']).std().iloc[-1]
        
        return {
            'upper': sma + std * self.cfg['bb_std'],
            'lower': sma - std * self.cfg['bb_std'],
            'middle': sma
        }
    
    def calc_position_size(self, price, stop_loss):
        risk_diff = abs(price - stop_loss)
        if self.account_balance <= 0 or risk_diff == 0: 
            return 0
        
        qty = (self.account_balance * self.cfg['risk_pct'] / 100) / risk_diff
        min_qty = self.cfg['min_notional'] / price
        return max(qty, min_qty)
    
    def calc_limit_price(self, market_price, side):
        offset = self.cfg['maker_offset_pct'] / 100
        mult = (1 - offset) if side in ['BUY', 'Buy'] else (1 + offset)
        return round(market_price * mult, 4)
    
    # === Trading Logic ===
    def signal(self, df):
        # Block signals if order pending or position exists
        if self.pending_order or self.position: 
            return None
        
        # Enforce minimum order interval
        if self.last_order_time:
            if (datetime.now() - self.last_order_time).total_seconds() < self.min_order_interval:
                return None
        
        if len(df) < self.cfg['regression_period']: 
            return None
        
        price = _float(df['close'].iloc[-1])
        
        # Update regression channel periodically
        if not self.last_channel_update or \
           (datetime.now() - self.last_channel_update).total_seconds() > 600:
            self.regression_channel = self.calc_regression(df['close'])
            self.last_channel_update = datetime.now()
        
        if not self.regression_channel: 
            return None
        
        bb = self.calc_bb(df['close'])
        if not bb: 
            return None
        
        # Calculate relative positions
        reg_range = self.regression_channel['upper'] - self.regression_channel['lower']
        bb_range = bb['upper'] - bb['lower']
        
        if reg_range == 0 or bb_range == 0:
            return None
            
        reg_pos = (price - self.regression_channel['lower']) / reg_range
        bb_pos = (price - bb['lower']) / bb_range
        
        # Generate signals
        if reg_pos <= 0.2 and bb_pos <= 0.3:
            return {
                'action': 'BUY', 'price': price,
                'reg': self.regression_channel['lower'],
                'bb': bb['lower'],
                'angle': self.regression_channel['angle']
            }
        elif reg_pos >= 0.8 and bb_pos >= 0.7:
            return {
                'action': 'SELL', 'price': price,
                'reg': self.regression_channel['upper'],
                'bb': bb['upper'],
                'angle': self.regression_channel['angle']
            }
        
        return None
    
    def should_close(self):
        if not self.position or not self.regression_channel: 
            return False, ""
        
        price = _float(self.price_data['close'].iloc[-1])
        entry = _float(self.position.get('avgPrice', 0))
        side = self.position.get('side', '')
        
        if entry == 0: 
            return False, ""
        
        # Calculate PnL percentage
        if side == "Buy":
            pnl_pct = (price - entry) / entry * 100
        else:
            pnl_pct = (entry - price) / entry * 100
        
        # Check exit conditions
        if pnl_pct >= self.cfg['net_take_profit']: 
            return True, "take_profit"
        if pnl_pct <= -self.cfg['net_stop_loss']: 
            return True, "stop_loss"
        
        # Channel midline exit
        if side == "Buy" and price >= self.regression_channel['midline']:
            return True, "channel_midline"
        if side == "Sell" and price <= self.regression_channel['midline']:
            return True, "channel_midline"
        
        # Opposite band exit
        bb = self.calc_bb(self.price_data['close'])
        if bb:
            if side == "Buy" and price >= bb['upper']:
                return True, "opposite_bb"
            if side == "Sell" and price <= bb['lower']:
                return True, "opposite_bb"
        
        return False, ""
    
    async def execute_trade(self, sig):
        # Enforce cooldown
        if time.time() - self.last_trade_time < self.trade_cooldown:
            return
        
        # Double-check no pending order
        if self.pending_order: 
            return
        
        # Set flags
        self.pending_order = True
        self.last_order_time = datetime.now()
        
        # Calculate order params
        sl = sig['reg'] * (0.995 if sig['action'] == 'BUY' else 1.005)
        qty = self.calc_position_size(sig['price'], sl)
        formatted_qty = self.fmt_qty(qty)
        
        # Validate position size
        if _float(formatted_qty) < (self.cfg['min_notional'] / sig['price']):
            self.pending_order = False
            return
        
        limit_price = self.calc_limit_price(sig['price'], sig['action'])
        
        try:
            order = self.exchange.place_order(
                category="linear", symbol=self.symbol,
                side="Buy" if sig['action'] == 'BUY' else "Sell",
                orderType="Limit", qty=formatted_qty,
                price=str(limit_price), timeInForce="PostOnly"
            )
            
            if order.get('retCode') == 0:
                self.last_trade_time = time.time()
                self.active_order_id = order['result']['orderId']
                
                # Calculate targets
                tp_mult = 1 + self.cfg['net_take_profit']/100 if sig['action'] == 'BUY' else 1 - self.cfg['net_take_profit']/100
                tp = limit_price * tp_mult
                
                # Log trade
                tags = f"reg:{sig['reg']:.4f}|bb:{sig['bb']:.4f}|angle:{sig['angle']:.1f}|risk_pct:{self.cfg['risk_pct']}"
                self.current_trade_id = self.logger.log_open(
                    "L" if sig['action'] == 'BUY' else "S",
                    sig['price'], limit_price, _float(formatted_qty),
                    sl, tp, self.account_balance, tags
                )
                
                print(f"✅ {sig['action']}: {formatted_qty} @ ${limit_price:.4f}")
            else:
                self.pending_order = False
                print(f"❌ Order failed: {order.get('retMsg')}")
        except Exception as e:
            self.pending_order = False
            print(f"❌ Trade error: {e}")
    
    async def close_position(self, reason):
        if not self.position: 
            return
        
        self.pending_order = True
        
        price = _float(self.price_data['close'].iloc[-1])
        side = "Sell" if self.position.get('side') == "Buy" else "Buy"
        qty = _float(self.position.get('size', 0))
        
        if qty == 0:
            self.pending_order = False
            return
            
        limit_price = self.calc_limit_price(price, side)
        
        try:
            order = self.exchange.place_order(
                category="linear", symbol=self.symbol, side=side,
                orderType="Limit", qty=self.fmt_qty(qty),
                price=str(limit_price), timeInForce="PostOnly",
                reduceOnly=True
            )
            
            if order.get('retCode') == 0:
                if self.current_trade_id:
                    self.logger.log_close(
                        self.current_trade_id, price, limit_price, reason,
                        self.cfg['maker_fee_pct'], self.cfg['maker_fee_pct'],
                        {"mode": "maker", "cooldown": self.trade_cooldown}
                    )
                    self.current_trade_id = None
                
                print(f"✅ CLOSE: {reason}")
                self.position = None
                self.pending_order = False
            else:
                self.pending_order = False
                print(f"❌ Close failed: {order.get('retMsg')}")
        except Exception as e:
            self.pending_order = False
            print(f"❌ Close error: {e}")
    
    # === Display ===
    def status(self):
        if self.price_data.empty: 
            return
        
        price = _float(self.price_data['close'].iloc[-1])
        
        # Status line
        print(f"\n💰 ${price:.4f} | Bal: ${self.account_balance:.2f} | PnL: ${self.logger.pnl_day:.2f} | Streak: {self.logger.streak_loss}")
        
        # Order status
        if self.pending_order: 
            print(f"⏳ ORDER: {self.active_order_id}")
        
        # Channel info
        if self.regression_channel:
            r = self.regression_channel
            print(f"📈 L:${r['lower']:.4f} M:${r['midline']:.4f} U:${r['upper']:.4f}")
        
        # Position info
        if self.position:
            side = self.position.get('side', '')
            size = self.position.get('size', '0')
            avg_price = _float(self.position.get('avgPrice', 0))
            pnl = _float(self.position.get('unrealisedPnl', 0))
            emoji = '🟢' if side == 'Buy' else '🔴'
            print(f"{emoji} {side}: {size} @ ${avg_price:.4f} PnL:${pnl:.2f}")
        
        print("-" * 50)
    
    # === Main Loop ===
    async def run_cycle(self):
        # Get market data
        if not await self.get_data():
            return
        
        # Update balance only when needed (no position)
        if not self.position:
            await self.get_balance()
        
        # Check states
        await self.check_orders()
        has_position = await self.check_position()
        
        if has_position:
            # Manage position
            should_exit, reason = self.should_close()
            if should_exit and not self.pending_order:
                await self.close_position(reason)
        else:
            # Look for entry
            if not self.pending_order:
                sig = self.signal(self.price_data)
                if sig:
                    await self.execute_trade(sig)
        
        self.status()
    
    async def run(self):
        if not self.connect():
            print("❌ Connection failed")
            return
        
        print(f"📊 Range Bot v{self.logger.ver} | {self.logger.sess} | {self.logger.env}")
        await self.get_balance()  # Initial balance
        
        try:
            while True:
                await self.run_cycle()
                await asyncio.sleep(5)
        except KeyboardInterrupt:
            print("\n🛑 Stopping...")
            if self.position:
                await self.close_position("manual_stop")
        except Exception as e:
            print(f"⚠️ Runtime error: {e}")
            await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(RangeBalancingBot().run())