import os, math, asyncio, json, time, warnings
from datetime import datetime, timezone
from collections import deque

import numpy as np
import pandas as pd
from pybit.unified_trading import HTTP
from dotenv import load_dotenv

warnings.filterwarnings('ignore')
load_dotenv()

# ---------------- Strategy ↔ Pair map ----------------
STRATEGY_PAIRS = {
    'LSTM_PREDICTION': {'symbol': 'BTCUSDT','name': 'Bitcoin','min_qty': 0.001,'qty_step': 0.001,'reason': 'High liquidity, extensive historical data for sequence learning'},
    'GRU_TREND':       {'symbol': 'ETHUSDT','name': 'Ethereum','min_qty': 0.01,'qty_step': 0.01,'reason': 'Strong trend patterns, second-largest market cap'},
    'CNN_PATTERN':     {'symbol': 'DOGEUSDT','name': 'Dogecoin','min_qty': 1.0,'qty_step': 1.0,'reason': 'Clear candlestick patterns, high volume for pattern recognition'},
    'TRANSFORMER':     {'symbol': 'BTCUSDT','name': 'Bitcoin','min_qty': 0.001,'qty_step': 0.001,'reason': 'Long-term dependencies, market leader dynamics'},
    'RANDOM_FOREST':   {'symbol': 'LTCUSDT','name': 'Litecoin','min_qty': 0.01,'qty_step': 0.01,'reason': 'Stable volatility, good for ensemble methods'},
    'XGBOOST':         {'symbol': 'XRPUSDT','name': 'Ripple','min_qty': 1.0,'qty_step': 1.0,'reason': 'High momentum patterns, technical indicator responsiveness'},
    'SVM':             {'symbol': 'ADAUSDT','name': 'Cardano','min_qty': 1.0,'qty_step': 1.0,'reason': 'Mid-cap with clear support/resistance levels'},
    'AUTOENCODER':     {'symbol': 'DOGEUSDT','name': 'Shiba Inu','min_qty': 1_000_000.0,'qty_step': 1_000_000.0,'reason': 'Meme coin volatility ideal for anomaly detection'},
    'DQN_RL':          {'symbol': 'SOLUSDT','name': 'Solana','min_qty': 0.01,'qty_step': 0.01,'reason': 'Fast blockchain, adaptive to market conditions'},
    'GENETIC_ALGO':    {'symbol': 'MATICUSDT','name': 'Polygon','min_qty': 1.0,'qty_step': 1.0,'reason': 'Layer-2 solution, good for parameter optimization'}
}

# ---------------- Shared utils ----------------
class SymbolInfo:
    def __init__(self, min_qty: float, qty_step: float, max_leverage=None):
        self.min_qty, self.qty_step, self.max_leverage = float(min_qty), float(qty_step), max_leverage

def _round_step(x: float, step: float) -> float:
    return float(x) if step <= 0 else math.floor(x / step + 1e-12) * step

def sized_qty(entry: float, stop: float, risk_usd: float, info: SymbolInfo) -> float:
    px_risk = abs(float(entry) - float(stop))
    if px_risk <= 0: return 0.0
    q = max(risk_usd / px_risk, info.min_qty)
    return _round_step(q, info.qty_step)

# ---------------- Logger ----------------
class TradeLogger:
    def __init__(self, bot_name, symbol):
        self.bot_name, self.symbol, self.currency = bot_name, symbol, 'USDT'
        self.open_trades, self.trade_id = {}, 1000
        self.daily_pnl, self.consecutive_losses, self.max_daily_loss = 0.0, 0, 50.0
        os.makedirs('logs', exist_ok=True)
        self.log_file = f'logs/{bot_name}_{symbol}.log'

    def _tid(self):
        self.trade_id += 1; return self.trade_id

    def log_trade_open(self, side, expected_price, actual_price, qty, stop_loss, take_profit, info=''):
        tid = self._tid(); now = datetime.now(timezone.utc)
        row = {
            'id': tid,'bot': self.bot_name,'symbol': self.symbol,
            'side': 'LONG' if side == 'BUY' else 'SHORT','action': 'OPEN','ts': now.isoformat(),
            'expected_price': round(float(expected_price), 6),'actual_price': round(float(actual_price), 6),
            'qty': round(float(qty), 6),'stop_loss': round(float(stop_loss), 6),'take_profit': round(float(take_profit), 6),
            'currency': self.currency,'info': info
        }
        self.open_trades[tid] = {'entry_time': now,'entry_price': float(actual_price),'side': side,'qty': float(qty),
                                 'stop_loss': float(stop_loss),'take_profit': float(take_profit)}
        with open(self.log_file, 'a', encoding='utf-8') as f: f.write(json.dumps(row)+'\n')
        return tid, row

    def log_trade_close(self, tid, expected_exit, actual_exit, reason, fees_entry=-0.04, fees_exit=-0.04):
        if tid not in self.open_trades: return None
        t = self.open_trades.pop(tid); now = datetime.now(timezone.utc)
        dur = (now - t['entry_time']).total_seconds(); entry, qty, side, exitp = map(float, (t['entry_price'], t['qty'], t['side']=='BUY', actual_exit))
        gross = (exitp - entry) * qty if side else (entry - exitp) * qty
        er, xr = entry * qty * abs(float(fees_entry)) / 100.0, exitp * qty * abs(float(fees_exit)) / 100.0
        net = gross + er + xr
        self.consecutive_losses = self.consecutive_losses + 1 if net < 0 else 0
        self.daily_pnl += net
        row = {
            'id': tid,'bot': self.bot_name,'symbol': self.symbol,
            'side': 'LONG' if side else 'SHORT','action': 'CLOSE','ts': now.isoformat(),'duration_sec': int(dur),
            'entry_price': round(entry, 6),'expected_exit': round(expected_exit, 6),'actual_exit': round(exitp, 6),
            'qty': round(qty, 6),'gross_pnl': round(gross, 6),
            'fee_rebates': {'entry': round(er, 6), 'exit': round(xr, 6), 'total': round(er + xr, 6)},
            'net_pnl': round(net, 6),'reason': reason,'currency': self.currency
        }
        with open(self.log_file, 'a', encoding='utf-8') as f: f.write(json.dumps(row)+'\n')
        return row

# ---------------- Base bot ----------------
class BaseAIBot:
    def __init__(self, strategy_name):
        self.strategy_name = strategy_name
        cfg = STRATEGY_PAIRS.get(strategy_name, STRATEGY_PAIRS['CNN_PATTERN'])
        self.symbol, self.crypto_name = cfg['symbol'], cfg['name']
        self.symbol_info = SymbolInfo(cfg['min_qty'], cfg['qty_step'])

        self.demo_mode = os.getenv('DEMO_MODE', 'true').lower() == 'true'
        pfx = 'TESTNET_' if self.demo_mode else 'LIVE_'
        self.api_key, self.api_secret = os.getenv(f'{pfx}BYBIT_API_KEY'), os.getenv(f'{pfx}BYBIT_API_SECRET')
        print(f"📌 API Key loaded: {self.api_key[:8]+'...' if self.api_key else '❌ missing'}")
        print(f"📌 API Secret loaded: {'***hidden***' if self.api_secret else '❌ missing'}")

        self.exchange = self.position = self.pending_order = None
        self.price_data, self.account_balance = pd.DataFrame(), 1000.0
        self.last_trade_time, self.trade_cooldown, self.last_signal_price, self.min_price_change_pct = 0.0, 30, 0.0, 0.1
        self.config, self.logger, self.current_trade_id = self.get_strategy_config(), TradeLogger(strategy_name, self.symbol), None

    def get_strategy_config(self):
        c = {'timeframe': '5','lookback': 120,'risk_per_trade_pct': 2.0,'maker_fee_pct': -0.04,
             'net_take_profit': 1.58,'net_stop_loss': 0.42,'maker_offset_pct': 0.01}
        if self.symbol in ['SHIBUSDT','PEPEUSDT','BONKUSDT']:
            c.update(net_take_profit=3.0, net_stop_loss=1.0, risk_per_trade_pct=1.0)
        elif self.symbol in ['BTCUSDT','ETHUSDT']:
            c.update(net_take_profit=1.2, net_stop_loss=0.3, lookback=200)
        elif self.symbol in ['SOLUSDT','AVAXUSDT']:
            c.update(timeframe='3', lookback=150)
        return c

    def connect(self):
        """Fixed connect method"""
        try:
            # Check credentials first
            if not (self.api_key and self.api_secret):
                print(f"❌ API credentials missing")
                print(f"   Required: {'TESTNET_' if self.demo_mode else 'LIVE_'}BYBIT_API_KEY")
                print(f"   Required: {'TESTNET_' if self.demo_mode else 'LIVE_'}BYBIT_API_SECRET")
                return False
                    
            # Create exchange connection - use 'demo' not 'testnet'
            self.exchange = HTTP(
                demo=self.demo_mode,  # FIXED: use 'demo' parameter
                api_key=self.api_key, 
                api_secret=self.api_secret
            )
            
            # Test basic connectivity
            server_time = self.exchange.get_server_time()
            if server_time.get('retCode') != 0:
                print(f"❌ Server connection failed: {server_time.get('retMsg')}")
                return False
                
            print(f"✅ Connected to {'Testnet' if self.demo_mode else 'Live'} Bybit")
            
            # Test authentication
            try:
                wallet = self.exchange.get_wallet_balance(accountType='UNIFIED')
                if wallet.get('retCode') == 401:
                    print("❌ API Authentication failed!")
                    return False
                elif wallet.get('retCode') == 0:
                    print("✅ API authentication successful")
                else:
                    print(f"⚠️ Auth warning: {wallet.get('retMsg')}")
                    
            except Exception as auth_e:
                print(f"❌ Authentication test failed: {auth_e}")
                return False
                
            # Get instrument info
            try:
                info = self.exchange.get_instruments_info(category='linear', symbol=self.symbol)
                if info.get('retCode') == 0 and info['result'].get('list'):
                    spec = info['result']['list'][0]
                    lot_filter = spec.get('lotSizeFilter', {})
                    min_qty = float(lot_filter.get('minOrderQty', self.symbol_info.min_qty))
                    qty_step = float(lot_filter.get('qtyStep', self.symbol_info.qty_step))
                    
                    if min_qty > 0: 
                        self.symbol_info.min_qty = min_qty
                    if qty_step > 0: 
                        self.symbol_info.qty_step = qty_step
                        
                    print(f"🔧 Instrument specs: min_qty={self.symbol_info.min_qty}, qty_step={self.symbol_info.qty_step}")
            except Exception as e:
                print(f"⚠️ Instrument info fetch failed: {e}")
                
            return True
            
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False


    async def check_position(self):
        """Fixed position check for testnet unified account"""
        try:
            # CRITICAL FIX: Use settleCoin='USDT' for testnet unified account
            positions = self.exchange.get_positions(
                category='linear',
                symbol=self.symbol,
                settleCoin='USDT'  # Required for testnet unified trading
            )
            
            if positions.get('retCode') == 0:
                pos_list = positions['result']['list']
                
                # Find position for our symbol with actual size
                for pos in pos_list:
                    if pos.get('symbol') == self.symbol and float(pos.get('size', 0)) > 0:
                        self.position = pos
                        return True
                        
                # No position found or size is 0
                self.position = None
                return False
                
            elif positions.get('retCode') == 401:
                print("❌ Position check failed: Authentication error")
                print("   Check API permissions and unified account setup")
                self.position = None
                return False
            else:
                print(f"❌ Position API error: {positions.get('retMsg','Unknown')} (Code: {positions.get('retCode')})")
                self.position = None
                return False
                
        except Exception as e:
            if '401' in str(e):
                print(f"❌ Auth failed during position check: Verify {'TESTNET_' if self.demo_mode else 'LIVE_'} API keys")
            else:
                print(f"❌ Position check error: {e}")
            self.position = None
            return False

    async def get_account_balance(self):
        """Fixed account balance method"""
        try:
            if not self.exchange: 
                print('❌ Not connected to exchange')
                return False
                
            wallet = self.exchange.get_wallet_balance(accountType='UNIFIED', coin='USDT')
            
            if wallet.get('retCode') == 0:
                balance_list = wallet['result']['list']
                if balance_list and len(balance_list) > 0:
                    for coin in balance_list[0].get('coin', []):
                        if coin.get('coin') == 'USDT':
                            # Try multiple balance fields (testnet may use different fields)
                            balance_fields = [
                                'availableToWithdraw',
                                'walletBalance', 
                                'equity',
                                'availableBalance',
                                'balance'
                            ]
                            
                            for field in balance_fields:
                                balance_str = coin.get(field, '')
                                if balance_str and str(balance_str).strip() != '':
                                    try:
                                        raw_balance = float(balance_str)
                                        # Cap balance for testnet (prevent huge positions)
                                        self.account_balance = min(raw_balance, 10000.0)
                                        return True
                                    except (ValueError, TypeError):
                                        continue
                                        
            elif wallet.get('retCode') == 401:
                print('❌ Balance check failed: Invalid API key/secret')
                return False
                
        except Exception as e:
            print(f"❌ Balance error: {e}")
        
        # Fallback for demo
        self.account_balance = 1000.0
        return True

    async def execute_trade(self, signal):
        """Fixed execute_trade with proper error handling"""
        # Check trading cooldown
        if time.time() - self.last_trade_time < self.trade_cooldown: 
            remaining = self.trade_cooldown - (time.time() - self.last_trade_time)
            print(f"⏰ Trade cooldown: wait {remaining:.0f}s")
            return
            
        # Update balance before trading
        await self.get_account_balance()
        
        # Calculate position sizing
        sl_price = signal['price'] * (1 - self.config['net_stop_loss']/100.0) if signal['action']=='BUY' else signal['price'] * (1 + self.config['net_stop_loss']/100.0)
        qty = self.calculate_position_size(signal['price'], sl_price)
        formatted_qty = self.format_qty(qty)
        
        if formatted_qty == '0' or float(formatted_qty) == 0: 
            print(f"❌ Position size too small: {qty:.6f}")
            return
        
        # Calculate limit price with maker offset
        offset_mult = 1 - self.config['maker_offset_pct']/100.0 if signal['action']=='BUY' else 1 + self.config['maker_offset_pct']/100.0
        limit_price = round(signal['price'] * offset_mult, 6)
        
        try:
            order = self.exchange.place_order(
                category='linear', 
                symbol=self.symbol, 
                side=('Buy' if signal['action']=='BUY' else 'Sell'), 
                orderType='Limit', 
                qty=formatted_qty, 
                price=str(limit_price), 
                timeInForce='PostOnly',  # Ensure maker rebate
                positionIdx=0, 
                reduceOnly=False
            )
            
            if order.get('retCode') == 0:
                self.last_trade_time = time.time()
                self.last_signal_price = signal['price']
                
                # Calculate TP/SL for logging
                tp_price = limit_price * (1 + self.config['net_take_profit']/100.0) if signal['action']=='BUY' else limit_price * (1 - self.config['net_take_profit']/100.0)
                sl_price = limit_price * (1 - self.config['net_stop_loss']/100.0) if signal['action']=='BUY' else limit_price * (1 + self.config['net_stop_loss']/100.0)
                
                self.current_trade_id, _ = self.logger.log_trade_open(
                    signal['action'], 
                    signal['price'], 
                    limit_price, 
                    float(formatted_qty), 
                    sl_price, 
                    tp_price, 
                    info=f"signal:{signal.get('confidence',0):.2f}_bal:{self.account_balance:.2f}"
                )
                
                print(f"✅ {self.strategy_name} {signal['action']}: {formatted_qty} {self.crypto_name} @ ${limit_price:.6f}")
                
            elif order.get('retCode') == 401:
                print("❌ Trade failed: Authentication error")
                print("   Check API permissions for Contract Trading")
            else:
                print(f"❌ Trade rejected (retCode={order.get('retCode')}): {order.get('retMsg')}")
                
        except Exception as e:
            print(f"❌ Trade execution failed: {e}")
            if '401' in str(e):
                print("   Verify API has Contract Trade permissions")


    async def get_account_balance(self):
        try:
            if not self.exchange: print('❌ Not connected to exchange'); return False
            w = self.exchange.get_wallet_balance(accountType='UNIFIED', coin='USDT')
            if w.get('retCode') == 0:
                for c in w['result']['list'][0]['coin']:
                    if c['coin'] == 'USDT': self.account_balance = float(c.get('walletBalance', 1000.0)); return True
            elif w.get('retCode') == 10002: print('❌ Invalid API key/secret. Check .env'); return False
            else: print(f"❌ Wallet API error: {w.get('retMsg','Unknown')}")
        except Exception as e:
            print(f"❌ Balance error: {e}")
        self.account_balance = 1000.0; return True

    def calculate_position_size(self, price, stop):
        """Fixed position sizing with limits"""
        if self.account_balance <= 0: 
            return 0.0
            
        risk = self.account_balance * (self.config['risk_per_trade_pct'] / 100.0)
        
        # Cap risk for large testnet balances
        max_risk = 100.0  # Max $100 risk per trade
        risk = min(risk, max_risk)
        
        return sized_qty(price, stop, risk, self.symbol_info)

    def format_qty(self, q):
        if q < self.symbol_info.min_qty: return '0'
        q = _round_step(q, self.symbol_info.qty_step)
        return str(int(q)) if self.symbol_info.qty_step >= 1 else f"{q:.{max(0, len(str(self.symbol_info.qty_step).split('.')[-1]))}f}"

    async def check_position(self):
        """Fixed position check"""
        try:
            # Use settleCoin='USDT' for testnet unified account
            positions = self.exchange.get_positions(
                category='linear',
                symbol=self.symbol,
                settleCoin='USDT'  # Required for testnet
            )
            
            if positions.get('retCode') == 0:
                pos_list = positions['result']['list']
                
                # Find position with actual size
                for pos in pos_list:
                    if pos.get('symbol') == self.symbol and float(pos.get('size', 0)) > 0:
                        self.position = pos
                        return True
                        
                # No position found
                self.position = None
                return False
                
            elif positions.get('retCode') == 401:
                print("❌ Position check failed: Authentication error")
                self.position = None
                return False
            else:
                print(f"❌ Position API error: {positions.get('retMsg')}")
                self.position = None
                return False
                
        except Exception as e:
            print(f"❌ Position check error: {e}")
            self.position = None
            return False


    async def get_market_data(self):
        try:
            k = self.exchange.get_kline(category='linear', symbol=self.symbol, interval=self.config['timeframe'], limit=self.config['lookback'])
            if k.get('retCode') != 0: return False
            df = pd.DataFrame(k['result']['list'], columns=['timestamp','open','high','low','close','volume','turnover'])
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype('int64'), unit='ms', utc=True)
            for c in ['open','high','low','close','volume']: df[c] = pd.to_numeric(df[c], errors='coerce')
            self.price_data = df.sort_values('timestamp').reset_index(drop=True); return True
        except Exception as e:
            print(f"❌ Market data error: {e}"); return False

    def should_close(self):
        if not self.position or self.price_data.empty: return (False, '')
        cp = float(self.price_data['close'].iloc[-1]); ep = float(self.position.get('avgPrice', 0)); side = self.position.get('side', '')
        if not ep: return (False, '')
        profit_pct = (cp - ep) / ep * 100.0 if side == 'Buy' else (ep - cp) / ep * 100.0
        if profit_pct >= self.config['net_take_profit']: return (True, 'take_profit')
        if profit_pct <= -self.config['net_stop_loss']: return (True, 'stop_loss')
        return (False, '')

    def _ok_for_signal(self, df):
        if len(df) < self.config['lookback'] or self.position: return None
        cp = float(df['close'].iloc[-1])
        if self.last_signal_price:
            if abs(cp - self.last_signal_price) / self.last_signal_price * 100.0 < self.min_price_change_pct: return None
        return cp

    async def execute_trade(self, signal):
        """Fixed execute_trade method"""
        # Check cooldown
        if time.time() - self.last_trade_time < self.trade_cooldown:
            remaining = self.trade_cooldown - (time.time() - self.last_trade_time)
            print(f"⏰ Trade cooldown: wait {remaining:.0f}s")
            return
            
        # Update balance
        await self.get_account_balance()
        
        # Calculate position
        sl_price = signal['price'] * (1 - self.config['net_stop_loss']/100.0) if signal['action']=='BUY' else signal['price'] * (1 + self.config['net_stop_loss']/100.0)
        qty = self.calculate_position_size(signal['price'], sl_price)
        formatted_qty = self.format_qty(qty)
        
        if formatted_qty == '0' or float(formatted_qty) == 0:
            print(f"❌ Position size too small: {qty:.6f}")
            return
            
        # Calculate limit price
        offset_mult = 1 - self.config['maker_offset_pct']/100.0 if signal['action']=='BUY' else 1 + self.config['maker_offset_pct']/100.0
        limit_price = round(signal['price'] * offset_mult, 6)
        
        try:
            order = self.exchange.place_order(
                category='linear',
                symbol=self.symbol,
                side='Buy' if signal['action']=='BUY' else 'Sell',
                orderType='Limit',
                qty=formatted_qty,
                price=str(limit_price),
                timeInForce='PostOnly',
                positionIdx=0,
                reduceOnly=False
            )
            
            if order.get('retCode') == 0:
                self.last_trade_time = time.time()
                self.last_signal_price = signal['price']
                
                # Calculate TP/SL for logging
                tp_price = limit_price * (1 + self.config['net_take_profit']/100.0) if signal['action']=='BUY' else limit_price * (1 - self.config['net_take_profit']/100.0)
                
                self.current_trade_id, _ = self.logger.log_trade_open(
                    signal['action'],
                    signal['price'],
                    limit_price,
                    float(formatted_qty),
                    sl_price,
                    tp_price,
                    info=f"signal:{signal.get('confidence',0):.2f}_bal:{self.account_balance:.2f}"
                )
                
                print(f"✅ {self.strategy_name} {signal['action']}: {formatted_qty} {self.crypto_name} @ ${limit_price:.6f}")
                
            elif order.get('retCode') == 401:
                print("❌ Trade failed: Authentication error")
            else:
                print(f"❌ Trade rejected: {order.get('retMsg')}")
                
        except Exception as e:
            print(f"❌ Trade execution failed: {e}")

    async def close_position(self, reason):
        """Fixed close position method"""
        if not self.position or self.price_data.empty:
            return
            
        current_price = float(self.price_data['close'].iloc[-1])
        side = 'Sell' if self.position.get('side') == 'Buy' else 'Buy'
        qty = float(self.position.get('size', 0))
        
        if qty <= 0:
            return
            
        # Calculate limit price with offset
        offset_mult = 1 + self.config['maker_offset_pct']/100.0 if side=='Sell' else 1 - self.config['maker_offset_pct']/100.0
        limit_price = round(current_price * offset_mult, 6)
        
        try:
            order = self.exchange.place_order(
                category='linear',
                symbol=self.symbol,
                side=side,
                orderType='Limit',
                qty=self.format_qty(qty),
                price=str(limit_price),
                timeInForce='PostOnly',
                positionIdx=0,
                reduceOnly=True
            )
            
            if order.get('retCode') == 0:
                if self.current_trade_id:
                    self.logger.log_trade_close(
                        self.current_trade_id,
                        current_price,
                        limit_price,
                        reason,
                        self.config['maker_fee_pct'],
                        self.config['maker_fee_pct']
                    )
                    self.current_trade_id = None
                    
                print(f"✅ Closed: {reason}")
                self.position = None
                
        except Exception as e:
            print(f"❌ Close failed: {e}")

    def show_status(self):
        if self.price_data.empty: return
        cp = float(self.price_data['close'].iloc[-1])
        print(f"\n🤖 {self.strategy_name} - {self.symbol} ({self.crypto_name})\n💰 Price: ${cp:.6f} | Balance: ${self.account_balance:.2f}")
        if self.position:
            ep = float(self.position.get('avgPrice', 0)); side = self.position.get('side', ''); size = self.position.get('size', '0'); pnl = float(self.position.get('unrealisedPnl', 0))
            print(f"{'🟢' if side=='Buy' else '🔴'} {side}: {size} @ ${ep:.6f} | PnL: ${pnl:.2f}")
        else: print('🔍 Analyzing...')
        print('-'*60)

    async def run_cycle(self):
        
        await self.check_position()
        if self.position:
            ok, rsn = self.should_close();
            if ok: await self.close_position(rsn)
        else:
            sig = self.generate_signal(self.price_data)
            if sig: await self.execute_trade(sig)
        self.show_status()

    def generate_signal(self, df): return None

    async def run(self):
        if not self.connect(): print('❌ Failed to connect'); return
        print(f"🤖 {self.strategy_name} Started on {self.symbol} ({self.crypto_name})\n   📊 Reason: {STRATEGY_PAIRS[self.strategy_name]['reason']}")
        try:
            while True:
                await self.run_cycle(); await asyncio.sleep(10)
        except KeyboardInterrupt:
            print('\n🛑 Bot stopped');
            if self.position: await self.close_position('manual_stop')

# ---------------- Strategies ----------------
class LSTMPredictionBot(BaseAIBot):
    def __init__(self): super().__init__('LSTM_PREDICTION'); self.lookback_window, self.predictions = 50, deque(maxlen=10)
    def simulate_lstm_prediction(self, df):
        if len(df) < self.lookback_window: return 0.5
        r = df['close'].iloc[-self.lookback_window:].pct_change().dropna(); m, v = r.mean(), r.std()
        t = (df['close'].iloc[-1] - df['close'].iloc[-self.lookback_window]) / df['close'].iloc[-self.lookback_window]
        return max(0, min(1, 0.5 + m*10 + t*5 - v*2))
    def generate_signal(self, df):
        cp = self._ok_for_signal(df); 
        if cp is None: return None
        s = self.simulate_lstm_prediction(df); self.predictions.append(s)
        if len(self.predictions) < 3: return None
        a = float(np.mean(self.predictions))
        return {'action':'BUY','price':cp,'confidence':a} if a>0.65 else ({'action':'SELL','price':cp,'confidence':1-a} if a<0.35 else None)

class GRUTrendBot(BaseAIBot):
    def __init__(self): super().__init__('GRU_TREND'); self.sequence_length, self.trend_buffer = 60, deque(maxlen=5)
    def simulate_gru_classification(self, df):
        if len(df) < self.sequence_length: return 0.5
        feats = [df['close'].pct_change().iloc[-20:].mean(),
                 (df['volume'].iloc[-20:].mean()/df['volume'].iloc[-60:-20].mean()) if len(df)>=60 else 1,
                 (df['high']-df['low']).iloc[-20:].mean(), df['close'].iloc[-1]/df['close'].iloc[-20]-1]
        score = sum(f*w for f,w in zip(feats,[0.4,0.2,0.2,0.2])); return 1/(1+np.exp(-score*10))
    def generate_signal(self, df):
        cp = self._ok_for_signal(df); 
        if cp is None: return None
        p = self.simulate_gru_classification(df); self.trend_buffer.append(p)
        if len(self.trend_buffer) < 3: return None
        a = float(np.mean(self.trend_buffer))
        return {'action':'BUY','price':cp,'confidence':a} if a>0.7 else ({'action':'SELL','price':cp,'confidence':1-a} if a<0.3 else None)

class CNNPatternBot(BaseAIBot):
    def __init__(self): super().__init__('CNN_PATTERN'); self.pattern_window, self.patterns_detected = 20, deque(maxlen=5)
    def detect_pattern(self, df):
        if len(df) < self.pattern_window: return 'neutral', 0.5
        r = df.iloc[-5:]; bodies = (r['close']-r['open']).abs(); vols = r['volume']
        ab, av = bodies.mean(), vols.mean()
        if bodies.iloc[-1] > ab*1.5 and vols.iloc[-1] > av*1.3:
            return ('bullish_engulfing',0.8) if r['close'].iloc[-1] > r['open'].iloc[-1] else ('bearish_engulfing',0.8)
        if bodies.iloc[-1] < ab*0.2: return 'doji', 0.6
        return 'neutral', 0.5
    def generate_signal(self, df):
        cp = self._ok_for_signal(df); 
        if cp is None: return None
        pat, conf = self.detect_pattern(df); self.patterns_detected.append((pat, conf))
        if len(self.patterns_detected) < 3: return None
        bup = sum(1 for p,_ in self.patterns_detected if 'bullish' in p); bdn = sum(1 for p,_ in self.patterns_detected if 'bearish' in p)
        return {'action':'BUY','price':cp,'confidence':0.75} if bup>=3 else ({'action':'SELL','price':cp,'confidence':0.75} if bdn>=3 else None)

class TransformerBot(BaseAIBot):
    def __init__(self): super().__init__('TRANSFORMER'); self.sequence_length, self.attention_scores = 100, deque(maxlen=10)
    def attention_score(self, df):
        if len(df) < self.sequence_length: return 0
        p = df['close'].iloc[-self.sequence_length:].values; v = df['volume'].iloc[-self.sequence_length:].values
        pn = (p-p.mean())/(p.std()+1e-8); vn = (v-v.mean())/(v.std()+1e-8); rr = df['close'].pct_change().iloc[-20:].values
        cp = np.corrcoef(pn[-20:], pn[-40:-20])[0,1] if len(pn)>=40 else 0
        cv = np.corrcoef(vn[-20:], vn[-40:-20])[0,1] if len(vn)>=40 else 0
        return ((cp+cv)/2)*np.mean(rr)
    def generate_signal(self, df):
        cp = self._ok_for_signal(df); 
        if cp is None: return None
        s = self.attention_score(df); self.attention_scores.append(s)
        if len(self.attention_scores) < 5: return None
        a = float(np.mean(self.attention_scores))
        if a>0.002: return {'action':'BUY','price':cp,'confidence':min(0.8,abs(a)*100)}
        if a<-0.002: return {'action':'SELL','price':cp,'confidence':min(0.8,abs(a)*100)}
        return None

class RandomForestBot(BaseAIBot):
    def __init__(self): super().__init__('RANDOM_FOREST'); self.predictions = deque(maxlen=5)
    def simulate_forest_prediction(self, df):
        if len(df) < 30: return 0.5
        d = df['close'].diff(); g = d.where(d>0,0).rolling(14).mean(); l = (-d.where(d<0,0)).rolling(14).mean(); rs = g/(l.replace(0,np.nan))
        rsi = (100 - 100/(1+rs)).iloc[-1]; r5, r10 = df['close'].pct_change(5).iloc[-1], df['close'].pct_change(10).iloc[-1]
        vr = (df['volume'].iloc[-5:].mean())/(df['volume'].iloc[-20:].mean()); vol = df['close'].pct_change().iloc[-20:].std()
        s = 0.5 + (0.2 if rsi<30 else -0.2 if rsi>70 else 0) + (0.15 if (r5>0 and r10>0) else -0.15 if (r5<0 and r10<0) else 0) + (0.1 if vr>1.5 else 0) + (0.05 if vol<0.01 else -0.05 if vol>0.03 else 0)
        return max(0, min(1, s))
    def generate_signal(self, df):
        cp = self._ok_for_signal(df); 
        if cp is None: return None
        p = self.simulate_forest_prediction(df); self.predictions.append(p)
        if len(self.predictions) < 3: return None
        a = float(np.mean(self.predictions))
        return {'action':'BUY','price':cp,'confidence':a} if a>0.65 else ({'action':'SELL','price':cp,'confidence':1-a} if a<0.35 else None)

class XGBoostBot(BaseAIBot):
    def __init__(self): super().__init__('XGBOOST'); self.return_predictions = deque(maxlen=5)
    def predict_return(self, df):
        if len(df) < 50: return 0
        feats = [df['close'].pct_change(l).iloc[-1] for l in [1,2,5,10,20]]
        ma20, ma50, vma20 = df['close'].rolling(20).mean().iloc[-1], df['close'].rolling(50).mean().iloc[-1], df['volume'].rolling(20).mean().iloc[-1]
        feats += [df['close'].iloc[-1]/(ma20 or df['close'].iloc[-1]) - 1, df['close'].iloc[-1]/(ma50 or df['close'].iloc[-1]) - 1, df['volume'].iloc[-1]/(vma20 or df['volume'].iloc[-1])]
        bb_std = df['close'].rolling(20).std().iloc[-1]
        feats.append((df['close'].iloc[-1]-ma20)/(2*bb_std) if bb_std and ma20 else 0)
        w = [0.2,0.15,0.15,0.1,0.05,0.1,0.05,0.1,0.1]
        pr = sum(f*wi for f,wi in zip(feats,w))
        if feats[0]>0 and feats[1]>0: pr*=1.2
        if feats[-1]>0.5: pr*=0.9
        elif feats[-1]<-0.5: pr*=1.1
        return pr
    def generate_signal(self, df):
        cp = self._ok_for_signal(df); 
        if cp is None: return None
        r = self.predict_return(df); self.return_predictions.append(r)
        if len(self.return_predictions) < 3: return None
        a = float(np.mean(self.return_predictions))
        if a>0.002: return {'action':'BUY','price':cp,'confidence':min(0.8,abs(a)*50)}
        if a<-0.002: return {'action':'SELL','price':cp,'confidence':min(0.8,abs(a)*50)}
        return None

class SVMBot(BaseAIBot):
    def __init__(self): super().__init__('SVM'); self.kernel_cache = deque(maxlen=10)
    def svm_decision(self, df):
        if len(df) < 30: return 0
        c,h,l = df['close'].values, df['high'].values, df['low'].values
        ll, hh = np.min(l[-14:]), np.max(h[-14:]); k = ((c[-1]-ll)/(hh-ll)) if hh!=ll else 0.5
        ema12, ema26 = df['close'].ewm(span=12).mean().iloc[-1], df['close'].ewm(span=26).mean().iloc[-1]
        macd = (ema12-ema26)/(c[-1] or 1)
        if len(c) >= 15:
            pc = c[-15:-1]; h14, l14 = h[-14:], l[-14:]
            tr = [max(h14[i]-l14[i], abs(h14[i]-pc[i]), abs(l14[i]-pc[i])) for i in range(14)]
            atr = np.mean(tr)/(c[-1] or 1)
        else: atr = 0.01
        x = np.array([k, macd, atr]); sv = np.array([[0.3,0.01,0.02],[0.7,-0.01,0.015],[0.5,0.0,0.025]]); a = np.array([1,-1,0.5])
        return float(sum(al*np.exp(-0.1*np.sum((x-s)**2)) for s,al in zip(sv,a)))
    def generate_signal(self, df):
        cp = self._ok_for_signal(df); 
        if cp is None: return None
        d = self.svm_decision(df); self.kernel_cache.append(d)
        if len(self.kernel_cache) < 3: return None
        a = float(np.mean(self.kernel_cache))
        if a>0.1: return {'action':'BUY','price':cp,'confidence':min(0.8,abs(a))}
        if a<-0.1: return {'action':'SELL','price':cp,'confidence':min(0.8,abs(a))}
        return None

class AutoencoderBot(BaseAIBot):
    def __init__(self): super().__init__('AUTOENCODER'); self.reconstruction_errors, self.anomaly_threshold = deque(maxlen=20), 0.02
    def reconstruction_error(self, df):
        if len(df) < 30: return 0
        r = df['close'].pct_change().iloc[-20:].values; v = df['volume'].iloc[-20:].values; rng = (df['high']-df['low']).iloc[-20:].values
        rn = (r-r.mean())/(r.std()+1e-8); vn = (v-v.mean())/(v.std()+1e-8); gn = (rng-rng.mean())/(rng.std()+1e-8)
        enc = np.mean([rn, vn, gn], axis=0)[:8]; rec = np.tile(np.mean(enc), 20)
        return float(np.mean((rn-rec)**2))
    def generate_signal(self, df):
        cp = self._ok_for_signal(df); 
        if cp is None: return None
        e = self.reconstruction_error(df); self.reconstruction_errors.append(e)
        if len(self.reconstruction_errors) < 5: return None
        a = float(np.mean(self.reconstruction_errors)); rr = df['close'].pct_change().iloc[-1]
        if a > self.anomaly_threshold:
            return {'action':'SELL','price':cp,'confidence':min(0.8,a*10)} if rr>0 else {'action':'BUY','price':cp,'confidence':min(0.8,a*10)}
        return None

class DQNBot(BaseAIBot):
    def __init__(self): super().__init__('DQN_RL'); self.epsilon=0.1; self.q_values={'BUY':deque(maxlen=100),'SELL':deque(maxlen=100),'HOLD':deque(maxlen=100)}; self.last_action='HOLD'; self.last_state=None
    def state_of(self, df):
        if len(df) < 20: return None
        s=[df['close'].pct_change().iloc[-1], df['close'].pct_change(5).iloc[-1], df['volume'].iloc[-1]/(df['volume'].rolling(20).mean().iloc[-1] or df['volume'].iloc[-1])]
        d=df['close'].diff(); g=d.where(d>0,0).rolling(14).mean(); l=(-d.where(d<0,0)).rolling(14).mean(); rs=g/(l.replace(0,np.nan)); rsi=(100-100/(1+rs)).iloc[-1]
        s.append(rsi/100); return np.array(s)
    def q(self,a): return 0 if not self.q_values[a] else float(np.mean(self.q_values[a]))
    def generate_signal(self, df):
        cp = self._ok_for_signal(df); 
        if cp is None: return None
        st = self.state_of(df); 
        if st is None: return None
        act = np.random.choice(['BUY','SELL','HOLD']) if np.random.random()<self.epsilon else ('BUY' if self.q('BUY')>max(self.q('SELL'),self.q('HOLD')) else ('SELL' if self.q('SELL')>max(self.q('BUY'),self.q('HOLD')) else 'HOLD'))
        if self.last_state is not None and self.last_action!='HOLD':
            r = st[0]*100 if self.last_action=='BUY' else -st[0]*100; self.q_values[self.last_action].append(r)
        self.last_state, self.last_action = st, act
        return {'action':act,'price':cp,'confidence':0.7} if act!='HOLD' else None

class GeneticAlgorithmBot(BaseAIBot):
    def __init__(self): super().__init__('GENETIC_ALGO'); self.population_size=50; self.population=self._init_pop(); self.generation=0; self.best_chromosome=None
    def _init_pop(self):
        return [{'rsi_buy':np.random.uniform(20,40),'rsi_sell':np.random.uniform(60,80),'ma_short':np.random.randint(5,15),'ma_long':np.random.randint(20,50),'volume_threshold':np.random.uniform(1.2,2.0)} for _ in range(self.population_size)]
    def fitness(self, ch, df):
        if len(df)<50: return 0
        d=df['close'].diff(); g=d.where(d>0,0).rolling(14).mean(); l=(-d.where(d<0,0)).rolling(14).mean(); rs=g/(l.replace(0,np.nan)); rsi=(100-100/(1+rs)).iloc[-1]
        ma_s, ma_l = df['close'].rolling(ch['ma_short']).mean().iloc[-1], df['close'].rolling(ch['ma_long']).mean().iloc[-1]
        vma20=df['volume'].rolling(20).mean().iloc[-1]; vr = df['volume'].iloc[-1]/(vma20 or df['volume'].iloc[-1])
        return 1 if (rsi<ch['rsi_buy'] and ma_s>ma_l and vr>ch['volume_threshold']) else (-1 if (rsi>ch['rsi_sell'] and ma_s<ma_l) else 0)
    def generate_signal(self, df):
        cp = self._ok_for_signal(df); 
        if cp is None: return None
        scores=[self.fitness(c,df) for c in self.population]; i=int(np.argmax(np.abs(scores))); self.best_chromosome=self.population[i]; best=scores[i]; self.generation+=1
        if self.generation%10==0: self.evolve(scores)
        return {'action':'BUY','price':cp,'confidence':0.75} if best>0 else ({'action':'SELL','price':cp,'confidence':0.75} if best<0 else None)
    def evolve(self, fit):
        idx=list(np.argsort(np.abs(fit)))[-self.population_size//2:]; parents=[self.population[i] for i in idx]; new=parents.copy()
        while len(new)<self.population_size:
            p1,p2=np.random.choice(parents,2,replace=False); ch={}
            for k in p1:
                v = p1[k] if np.random.random()>0.5 else p2[k]
                if np.random.random()<0.1:
                    if 'rsi' in k: v += np.random.uniform(-5,5)
                    elif 'ma' in k: v = max(1, int(v + np.random.randint(-2,3)))
                    else: v += np.random.uniform(-0.1,0.1)
                ch[k]=v
            new.append(ch)
        self.population=new

# ---------------- Launcher ----------------
async def run_single_bot(bot_class, name):
    print(f"\n🚀 Starting {name}..."); bot = bot_class(); await bot.run()

async def run_all_bots_parallel():
    print("\n🚀 Starting ALL 10 strategies with optimized crypto pairs...\n" + "="*70)
    strategies=[(LSTMPredictionBot,'LSTM → BTC'),(GRUTrendBot,'GRU → ETH'),(CNNPatternBot,'CNN → DOGE'),(TransformerBot,'Transformer → BTC'),(RandomForestBot,'RandomForest → LTC'),(XGBoostBot,'XGBoost → XRP'),(SVMBot,'SVM → ADA'),(AutoencoderBot,'Autoencoder → SHIB'),(DQNBot,'DQN → SOL'),(GeneticAlgorithmBot,'GeneticAlgo → MATIC')]
    print("\n📊 Strategy-Pair Assignments:\n"+"-"*70)
    for bc, d in strategies:
        bn = bc.__name__.replace('Bot','').replace('Prediction','').replace('Algorithm','')
        pi = STRATEGY_PAIRS.get(bn.upper(), STRATEGY_PAIRS.get(f"{bn.upper()}_PREDICTION", {})) or next((STRATEGY_PAIRS[k] for k in STRATEGY_PAIRS if bn.upper() in k), {})
        if pi: print(f"  • {d:20} → {pi['symbol']:10} ({pi['name']})")
    print("-"*70)
    tasks=[]
    for bc, name in strategies:
        print(f"  • Launching {name}..."); b=bc(); b.trade_cooldown, b.min_price_change_pct = 60, 0.2; tasks.append(asyncio.create_task(b.run())); await asyncio.sleep(2)
    print(f"\n✅ All {len(tasks)} bots running with optimized pairs!\nPress Ctrl+C to stop all bots\n")
    try: await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        print("\n🛑 Stopping all bots..."); [t.cancel() for t in tasks]; await asyncio.gather(*tasks, return_exceptions=True); print('✅ All bots stopped')

async def main():
    print('🤖 AI Trading Bot Launcher - Multi-Pair Edition')
    print('='*70)
    print('Select option:\n0. START ALL STRATEGIES (10 bots, 10 different pairs)\n' + '-'*70)
    print('Individual strategies with optimized pairs:')
    for i,s in enumerate(['LSTM Price Prediction → BTCUSDT (Bitcoin)','GRU Trend Classifier → ETHUSDT (Ethereum)','CNN Candle Pattern → DOGEUSDT (Dogecoin)','Transformer Forecasting → BTCUSDT (Bitcoin)','Random Forest Classifier → LTCUSDT (Litecoin)','XGBoost Regression → XRPUSDT (Ripple)','SVM Classifier → ADAUSDT (Cardano)','Autoencoder Anomaly → SHIBUSDT (Shiba Inu)','DQN Reinforcement → SOLUSDT (Solana)','Genetic Algorithm → MATICUSDT (Polygon)'],1): print(f"{i}. {s}")
    print('='*70)
    ch = input('Enter choice (0 for ALL, 1-10 for single): ').strip()
    if ch=='0': await run_all_bots_parallel(); return
    mp={'1':(LSTMPredictionBot,'LSTM Price Prediction'),'2':(GRUTrendBot,'GRU Trend Classifier'),'3':(CNNPatternBot,'CNN Candle Pattern'),'4':(TransformerBot,'Transformer Forecasting'),'5':(RandomForestBot,'Random Forest'),'6':(XGBoostBot,'XGBoost Regression'),'7':(SVMBot,'SVM Classifier'),'8':(AutoencoderBot,'Autoencoder Anomaly'),'9':(DQNBot,'DQN Reinforcement Learning'),'10':(GeneticAlgorithmBot,'Genetic Algorithm')}
    if ch in mp: bc,n=mp[ch]; await run_single_bot(bc,n)
    else: print('❌ Invalid choice')

if __name__ == '__main__':
    import sys
    if any(a in sys.argv for a in ('--all','-a')):
        async def auto_start(): print('\n🤖 AUTO-START: Launching all strategies with optimized pairs...'); await run_all_bots_parallel()
        asyncio.run(auto_start())
    else:
        asyncio.run(main())
