#!/usr/bin/env python3
import os, sys, time, threading, logging
from datetime import datetime
import numpy as np
logging.basicConfig(level=logging.CRITICAL)
for logger in ['werkzeug', 'dash']: logging.getLogger(logger).setLevel(logging.CRITICAL)

import dash
from dash import dcc, html, Input, Output, dash_table, no_update, State, callback_context
import plotly.graph_objects as go
import pandas as pd
from dotenv import load_dotenv
from pybit.unified_trading import HTTP
import yfinance as yf

load_dotenv(override=True)

class BybitDataProvider:
    def __init__(self, demo=True):
        self.demo, prefix = demo, "TESTNET_" if demo else "LIVE_"
        key, secret = os.getenv(f"{prefix}BYBIT_API_KEY", ""), os.getenv(f"{prefix}BYBIT_API_SECRET", "")
        if not (key and secret): raise ValueError(f"Missing {prefix} API credentials")
        self.ex = HTTP(demo=demo, api_key=key, api_secret=secret)
        if self.ex.get_server_time().get('retCode'): raise ConnectionError("Connection failed")
        print(f"✅ {'Testnet' if demo else 'Live'} connected")
        self.data = {"positions": [], "account": {}, "last_update": None}
        self.market = {"SPY": {}, "BTC": {}}
        self.indicators = {
            "trend": "ANALYZING", "trend_color": "#8892b0", 
            "volatility": "CHECKING", "vol_color": "#8892b0", 
            "atr_pct": 0, "adx": 0, "ema_slope": 0
        }
        self.equity0, self.reset_date, self.last_call = None, datetime.now().strftime('%Y-%m-%d'), 0
    
    def sf(self, v, d=0): return float(v) if v not in [None, '', 'null'] else d
    def rl(self): time.sleep(max(0, 0.2 - (time.time() - self.last_call))); self.last_call = time.time()
    
    def fetch_indicators(self):
        """Calculate trend and volatility indicators for BTCUSDT
        - Trend: Based on 200-EMA position and slope
        - Volatility: Based on ATR(14) as percentage of price
        - All calculations use BTCUSDT hourly candles
        """
        try:
            # Get 200 1-hour candles for BTC
            self.rl()
            klines = self.ex.get_kline(category="linear", symbol="BTCUSDT", interval="60", limit=200)
            if klines.get("retCode") != 0 or not klines.get("result", {}).get("list"):
                return
            
            candles = klines["result"]["list"][::-1]  # Reverse to chronological order
            closes = np.array([float(c[4]) for c in candles])
            highs = np.array([float(c[2]) for c in candles])
            lows = np.array([float(c[3]) for c in candles])
            
            current_price = closes[-1]
            
            # Calculate 200 EMA
            ema200 = current_price
            if len(closes) >= 200:
                alpha = 2 / (200 + 1)
                ema = closes[0]
                for price in closes[1:]:
                    ema = price * alpha + ema * (1 - alpha)
                ema200 = ema
            
            # Calculate EMA slope (using last 20 periods for comparison)
            ema_slope = 0
            if len(closes) >= 200:
                # Calculate EMA 20 periods ago
                ema_prev = closes[0]
                for price in closes[1:-20]:
                    ema_prev = price * alpha + ema_prev * (1 - alpha)
                # Slope as percentage change
                ema_slope = ((ema200 - ema_prev) / ema_prev) * 100
            
            # Calculate ATR(14)
            tr_list = []
            for i in range(1, min(15, len(closes))):
                high_low = highs[i] - lows[i]
                high_close = abs(highs[i] - closes[i-1])
                low_close = abs(lows[i] - closes[i-1])
                tr = max(high_low, high_close, low_close)
                tr_list.append(tr)
            
            atr = np.mean(tr_list) if tr_list else 0
            atr_pct = (atr / current_price * 100) if current_price > 0 else 0
            
            # Calculate ADX(14) - simplified version
            adx = 0
            if len(highs) > 14 and len(lows) > 14:
                # Calculate directional movements
                plus_dm = []
                minus_dm = []
                for i in range(1, min(15, len(highs))):
                    high_diff = highs[i] - highs[i-1]
                    low_diff = lows[i-1] - lows[i]
                    
                    if high_diff > low_diff and high_diff > 0:
                        plus_dm.append(high_diff)
                    else:
                        plus_dm.append(0)
                    
                    if low_diff > high_diff and low_diff > 0:
                        minus_dm.append(low_diff)
                    else:
                        minus_dm.append(0)
                
                if tr_list:
                    atr_val = np.mean(tr_list)
                    plus_di = 100 * np.mean(plus_dm) / atr_val if atr_val > 0 else 0
                    minus_di = 100 * np.mean(minus_dm) / atr_val if atr_val > 0 else 0
                    
                    if (plus_di + minus_di) > 0:
                        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
                        adx = dx
            
            # Determine trend based on 200-EMA (for BTC trading)
            if current_price > ema200 and ema_slope > 0.1:  # Price above EMA and slope is positive
                trend = "LONG ONLY"
                trend_color = "#00d4aa"
            elif current_price < ema200 and ema_slope < -0.1:  # Price below EMA and slope is negative
                trend = "SHORT ONLY"
                trend_color = "#f6465d"
            elif adx < 20:  # Low ADX indicates ranging market
                trend = "RANGE"
                trend_color = "#ffa500"
            else:
                trend = "NEUTRAL"
                trend_color = "#8892b0"
            
            # Determine volatility status based on ATR% (for BTC trading)
            if atr_pct < 0.6:
                volatility = "TOO LOW"
                vol_color = "#f6465d"
            elif atr_pct > 2.5:
                volatility = "TOO HIGH"
                vol_color = "#f6465d"
            else:
                volatility = "TRADABLE"
                vol_color = "#00d4aa"
            
            self.indicators = {
                "trend": trend,
                "trend_color": trend_color,
                "volatility": volatility,
                "vol_color": vol_color,
                "atr_pct": atr_pct,
                "adx": adx,
                "ema_slope": ema_slope
            }
            
        except Exception as e:
            print(f"Indicators error: {e}")
            self.indicators = {
                "trend": "ERROR", "trend_color": "#f6465d",
                "volatility": "ERROR", "vol_color": "#f6465d",
                "atr_pct": 0, "adx": 0, "ema_slope": 0
            }
    
    def fees(self, sym, sz, ap):
        try:
            self.rl(); orders = self.ex.get_order_history(category="linear", symbol=sym, orderStatus="Filled", limit=50)
            if orders.get("retCode"): return sz * ap * 0.0002
            fees, covered = 0, 0
            for o in orders.get("result", {}).get("list", [])[:50]:
                if covered >= sz: break
                oq, of = self.sf(o.get("cumExecQty")), self.sf(o.get("cumExecFee"))
                qty = min(oq, sz - covered)
                if oq > 0: fees += abs(of * qty / oq); covered += qty
            return fees + (sz - covered) * ap * 0.0002
        except: return sz * ap * 0.0002
    
    def liquidate_position(self, symbol, side, size):
        try:
            self.rl()
            pos_resp = self.ex.get_positions(category="linear", symbol=symbol)
            if pos_resp.get("retCode") == 0:
                positions = pos_resp.get("result", {}).get("list", [])
                if positions and self.sf(positions[0].get("size")) > 0:
                    actual_size = self.sf(positions[0].get("size"))
                    actual_side = positions[0].get("side")
                    mark_price = self.sf(positions[0].get("markPrice"))
                    close_side = "Sell" if actual_side == "Buy" else "Buy"
                    
                    # Post-only limit order: place slightly away from mark price
                    offset = mark_price * 0.001  # 0.1% offset
                    limit_price = mark_price + offset if actual_side == "Buy" else mark_price - offset
                    
                    # Get tick size for proper price rounding
                    self.rl()
                    instruments = self.ex.get_instruments_info(category="linear", symbol=symbol)
                    tick_size = 0.01
                    if instruments.get("retCode") == 0:
                        tick_size = float(instruments.get("result", {}).get("list", [{}])[0].get("priceFilter", {}).get("tickSize", 0.01))
                    
                    # Round price to tick size
                    limit_price = round(limit_price / tick_size) * tick_size
                    
                    self.rl()
                    order = self.ex.place_order(
                        category="linear", symbol=symbol, side=close_side,
                        orderType="Limit", qty=str(actual_size), price=str(limit_price),
                        reduceOnly=True, timeInForce="PostOnly"
                    )
                    if order.get("retCode") == 0: return True, "Limit order placed"
                    else: return False, f"Error: {order.get('retMsg', 'Unknown')}"
                else: return False, "Position not found"
            else: return False, f"Failed: {pos_resp.get('retMsg', 'Unknown')}"
        except Exception as e: return False, f"Exception: {str(e)}"
    
    def close_all_limit(self):
        try:
            self.rl()
            pos_resp = self.ex.get_positions(category="linear", settleCoin="USDT")
            if pos_resp.get("retCode") != 0: return False, "Failed to fetch"
            
            results = []
            for p in pos_resp.get("result", {}).get("list", []):
                if not (size := self.sf(p.get("size"))): continue
                
                self.rl()
                symbol = p.get("symbol")
                side = p.get("side")
                mark_price = self.sf(p.get("markPrice"))
                close_side = "Sell" if side == "Buy" else "Buy"
                offset = mark_price * 0.001
                limit_price = mark_price + offset if side == "Buy" else mark_price - offset
                
                self.rl()
                instruments = self.ex.get_instruments_info(category="linear", symbol=symbol)
                tick_size = 0.01
                if instruments.get("retCode") == 0:
                    tick_size = float(instruments.get("result", {}).get("list", [{}])[0].get("priceFilter", {}).get("tickSize", 0.01))
                
                limit_price = round(limit_price / tick_size) * tick_size
                
                order = self.ex.place_order(
                    category="linear", symbol=symbol, side=close_side,
                    orderType="Limit", qty=str(size), price=str(limit_price),
                    reduceOnly=True, timeInForce="PostOnly"
                )
                
                if order.get("retCode") == 0: results.append(f"✅ {symbol}")
                else: results.append(f"❌ {symbol}")
            
            return True if results else False, ", ".join(results) if results else "No positions"
        except Exception as e: return False, f"Exception: {str(e)}"
    
    def fetch_market(self, period='24h'):
        try:
            spy = yf.Ticker("SPY").history(period='2d', interval='1d')
            if len(spy) >= 2:
                curr, prev = spy['Close'].iloc[-1], spy['Close'].iloc[-2]
                self.market["SPY"] = {"price": curr, "change_pct": ((curr - prev) / prev) * 100}
        except: self.market["SPY"] = {"price": 0, "change_pct": 0}
        try:
            self.rl(); btc = self.ex.get_tickers(category="linear", symbol="BTCUSDT")
            if not btc.get("retCode") and btc.get("result", {}).get("list"):
                b = btc["result"]["list"][0]
                self.market["BTC"] = {"price": self.sf(b.get("lastPrice")), "change_pct": self.sf(b.get("price24hPcnt")) * 100}
        except: self.market["BTC"] = {"price": 0, "change_pct": 0}
        self.market["update"] = datetime.now().strftime('%H:%M:%S')
    
    def fetch(self):
        try:
            date = datetime.now().strftime('%Y-%m-%d')
            if self.reset_date != date: self.equity0, self.reset_date = None, date
            
            self.rl(); pos_resp = self.ex.get_positions(category="linear", settleCoin="USDT")
            positions = []
            if not pos_resp.get("retCode"):
                for p in pos_resp.get("result", {}).get("list", []):
                    if not (sz := self.sf(p.get("size"))): continue
                    sym, side, ap, mp = p.get("symbol", ""), p.get("side", ""), self.sf(p.get("avgPrice")), self.sf(p.get("markPrice"))
                    lev = self.sf(p.get("leverage", 1))
                    upnl = self.sf(p.get("unrealisedPnl")) or ((mp - ap) if side == "Buy" else (ap - mp)) * sz
                    fee = self.fees(sym, sz, ap)
                    val = ap * sz
                    be_fee = (fee + sz * mp * 0.0002) / sz if sz else 0
                    positions.append({
                        "symbol": sym, "side": side, "size": sz, "avg_price": ap, "mark_price": mp,
                        "pnl": upnl, "fees": fee, "net_pnl": upnl - fee,
                        "breakeven": ap + (be_fee if side == "Buy" else -be_fee),
                        "pnl_pct": (upnl / val * 100) if val else 0,
                        "value": sz * mp, "liq_price": self.sf(p.get("liqPrice")), "leverage": lev
                    })
            
            self.rl(); acc = self.ex.get_wallet_balance(accountType="UNIFIED", coin="USDT")
            account, equity = {}, 0
            if not acc.get("retCode"):
                for c in acc.get("result", {}).get("list", [{}])[0].get("coin", []):
                    if c.get("coin") == "USDT":
                        equity = self.sf(c.get("equity"))
                        account = {"equity": equity, "available": self.sf(c.get("availableToWithdraw")), "unrealized_pnl": self.sf(c.get("unrealisedPnl"))}
                        break
            
            if self.equity0 is None and equity: self.equity0 = equity
            pos_pnl = sum(p["net_pnl"] for p in positions)
            self.data = {
                "positions": positions, "account": account, "positions_pnl": pos_pnl,
                "positions_pct": (pos_pnl / equity * 100) if equity else 0,
                "daily_change": equity - (self.equity0 or equity),
                "daily_change_pct": ((equity - (self.equity0 or equity)) / (self.equity0 or 1)) * 100,
                "starting_equity": self.equity0, "position_count": len(positions),
                "last_update": datetime.now().strftime('%H:%M:%S'), "demo_mode": self.demo
            }
        except Exception as e: 
            if self.data: self.data["last_update"] = f"ERROR: {datetime.now().strftime('%H:%M:%S')}"

prov = BybitDataProvider(demo=True)
prov.fetch_indicators()  # Initialize indicators on startup

def loop():
    counter = 0
    while True:
        try: 
            prov.fetch()
            prov.fetch_market() if not prov.market.get("update") else None
            if counter % 10 == 0:  # Fetch indicators every 30 seconds
                prov.fetch_indicators()
            counter += 1
            time.sleep(3)
        except: time.sleep(10)

threading.Thread(target=loop, daemon=True).start()

app = dash.Dash(__name__, suppress_callback_exceptions=True, meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1.0'}])
app.title = "Bybit Dashboard"

def col(v): return '#00d4aa' if v >= 0 else '#f6465d'

app.layout = html.Div([
    dcc.Interval(id='int', interval=3000),
    html.Div([
        html.H1("📈 Bybit", className="hd"),
        html.Div(id="mode", className="md")
    ], className="top"),
    
    html.Div([
        html.Div([html.Div("Equity", className="sl"), html.Div(id="eq", className="sv")], className="sc"),
        html.Div([html.Div("Available", className="sl"), html.Div(id="av", className="sv")], className="sc"),
        html.Div([html.Div("PnL", className="sl"), html.Div(id="pnl", className="sv")], className="sc"),
        html.Div([html.Div("Daily", className="sl"), html.Div(id="day", className="sv")], className="sc")
    ], className="stats"),
    
    html.Div([
        html.Div([html.Span("SPY", className="ml"), html.Span(id="spy", className="mv")], className="mi"),
        html.Div([html.Span("BTC", className="ml"), html.Span(id="btc", className="mv")], className="mi")
    ], className="mkt"),
    
    html.Div([
        html.Div([
            html.Div("📊 BTC TREND", className="ind-label"),
            html.Div(id="trend", className="ind-value")
        ], className="indicator"),
        html.Div([
            html.Div("📈 BTC VOLATILITY", className="ind-label"),
            html.Div(id="volatility", className="ind-value")
        ], className="indicator")
    ], className="indicators"),
    
    html.Button("🚫 Close All (Limit)", id="cls", className="btn", title="Close all positions with limit orders post-only"),
    
    html.Div(id="pos"),
    
    dcc.Graph(id="chart", config={'displayModeBar': False}),
    
    html.Div(id="msg", className="msg"),
    html.Div(id="ft", className="ft")
], className="app")

@app.callback(
    [Output('mode', 'children'), Output('eq', 'children'), Output('av', 'children'), 
     Output('pnl', 'children'), Output('day', 'children'), Output('spy', 'children'),
     Output('btc', 'children'), Output('trend', 'children'), Output('volatility', 'children'),
     Output('pos', 'children'), Output('chart', 'figure'), Output('ft', 'children')],
    [Input('int', 'n_intervals')]
)
def update(n):
    d, m, ind = prov.data, prov.market, prov.indicators
    acc = d.get("account", {})
    
    eq = f"${acc.get('equity', 0):,.0f}"
    av = f"${acc.get('available', 0):,.0f}"
    pnl = html.Span(f"${acc.get('unrealized_pnl', 0):+.0f}", style={'color': col(acc.get('unrealized_pnl', 0))})
    day = html.Span(f"${d.get('daily_change', 0):+.0f}", style={'color': col(d.get('daily_change', 0))})
    
    spy = html.Span(f"${m.get('SPY', {}).get('price', 0):.0f} ({m.get('SPY', {}).get('change_pct', 0):+.1f}%)", 
                    style={'color': col(m.get('SPY', {}).get('change_pct', 0))})
    btc = html.Span(f"${m.get('BTC', {}).get('price', 0):,.0f} ({m.get('BTC', {}).get('change_pct', 0):+.1f}%)",
                    style={'color': col(m.get('BTC', {}).get('change_pct', 0))})
    
    # Trend indicator with ADX for BTC
    trend = html.Div([
        html.Span(ind.get("trend", "ANALYZING"), style={'color': ind.get("trend_color", "#8892b0"), 'font-weight': 'bold'}),
        html.Span(f" (ADX: {ind.get('adx', 0):.0f})", style={'color': '#8892b0', 'font-size': '11px'})
    ])
    
    # Volatility indicator with ATR% for BTC
    volatility = html.Div([
        html.Span(ind.get("volatility", "CHECKING"), style={'color': ind.get("vol_color", "#8892b0"), 'font-weight': 'bold'}),
        html.Span(f" (ATR: {ind.get('atr_pct', 0):.2f}%)", style={'color': '#8892b0', 'font-size': '11px'})
    ])
    
    pos = d.get("positions", [])
    if not pos:
        table = html.Div("No positions", className="np")
    else:
        rows = []
        for i, p in enumerate(pos):
            rows.append(html.Div([
                html.Div([
                    html.Div([
                        html.Span(p['symbol'], className="ps"),
                        html.Span(p['side'].upper(), className="pd", style={'color': '#00d4aa' if p['side']=='Buy' else '#f6465d'})
                    ], className="pr1"),
                    html.Div([
                        html.Span(f"${p['value']:,.2f} @ {p.get('leverage', 1):.0f}x", className="pz")
                    ], className="pr2")
                ], className="pl"),
                html.Div([
                    html.Div(f"${p['net_pnl']:+.1f}", className="ppnl", style={'color': col(p['net_pnl'])}),
                    html.Button("X", id={'type': 'liq', 'index': i}, className="liq", title="Close with Limit Order Post-Only")
                ], className="pr")
            ], className="row"))
        table = html.Div(rows)
    
    if pos:
        df = pd.DataFrame(pos)
        fig = go.Figure([go.Bar(x=df['symbol'], y=df['net_pnl'], marker_color=[col(p) for p in df['net_pnl']])])
        fig.update_layout(
            height=200, margin=dict(l=0, r=0, t=0, b=30),
            plot_bgcolor='#1a1f3a', paper_bgcolor='#0a0e27',
            font=dict(color='#fff', size=10), showlegend=False
        )
        fig.update_xaxes(gridcolor='#2a3050'); fig.update_yaxes(gridcolor='#2a3050')
    else: 
        fig = go.Figure()
        fig.update_layout(height=200, plot_bgcolor='#1a1f3a', paper_bgcolor='#0a0e27')
    
    ft = f"⏱ {d.get('last_update', 'Never')}"
    mode = "TEST" if d.get("demo_mode") else "LIVE"
    
    return mode, eq, av, pnl, day, spy, btc, trend, volatility, table, fig, ft

@app.callback(
    Output('msg', 'children'),
    [Input('cls', 'n_clicks'), Input({'type': 'liq', 'index': dash.dependencies.ALL}, 'n_clicks')],
    prevent_initial_call=True
)
def handle(n_cls, n_liq):
    ctx = callback_context
    if not ctx.triggered: return no_update
    
    triggered_id = ctx.triggered[0]['prop_id']
    
    if triggered_id == 'cls.n_clicks':
        success, msg = prov.close_all_limit()
        prov.fetch()
        return f"{'✅' if success else '❌'} {msg}"
    
    # Handle liquidation button clicks
    if 'liq' in triggered_id:
        # Find which button was clicked
        for i, clicks in enumerate(n_liq):
            if clicks:
                positions = prov.data.get("positions", [])
                if i < len(positions):
                    p = positions[i]
                    success, msg = prov.liquidate_position(p['symbol'], p['side'], p['size'])
                    prov.fetch()
                    return f"{'✅' if success else '❌'} {p['symbol']}: {msg}"
    
    return no_update

app.index_string = '''<!DOCTYPE html><html><head>{%metas%}<title>{%title%}</title>{%favicon%}{%css%}<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:system-ui;background:#0a0e27;color:#fff;overflow-x:hidden}
.app{padding:10px;max-width:100vw}
.top{background:linear-gradient(135deg,#667eea,#764ba2);padding:15px;border-radius:8px;margin-bottom:10px;display:flex;justify-content:space-between;align-items:center}
.hd{font-size:24px}
.md{background:rgba(0,0,0,0.3);padding:4px 8px;border-radius:4px;font-size:12px;font-weight:bold}
.stats{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:10px}
.sc{background:#1a1f3a;padding:12px;border-radius:8px;border:1px solid #2a3050}
.sl{color:#8892b0;font-size:11px;text-transform:uppercase}
.sv{font-size:18px;font-weight:bold;margin-top:4px}
.mkt{display:flex;gap:10px;margin-bottom:10px}
.mi{background:#1a1f3a;padding:10px;border-radius:8px;flex:1;display:flex;justify-content:space-between;align-items:center}
.ml{color:#8892b0;font-size:12px;text-transform:uppercase}
.mv{font-size:14px;font-weight:bold}
.indicators{display:flex;gap:10px;margin-bottom:10px}
.indicator{background:#1a1f3a;padding:12px;border-radius:8px;flex:1;border:2px solid #2a3050}
.ind-label{color:#8892b0;font-size:11px;text-transform:uppercase;margin-bottom:5px;font-weight:bold}
.ind-value{font-size:14px}
.btn{width:100%;background:#ff9800;color:#fff;border:none;padding:12px;border-radius:8px;font-size:16px;font-weight:bold;margin-bottom:10px;cursor:pointer}
.btn:hover{background:#e68a00}
.row{background:#1a1f3a;padding:10px;border-radius:8px;margin-bottom:8px;display:flex;justify-content:space-between;align-items:center;border:1px solid #2a3050}
.pl{flex:1}
.pr1{display:flex;gap:10px;margin-bottom:4px}
.ps{font-weight:bold;font-size:14px}
.pd{font-size:12px;text-transform:uppercase;font-weight:bold}
.pz{color:#8892b0;font-size:11px}
.pr{display:flex;align-items:center;gap:10px}
.ppnl{font-size:16px;font-weight:bold}
.liq{background:#f6465d;color:#fff;border:none;padding:8px 12px;border-radius:4px;font-size:14px;font-weight:bold;cursor:pointer;min-width:35px}
.liq:hover{background:#d63547}
.np{text-align:center;padding:30px;background:#1a1f3a;border-radius:8px;color:#8892b0}
.msg{padding:10px;border-radius:8px;margin:10px 0;font-size:14px;text-align:center;background:#1a1f3a}
.ft{text-align:center;color:#8892b0;font-size:11px;margin-top:10px}
@media (max-width:500px){
    .stats{grid-template-columns:1fr 1fr}
    .sv{font-size:16px}
    .indicators{flex-direction:column}
}
</style></head><body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body></html>'''

if __name__ == '__main__':
    os.system('clear' if os.name == 'posix' else 'cls')
    print("🚀 Mobile Bybit Dashboard → http://localhost:8050")
    print("📊 With BTC Trend & Volatility Indicators")
    print("✅ Optimized for mobile!\n")
    print("Features:")
    print("  • All position closures use LIMIT orders (Post-Only)")
    print("  • BTC TREND: Based on 200-EMA position and slope")
    print("  • BTC VOLATILITY: ATR(14) as % of BTC price")
    print("  • Indicators calculated from BTCUSDT hourly data")
    print("  • Updates every 30 seconds\n")
    app.run(host='0.0.0.0', port=8050, debug=False)