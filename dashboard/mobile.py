#!/usr/bin/env python3
import os, time, threading, json, logging
from datetime import datetime
import numpy as np, pandas as pd
from dotenv import load_dotenv
from pybit.unified_trading import HTTP
import yfinance as yf
import dash
from dash import dcc, html, Input, Output, State, no_update, callback_context
import dash.dependencies
import plotly.graph_objects as go

logging.basicConfig(level=logging.CRITICAL)
for lg in ['werkzeug','dash']: logging.getLogger(lg).setLevel(logging.CRITICAL)
load_dotenv(override=True)

# ---------- Data Provider ----------
class BybitDataProvider:
    def __init__(self, demo=True):
        self.demo = demo
        pref = "TESTNET_" if demo else "LIVE_"
        key, sec = os.getenv(f"{pref}BYBIT_API_KEY",""), os.getenv(f"{pref}BYBIT_API_SECRET","")
        if not (key and sec): raise ValueError(f"Missing {pref} API credentials")
        self.ex = HTTP(demo=demo, api_key=key, api_secret=sec)
        if self.ex.get_server_time().get('retCode'): raise ConnectionError("Connection failed")
        print(f"✅ {'Testnet' if demo else 'Live'} connected")
        self.data = {"positions": [], "account": {}, "last_update": None, "demo_mode": demo}
        self.market = {"SPY": {}, "BTC": {}}
        self.indicators = {"trend":"ANALYZING","trend_color":"#8892b0","volatility":"CHECKING","vol_color":"#8892b0","atr_pct":0,"adx":0,"ema_slope":0}
        self.equity0, self.reset_date, self.last_call = None, datetime.now().strftime('%Y-%m-%d'), 0
        self.cache = {"tick":{}}
        self.maker_offset_pct = 0.01

    @staticmethod
    def sf(v, d=0): return float(v) if v not in [None,'','null'] else d
    def rl(self): time.sleep(max(0, 0.2 - (time.time() - self.last_call))); self.last_call = time.time()
    
    def tick(self, symbol):
        if symbol in self.cache["tick"]: return self.cache["tick"][symbol]
        self.rl()
        info = self.ex.get_instruments_info(category="linear", symbol=symbol)
        t = float(info.get("result",{}).get("list",[{}])[0].get("priceFilter",{}).get("tickSize",0.01)) if info.get("retCode")==0 else 0.01
        self.cache["tick"][symbol] = t
        return t
    
    def round_to(self, price, tick): return round(price / tick) * tick

    def fetch_indicators(self):
        try:
            self.rl()
            r = self.ex.get_kline(category="linear", symbol="BTCUSDT", interval="60", limit=200)
            if r.get("retCode")!=0 or not r.get("result",{}).get("list"): return
            rows = r["result"]["list"][::-1]
            cls = np.asarray([float(x[4]) for x in rows]); hi = np.asarray([float(x[2]) for x in rows]); lo = np.asarray([float(x[3]) for x in rows])
            px = cls[-1]
            alpha = 2/(200+1); ema = cls[0]
            for p in cls[1:]: ema = p*alpha + ema*(1-alpha)
            ema200 = ema
            ema_prev = cls[0]
            for p in cls[1:-20]: ema_prev = p*alpha + ema_prev*(1-alpha)
            ema_slope = ((ema200-ema_prev)/max(1e-9,abs(ema_prev)))*100 if len(cls)>=200 else 0
            tr = np.maximum(hi[1:]-lo[1:], np.maximum(np.abs(hi[1:]-cls[:-1]), np.abs(lo[1:]-cls[:-1])))
            atr = float(np.mean(tr[-14:])) if len(tr)>=14 else float(np.mean(tr)) if len(tr)>0 else 0
            atr_pct = (atr/px*100) if px>0 else 0
            plus_dm = np.maximum(hi[1:]-hi[:-1], 0)
            minus_dm = np.maximum(lo[:-1]-lo[1:], 0)
            plus_dm[plus_dm < minus_dm] = 0
            minus_dm[minus_dm <= plus_dm] = minus_dm[minus_dm <= plus_dm]
            atr_s = np.mean(tr[-14:]) if len(tr)>=14 else (np.mean(tr) if len(tr)>0 else 0)
            plus_di = 100*np.mean(plus_dm[-14:])/atr_s if atr_s>0 else 0
            minus_di = 100*np.mean(minus_dm[-14:])/atr_s if atr_s>0 else 0
            dx = 100*abs(plus_di-minus_di)/max(1e-9,(plus_di+minus_di))
            adx = float(dx)
            if px>ema200 and ema_slope>0.1: trend, tcol = "LONG ONLY", "#00d4aa"
            elif px<ema200 and ema_slope<-0.1: trend, tcol = "SHORT ONLY", "#f6465d"
            elif adx<20: trend, tcol = "RANGE", "#ffa500"
            else: trend, tcol = "NEUTRAL", "#8892b0"
            if atr_pct<0.6: vol, vcol = "TOO LOW", "#f6465d"
            elif atr_pct>2.5: vol, vcol = "TOO HIGH", "#f6465d"
            else: vol, vcol = "TRADABLE", "#00d4aa"
            self.indicators = {"trend":trend,"trend_color":tcol,"volatility":vol,"vol_color":vcol,"atr_pct":atr_pct,"adx":adx,"ema_slope":ema_slope}
        except Exception as e:
            print("Indicators error:", e)
            self.indicators = {"trend":"ERROR","trend_color":"#f6465d","volatility":"ERROR","vol_color":"#f6465d","atr_pct":0,"adx":0,"ema_slope":0}

    def fees(self, sym, sz, ap):
        try:
            self.rl()
            r = self.ex.get_order_history(category="linear", symbol=sym, orderStatus="Filled", limit=50)
            if r.get("retCode"): return sz*ap*0.0002
            fees, covered = 0.0, 0.0
            for o in r.get("result",{}).get("list",[]):
                if covered>=sz: break
                oq, of = self.sf(o.get("cumExecQty")), self.sf(o.get("cumExecFee"))
                if oq>0:
                    q = min(oq, sz-covered); fees += abs(of*(q/oq)); covered += q
            return fees + max(0.0, sz-covered)*ap*0.0002
        except: return sz*ap*0.0002

    def _postonly_close(self, symbol, side, size, mark_price):
        close_side = "Sell" if side == "Buy" else "Buy"
        if close_side == "Sell":
            offset_mult = 1 + (self.maker_offset_pct / 100.0)
        else:
            offset_mult = 1 - (self.maker_offset_pct / 100.0)
        price = mark_price * offset_mult
        px = self.round_to(price, self.tick(symbol))
        self.rl()
        return self.ex.place_order(
            category="linear",
            symbol=symbol,
            side=close_side,
            orderType="Limit",
            qty=str(size),
            price=str(px),
            reduceOnly=True,
            timeInForce="PostOnly"
        )

    def liquidate_position(self, symbol, side, size):
        try:
            self.rl()
            ticker_res = self.ex.get_tickers(category="linear", symbol=symbol)
            if ticker_res.get("retCode") != 0:
                return False, ticker_res.get("retMsg", "Failed to get ticker")
            ticker_list = ticker_res.get("result", {}).get("list", [])
            if not ticker_list:
                return False, "Ticker not found"
            ticker = ticker_list[0]
            mp = self.sf(ticker.get("markPrice", ticker.get("lastPrice")))
            if mp <= 0:
                return False, "Invalid price"
            od = self._postonly_close(symbol, side, size, mp)
            if od.get("retCode") == 0:
                order_price = self.sf(od.get('result', {}).get('price', mp))
                order_id = od.get('result', {}).get('orderId', 'N/A')
                return True, f"PostOnly Limit @ {order_price:.4f} (ID: {order_id})"
            else:
                return False, od.get("retMsg", "Order failed")
        except Exception as e:
            return False, f"Exception: {e}"

    def close_all_limit(self):
        try:
            self.rl()
            pr = self.ex.get_positions(category="linear", settleCoin="USDT")
            if pr.get("retCode") != 0:
                return False, pr.get("retMsg", "Fetch failed")
            positions = pr.get("result", {}).get("list", [])
            if not positions:
                return False, "No positions to close"
            ok, failed = 0, 0
            msgs = []
            for p in positions:
                sz = self.sf(p.get("size"))
                if not sz:
                    continue
                symbol = p.get("symbol")
                side = p.get("side")
                mp = self.sf(p.get("markPrice"))
                if mp <= 0:
                    msgs.append(f"⚠️ {symbol}: Invalid mark price")
                    continue
                try:
                    r = self._postonly_close(symbol, side, sz, mp)
                    if r.get("retCode") == 0:
                        ok += 1
                        order_price = self.sf(r.get('result', {}).get('price', mp))
                        msgs.append(f"✅ {symbol} @ {order_price:.4f}")
                    else:
                        failed += 1
                        msgs.append(f"❌ {symbol}: {r.get('retMsg', 'Failed')}")
                except Exception as e:
                    failed += 1
                    msgs.append(f"❌ {symbol}: {str(e)}")
            if ok == 0 and failed == 0:
                return False, "No positions found"
            status = f"Placed {ok}/{ok+failed} orders"
            if msgs:
                status += f": {', '.join(msgs[:3])}"
                if len(msgs) > 3:
                    status += f" (+{len(msgs)-3} more)"
            return (ok > 0, status)
        except Exception as e:
            return False, f"Exception: {e}"

    def fetch_market(self):
        try:
            h = yf.Ticker("SPY").history(period='2d', interval='1d')
            if len(h)>=2: c0, c1 = float(h['Close'].iloc[-2]), float(h['Close'].iloc[-1]); self.market["SPY"] = {"price":c1,"change_pct":(c1-c0)/c0*100}
        except: self.market["SPY"]={"price":0,"change_pct":0}
        try:
            self.rl(); r = self.ex.get_tickers(category="linear", symbol="BTCUSDT")
            if r.get("retCode")==0 and r.get("result",{}).get("list"):
                b = r["result"]["list"][0]; self.market["BTC"]={"price":self.sf(b.get("lastPrice")), "change_pct": self.sf(b.get("price24hPcnt"))*100}
        except: self.market["BTC"]={"price":0,"change_pct":0}
        self.market["update"] = datetime.now().strftime('%H:%M:%S')

    def fetch(self):
        try:
            today = datetime.now().strftime('%Y-%m-%d')
            if self.reset_date!=today: self.equity0, self.reset_date = None, today
            self.rl(); pr = self.ex.get_positions(category="linear", settleCoin="USDT")
            pos = []
            if pr.get("retCode")==0:
                for p in pr.get("result",{}).get("list",[]):
                    sz = self.sf(p.get("size")); 
                    if not sz: continue
                    sym, side = p.get("symbol",""), p.get("side","")
                    ap, mp = self.sf(p.get("avgPrice")), self.sf(p.get("markPrice"))
                    lev = self.sf(p.get("leverage",1))
                    upnl = self.sf(p.get("unrealisedPnl")) or ((mp-ap) if side=="Buy" else (ap-mp))*sz
                    fee = self.fees(sym, sz, ap); val = ap*sz
                    be_f = (fee + sz*mp*0.0002)/max(1e-9,sz)
                    pos.append({"symbol":sym,"side":side,"size":sz,"avg_price":ap,"mark_price":mp,"pnl":upnl,"fees":fee,
                                "net_pnl":upnl-fee,"breakeven": ap + (be_f if side=="Buy" else -be_f),
                                "pnl_pct": (upnl/max(1e-9,val))*100, "value": sz*mp, "liq_price": self.sf(p.get("liqPrice")), "leverage": lev})
            self.rl(); wb = self.ex.get_wallet_balance(accountType="UNIFIED", coin="USDT")
            acct, eq = {}, 0.0
            if wb.get("retCode")==0:
                for c in wb.get("result",{}).get("list",[{}])[0].get("coin",[]):
                    if c.get("coin")=="USDT":
                        eq = self.sf(c.get("equity")); acct={"equity":eq,"available":self.sf(c.get("availableToWithdraw")),"unrealized_pnl":self.sf(c.get("unrealisedPnl"))}; break
            if self.equity0 is None and eq: self.equity0 = eq
            pos_pnl = sum(p["net_pnl"] for p in pos)
            self.data = {"positions":pos, "account":acct, "positions_pnl":pos_pnl, "positions_pct": (pos_pnl/max(1e-9,eq))*100 if eq else 0,
                         "daily_change": eq - (self.equity0 or eq), "daily_change_pct": ((eq-(self.equity0 or eq))/max(1.0,self.equity0 or 1))*100,
                         "starting_equity": self.equity0, "position_count": len(pos), "last_update": datetime.now().strftime('%H:%M:%S'),
                         "demo_mode": self.demo}
        except Exception as e:
            print("Fetch error:", e)
            if self.data: self.data["last_update"] = f"ERROR: {datetime.now().strftime('%H:%M:%S')}"

# ---------- Runtime loop ----------
prov = BybitDataProvider(demo=True)
prov.fetch_indicators()
def loop():
    k=0
    while True:
        try:
            prov.fetch()
            if not prov.market.get("update"): prov.fetch_market()
            if k%10==0: prov.fetch_indicators()
            k+=1; time.sleep(3)
        except: time.sleep(10)
threading.Thread(target=loop, daemon=True).start()

# ---------- Dash App ----------
app = dash.Dash(__name__, suppress_callback_exceptions=True, meta_tags=[{'name':'viewport','content':'width=device-width, initial-scale=1.0'}])
app.title = "Bybit Dashboard"
col = lambda v: '#00d4aa' if v>=0 else '#f6465d'

app.layout = html.Div([
    dcc.Interval(id='int', interval=3000),
    html.Div([html.H1("📈 Bybit", style={'margin':'0'}), html.Div(id="mode", style={'background':'rgba(0,0,0,.3)','padding':'4px 8px','border-radius':'4px','font-size':'12px','font-weight':'bold'})], style={'background':'linear-gradient(135deg,#667eea,#764ba2)','padding':'15px','border-radius':'8px','margin-bottom':'10px','display':'flex','justify-content':'space-between','align-items':'center'}),
    html.Div([
        html.Div([html.Div("Equity", style={'color':'#8892b0','font-size':'11px','text-transform':'uppercase'}), html.Div(id="eq", style={'font-size':'18px','font-weight':'bold','margin-top':'4px'})], style={'background':'#1a1f3a','padding':'12px','border-radius':'8px','border':'1px solid #2a3050'}),
        html.Div([html.Div("Available", style={'color':'#8892b0','font-size':'11px','text-transform':'uppercase'}), html.Div(id="av", style={'font-size':'18px','font-weight':'bold','margin-top':'4px'})], style={'background':'#1a1f3a','padding':'12px','border-radius':'8px','border':'1px solid #2a3050'}),
        html.Div([html.Div("PnL", style={'color':'#8892b0','font-size':'11px','text-transform':'uppercase'}), html.Div(id="pnl", style={'font-size':'18px','font-weight':'bold','margin-top':'4px'})], style={'background':'#1a1f3a','padding':'12px','border-radius':'8px','border':'1px solid #2a3050'}),
        html.Div([html.Div("Daily", style={'color':'#8892b0','font-size':'11px','text-transform':'uppercase'}), html.Div(id="day", style={'font-size':'18px','font-weight':'bold','margin-top':'4px'})], style={'background':'#1a1f3a','padding':'12px','border-radius':'8px','border':'1px solid #2a3050'})
    ], style={'display':'grid','grid-template-columns':'1fr 1fr','gap':'8px','margin-bottom':'10px'}),
    html.Div([
        html.Div([html.Span("SPY", style={'color':'#8892b0','font-size':'12px','text-transform':'uppercase'}), html.Span(id="spy", style={'font-size':'14px','font-weight':'bold'})], style={'background':'#1a1f3a','padding':'10px','border-radius':'8px','flex':'1','display':'flex','justify-content':'space-between','align-items':'center'}),
        html.Div([html.Span("BTC", style={'color':'#8892b0','font-size':'12px','text-transform':'uppercase'}), html.Span(id="btc", style={'font-size':'14px','font-weight':'bold'})], style={'background':'#1a1f3a','padding':'10px','border-radius':'8px','flex':'1','display':'flex','justify-content':'space-between','align-items':'center'})
    ], style={'display':'flex','gap':'10px','margin-bottom':'10px'}),
    html.Div([
        html.Div([html.Div("📊 BTC TREND", style={'color':'#8892b0','font-size':'11px','text-transform':'uppercase','margin-bottom':'5px','font-weight':'bold'}), html.Div(id="trend", style={'font-size':'14px'})], style={'background':'#1a1f3a','padding':'12px','border-radius':'8px','flex':'1','border':'2px solid #2a3050'}),
        html.Div([html.Div("📈 BTC VOLATILITY", style={'color':'#8892b0','font-size':'11px','text-transform':'uppercase','margin-bottom':'5px','font-weight':'bold'}), html.Div(id="volatility", style={'font-size':'14px'})], style={'background':'#1a1f3a','padding':'12px','border-radius':'8px','flex':'1','border':'2px solid #2a3050'})
    ], style={'display':'flex','gap':'10px','margin-bottom':'10px'}),
    html.Button("🚫 Close All (PostOnly)", id="cls", style={'width':'100%','background':'#ff9800','color':'#fff','border':'none','padding':'15px','border-radius':'8px','font-size':'16px','font-weight':'bold','margin-bottom':'10px','cursor':'pointer'}, title="Close all positions with PostOnly limit orders"),
    html.Div(id="pos"),
    dcc.Graph(id="chart", config={'displayModeBar': False}),
    html.Div(id="msg", style={'padding':'10px','border-radius':'8px','margin':'10px 0','font-size':'14px','text-align':'center','background':'#1a1f3a','min-height':'20px','border':'1px solid #2a3050'}),
    html.Div(id="ft", style={'text-align':'center','color':'#8892b0','font-size':'11px','margin-top':'10px'})
], style={'padding':'10px','max-width':'100vw','font-family':'system-ui','background':'#0a0e27','color':'#fff','min-height':'100vh'})

@app.callback(
    [Output('mode','children'),Output('eq','children'),Output('av','children'),
     Output('pnl','children'),Output('day','children'),Output('spy','children'),
     Output('btc','children'),Output('trend','children'),Output('volatility','children'),
     Output('pos','children'),Output('chart','figure'),Output('ft','children')],
    [Input('int','n_intervals')]
)
def update(_):
    d, m, ind = prov.data, prov.market, prov.indicators
    acc = d.get("account", {})
    eq = f"${acc.get('equity',0):,.0f}"
    av = f"${acc.get('available',0):,.0f}"
    pnl = html.Span(f"${acc.get('unrealized_pnl',0):+.0f}", style={'color': col(acc.get('unrealized_pnl',0))})
    day = html.Span(f"${d.get('daily_change',0):+.0f}", style={'color': col(d.get('daily_change',0))})
    spy = html.Span(f"${m.get('SPY',{}).get('price',0):.0f} ({m.get('SPY',{}).get('change_pct',0):+.1f}%)", style={'color': col(m.get('SPY',{}).get('change_pct',0))})
    btc = html.Span(f"${m.get('BTC',{}).get('price',0):,.0f} ({m.get('BTC',{}).get('change_pct',0):+.1f}%)", style={'color': col(m.get('BTC',{}).get('change_pct',0))})
    trend = html.Span([
        html.Span(ind.get("trend","ANALYZING"), style={'color': ind.get("trend_color","#8892b0"), 'font-weight':'bold'}),
        html.Span(f" (ADX: {ind.get('adx',0):.0f})", style={'color':'#8892b0','font-size':'11px'})
    ])
    volatility = html.Span([
        html.Span(ind.get("volatility","CHECKING"), style={'color': ind.get("vol_color","#8892b0"), 'font-weight':'bold'}),
        html.Span(f" (ATR: {ind.get('atr_pct',0):.2f}%)", style={'color':'#8892b0','font-size':'11px'})
    ])

    pos = d.get("positions", [])
    if pos:
        rows = [html.Div([
                    html.Div([
                        html.Div([html.Span(p['symbol'], style={'font-weight':'bold','font-size':'14px'}),
                                  html.Span(f"${p['net_pnl']:+.2f}", style={'font-size':'16px','font-weight':'bold','color': col(p['net_pnl'])})], style={'display':'flex','gap':'10px','margin-bottom':'4px'}),
                        html.Div([html.Span(f"${p['value']:,.2f} @ {p.get('leverage',1):.0f}x", style={'color':'#8892b0','font-size':'11px'}),
                                  html.Span(f"({p['pnl_pct']:+.1f}%)", style={'font-size':'12px','font-weight':'bold','color': col(p['pnl_pct'])})], style={'display':'flex','gap':'10px'})
                    ], style={'flex':'1'}),
                    html.Button("X", id={'type':'liq','index':i}, style={'background':'#f6465d','color':'#fff','border':'none','padding':'8px 14px','border-radius':'4px','font-size':'16px','font-weight':'bold','cursor':'pointer','min-width':'40px'}, title="Close with PostOnly Limit Order")
                ], style={'background':'#1a1f3a','padding':'10px','border-radius':'8px','margin-bottom':'8px','display':'flex','justify-content':'space-between','align-items':'center','border':'1px solid #2a3050'}) for i,p in enumerate(pos)]
        table = html.Div(rows)
        df = pd.DataFrame(pos)
        fig = go.Figure([go.Bar(x=df['symbol'], y=df['net_pnl'], marker_color=[col(v) for v in df['net_pnl']])])
        fig.update_layout(height=200, margin=dict(l=0,r=0,t=0,b=30), plot_bgcolor='#1a1f3a', paper_bgcolor='#0a0e27', font=dict(color='#fff',size=10), showlegend=False)
        fig.update_xaxes(gridcolor='#2a3050'); fig.update_yaxes(gridcolor='#2a3050')
    else:
        table = html.Div("No positions", style={'text-align':'center','padding':'30px','background':'#1a1f3a','border-radius':'8px','color':'#8892b0'})
        fig = go.Figure(); fig.update_layout(height=200, plot_bgcolor='#1a1f3a', paper_bgcolor='#0a0e27')

    ft = f"⏱ {d.get('last_update','Never')}"
    mode = "TEST" if d.get("demo_mode") else "LIVE"
    return mode, eq, av, pnl, day, spy, btc, trend, volatility, table, fig, ft

@app.callback(
    Output('msg','children'),
    [Input('cls','n_clicks'), Input({'type':'liq','index':dash.dependencies.ALL}, 'n_clicks')],
    [State({'type':'liq','index':dash.dependencies.ALL}, 'id')],
    prevent_initial_call=True
)
def handle(n_cls, n_liq, ids):
    ctx = callback_context
    if not ctx.triggered: return no_update
    trig = ctx.triggered[0]['prop_id']; val = ctx.triggered[0]['value']
    if trig=='cls.n_clicks' and val:
        ok, msg = prov.close_all_limit()
        prov.fetch()
        return f"{'✅' if ok else '❌'} {msg}"
    if '"type":"liq"' in trig and val:
        try:
            idx = json.loads(trig.split('.')[0])['index']
            pos = prov.data.get("positions", [])
            if 0<=idx<len(pos):
                p = pos[idx]
                ok, msg = prov.liquidate_position(p['symbol'], p['side'], p['size'])
                prov.fetch()
                return f"{'✅' if ok else '❌'} {p['symbol']}: {msg}"
            return f"❌ Invalid index: {idx}"
        except Exception as e: return f"❌ Error: {e}"
    return no_update

# ---------- External CSS from CDN ----------
app.index_string = '''<!DOCTYPE html><html><head>{%metas%}<title>{%title%}</title>{%favicon%}{%css%}
<link href="https://cdnjs.cloudflare.com/ajax/libs/normalize/8.0.1/normalize.min.css" rel="stylesheet">
</head><body style="margin:0;padding:0">{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body></html>'''

if __name__ == '__main__':
    os.system('clear' if os.name=='posix' else 'cls')
    print("🚀 Mobile Bybit Dashboard → http://localhost:8050")
    print("📊 With BTC Trend & Volatility Indicators")
    print("✅ Using Bot's PostOnly Limit Order Logic\n")
    print("  • PostOnly orders with 0.01% offset")
    print("  • reduceOnly=True for closing positions")
    print("  • Better ticker-based price fetching\n")
    app.run(host='0.0.0.0', port=8050, debug=False)