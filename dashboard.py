#!/usr/bin/env python3
import os, sys, time, threading, logging
from datetime import datetime
logging.basicConfig(level=logging.CRITICAL)
for logger in ['werkzeug', 'dash']: logging.getLogger(logger).setLevel(logging.CRITICAL)

import dash
from dash import dcc, html, Input, Output, dash_table, no_update, State, callback_context
import dash_ag_grid as dag
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
        self.equity0, self.reset_date, self.last_call = None, datetime.now().strftime('%Y-%m-%d'), 0
    
    def sf(self, v, d=0): return float(v) if v not in [None, '', 'null'] else d
    def rl(self): time.sleep(max(0, 0.2 - (time.time() - self.last_call))); self.last_call = time.time()
    
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
            close_side = "Sell" if side == "Buy" else "Buy"
            order = self.ex.place_order(
                category="linear",
                symbol=symbol,
                side=close_side,
                orderType="Market",
                qty=str(size),
                reduceOnly=True
            )
            if order.get("retCode") == 0:
                return True, "Position liquidated successfully"
            else:
                return False, f"Error: {order.get('retMsg', 'Unknown error')}"
        except Exception as e:
            return False, f"Exception: {str(e)}"
    
    def close_all_limit(self):
        try:
            results = []
            for p in self.data.get("positions", []):
                self.rl()
                symbol, side, size, mark_price = p['symbol'], p['side'], p['size'], p['mark_price']
                close_side = "Sell" if side == "Buy" else "Buy"
                # Post-only: place order slightly away from mark price
                offset = mark_price * 0.001  # 0.1% offset
                limit_price = mark_price + offset if side == "Buy" else mark_price - offset
                
                order = self.ex.place_order(
                    category="linear",
                    symbol=symbol,
                    side=close_side,
                    orderType="Limit",
                    qty=str(size),
                    price=str(round(limit_price, 2)),
                    reduceOnly=True,
                    timeInForce="PostOnly"
                )
                
                if order.get("retCode") == 0:
                    results.append(f"✅ {symbol}")
                else:
                    results.append(f"❌ {symbol}: {order.get('retMsg', 'Unknown')}")
            
            return True if results else False, ", ".join(results) if results else "No positions to close"
        except Exception as e:
            return False, f"Exception: {str(e)}"
    
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
                        "value": sz * mp, "liq_price": self.sf(p.get("liqPrice")),
                        "leverage": lev
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

def loop():
    while True:
        try: prov.fetch(); prov.fetch_market() if not prov.market.get("update") else None; time.sleep(3)
        except: time.sleep(10)

threading.Thread(target=loop, daemon=True).start()

class FilteredStream:
    def __init__(self, s): self.s, self.suppress = s, False
    def write(self, d): 
        if any(x in str(d) for x in ['stats-cards', 'market-comparison', 'positions-table', 'Callback function']): self.suppress = True; return
        if self.suppress and ('POST /_dash-update' in str(d) or 'GET /' in str(d)): self.suppress = False; return
        if not self.suppress: self.s.write(d)
    def flush(self): self.s.flush()

sys.stderr = FilteredStream(sys.__stderr__)

app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Bybit Dashboard"

def col(v): return '#00d4aa' if v >= 0 else '#f6465d'
def card(l, v, c=None): return html.Div([html.Div(l, className="sl"), html.Div(v, className="sv", style={'color': c} if c else {})], className="sc")
def perf(l, v, p): return html.Div(f"{l}: {v} ({p:+.2f}%)", style={'color': col(p)})

app.layout = html.Div([
    dcc.Interval(id='int', interval=3000),
    html.Div([
        html.H1("📈 Bybit Dashboard", style={'color': '#fff', 'margin': 0}),
        html.Div([html.Div(id="mode"), dcc.Dropdown(id='time', options=[{'label': p, 'value': p.lower()} for p in ['1H', '4H', '12H', '24H']], value='24h', style={'width': '80px', 'color': '#000'}, clearable=False)], 
                 style={'display': 'flex', 'gap': '20px', 'align-items': 'center'})
    ], style={'background': 'linear-gradient(135deg, #667eea, #764ba2)', 'padding': '20px', 'border-radius': '10px', 'margin-bottom': '20px', 'display': 'flex', 'justify-content': 'space-between', 'align-items': 'center'}),
    html.Div(id="stats"), html.Div([html.H3("📊 Performance", style={'color': '#fff'}), html.Div(id="perf")], style={'margin': '20px 0'}),
    html.Div([
        html.Button("🚫 Close All (Limit Post-Only)", id="close-all-btn", 
                   style={'background': '#ff9800', 'color': '#fff', 'border': 'none', 'padding': '10px 20px', 
                          'border-radius': '5px', 'cursor': 'pointer', 'margin-bottom': '10px', 'font-weight': 'bold'}),
        html.Div(id="pos")
    ]),
    html.Div([html.H3("PnL Chart", style={'color': '#fff'}), dcc.Graph(id="chart")]), html.Div(id="foot"),
    html.Div(id="liq-msg", style={'color': '#fff', 'margin': '10px 0'})
], style={'font-family': 'system-ui', 'background': '#0a0e27', 'color': '#fff', 'padding': '20px', 'min-height': '100vh'})

@app.callback([Output('mode', 'children'), Output('stats', 'children'), Output('perf', 'children'), Output('pos', 'children'), Output('chart', 'figure'), Output('foot', 'children')],
              [Input('int', 'n_intervals'), Input('time', 'value')])
def update(n, period):
    d, m = prov.data, prov.market
    prov.fetch_market(period)
    acc, fees = d.get("account", {}), sum(p.get("fees", 0) for p in d.get("positions", []))
    
    stats = html.Div([
        html.Div([card("Equity", f"${acc.get('equity', 0):.2f}"), card("Available", f"${acc.get('available', 0):.2f}"), 
                  card("Unrealized PnL", f"${acc.get('unrealized_pnl', 0):.2f}", col(acc.get('unrealized_pnl', 0)))], className="sr"),
        html.Div([card("Fees Paid", f"${fees:.2f}", '#ffa500'), card("Positions", str(d.get("position_count", 0))),
                  card("Daily Change", f"${d.get('daily_change', 0):+.2f}", col(d.get('daily_change', 0)))], className="sr")
    ])
    
    spy, btc = m.get("SPY", {}), m.get("BTC", {})
    perf_div = html.Div([
        html.Div([perf("SPY", f"${spy.get('price', 0):,.2f}", spy.get('change_pct', 0)), perf("BTC", f"${btc.get('price', 0):,.0f}", btc.get('change_pct', 0))], 
                 style={'display': 'flex', 'justify-content': 'space-around', 'margin-bottom': '10px'}),
        html.Div([perf("Open Positions", f"${d.get('positions_pnl', 0):+.2f}", d.get('positions_pct', 0)), 
                  perf("Daily Total", f"${d.get('daily_change', 0):+.2f}", d.get('daily_change_pct', 0))], 
                 style={'display': 'flex', 'justify-content': 'space-around'})
    ], className="pc")
    
    pos = d.get("positions", [])
    if not pos: 
        table = html.Div("No positions", className="np")
    else:
        td = []
        for p in pos:
            row = {
                'Symbol': p['symbol'], 
                'Side': p['side'], 
                'Size': f"{p['size']:.4f}", 
                'Leverage': f"{p.get('leverage', 1):.1f}x",
                'Entry': f"${p['avg_price']:.4f}", 
                'Mark': f"${p['mark_price']:.4f}",
                'PnL': f"${p['pnl']:.2f}", 
                'Fees': f"${p['fees']:.2f}", 
                'Net PnL': f"${p['net_pnl']:.2f}", 
                'PnL %': f"{p['pnl_pct']:.2f}%",
                'Break-even': f"${p['breakeven']:.4f}", 
                'Value': f"${p['value']:.2f}",
                'Action': 'Liquidate'
            }
            td.append(row)
        
        columnDefs = [
            {"field": "Symbol"}, {"field": "Side"}, {"field": "Size"}, {"field": "Leverage"},
            {"field": "Entry"}, {"field": "Mark"},
            {"field": "PnL"}, {"field": "Fees"}, {"field": "Net PnL"}, {"field": "PnL %"},
            {"field": "Break-even"}, {"field": "Value"},
            {"field": "Action", "cellStyle": {"backgroundColor": "#f6465d", "color": "#fff", "cursor": "pointer", "textAlign": "center"}}
        ]
        
        table = dag.AgGrid(
            id="positions-grid",
            rowData=td,
            columnDefs=columnDefs,
            defaultColDef={"resizable": True, "sortable": True},
            dashGridOptions={"domLayout": "autoHeight"},
            style={"height": None},
            className="ag-theme-alpine-dark"
        )
    
    if pos:
        df = pd.DataFrame(pos)
        fig = go.Figure([go.Bar(x=df['symbol'], y=df['net_pnl'], marker_color=[col(p) for p in df['net_pnl']], 
                               text=[f"${p:.2f}" for p in df['net_pnl']], textposition='auto')])
        fig.update_layout(title="Net PnL by Position", plot_bgcolor='#1a1f3a', paper_bgcolor='#1a1f3a', font=dict(color='#fff'), 
                         showlegend=False, xaxis_title="Symbol", yaxis_title="Net PnL ($)")
        fig.update_xaxes(gridcolor='#2a3050'); fig.update_yaxes(gridcolor='#2a3050', zeroline=True, zerolinecolor='#8892b0')
    else: fig = go.Figure(); fig.update_layout(title="No positions", plot_bgcolor='#1a1f3a', paper_bgcolor='#1a1f3a', font=dict(color='#fff'))
    
    foot = f"Updated: {d.get('last_update', 'Never')} | Market: {m.get('update', 'Never')}" + (f" | Baseline: ${d.get('starting_equity'):.2f}" if d.get('starting_equity') else "")
    return "TESTNET" if d.get("demo_mode") else "LIVE", stats, perf_div, table, fig, foot

@app.callback(Output('liq-msg', 'children'),
              [Input('positions-grid', 'cellClicked'), Input('close-all-btn', 'n_clicks')],
              State('positions-grid', 'rowData'),
              prevent_initial_call=True)
def handle_actions(cell, n_clicks, row_data):
    ctx = callback_context
    if not ctx.triggered: return no_update
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if trigger_id == 'close-all-btn':
        success, msg = prov.close_all_limit()
        if success:
            prov.fetch()
            return html.Div(f"📝 Limit orders placed: {msg}", style={'color': '#ff9800'})
        else:
            return html.Div(f"❌ {msg}", style={'color': '#f6465d'})
    
    elif trigger_id == 'positions-grid' and cell and cell.get('colId') == 'Action':
        row_index = cell.get('rowIndex')
        if row_data and row_index < len(row_data):
            symbol = row_data[row_index].get('Symbol')
            for p in prov.data.get("positions", []):
                if p['symbol'] == symbol:
                    success, msg = prov.liquidate_position(p['symbol'], p['side'], p['size'])
                    if success:
                        prov.fetch()
                        return html.Div(f"✅ {symbol} {msg}", style={'color': '#00d4aa'})
                    else:
                        return html.Div(f"❌ {symbol} {msg}", style={'color': '#f6465d'})
            return html.Div(f"❌ Position not found: {symbol}", style={'color': '#f6465d'})
    
    return no_update

app.index_string = '''<!DOCTYPE html><html><head>{%metas%}<title>{%title%}</title>{%favicon%}{%css%}<style>
.sr{display:grid;grid-template-columns:repeat(3,1fr);gap:15px;margin-bottom:15px}
.sc{background:#1a1f3a;padding:20px;border-radius:10px;border:1px solid #2a3050}
.sl{color:#8892b0;font-size:12px;text-transform:uppercase;margin-bottom:5px}
.sv{font-size:24px;font-weight:bold}
.pc{background:#1a1f3a;padding:20px;border-radius:10px;border:1px solid #2a3050}
.np{text-align:center;padding:40px;background:#1a1f3a;border-radius:10px;border:1px solid #2a3050}
.ag-cell[col-id="Action"]{padding:5px!important;border-radius:4px}
.ag-cell[col-id="Action"]:hover{background:#d63547!important;transform:scale(0.98)}
#close-all-btn:hover{background:#e68a00!important}
</style></head><body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body></html>'''

if __name__ == '__main__':
    os.system('clear' if os.name == 'posix' else 'cls')
    print("🚀 Bybit Dashboard → http://localhost:8050\n✅ Dashboard is functional!\n💡 Use incognito/private mode\n")
    app.run(host='0.0.0.0', port=8050, debug=False)