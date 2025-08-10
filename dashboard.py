#!/usr/bin/env python3
"""
Streamlined Bybit Plotly Dashboard
"""

import os
import time
import threading
from datetime import datetime
import dash
from dash import dcc, html, Input, Output, dash_table
import plotly.graph_objects as go
import pandas as pd
from dotenv import load_dotenv
from pybit.unified_trading import HTTP
import yfinance as yf

load_dotenv(override=True)

class BybitDataProvider:
    def __init__(self, demo_mode=True):
        self.demo_mode = demo_mode
        prefix = "TESTNET_" if demo_mode else "LIVE_"
        
        api_key = os.getenv(f"{prefix}BYBIT_API_KEY", "")
        api_secret = os.getenv(f"{prefix}BYBIT_API_SECRET", "")
        
        if not api_key or not api_secret:
            raise ValueError(f"Missing API credentials. Set {prefix}BYBIT_API_KEY and {prefix}BYBIT_API_SECRET")
        
        self.exchange = HTTP(demo=demo_mode, api_key=api_key, api_secret=api_secret)
        
        # Test connection
        try:
            server_time = self.exchange.get_server_time()
            if server_time.get('retCode') != 0:
                raise ConnectionError("Failed to connect to Bybit")
            print(f"✅ Connected to {'Testnet' if demo_mode else 'Live'} Bybit")
        except Exception as e:
            raise ConnectionError(f"Connection failed: {e}")
        
        self.data = {"positions": [], "account": {}, "last_update": None}
        self.market_data = {"SPY": {}, "BTC": {}, "last_market_update": None}
        self.starting_equity = None
        self.last_reset_date = datetime.now().strftime('%Y-%m-%d')
        self.last_api_call = 0
    
    def safe_float(self, val, default=0):
        try:
            return float(val) if val not in [None, '', 'null'] else default
        except:
            return default
    
    def _rate_limit(self):
        """Simple rate limiting - 200ms between API calls"""
        now = time.time()
        if now - self.last_api_call < 0.2:
            time.sleep(0.2)
        self.last_api_call = time.time()
    
    def get_position_fees(self, symbol, size, avg_price):
        """Simplified fee calculation - uses actual fees from recent orders"""
        try:
            self._rate_limit()
            orders = self.exchange.get_order_history(
                category="linear", symbol=symbol, orderStatus="Filled", limit=50
            )
            
            if orders.get("retCode") == 0:
                total_fees = 0
                qty_covered = 0
                
                for order in orders.get("result", {}).get("list", []):
                    if qty_covered >= size:
                        break
                        
                    order_qty = self.safe_float(order.get("cumExecQty"))
                    order_fee = self.safe_float(order.get("cumExecFee"))
                    
                    qty_to_count = min(order_qty, size - qty_covered)
                    fee_proportion = qty_to_count / order_qty if order_qty > 0 else 0
                    total_fees += abs(order_fee * fee_proportion)
                    qty_covered += qty_to_count
                
                # Use maker fee for uncovered quantity
                if qty_covered < size:
                    remaining_qty = size - qty_covered
                    total_fees += remaining_qty * avg_price * 0.0002
                
                return total_fees
        except Exception as e:
            print(f"Fee calculation error for {symbol}: {e}")
        
        # Fallback to maker fee estimate
        return size * avg_price * 0.0002
    
    def fetch_market_data(self, time_period='24h'):
        """Simplified market data fetching with fallbacks"""
        try:
            # SPY from yfinance
            try:
                spy = yf.Ticker("SPY").history(period='2d', interval='1d')
                if len(spy) >= 2:
                    spy_current = spy['Close'].iloc[-1]
                    spy_previous = spy['Close'].iloc[-2]
                    spy_change_pct = ((spy_current - spy_previous) / spy_previous) * 100
                    
                    self.market_data["SPY"] = {
                        "price": spy_current,
                        "change_pct": spy_change_pct,
                        "time_period": time_period
                    }
                else:
                    raise Exception("Insufficient SPY data")
            except Exception as e:
                print(f"SPY data error: {e}")
                # Fallback values
                self.market_data["SPY"] = {
                    "price": 0,
                    "change_pct": 0,
                    "time_period": time_period
                }
            
            # BTC from Bybit
            try:
                self._rate_limit()
                btc_ticker = self.exchange.get_tickers(category="linear", symbol="BTCUSDT")
                if btc_ticker.get("retCode") == 0:
                    btc_data = btc_ticker.get("result", {}).get("list", [])
                    if btc_data:
                        btc_info = btc_data[0]
                        btc_current = self.safe_float(btc_info.get("lastPrice"))
                        btc_change_pct = self.safe_float(btc_info.get("price24hPcnt")) * 100
                        
                        self.market_data["BTC"] = {
                            "price": btc_current,
                            "change_pct": btc_change_pct,
                            "time_period": time_period
                        }
                    else:
                        raise Exception("No BTC data")
                else:
                    raise Exception(f"API error: {btc_ticker.get('retMsg', 'Unknown')}")
            except Exception as e:
                print(f"BTC data error: {e}")
                # Fallback values
                self.market_data["BTC"] = {
                    "price": 0,
                    "change_pct": 0,
                    "time_period": time_period
                }
            
            self.market_data["last_market_update"] = datetime.now().strftime('%H:%M:%S')
            
        except Exception as e:
            print(f"Market data fetch error: {e}")
            # Ensure we have default market data structure
            if "SPY" not in self.market_data:
                self.market_data["SPY"] = {"price": 0, "change_pct": 0, "time_period": time_period}
            if "BTC" not in self.market_data:
                self.market_data["BTC"] = {"price": 0, "change_pct": 0, "time_period": time_period}
    
    def fetch_data(self):
        try:
            # Reset daily tracking if new day
            current_date = datetime.now().strftime('%Y-%m-%d')
            if self.last_reset_date != current_date:
                self.starting_equity = None
                self.last_reset_date = current_date
            
            # Get positions
            self._rate_limit()
            pos_resp = self.exchange.get_positions(category="linear", settleCoin="USDT")
            positions = []
            
            if pos_resp.get("retCode") == 0:
                for p in pos_resp.get("result", {}).get("list", []):
                    try:
                        size = self.safe_float(p.get("size"))
                        if size > 0:
                            symbol = p.get("symbol", "")
                            side = p.get("side", "")
                            avg_price = self.safe_float(p.get("avgPrice"))
                            mark_price = self.safe_float(p.get("markPrice"))
                            unrealized_pnl = self.safe_float(p.get("unrealisedPnl"))
                            
                            # Calculate PnL if not provided
                            if unrealized_pnl == 0:
                                if side == "Buy":
                                    unrealized_pnl = (mark_price - avg_price) * size
                                else:
                                    unrealized_pnl = (avg_price - mark_price) * size
                            
                            fees = self.get_position_fees(symbol, size, avg_price)
                            net_pnl = unrealized_pnl - fees
                            position_value = avg_price * size
                            pnl_pct = (unrealized_pnl / position_value) * 100 if position_value > 0 else 0
                            
                            # Calculate breakeven
                            exit_fee = size * mark_price * 0.0002  # Estimated exit fee
                            total_fees = fees + exit_fee
                            fee_per_unit = total_fees / size if size > 0 else 0
                            
                            if side == "Buy":
                                breakeven = avg_price + fee_per_unit
                            else:
                                breakeven = avg_price - fee_per_unit
                            
                            positions.append({
                                "symbol": symbol,
                                "side": side,
                                "size": size,
                                "avg_price": avg_price,
                                "mark_price": mark_price,
                                "pnl": unrealized_pnl,
                                "fees": fees,
                                "net_pnl": net_pnl,
                                "breakeven": breakeven,
                                "pnl_pct": pnl_pct,
                                "value": size * mark_price,
                                "liq_price": self.safe_float(p.get("liqPrice"))
                            })
                    except Exception as e:
                        print(f"Error processing position {p.get('symbol', 'unknown')}: {e}")
                        continue
            
            # Get account data
            self._rate_limit()
            acc_resp = self.exchange.get_wallet_balance(accountType="UNIFIED", coin="USDT")
            account = {}
            current_equity = 0
            
            if acc_resp.get("retCode") == 0:
                try:
                    for coin in acc_resp.get("result", {}).get("list", [{}])[0].get("coin", []):
                        if coin.get("coin") == "USDT":
                            current_equity = self.safe_float(coin.get("equity"))
                            account = {
                                "equity": current_equity,
                                "available": self.safe_float(coin.get("availableToWithdraw")),
                                "unrealized_pnl": self.safe_float(coin.get("unrealisedPnl"))
                            }
                            break
                except Exception as e:
                    print(f"Error processing account data: {e}")
            
            # Set starting equity on first run of day
            if self.starting_equity is None and current_equity > 0:
                self.starting_equity = current_equity
            
            # Calculate daily performance
            daily_change = current_equity - (self.starting_equity or current_equity)
            daily_change_pct = (daily_change / (self.starting_equity or current_equity)) * 100 if (self.starting_equity or current_equity) > 0 else 0
            
            # Current positions PnL
            positions_pnl = sum(p["net_pnl"] for p in positions)
            positions_pct = (positions_pnl / current_equity) * 100 if current_equity > 0 else 0
            
            self.data = {
                "positions": positions,
                "account": account,
                "positions_pnl": positions_pnl,
                "positions_pct": positions_pct,
                "daily_change": daily_change,
                "daily_change_pct": daily_change_pct,
                "starting_equity": self.starting_equity,
                "position_count": len(positions),
                "last_update": datetime.now().strftime('%H:%M:%S'),
                "demo_mode": self.demo_mode
            }
            
        except Exception as e:
            print(f"Data fetch error: {e}")
            # Keep existing data on error, just update timestamp
            if hasattr(self, 'data') and self.data:
                self.data["last_update"] = f"ERROR: {datetime.now().strftime('%H:%M:%S')}"

# Initialize
provider = BybitDataProvider(demo_mode=True)

def data_loop():
    """Background thread for data fetching with error handling"""
    error_count = 0
    max_errors = 5
    
    while True:
        try:
            provider.fetch_data()
            if not provider.market_data.get("last_market_update"):
                provider.fetch_market_data('24h')
            
            # Reset error count on success
            error_count = 0
            time.sleep(3)
            
        except Exception as e:
            error_count += 1
            print(f"Data loop error ({error_count}/{max_errors}): {e}")
            
            if error_count >= max_errors:
                print("❌ Too many errors in data loop, using exponential backoff")
                time.sleep(min(60 * error_count, 300))  # Cap at 5 minutes
                error_count = 0  # Reset after long sleep
            else:
                time.sleep(10)  # Short sleep for minor errors

threading.Thread(target=data_loop, daemon=True).start()

# Dash App
app = dash.Dash(__name__)
app.title = "Bybit Dashboard"

app.layout = html.Div([
    dcc.Interval(id='interval', interval=3000, n_intervals=0),
    
    # Header
    html.Div([
        html.H1("📈 Bybit Dashboard", style={'color': '#fff', 'margin': 0}),
        html.Div([
            html.Div(id="mode-badge"),
            dcc.Dropdown(
                id='time-dropdown',
                options=[
                    {'label': '1H', 'value': '1h'},
                    {'label': '4H', 'value': '4h'},
                    {'label': '12H', 'value': '12h'},
                    {'label': '24H', 'value': '24h'}
                ],
                value='24h',
                style={'width': '80px', 'color': '#000'},
                clearable=False
            )
        ], style={'display': 'flex', 'gap': '20px', 'align-items': 'center'})
    ], style={
        'background': 'linear-gradient(135deg, #667eea, #764ba2)',
        'padding': '20px',
        'border-radius': '10px',
        'margin-bottom': '20px',
        'display': 'flex',
        'justify-content': 'space-between',
        'align-items': 'center'
    }),
    
    # Stats
    html.Div(id="stats"),
    
    # Performance
    html.Div([
        html.H3("📊 Performance", style={'color': '#fff'}),
        html.Div(id="performance")
    ], style={'margin': '20px 0'}),
    
    # Positions
    html.Div(id="positions"),
    
    # Chart
    html.Div([
        html.H3("PnL Chart", style={'color': '#fff'}),
        dcc.Graph(id="chart")
    ]),
    
    # Footer
    html.Div(id="footer")
    
], style={
    'font-family': 'system-ui',
    'background': '#0a0e27',
    'color': '#fff',
    'padding': '20px',
    'min-height': '100vh'
})

@app.callback(
    [Output('mode-badge', 'children'),
     Output('stats', 'children'),
     Output('performance', 'children'),
     Output('positions', 'children'),
     Output('chart', 'figure'),
     Output('footer', 'children')],
    [Input('interval', 'n_intervals'),
     Input('time-dropdown', 'value')]
)
def update_all(n, time_period):
    data = provider.data
    
    # Update market data for selected period
    provider.fetch_market_data(time_period)
    market = provider.market_data
    
    # Mode badge
    mode = "TESTNET" if data.get("demo_mode", True) else "LIVE"
    
    # Stats cards
    account = data.get("account", {})
    total_fees = sum(p.get("fees", 0) for p in data.get("positions", []))
    
    stats = html.Div([
        # Row 1
        html.Div([
            html.Div([
                html.Div("Equity", className="stat-label"),
                html.Div(f"${account.get('equity', 0):.2f}", className="stat-value")
            ], className="stat-card"),
            html.Div([
                html.Div("Available", className="stat-label"),
                html.Div(f"${account.get('available', 0):.2f}", className="stat-value")
            ], className="stat-card"),
            html.Div([
                html.Div("Unrealized PnL", className="stat-label"),
                html.Div(f"${account.get('unrealized_pnl', 0):.2f}", 
                        className="stat-value",
                        style={'color': '#00d4aa' if account.get('unrealized_pnl', 0) >= 0 else '#f6465d'})
            ], className="stat-card")
        ], className="stat-row"),
        
        # Row 2
        html.Div([
            html.Div([
                html.Div("Fees Paid", className="stat-label"),
                html.Div(f"${total_fees:.2f}", className="stat-value", style={'color': '#ffa500'})
            ], className="stat-card"),
            html.Div([
                html.Div("Positions", className="stat-label"),
                html.Div(str(data.get("position_count", 0)), className="stat-value")
            ], className="stat-card"),
            html.Div([
                html.Div("Daily Change", className="stat-label"),
                html.Div(f"${data.get('daily_change', 0):+.2f}", 
                        className="stat-value",
                        style={'color': '#00d4aa' if data.get('daily_change', 0) >= 0 else '#f6465d'})
            ], className="stat-card")
        ], className="stat-row")
    ])
    
    # Performance comparison
    spy_data = market.get("SPY", {})
    btc_data = market.get("BTC", {})
    
    performance = html.Div([
        html.Div([
            html.Div(f"SPY: ${spy_data.get('price', 0):,.2f} ({spy_data.get('change_pct', 0):+.2f}%)",
                    style={'color': '#00d4aa' if spy_data.get('change_pct', 0) >= 0 else '#f6465d'}),
            html.Div(f"BTC: ${btc_data.get('price', 0):,.0f} ({btc_data.get('change_pct', 0):+.2f}%)",
                    style={'color': '#00d4aa' if btc_data.get('change_pct', 0) >= 0 else '#f6465d'})
        ], style={'display': 'flex', 'justify-content': 'space-around', 'margin-bottom': '10px'}),
        
        html.Div([
            html.Div(f"Open Positions: ${data.get('positions_pnl', 0):+.2f} ({data.get('positions_pct', 0):+.2f}%)",
                    style={'color': '#00d4aa' if data.get('positions_pnl', 0) >= 0 else '#f6465d'}),
            html.Div(f"Daily Total: ${data.get('daily_change', 0):+.2f} ({data.get('daily_change_pct', 0):+.2f}%)",
                    style={'color': '#00d4aa' if data.get('daily_change', 0) >= 0 else '#f6465d'})
        ], style={'display': 'flex', 'justify-content': 'space-around'})
    ], className="performance-card")
    
    # Positions table
    positions = data.get("positions", [])
    if not positions:
        table = html.Div("No positions", className="no-positions")
    else:
        table_data = []
        for p in positions:
            table_data.append({
                'Symbol': p['symbol'],
                'Side': p['side'],
                'Size': f"{p['size']:.4f}",
                'Entry': f"${p['avg_price']:.4f}",
                'Mark': f"${p['mark_price']:.4f}",
                'PnL': f"${p['pnl']:.2f}",
                'Fees': f"${p['fees']:.2f}",
                'Net PnL': f"${p['net_pnl']:.2f}",
                'PnL %': f"{p['pnl_pct']:.2f}%",
                'Break-even': f"${p['breakeven']:.4f}",
                'Value': f"${p['value']:.2f}"
            })
        
        # Add total
        table_data.append({
            'Symbol': 'TOTAL', 'Side': '', 'Size': '', 'Entry': '', 'Mark': '', 'PnL': '',
            'Fees': '', 'Net PnL': f"${data.get('positions_pnl', 0):.2f}",
            'PnL %': '', 'Break-even': '', 'Value': ''
        })
        
        table = dash_table.DataTable(
            data=table_data,
            columns=[{"name": i, "id": i} for i in table_data[0].keys()],
            style_table={'background': '#1a1f3a', 'border-radius': '10px'},
            style_header={'background': '#0f1529', 'color': '#8892b0', 'font-weight': 'bold'},
            style_cell={'background': '#1a1f3a', 'color': '#fff', 'padding': '12px', 'border': '1px solid #2a3050'},
            style_data_conditional=[
                {'if': {'row_index': len(positions)}, 'background': '#0f1529', 'font-weight': 'bold'},
                {'if': {'filter_query': '{Side} = Buy'}, 'color': '#00d4aa'},
                {'if': {'filter_query': '{Side} = Sell'}, 'color': '#f6465d'}
            ]
        )
    
    # PnL Chart
    if positions:
        df = pd.DataFrame(positions)
        colors = ['#00d4aa' if pnl >= 0 else '#f6465d' for pnl in df['net_pnl']]
        
        fig = go.Figure(data=[
            go.Bar(x=df['symbol'], y=df['net_pnl'], marker_color=colors,
                  text=[f"${pnl:.2f}" for pnl in df['net_pnl']], textposition='auto')
        ])
        
        fig.update_layout(
            title="Net PnL by Position",
            plot_bgcolor='#1a1f3a', paper_bgcolor='#1a1f3a',
            font=dict(color='#fff'), showlegend=False,
            xaxis_title="Symbol", yaxis_title="Net PnL ($)"
        )
        fig.update_xaxes(gridcolor='#2a3050')
        fig.update_yaxes(gridcolor='#2a3050', zeroline=True, zerolinecolor='#8892b0')
    else:
        fig = go.Figure()
        fig.update_layout(title="No positions", plot_bgcolor='#1a1f3a', paper_bgcolor='#1a1f3a', font=dict(color='#fff'))
    
    # Footer
    footer = f"Updated: {data.get('last_update', 'Never')} | Market: {market.get('last_market_update', 'Never')}"
    if data.get('starting_equity'):
        footer += f" | Baseline: ${data.get('starting_equity'):.2f}"
    
    return mode, stats, performance, table, fig, footer

# CSS
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            .stat-row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin-bottom: 15px; }
            .stat-card { background: #1a1f3a; padding: 20px; border-radius: 10px; border: 1px solid #2a3050; }
            .stat-label { color: #8892b0; font-size: 12px; text-transform: uppercase; margin-bottom: 5px; }
            .stat-value { font-size: 24px; font-weight: bold; }
            .performance-card { background: #1a1f3a; padding: 20px; border-radius: 10px; border: 1px solid #2a3050; }
            .no-positions { text-align: center; padding: 40px; background: #1a1f3a; border-radius: 10px; border: 1px solid #2a3050; }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

if __name__ == '__main__':
    print("🚀 Bybit Dashboard → http://localhost:8050")
    app.run(host='0.0.0.0', port=8050, debug=False)