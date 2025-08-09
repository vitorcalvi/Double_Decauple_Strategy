#!/usr/bin/env python3
"""
Bybit Plotly Dashboard - Fixed Callback Issue
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
        self.exchange = HTTP(
            demo=demo_mode,
            api_key=os.getenv(f"{prefix}BYBIT_API_KEY", ""),
            api_secret=os.getenv(f"{prefix}BYBIT_API_SECRET", "")
        )
        self.data = {"positions": [], "account": {}, "last_update": None}
        self.market_data = {"SPY": {}, "BTC": {}, "last_market_update": None}
        self.starting_equity = None
        self.last_reset_date = None
    
    def safe_float(self, val, default=0):
        if val in [None, '', 'null']:
            return default
        try:
            return float(val)
        except (ValueError, TypeError):
            return default
    
    def safe_divide(self, numerator, denominator, default=0):
        return (numerator / denominator) if denominator > 0 else default
    
    def get_position_fees(self, symbol, size, avg_price):
        try:
            orders = self.exchange.get_order_history(
                category="linear", symbol=symbol, orderStatus="Filled", limit=200
            )
            
            if orders.get("retCode") != 0:
                return size * avg_price * 0.0002
            
            total_fees = 0
            total_qty = 0
            
            for order in orders.get("result", {}).get("list", []):
                order_qty = self.safe_float(order.get("cumExecQty"))
                order_fee = self.safe_float(order.get("cumExecFee"))
                
                if total_qty < size:
                    qty_to_count = min(order_qty, size - total_qty)
                    fee_proportion = self.safe_divide(qty_to_count, order_qty)
                    total_fees += abs(order_fee * fee_proportion)
                    total_qty += qty_to_count
                    
                    if total_qty >= size:
                        break
            
            if total_qty < size:
                remaining_qty = size - total_qty
                total_fees += remaining_qty * avg_price * 0.0002
            
            return total_fees
            
        except Exception as e:
            print(f"Error getting fees for {symbol}: {e}")
            return size * avg_price * 0.0002
    
    def calculate_breakeven(self, entry_price, size, fees, side):
        if size <= 0:
            return entry_price
        
        exit_fee_estimate = size * entry_price * 0.0002
        total_fees = fees + exit_fee_estimate
        fee_per_unit = self.safe_divide(total_fees, size)
        
        return entry_price + fee_per_unit if side == "Buy" else entry_price - fee_per_unit
    
    def fetch_market_data(self, time_period='24h'):
        try:
            period_map = {'1h': '1d', '4h': '1d', '12h': '1d', '24h': '2d'}
            interval_map = {'1h': '1h', '4h': '1h', '12h': '1h', '24h': '1d'}
            
            period = period_map.get(time_period, '2d')
            interval = interval_map.get(time_period, '1d')
            
            # SPY data
            spy = yf.Ticker("SPY")
            spy_info = spy.history(period=period, interval=interval)
            
            if len(spy_info) >= 2:
                spy_current = spy_info['Close'].iloc[-1]
                
                lookback_map = {'1h': -2, '4h': -5, '12h': -13, '24h': -2}
                lookback = lookback_map.get(time_period, -2)
                
                if len(spy_info) >= abs(lookback):
                    spy_previous = spy_info['Close'].iloc[lookback]
                else:
                    spy_previous = spy_info['Close'].iloc[-2] if len(spy_info) >= 2 else spy_current
                
                spy_change = spy_current - spy_previous
                spy_change_pct = self.safe_divide(spy_change, spy_previous) * 100
                
                self.market_data["SPY"] = {
                    "price": spy_current,
                    "change": spy_change,
                    "change_pct": spy_change_pct
                }
            
            # BTC data
            try:
                btc_ticker = self.exchange.get_tickers(category="linear", symbol="BTCUSDT")
                if btc_ticker.get("retCode") == 0:
                    btc_data = btc_ticker.get("result", {}).get("list", [])
                    if btc_data:
                        btc_info = btc_data[0]
                        btc_current = self.safe_float(btc_info.get("lastPrice"))
                        
                        if time_period == '24h':
                            btc_change_pct = self.safe_float(btc_info.get("price24hPcnt")) * 100
                            btc_change = self.safe_float(btc_info.get("price24h"))
                        else:
                            interval_map_bybit = {'1h': '60', '4h': '240', '12h': '720'}
                            bybit_interval = interval_map_bybit.get(time_period, '60')
                            
                            kline_resp = self.exchange.get_kline(
                                category="linear", symbol="BTCUSDT", 
                                interval=bybit_interval, limit=2
                            )
                            
                            if kline_resp.get("retCode") == 0:
                                klines = kline_resp.get("result", {}).get("list", [])
                                if len(klines) >= 2:
                                    current_close = float(klines[0][4])
                                    previous_close = float(klines[1][4])
                                    btc_change = current_close - previous_close
                                    btc_change_pct = self.safe_divide(btc_change, previous_close) * 100
                                    btc_current = current_close
                                else:
                                    btc_change = btc_change_pct = 0
                            else:
                                btc_change = btc_change_pct = 0
                        
                        self.market_data["BTC"] = {
                            "price": btc_current,
                            "change": btc_change,
                            "change_pct": btc_change_pct
                        }
                    else:
                        raise Exception("No BTC data")
                else:
                    raise Exception(f"Bybit API error: {btc_ticker.get('retMsg', 'Unknown')}")
            except Exception as e:
                print(f"BTC fetch failed: {e}")
                self.market_data["BTC"] = {"price": 0, "change": 0, "change_pct": 0}
            
            self.market_data["last_market_update"] = datetime.now().strftime('%H:%M:%S')
            
        except Exception as e:
            print(f"Market data error: {e}")
    
    def fetch_data(self):
        try:
            current_date = datetime.now().strftime('%Y-%m-%d')
            if self.last_reset_date != current_date:
                self.starting_equity = None
                self.last_reset_date = current_date
            
            # Fetch positions
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
                            
                            if unrealized_pnl == 0:
                                is_buy = side == "Buy"
                                unrealized_pnl = (mark_price - avg_price) * size if is_buy else (avg_price - mark_price) * size
                            
                            fees = self.get_position_fees(symbol, size, avg_price)
                            breakeven = self.calculate_breakeven(avg_price, size, fees, side)
                            net_pnl = unrealized_pnl - fees
                            position_value = avg_price * size
                            pnl_pct = self.safe_divide(unrealized_pnl, position_value) * 100
                            
                            positions.append({
                                "symbol": symbol, "side": side, "size": size,
                                "avg_price": avg_price, "mark_price": mark_price,
                                "pnl": unrealized_pnl, "fees": fees, "net_pnl": net_pnl,
                                "breakeven": breakeven, "pnl_pct": pnl_pct,
                                "value": size * mark_price,
                                "liq_price": self.safe_float(p.get("liqPrice"))
                            })
                    except Exception as e:
                        print(f"Error processing position: {e}")
                        continue
            
            # Fetch account
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
                except Exception:
                    pass
            
            if self.starting_equity is None and current_equity > 0:
                self.starting_equity = current_equity
                print(f"Starting equity set for {current_date}: ${self.starting_equity:.2f}")
            
            daily_equity_change = current_equity - (self.starting_equity or current_equity)
            daily_equity_change_pct = self.safe_divide(daily_equity_change, self.starting_equity or current_equity) * 100
            
            current_positions_pnl = sum(p["net_pnl"] for p in positions)
            current_positions_pct = self.safe_divide(current_positions_pnl, current_equity) * 100
            
            self.data = {
                "positions": positions, "account": account,
                "current_positions_pnl": current_positions_pnl,
                "current_positions_pct": current_positions_pct,
                "daily_equity_change": daily_equity_change,
                "daily_equity_change_pct": daily_equity_change_pct,
                "starting_equity": self.starting_equity,
                "position_count": len(positions),
                "last_update": datetime.now().strftime('%H:%M:%S'),
                "demo_mode": self.demo_mode
            }
            
        except Exception as e:
            print(f"Fetch error: {e}")

# Initialize
provider = BybitDataProvider(demo_mode=True)

def update_loop():
    while True:
        provider.fetch_data()
        if not provider.market_data.get("last_market_update"):
            provider.fetch_market_data('24h')
        time.sleep(2)

threading.Thread(target=update_loop, daemon=True).start()

# Initialize Dash app - CRITICAL: Disable hot reload to prevent callback conflicts
app = dash.Dash(__name__)
app.title = "Bybit Dashboard"

# CRITICAL: Suppress callback exceptions to prevent timing issues
app.config.suppress_callback_exceptions = True

# Styles
CARD_STYLE = {
    'background': '#1a1f3a', 'padding': '20px', 'border-radius': '10px', 
    'border': '1px solid #2a3050'
}

# Layout - ENSURE all IDs are unique and match callback exactly
app.layout = html.Div([
    dcc.Interval(id='interval-component', interval=2000, n_intervals=0),
    
    # Header
    html.Div([
        html.H1("📈 Bybit Position Dashboard", style={'color': '#fff', 'margin': 0}),
        html.Div([
            html.Div(id="mode-badge", children="Loading...", style={
                'background': 'rgba(255,255,255,0.2)', 'padding': '5px 15px', 
                'border-radius': '20px', 'font-weight': 'bold', 'margin-right': '15px'
            }),
            html.Div([
                html.Label("Time Period:", style={'color': '#8892b0', 'margin-right': '10px', 'font-size': '14px'}),
                dcc.Dropdown(
                    id='time-period-dropdown',
                    options=[
                        {'label': '1 Hour', 'value': '1h'},
                        {'label': '4 Hours', 'value': '4h'},
                        {'label': '12 Hours', 'value': '12h'},
                        {'label': '24 Hours', 'value': '24h'}
                    ],
                    value='24h',
                    style={'width': '120px', 'color': '#000'},
                    clearable=False
                )
            ], style={'display': 'flex', 'align-items': 'center'})
        ], style={'display': 'flex', 'align-items': 'center'})
    ], style={'background': 'linear-gradient(135deg, #667eea, #764ba2)', 'padding': '20px', 'border-radius': '10px', 'margin-bottom': '20px', 'display': 'flex', 'justify-content': 'space-between', 'align-items': 'center'}),
    
    # Info note
    html.Div("ℹ️ Fees shown are actual fees paid for current positions (limit orders: 0.02%, market orders: 0.055%)", 
             style={**CARD_STYLE, 'margin-bottom': '20px', 'font-size': '12px', 'color': '#8892b0'}),
    
    # Components with default content to prevent initial errors
    html.Div(id="stats-cards", children=[], style={
        'display': 'grid', 'grid-template-columns': 'repeat(auto-fit, minmax(200px, 1fr))', 
        'gap': '15px', 'margin-bottom': '20px'
    }),
    
    html.Div([
        html.H3(id="performance-title", children="📊 Daily Performance", style={'color': '#fff', 'margin-bottom': '15px'}),
        html.Div(id="market-comparison", children="Loading market data...")
    ], style={'margin-bottom': '30px'}),
    
    html.Div(id="positions-table", children="Loading positions..."),
    
    html.Div([
        html.H3("PnL Distribution", style={'color': '#fff', 'margin-bottom': '10px'}),
        dcc.Graph(id="pnl-chart", figure=go.Figure())
    ], style={'margin-top': '30px'}),
    
    html.Div(id="update-time", children="Initializing...", style={
        'text-align': 'center', 'color': '#8892b0', 'margin-top': '20px', 'font-size': '12px'
    })
    
], style={
    'font-family': 'system-ui, -apple-system, sans-serif', 'background': '#0a0e27', 
    'color': '#fff', 'padding': '20px', 'min-height': '100vh'
})

# CRITICAL: Single callback with exact ID matching and proper @ decorator
@app.callback(
    [Output('mode-badge', 'children'),
     Output('stats-cards', 'children'),
     Output('performance-title', 'children'),
     Output('market-comparison', 'children'),
     Output('positions-table', 'children'),
     Output('pnl-chart', 'figure'),
     Output('update-time', 'children')],
    [Input('interval-component', 'n_intervals'),
     Input('time-period-dropdown', 'value')],
    prevent_initial_call=False
)
def update_dashboard(n, time_period):
    # Input validation with proper defaults
    if time_period is None or time_period not in ['1h', '4h', '12h', '24h']:
        time_period = '24h'
    
    time_labels = {'1h': '1 Hour', '4h': '4 Hours', '12h': '12 Hours', '24h': 'Daily'}
    
    try:
        data = provider.data
        provider.fetch_market_data(time_period)
        market_data = provider.market_data
        
        # Mode badge
        mode = "TESTNET" if data.get("demo_mode", True) else "LIVE"
        
        # Performance title
        performance_title = f"📊 {time_labels.get(time_period, 'Daily')} Performance"
        
        # Stats cards
        account = data.get("account", {})
        total_fees = sum(p.get("fees", 0) for p in data.get("positions", []))
        
        def create_stat_card(title, value, color='#fff'):
            return html.Div([
                html.Div(title, style={
                    'color': '#8892b0', 'font-size': '12px', 
                    'text-transform': 'uppercase', 'margin-bottom': '5px'
                }),
                html.Div(value, style={'font-size': '24px', 'font-weight': 'bold', 'color': color})
            ], style=CARD_STYLE)
        
        unrealized_pnl = account.get('unrealized_pnl', 0)
        stats_cards = [
            create_stat_card("Equity", f"${account.get('equity', 0):.2f}"),
            create_stat_card("Available", f"${account.get('available', 0):.2f}"),
            create_stat_card("Unrealized PnL", f"${unrealized_pnl:.2f}", 
                           '#00d4aa' if unrealized_pnl >= 0 else '#f6465d'),
            create_stat_card("Total Fees Paid", f"${total_fees:.2f}", '#ffa500'),
            create_stat_card("Positions", str(data.get("position_count", 0)))
        ]
        
        # Market comparison
        spy_data = market_data.get("SPY", {})
        btc_data = market_data.get("BTC", {})
        
        def format_performance(price, change_pct, name):
            if price == 0:
                return html.Div(f"{name}: Data unavailable", style={'color': '#8892b0'})
            color = '#00d4aa' if change_pct >= 0 else '#f6465d'
            time_label = time_labels.get(time_period, 'Daily')
            return html.Div([
                html.Span(f"{name}: ", style={'color': '#8892b0'}),
                html.Span(f"${price:,.2f} ", style={'color': '#fff'}),
                html.Span(f"({change_pct:+.2f}%)", style={'color': color}),
                html.Span(f" {time_label.lower()}", style={'color': '#8892b0', 'font-size': '11px'})
            ])
        
        current_positions_pnl = data.get("current_positions_pnl", 0)
        current_positions_pct = data.get("current_positions_pct", 0)
        daily_equity_change = data.get("daily_equity_change", 0)
        daily_equity_change_pct = data.get("daily_equity_change_pct", 0)
        
        market_comparison = html.Div([
            html.Div([
                format_performance(spy_data.get('price', 0), spy_data.get('change_pct', 0), "SPY"),
                format_performance(btc_data.get('price', 0), btc_data.get('change_pct', 0), "BTC"),
            ], style={'display': 'flex', 'justify-content': 'space-around', 'margin-bottom': '10px'}),
            
            html.Div([
                html.Div([
                    html.Span("Open Positions: ", style={'color': '#8892b0'}),
                    html.Span(f"${current_positions_pnl:+.2f} ", 
                             style={'color': '#00d4aa' if current_positions_pnl >= 0 else '#f6465d'}),
                    html.Span(f"({current_positions_pct:+.2f}%)", 
                             style={'color': '#00d4aa' if current_positions_pct >= 0 else '#f6465d'})
                ]),
                html.Div([
                    html.Span("Daily Total: ", style={'color': '#8892b0'}),
                    html.Span(f"${daily_equity_change:+.2f} ", 
                             style={'color': '#00d4aa' if daily_equity_change >= 0 else '#f6465d'}),
                    html.Span(f"({daily_equity_change_pct:+.2f}%)", 
                             style={'color': '#00d4aa' if daily_equity_change_pct >= 0 else '#f6465d'})
                ])
            ], style={'display': 'flex', 'justify-content': 'space-around'})
        ], style=CARD_STYLE)
        
        # Positions table
        positions = data.get("positions", [])
        if not positions:
            table = html.Div("No open positions", style={
                **CARD_STYLE, 'text-align': 'center', 'padding': '40px'
            })
        else:
            table_data = []
            for p in positions:
                table_data.append({
                    'Symbol': p['symbol'], 'Side': p['side'], 'Size': f"{p['size']:.4f}",
                    'Entry': f"${p['avg_price']:.4f}", 'Mark': f"${p['mark_price']:.4f}",
                    'Unrealized PnL': f"${p['pnl']:.2f}", 'Fees Paid': f"${p['fees']:.2f}",
                    'Net PnL': f"${p['net_pnl']:.2f}", 'PnL %': f"{p['pnl_pct']:.2f}%",
                    'Break-even': f"${p['breakeven']:.4f}", 'Value': f"${p['value']:.2f}",
                    'Liq Price': f"${p['liq_price']:.4f}" if p['liq_price'] > 0 else "—"
                })
            
            table_data.append({
                'Symbol': 'TOTAL', 'Side': '', 'Size': '', 'Entry': '', 'Mark': '',
                'Unrealized PnL': '', 'Fees Paid': '', 'Net PnL': f"${current_positions_pnl:.2f}",
                'PnL %': '', 'Break-even': '', 'Value': '', 'Liq Price': ''
            })
            
            table = dash_table.DataTable(
                data=table_data,
                columns=[{"name": i, "id": i} for i in table_data[0].keys()],
                style_table={'background': '#1a1f3a', 'border': '1px solid #2a3050', 'border-radius': '10px', 'overflow': 'hidden'},
                style_header={
                    'background': '#0f1529', 'color': '#8892b0', 'font-size': '11px',
                    'text-transform': 'uppercase', 'padding': '12px', 'font-weight': 'bold'
                },
                style_cell={
                    'background': '#1a1f3a', 'color': '#fff', 'padding': '12px',
                    'font-size': '14px', 'border': '1px solid #2a3050'
                },
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
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=df['symbol'], y=df['net_pnl'], marker_color=colors,
                text=[f"${pnl:.2f}" for pnl in df['net_pnl']], textposition='auto'
            ))
            
            fig.update_layout(
                title="Net PnL by Position", xaxis_title="Symbol", yaxis_title="Net PnL ($)",
                plot_bgcolor='#1a1f3a', paper_bgcolor='#1a1f3a', font=dict(color='#fff'),
                showlegend=False
            )
            fig.update_xaxes(gridcolor='#2a3050')
            fig.update_yaxes(gridcolor='#2a3050', zeroline=True, zerolinecolor='#8892b0')
        else:
            fig = go.Figure()
            fig.update_layout(
                title="No positions to display", plot_bgcolor='#1a1f3a',
                paper_bgcolor='#1a1f3a', font=dict(color='#fff')
            )
        
        # Update time
        starting_equity = data.get("starting_equity", 0)
        update_time = f"Last update: {data.get('last_update', 'Never')} | Market: {market_data.get('last_market_update', 'Never')}"
        if starting_equity:
            update_time += f" | Daily baseline: ${starting_equity:.2f}"
        
        return mode, stats_cards, performance_title, market_comparison, table, fig, update_time
        
    except Exception as e:
        print(f"Callback error: {e}")
        # Return safe defaults that match expected structure
        return "ERROR", [], "Error", html.Div("Error loading data"), html.Div("Error"), go.Figure(), "Error"

if __name__ == '__main__':
    print("🚀 Starting Plotly Dashboard → http://localhost:8050")
    print("📦 Required packages: pip install dash plotly pandas yfinance python-dotenv pybit")
    app.run(host='0.0.0.0', port=8050, debug=False)