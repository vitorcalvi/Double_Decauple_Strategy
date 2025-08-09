#!/usr/bin/env python3
"""
Bybit Plotly Dashboard - With actual fee tracking from executed orders
"""

import os
import time
import threading
from datetime import datetime
import dash
from dash import dcc, html, Input, Output, callback, dash_table
import plotly.graph_objects as go
import plotly.express as px
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
    
    def safe_float(self, val, default=0):
        """Convert to float, handling empty strings and None"""
        if val in [None, '', 'null']:
            return default
        try:
            return float(val)
        except (ValueError, TypeError):
            return default
    
    def get_position_fees(self, symbol, size, avg_price):
        """Get actual fees paid for current position from recent executed orders"""
        try:
            orders = self.exchange.get_order_history(
                category="linear",
                symbol=symbol,
                orderStatus="Filled",
                limit=200
            )
            
            if orders.get("retCode") != 0:
                maker_fee_rate = 0.0002
                return size * avg_price * maker_fee_rate
            
            total_fees = 0
            total_qty = 0
            order_list = orders.get("result", {}).get("list", [])
            
            for order in order_list:
                order_qty = self.safe_float(order.get("cumExecQty"))
                order_fee = self.safe_float(order.get("cumExecFee"))
                
                if total_qty < size:
                    qty_to_count = min(order_qty, size - total_qty)
                    fee_proportion = qty_to_count / order_qty if order_qty > 0 else 0
                    total_fees += abs(order_fee * fee_proportion)
                    total_qty += qty_to_count
                    
                    if total_qty >= size:
                        break
            
            if total_qty < size:
                remaining_qty = size - total_qty
                maker_fee_rate = 0.0002
                total_fees += remaining_qty * avg_price * maker_fee_rate
            
            return total_fees
            
        except Exception as e:
            print(f"Error getting fees for {symbol}: {e}")
            maker_fee_rate = 0.0002
            return size * avg_price * maker_fee_rate
    
    def calculate_breakeven(self, entry_price, size, fees, side):
        """Calculate break-even price including actual fees paid"""
        if size <= 0:
            return entry_price
        
        entry_fee_paid = fees
        exit_fee_estimate = size * entry_price * 0.0002
        total_fees = entry_fee_paid + exit_fee_estimate
        fee_per_unit = total_fees / size
        
        if side == "Buy":
            return entry_price + fee_per_unit
        else:
            return entry_price - fee_per_unit
    
    def fetch_market_data(self):
        """Fetch SPY and BTC real-time data"""
        try:
            # Fetch SPY (S&P 500)
            spy = yf.Ticker("SPY")
            spy_info = spy.history(period="2d")  # Get 2 days to calculate daily change
            if len(spy_info) >= 2:
                spy_current = spy_info['Close'].iloc[-1]
                spy_previous = spy_info['Close'].iloc[-2]
                spy_change = spy_current - spy_previous
                spy_change_pct = (spy_change / spy_previous) * 100
                
                self.market_data["SPY"] = {
                    "price": spy_current,
                    "change": spy_change,
                    "change_pct": spy_change_pct,
                    "previous_close": spy_previous
                }
            
            # Fetch BTC
            btc = yf.Ticker("BTC-USD")
            btc_info = btc.history(period="2d")
            if len(btc_info) >= 2:
                btc_current = btc_info['Close'].iloc[-1]
                btc_previous = btc_info['Close'].iloc[-2]
                btc_change = btc_current - btc_previous
                btc_change_pct = (btc_change / btc_previous) * 100
                
                self.market_data["BTC"] = {
                    "price": btc_current,
                    "change": btc_change,
                    "change_pct": btc_change_pct,
                    "previous_close": btc_previous
                }
            
            self.market_data["last_market_update"] = datetime.now().strftime('%H:%M:%S')
            
        except Exception as e:
            print(f"Market data fetch error: {e}")
    
    
    def fetch_data(self):
        try:
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
                            pnl_pct = (unrealized_pnl / position_value) * 100 if position_value > 0 else 0
                            
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
                        print(f"Error processing position: {e}")
                        continue
            
            # Fetch account
            acc_resp = self.exchange.get_wallet_balance(accountType="UNIFIED", coin="USDT")
            account = {}
            
            if acc_resp.get("retCode") == 0:
                try:
                    for coin in acc_resp.get("result", {}).get("list", [{}])[0].get("coin", []):
                        if coin.get("coin") == "USDT":
                            account = {
                                "equity": self.safe_float(coin.get("equity")),
                                "available": self.safe_float(coin.get("availableToWithdraw")),
                                "unrealized_pnl": self.safe_float(coin.get("unrealisedPnl"))
                            }
                            break
                except Exception:
                    pass
            
            self.data = {
                "positions": positions,
                "account": account,
                "total_pnl": sum(p["net_pnl"] for p in positions),
                "position_count": len(positions),
                "last_update": datetime.now().strftime('%H:%M:%S'),
                "demo_mode": self.demo_mode
            }
            
            # Also fetch market data
            self.fetch_market_data()
            
        except Exception as e:
            print(f"Fetch error: {e}")
    
    def close_position(self, symbol):
        """Close a specific position"""
        try:
            pos_resp = self.exchange.get_positions(category="linear", symbol=symbol)
            
            if pos_resp.get("retCode") != 0:
                return {"success": False, "message": "Failed to get position"}
            
            positions = pos_resp.get("result", {}).get("list", [])
            if not positions or self.safe_float(positions[0].get("size")) == 0:
                return {"success": False, "message": "No open position"}
            
            position = positions[0]
            side = "Sell" if position.get("side") == "Buy" else "Buy"
            qty = position.get("size")
            
            order_resp = self.exchange.place_order(
                category="linear",
                symbol=symbol,
                side=side,
                orderType="Market",
                qty=qty,
                reduceOnly=True
            )
            
            if order_resp.get("retCode") == 0:
                return {"success": True, "message": f"Position closed for {symbol}"}
            else:
                return {"success": False, "message": order_resp.get("retMsg", "Failed to close")}
                
        except Exception as e:
            return {"success": False, "message": str(e)}

# Initialize provider
provider = BybitDataProvider(demo_mode=True)

def update_loop():
    while True:
        provider.fetch_data()
        time.sleep(2)

threading.Thread(target=update_loop, daemon=True).start()

# Initialize Dash app
app = dash.Dash(__name__)
app.title = "Bybit Dashboard"

app.layout = html.Div([
    dcc.Interval(id='interval-component', interval=2000, n_intervals=0),
    
    # Header
    html.Div([
        html.H1("📈 Bybit Position Dashboard", style={'color': '#fff', 'margin': 0}),
        html.Div(id="mode-badge", style={'background': 'rgba(255,255,255,0.2)', 'padding': '5px 15px', 'border-radius': '20px', 'font-weight': 'bold'})
    ], style={'background': 'linear-gradient(135deg, #667eea, #764ba2)', 'padding': '20px', 'border-radius': '10px', 'margin-bottom': '20px', 'display': 'flex', 'justify-content': 'space-between', 'align-items': 'center'}),
    
    # Info note
    html.Div("ℹ️ Fees shown are actual fees paid for current positions (limit orders: 0.02%, market orders: 0.055%)", 
             style={'background': '#1f2547', 'padding': '10px', 'border-radius': '5px', 'margin-bottom': '20px', 'font-size': '12px', 'color': '#8892b0'}),
    
    # Stats cards
    html.Div(id="stats-cards", style={'display': 'grid', 'grid-template-columns': 'repeat(auto-fit, minmax(200px, 1fr))', 'gap': '15px', 'margin-bottom': '20px'}),
    
    # Market comparison
    html.Div([
        html.H3("📊 Buy & Hold vs Trading (Today)", style={'color': '#fff', 'margin-bottom': '15px'}),
        html.Div(id="market-comparison", style={'display': 'grid', 'grid-template-columns': 'repeat(auto-fit, minmax(250px, 1fr))', 'gap': '15px'})
    ], style={'margin-bottom': '30px'}),
    
    # Positions table
    html.Div(id="positions-table"),
    
    # PnL Chart
    html.Div([
        html.H3("PnL Distribution", style={'color': '#fff', 'margin-bottom': '10px'}),
        dcc.Graph(id="pnl-chart")
    ], style={'margin-top': '30px'}),
    
    # Update timestamp
    html.Div(id="update-time", style={'text-align': 'center', 'color': '#8892b0', 'margin-top': '20px', 'font-size': '12px'})
    
], style={'font-family': 'system-ui, -apple-system, sans-serif', 'background': '#0a0e27', 'color': '#fff', 'padding': '20px', 'min-height': '100vh'})

@callback(
    Output('mode-badge', 'children'),
    Output('stats-cards', 'children'), 
    Output('market-comparison', 'children'),
    Output('positions-table', 'children'),
    Output('pnl-chart', 'figure'),
    Output('update-time', 'children'),
    Input('interval-component', 'n_intervals')
)
def update_dashboard(n):
    data = provider.data
    market_data = provider.market_data
    
    # Mode badge
    mode = "TESTNET" if data.get("demo_mode", True) else "LIVE"
    
    # Stats cards
    account = data.get("account", {})
    total_fees = sum(p.get("fees", 0) for p in data.get("positions", []))
    
    stats_cards = [
        html.Div([
            html.Div("Equity", style={'color': '#8892b0', 'font-size': '12px', 'text-transform': 'uppercase', 'margin-bottom': '5px'}),
            html.Div(f"${account.get('equity', 0):.2f}", style={'font-size': '24px', 'font-weight': 'bold'})
        ], style={'background': '#1a1f3a', 'padding': '20px', 'border-radius': '10px', 'border': '1px solid #2a3050'}),
        
        html.Div([
            html.Div("Available", style={'color': '#8892b0', 'font-size': '12px', 'text-transform': 'uppercase', 'margin-bottom': '5px'}),
            html.Div(f"${account.get('available', 0):.2f}", style={'font-size': '24px', 'font-weight': 'bold'})
        ], style={'background': '#1a1f3a', 'padding': '20px', 'border-radius': '10px', 'border': '1px solid #2a3050'}),
        
        html.Div([
            html.Div("Unrealized PnL", style={'color': '#8892b0', 'font-size': '12px', 'text-transform': 'uppercase', 'margin-bottom': '5px'}),
            html.Div(f"${account.get('unrealized_pnl', 0):.2f}", 
                    style={'font-size': '24px', 'font-weight': 'bold', 'color': '#00d4aa' if account.get('unrealized_pnl', 0) >= 0 else '#f6465d'})
        ], style={'background': '#1a1f3a', 'padding': '20px', 'border-radius': '10px', 'border': '1px solid #2a3050'}),
        
        html.Div([
            html.Div("Total Fees Paid", style={'color': '#8892b0', 'font-size': '12px', 'text-transform': 'uppercase', 'margin-bottom': '5px'}),
            html.Div(f"${total_fees:.2f}", style={'font-size': '24px', 'font-weight': 'bold', 'color': '#ffa500'})
        ], style={'background': '#1a1f3a', 'padding': '20px', 'border-radius': '10px', 'border': '1px solid #2a3050'}),
        
        html.Div([
            html.Div("Positions", style={'color': '#8892b0', 'font-size': '12px', 'text-transform': 'uppercase', 'margin-bottom': '5px'}),
            html.Div(str(data.get("position_count", 0)), style={'font-size': '24px', 'font-weight': 'bold'})
        ], style={'background': '#1a1f3a', 'padding': '20px', 'border-radius': '10px', 'border': '1px solid #2a3050'})
    ]
    
    # Market comparison cards
    spy_data = market_data.get("SPY", {})
    btc_data = market_data.get("BTC", {})
    total_trading_pnl = data.get("total_pnl", 0)
    
    # Calculate trading performance as percentage (assume $10k account for comparison)
    account_equity = data.get("account", {}).get("equity", 10000)
    trading_pnl_pct = (total_trading_pnl / account_equity) * 100 if account_equity > 0 else 0
    
    market_comparison = [
        # SPY Card
        html.Div([
            html.Div([
                html.H4("SPY (S&P 500)", style={'margin': 0, 'color': '#fff'}),
                html.Div("Buy & Hold", style={'color': '#8892b0', 'font-size': '12px'})
            ]),
            html.Div([
                html.Div(f"${spy_data.get('price', 0):.2f}", style={'font-size': '24px', 'font-weight': 'bold', 'color': '#fff'}),
                html.Div([
                    html.Span(f"${spy_data.get('change', 0):+.2f} ", style={'color': '#00d4aa' if spy_data.get('change', 0) >= 0 else '#f6465d'}),
                    html.Span(f"({spy_data.get('change_pct', 0):+.2f}%)", style={'color': '#00d4aa' if spy_data.get('change_pct', 0) >= 0 else '#f6465d'})
                ], style={'font-size': '14px'})
            ])
        ], style={'background': '#1a1f3a', 'padding': '20px', 'border-radius': '10px', 'border': '1px solid #2a3050'}),
        
        # BTC Card
        html.Div([
            html.Div([
                html.H4("BTC", style={'margin': 0, 'color': '#fff'}),
                html.Div("Buy & Hold", style={'color': '#8892b0', 'font-size': '12px'})
            ]),
            html.Div([
                html.Div(f"${btc_data.get('price', 0):,.2f}", style={'font-size': '24px', 'font-weight': 'bold', 'color': '#fff'}),
                html.Div([
                    html.Span(f"${btc_data.get('change', 0):+,.2f} ", style={'color': '#00d4aa' if btc_data.get('change', 0) >= 0 else '#f6465d'}),
                    html.Span(f"({btc_data.get('change_pct', 0):+.2f}%)", style={'color': '#00d4aa' if btc_data.get('change_pct', 0) >= 0 else '#f6465d'})
                ], style={'font-size': '14px'})
            ])
        ], style={'background': '#1a1f3a', 'padding': '20px', 'border-radius': '10px', 'border': '1px solid #2a3050'}),
        
        # Trading Performance Card
        html.Div([
            html.Div([
                html.H4("Your Trading", style={'margin': 0, 'color': '#fff'}),
                html.Div("1-Day Positions", style={'color': '#8892b0', 'font-size': '12px'})
            ]),
            html.Div([
                html.Div(f"${total_trading_pnl:+.2f}", style={'font-size': '24px', 'font-weight': 'bold', 'color': '#00d4aa' if total_trading_pnl >= 0 else '#f6465d'}),
                html.Div(f"({trading_pnl_pct:+.2f}%)", style={'font-size': '14px', 'color': '#00d4aa' if trading_pnl_pct >= 0 else '#f6465d'})
            ])
        ], style={'background': '#1a1f3a', 'padding': '20px', 'border-radius': '10px', 'border': '1px solid #764ba2'})  # Different border for trading
    ]
    
    # Positions table
    positions = data.get("positions", [])
    if not positions:
        table = html.Div("No open positions", style={'text-align': 'center', 'padding': '40px', 'background': '#1a1f3a', 'border-radius': '10px', 'border': '1px solid #2a3050'})
    else:
        # Prepare data for table
        table_data = []
        for p in positions:
            table_data.append({
                'Symbol': p['symbol'],
                'Side': p['side'],
                'Size': f"{p['size']:.4f}",
                'Entry': f"${p['avg_price']:.4f}",
                'Mark': f"${p['mark_price']:.4f}",
                'Unrealized PnL': f"${p['pnl']:.2f}",
                'Fees Paid': f"${p['fees']:.2f}",
                'Net PnL': f"${p['net_pnl']:.2f}",
                'PnL %': f"{p['pnl_pct']:.2f}%",
                'Break-even': f"${p['breakeven']:.4f}",
                'Value': f"${p['value']:.2f}",
                'Liq Price': f"${p['liq_price']:.4f}" if p['liq_price'] > 0 else "—"
            })
        
        # Add total row
        total_pnl = data.get("total_pnl", 0)
        table_data.append({
            'Symbol': 'TOTAL',
            'Side': '',
            'Size': '',
            'Entry': '',
            'Mark': '',
            'Unrealized PnL': '',
            'Fees Paid': '',
            'Net PnL': f"${total_pnl:.2f}",
            'PnL %': '',
            'Break-even': '',
            'Value': '',
            'Liq Price': ''
        })
        
        table = dash_table.DataTable(
            data=table_data,
            columns=[{"name": i, "id": i} for i in table_data[0].keys()],
            style_table={'background': '#1a1f3a', 'border': '1px solid #2a3050', 'border-radius': '10px', 'overflow': 'hidden'},
            style_header={'background': '#0f1529', 'color': '#8892b0', 'font-size': '11px', 'text-transform': 'uppercase', 'padding': '12px', 'font-weight': 'bold'},
            style_cell={'background': '#1a1f3a', 'color': '#fff', 'padding': '12px', 'font-size': '14px', 'border': '1px solid #2a3050'},
            style_data_conditional=[
                {
                    'if': {'row_index': len(positions)},  # Total row
                    'background': '#0f1529',
                    'font-weight': 'bold'
                },
                {
                    'if': {'filter_query': '{Side} = Buy'},
                    'color': '#00d4aa'
                },
                {
                    'if': {'filter_query': '{Side} = Sell'},
                    'color': '#f6465d'
                }
            ]
        )
    
    # PnL Chart
    if positions:
        df = pd.DataFrame(positions)
        
        # Create bar chart for PnL by symbol
        fig = go.Figure()
        
        colors = ['#00d4aa' if pnl >= 0 else '#f6465d' for pnl in df['net_pnl']]
        
        fig.add_trace(go.Bar(
            x=df['symbol'],
            y=df['net_pnl'],
            marker_color=colors,
            text=[f"${pnl:.2f}" for pnl in df['net_pnl']],
            textposition='auto'
        ))
        
        fig.update_layout(
            title="Net PnL by Position",
            xaxis_title="Symbol",
            yaxis_title="Net PnL ($)",
            plot_bgcolor='#1a1f3a',
            paper_bgcolor='#1a1f3a',
            font=dict(color='#fff'),
            showlegend=False
        )
        
        fig.update_xaxes(gridcolor='#2a3050')
        fig.update_yaxes(gridcolor='#2a3050', zeroline=True, zerolinecolor='#8892b0')
    else:
        fig = go.Figure()
        fig.update_layout(
            title="No positions to display",
            plot_bgcolor='#1a1f3a',
            paper_bgcolor='#1a1f3a',
            font=dict(color='#fff')
        )
    
    # Update time
    update_time = f"Last update: {data.get('last_update', 'Never')} | Market: {market_data.get('last_market_update', 'Never')}"
    
    return mode, stats_cards, market_comparison, table, fig, update_time

if __name__ == '__main__':
    print("🚀 Starting Plotly Dashboard → http://localhost:8050")
    print("📦 Required packages: pip install dash plotly pandas yfinance python-dotenv pybit")
    app.run(host='0.0.0.0', port=8050, debug=False)