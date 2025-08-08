#!/usr/bin/env python3
"""
Bybit Web Dashboard - With actual fee tracking from executed orders
"""

import os
import time
import threading
from datetime import datetime
from flask import Flask, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv
from pybit.unified_trading import HTTP

load_dotenv(override=True)

app = Flask(__name__)
CORS(app)

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
            # Get filled orders for this symbol
            orders = self.exchange.get_order_history(
                category="linear",
                symbol=symbol,
                orderStatus="Filled",
                limit=200  # Increased to catch more orders
            )
            
            if orders.get("retCode") != 0:
                # Fallback to estimate
                maker_fee_rate = 0.0002  # 0.02% limit order
                return size * avg_price * maker_fee_rate
            
            # Calculate total fees from recent orders that match position size
            total_fees = 0
            total_qty = 0
            
            # Get orders sorted by time (most recent first)
            order_list = orders.get("result", {}).get("list", [])
            
            for order in order_list:
                order_qty = self.safe_float(order.get("cumExecQty"))
                order_fee = self.safe_float(order.get("cumExecFee"))
                
                # Add fees from orders until we match position size
                if total_qty < size:
                    # Check if this order contributes to current position
                    qty_to_count = min(order_qty, size - total_qty)
                    fee_proportion = qty_to_count / order_qty if order_qty > 0 else 0
                    
                    total_fees += abs(order_fee * fee_proportion)
                    total_qty += qty_to_count
                    
                    if total_qty >= size:
                        break
            
            # If we couldn't find enough orders, estimate remaining fees
            if total_qty < size:
                remaining_qty = size - total_qty
                maker_fee_rate = 0.0002  # 0.02% for limit orders
                total_fees += remaining_qty * avg_price * maker_fee_rate
            
            return total_fees
            
        except Exception as e:
            print(f"Error getting fees for {symbol}: {e}")
            # Fallback: estimate based on limit order fee
            maker_fee_rate = 0.0002
            return size * avg_price * maker_fee_rate
    
    def calculate_breakeven(self, entry_price, size, fees, side):
        """Calculate break-even price including actual fees paid"""
        if size <= 0:
            return entry_price
        
        # Use actual fees paid + estimated exit fee
        entry_fee_paid = fees
        # Estimate exit fee at limit order rate (0.02%)
        exit_fee_estimate = size * entry_price * 0.0002
        total_fees = entry_fee_paid + exit_fee_estimate
        
        # Calculate fee impact per unit
        fee_per_unit = total_fees / size
        
        if side == "Buy":
            # For long: need price to go up to cover fees
            return entry_price + fee_per_unit
        else:
            # For short: need price to go down to cover fees  
            return entry_price - fee_per_unit
    
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
                            
                            # Use unrealisedPnl from API if available
                            unrealized_pnl = self.safe_float(p.get("unrealisedPnl"))
                            
                            # If not available, calculate it
                            if unrealized_pnl == 0:
                                is_buy = side == "Buy"
                                unrealized_pnl = (mark_price - avg_price) * size if is_buy else (avg_price - mark_price) * size
                            
                            # Get actual fees for this position
                            fees = self.get_position_fees(symbol, size, avg_price)
                            
                            # Calculate break-even
                            breakeven = self.calculate_breakeven(avg_price, size, fees, side)
                            
                            # Net PnL after estimated fees
                            net_pnl = unrealized_pnl - fees
                            
                            # PnL percentage
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
        except Exception as e:
            print(f"Fetch error: {e}")
    
    def close_position(self, symbol):
        """Close a specific position"""
        try:
            # Get current position
            pos_resp = self.exchange.get_positions(
                category="linear",
                symbol=symbol
            )
            
            if pos_resp.get("retCode") != 0:
                return {"success": False, "message": "Failed to get position"}
            
            positions = pos_resp.get("result", {}).get("list", [])
            if not positions or self.safe_float(positions[0].get("size")) == 0:
                return {"success": False, "message": "No open position"}
            
            position = positions[0]
            side = "Sell" if position.get("side") == "Buy" else "Buy"
            qty = position.get("size")
            
            # Place market order to close
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

# Initialize and start background updates
provider = BybitDataProvider(demo_mode=True)

def update_loop():
    while True:
        provider.fetch_data()
        time.sleep(2)

threading.Thread(target=update_loop, daemon=True).start()

@app.route('/')
def index():
    return '''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Bybit Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: system-ui, -apple-system, sans-serif; background: #0a0e27; color: #fff; padding: 20px; }
        .container { max-width: 1600px; margin: 0 auto; }
        .header { background: linear-gradient(135deg, #667eea, #764ba2); padding: 20px; border-radius: 10px; margin-bottom: 20px; display: flex; justify-content: space-between; align-items: center; }
        .mode { background: rgba(255,255,255,0.2); padding: 5px 15px; border-radius: 20px; font-weight: bold; }
        .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 30px; }
        .card { background: #1a1f3a; padding: 20px; border-radius: 10px; border: 1px solid #2a3050; }
        .label { color: #8892b0; font-size: 12px; text-transform: uppercase; margin-bottom: 5px; }
        .value { font-size: 24px; font-weight: bold; }
        .positive { color: #00d4aa; }
        .negative { color: #f6465d; }
        table { width: 100%; background: #1a1f3a; border-radius: 10px; overflow: hidden; border: 1px solid #2a3050; border-collapse: collapse; }
        th { background: #0f1529; padding: 12px; text-align: left; color: #8892b0; font-size: 11px; text-transform: uppercase; white-space: nowrap; }
        td { padding: 12px; border-top: 1px solid #2a3050; font-size: 14px; }
        tr:hover { background: rgba(102, 126, 234, 0.1); }
        .buy { color: #00d4aa; font-weight: bold; }
        .sell { color: #f6465d; font-weight: bold; }
        .update { text-align: center; color: #8892b0; margin-top: 20px; font-size: 12px; }
        .total { background: #0f1529; font-weight: bold; }
        .btn-close { 
            background: #f6465d; 
            color: white; 
            border: none; 
            padding: 6px 12px; 
            border-radius: 5px; 
            cursor: pointer; 
            font-size: 12px;
            font-weight: bold;
            transition: all 0.2s;
        }
        .btn-close:hover { background: #d73547; transform: scale(1.05); }
        .btn-close:disabled { background: #666; cursor: not-allowed; opacity: 0.5; }
        .fees { color: #ffa500; }
        .breakeven { color: #9b59b6; }
        .note { background: #1f2547; padding: 10px; border-radius: 5px; margin-bottom: 20px; font-size: 12px; color: #8892b0; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📈 Bybit Position Dashboard</h1>
            <div class="mode" id="mode">TESTNET</div>
        </div>
        <div class="note">ℹ️ Fees shown are actual fees paid for current positions (limit orders: 0.02%, market orders: 0.055%)</div>
        <div class="stats" id="stats"></div>
        <table>
            <thead>
                <tr>
                    <th>Symbol</th>
                    <th>Side</th>
                    <th>Size</th>
                    <th>Entry</th>
                    <th>Mark</th>
                    <th>Unrealized PnL</th>
                    <th>Fees Paid</th>
                    <th>Net PnL</th>
                    <th>PnL %</th>
                    <th>Break-even</th>
                    <th>Value</th>
                    <th>Liq Price</th>
                    <th>Action</th>
                </tr>
            </thead>
            <tbody id="positions"></tbody>
        </table>
        <div class="update" id="update">Loading...</div>
    </div>
    <script>
        const fmt = (n, d=2) => n?.toFixed(d) || '0.00';
        const cls = (v) => v >= 0 ? 'positive' : 'negative';
        
        async function closePosition(symbol) {
            if (!confirm(`Close position for ${symbol}?`)) return;
            
            const btn = document.getElementById(`btn-${symbol}`);
            btn.disabled = true;
            btn.textContent = 'Closing...';
            
            try {
                const response = await fetch('/api/close', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({symbol})
                });
                const result = await response.json();
                
                if (result.success) {
                    alert(result.message);
                } else {
                    alert(`Error: ${result.message}`);
                    btn.disabled = false;
                    btn.textContent = 'Close';
                }
            } catch (e) {
                alert('Failed to close position');
                btn.disabled = false;
                btn.textContent = 'Close';
            }
        }
        
        async function update() {
            try {
                const data = await (await fetch('/api/data')).json();
                
                document.getElementById('mode').textContent = data.demo_mode ? 'TESTNET' : 'LIVE';
                
                const totalFees = data.positions.reduce((sum, p) => sum + p.fees, 0);
                
                document.getElementById('stats').innerHTML = `
                    <div class="card"><div class="label">Equity</div><div class="value">${fmt(data.account.equity)}</div></div>
                    <div class="card"><div class="label">Available</div><div class="value">${fmt(data.account.available)}</div></div>
                    <div class="card"><div class="label">Unrealized PnL</div><div class="value ${cls(data.account.unrealized_pnl)}">${fmt(data.account.unrealized_pnl)}</div></div>
                    <div class="card"><div class="label">Total Fees Paid</div><div class="value fees">${fmt(totalFees)}</div></div>
                    <div class="card"><div class="label">Positions</div><div class="value">${data.position_count}</div></div>
                `;
                
                const tbody = document.getElementById('positions');
                if (data.positions.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="13" style="text-align:center">No open positions</td></tr>';
                } else {
                    tbody.innerHTML = data.positions.map(p => `
                        <tr>
                            <td><b>${p.symbol}</b></td>
                            <td class="${p.side.toLowerCase()}">${p.side}</td>
                            <td>${fmt(p.size, 4)}</td>
                            <td>$${fmt(p.avg_price, 4)}</td>
                            <td>$${fmt(p.mark_price, 4)}</td>
                            <td class="${cls(p.pnl)}">${p.pnl >= 0 ? '+' : ''}$${fmt(p.pnl)}</td>
                            <td class="fees">$${fmt(p.fees)}</td>
                            <td class="${cls(p.net_pnl)}">${p.net_pnl >= 0 ? '+' : ''}$${fmt(p.net_pnl)}</td>
                            <td class="${cls(p.pnl_pct)}">${p.pnl_pct >= 0 ? '+' : ''}${fmt(p.pnl_pct)}%</td>
                            <td class="breakeven">$${fmt(p.breakeven, 4)}</td>
                            <td>$${fmt(p.value)}</td>
                            <td>${p.liq_price > 0 ? '$' + fmt(p.liq_price, 4) : '—'}</td>
                            <td>
                                <button class="btn-close" id="btn-${p.symbol}" onclick="closePosition('${p.symbol}')">Close</button>
                            </td>
                        </tr>
                    `).join('') + `
                        <tr class="total">
                            <td colspan="7">TOTAL</td>
                            <td class="${cls(data.total_pnl)}">${data.total_pnl >= 0 ? '+' : ''}$${fmt(data.total_pnl)}</td>
                            <td colspan="5"></td>
                        </tr>
                    `;
                }
                
                document.getElementById('update').textContent = `Last update: ${data.last_update}`;
            } catch (e) {
                console.error(e);
            }
        }
        
        setInterval(update, 2000);
        update();
    </script>
</body>
</html>'''

@app.route('/api/data')
def get_data():
    return jsonify(provider.data)

@app.route('/api/close', methods=['POST'])
def close_position():
    try:
        symbol = request.json.get('symbol')
        if not symbol:
            return jsonify({"success": False, "message": "Symbol required"})
        
        result = provider.close_position(symbol)
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})

if __name__ == '__main__':
    print("🚀 Starting Fixed Dashboard → http://localhost:5501")
    app.run(host='0.0.0.0', port=5501, debug=False)