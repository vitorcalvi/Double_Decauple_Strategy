#!/usr/bin/env python3
"""
Bybit Position Dashboard - Monitor all positions in real-time
"""

import os
import time
import shutil
from datetime import datetime
from dotenv import load_dotenv
from pybit.unified_trading import HTTP
import pandas as pd

load_dotenv(override=True)


class PositionDashboard:
    def __init__(self, demo_mode=True):
        self.demo_mode = demo_mode
        prefix = "TESTNET_" if demo_mode else "LIVE_"
        self.api_key = os.getenv(f"{prefix}BYBIT_API_KEY", "")
        self.api_secret = os.getenv(f"{prefix}BYBIT_API_SECRET", "")
        self.exchange = None
        self.positions = []
        self.account_data = {}
        self.refresh_interval = 2  # seconds
        
    def connect(self):
        """Connect to Bybit API"""
        try:
            self.exchange = HTTP(
                demo=self.demo_mode,
                api_key=self.api_key,
                api_secret=self.api_secret
            )
            resp = self.exchange.get_server_time()
            return resp.get("retCode") == 0
        except Exception as e:
            print(f"Connection error: {e}")
            return False
    
    def get_positions(self):
        """Fetch all open positions"""
        try:
            resp = self.exchange.get_positions(
                category="linear",
                settleCoin="USDT"
            )
            if resp.get("retCode") == 0:
                positions = resp["result"]["list"]
                # Filter only positions with size > 0
                self.positions = [p for p in positions if float(p.get("size", 0)) > 0]
                return True
            return False
        except Exception as e:
            print(f"Error fetching positions: {e}")
            return False
    
    def get_account_info(self):
        """Fetch account balance and info"""
        try:
            resp = self.exchange.get_wallet_balance(
                accountType="UNIFIED",
                coin="USDT"
            )
            if resp.get("retCode") == 0:
                for coin_data in resp["result"]["list"][0]["coin"]:
                    if coin_data["coin"] == "USDT":
                        self.account_data = {
                            "equity": float(coin_data["equity"]),
                            "available": float(coin_data["availableToWithdraw"]),
                            "used_margin": float(coin_data["locked"]),
                            "unrealized_pnl": float(coin_data["unrealisedPnl"]),
                            "realized_pnl": float(coin_data["cumRealisedPnl"])
                        }
                return True
        except Exception:
            return False
    
    def get_current_price(self, symbol):
        """Get current market price for symbol"""
        try:
            resp = self.exchange.get_tickers(
                category="linear",
                symbol=symbol
            )
            if resp.get("retCode") == 0:
                return float(resp["result"]["list"][0]["lastPrice"])
        except Exception:
            return 0.0
    
    def calculate_position_metrics(self, position):
        """Calculate PnL and other metrics for a position"""
        symbol = position["symbol"]
        side = position["side"]
        size = float(position["size"])
        avg_price = float(position["avgPrice"])
        mark_price = float(position["markPrice"])
        
        # PnL calculation
        if side == "Buy":
            pnl = (mark_price - avg_price) * size
            pnl_pct = ((mark_price - avg_price) / avg_price) * 100
        else:
            pnl = (avg_price - mark_price) * size
            pnl_pct = ((avg_price - mark_price) / avg_price) * 100
        
        return {
            "symbol": symbol,
            "side": side,
            "size": size,
            "avg_price": avg_price,
            "mark_price": mark_price,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "value": size * mark_price,
            "margin": float(position.get("positionIM", 0)),
            "liq_price": float(position.get("liqPrice", 0) or 0)
        }
    
    def clear_screen(self):
        """Clear terminal screen"""
        print("\x1b[2J\x1b[H", end="")
    
    def print_dashboard(self):
        """Print formatted dashboard"""
        cols = shutil.get_terminal_size((120, 30)).columns
        
        self.clear_screen()
        
        # Header
        print("=" * cols)
        mode = "TESTNET" if self.demo_mode else "LIVE"
        print(f" BYBIT POSITION DASHBOARD [{mode}] - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ".center(cols))
        print("=" * cols)
        
        # Account Summary
        if self.account_data:
            print("\n📊 ACCOUNT SUMMARY")
            print("-" * cols)
            print(f"Equity: ${self.account_data['equity']:.2f} | "
                  f"Available: ${self.account_data['available']:.2f} | "
                  f"Margin Used: ${self.account_data['used_margin']:.2f}")
            print(f"Unrealized PnL: ${self.account_data['unrealized_pnl']:+.2f} | "
                  f"Realized PnL Today: ${self.account_data['realized_pnl']:+.2f}")
        
        # Positions
        print("\n📈 OPEN POSITIONS")
        print("-" * cols)
        
        if not self.positions:
            print("No open positions".center(cols))
        else:
            # Create DataFrame for better formatting
            data = []
            total_pnl = 0
            
            for pos in self.positions:
                metrics = self.calculate_position_metrics(pos)
                total_pnl += metrics["pnl"]
                data.append([
                    metrics["symbol"],
                    metrics["side"],
                    f"{metrics['size']:.4f}",
                    f"${metrics['avg_price']:.4f}",
                    f"${metrics['mark_price']:.4f}",
                    f"${metrics['pnl']:+.2f}",
                    f"{metrics['pnl_pct']:+.2f}%",
                    f"${metrics['value']:.2f}",
                    f"${metrics['liq_price']:.4f}" if metrics['liq_price'] > 0 else "—"
                ])
            
            # Print table header
            headers = ["Symbol", "Side", "Size", "Entry", "Mark", "PnL $", "PnL %", "Value", "Liq Price"]
            col_widths = [12, 6, 12, 12, 12, 12, 10, 12, 12]
            
            header_line = ""
            for header, width in zip(headers, col_widths):
                header_line += header.ljust(width)
            print(header_line)
            print("-" * sum(col_widths))
            
            # Print positions
            for row in data:
                line = ""
                for item, width in zip(row, col_widths):
                    # Color PnL
                    if "+" in str(item) and "$" in str(item):
                        line += f"\033[92m{item.ljust(width)}\033[0m"  # Green
                    elif "-" in str(item) and "$" in str(item):
                        line += f"\033[91m{item.ljust(width)}\033[0m"  # Red
                    elif "+" in str(item) and "%" in str(item):
                        line += f"\033[92m{item.ljust(width)}\033[0m"  # Green
                    elif "-" in str(item) and "%" in str(item):
                        line += f"\033[91m{item.ljust(width)}\033[0m"  # Red
                    else:
                        line += str(item).ljust(width)
                print(line)
            
            # Summary
            print("-" * sum(col_widths))
            color = "\033[92m" if total_pnl >= 0 else "\033[91m"
            print(f"TOTAL PnL: {color}${total_pnl:+.2f}\033[0m | Positions: {len(self.positions)}")
        
        print("\n" + "=" * cols)
        print("Press Ctrl+C to exit | Refreshing every 2 seconds")
    
    def run(self):
        """Main loop"""
        if not self.connect():
            print("❌ Failed to connect to Bybit")
            return
        
        print("✅ Connected to Bybit")
        print("Loading positions...")
        
        try:
            while True:
                # Fetch data
                self.get_positions()
                self.get_account_info()
                
                # Display
                self.print_dashboard()
                
                # Wait
                time.sleep(self.refresh_interval)
                
        except KeyboardInterrupt:
            print("\n\n👋 Dashboard stopped")


if __name__ == "__main__":
    # Set to False for LIVE trading
    dashboard = PositionDashboard(demo_mode=True)
    dashboard.run()