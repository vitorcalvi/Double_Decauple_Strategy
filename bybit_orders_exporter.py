#!/usr/bin/env python3
"""
Bybit Transaction Exporter to CSV
Exports all transactions from Bybit using API v5
"""

import csv
import time
import sys
from datetime import datetime, timedelta
from pybit.unified_trading import HTTP
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()

# Configuration
USE_TESTNET = True  # Set to False for mainnet
prefix = 'TESTNET_' if USE_TESTNET else ''
API_KEY = os.getenv(f'{prefix}BYBIT_API_KEY')
API_SECRET = os.getenv(f'{prefix}BYBIT_API_SECRET')

# Days to fetch (can be overridden by command line argument)
DAYS_TO_FETCH = 2
if len(sys.argv) > 1:
    try:
        DAYS_TO_FETCH = int(sys.argv[1])
    except ValueError:
        print(f"Invalid days argument. Using default: {DAYS_TO_FETCH}")

class BybitExporter:
    def __init__(self, api_key, api_secret, testnet=True, days=2):
        """Initialize Bybit session"""
        if not api_key or not api_secret:
            raise ValueError("API_KEY and API_SECRET must be provided")
            
        # Fixed: Use demo= instead of testnet=
        self.session = HTTP(
            demo=testnet,  # Changed from testnet= to demo=
            api_key=api_key,
            api_secret=api_secret
        )
        self.transactions = []
        
        # Calculate start time for filtering
        self.days = days
        self.start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
        
        # Test connection
        self.test_connection()
        
    def test_connection(self):
        """Test API connection and permissions"""
        try:
            response = self.session.get_wallet_balance(accountType="UNIFIED")
            if response["retCode"] == 0:
                print("✓ API connection successful")
                return True
            else:
                print(f"✗ API Error: {response['retMsg']}")
                return False
        except Exception as e:
            print(f"✗ Connection failed: {e}")
            return False
            
    def get_executions(self, category, limit=200):
        """Get trade execution history"""
        executions = []
        cursor = None
        
        while True:
            try:
                params = {
                    "category": category,
                    "limit": limit,
                    "startTime": self.start_time  # Filter by days
                }
                if cursor:
                    params["cursor"] = cursor
                    
                response = self.session.get_executions(**params)
                
                if response["retCode"] == 0:
                    data = response["result"]
                    if data["list"]:
                        executions.extend(data["list"])
                        print(f"  Found {len(data['list'])} {category} executions")
                    
                    cursor = data.get("nextPageCursor")
                    if not cursor:
                        break
                elif response["retCode"] == 10002:  # No data
                    print(f"  No {category} executions found")
                    break
                else:
                    print(f"  Error: {response['retMsg']}")
                    break
                    
                time.sleep(0.1)  # Rate limit
                
            except Exception as e:
                print(f"  Exception: {e}")
                break
                
        return executions
    
    def get_closed_pnl(self, category, limit=200):
        """Get closed P&L history"""
        pnl_records = []
        cursor = None
        
        while True:
            try:
                params = {
                    "category": category,
                    "limit": limit,
                    "startTime": self.start_time  # Filter by days
                }
                if cursor:
                    params["cursor"] = cursor
                    
                response = self.session.get_closed_pnl(**params)
                
                if response["retCode"] == 0:
                    data = response["result"]
                    if data["list"]:
                        pnl_records.extend(data["list"])
                        print(f"  Found {len(data['list'])} {category} P&L records")
                    
                    cursor = data.get("nextPageCursor")
                    if not cursor:
                        break
                elif response["retCode"] == 10002:  # No data
                    print(f"  No {category} P&L records found")
                    break
                else:
                    print(f"  Error: {response['retMsg']}")
                    break
                    
                time.sleep(0.1)  # Rate limit
                
            except Exception as e:
                print(f"  Exception: {e}")
                break
                
        return pnl_records
    
    def get_deposit_records(self, limit=50):
        """Get deposit history"""
        deposits = []
        cursor = None
        
        while True:
            try:
                params = {
                    "limit": limit,
                    "startTime": self.start_time  # Filter by days
                }
                if cursor:
                    params["cursor"] = cursor
                    
                response = self.session.get_deposit_records(**params)
                
                if response["retCode"] == 0:
                    data = response["result"]
                    if data["rows"]:
                        deposits.extend(data["rows"])
                        print(f"  Found {len(data['rows'])} deposits")
                    
                    cursor = data.get("nextPageCursor")
                    if not cursor:
                        break
                else:
                    break
                    
                time.sleep(0.1)
                
            except:
                break
                
        return deposits
    
    def get_withdrawal_records(self, limit=50):
        """Get withdrawal history"""
        withdrawals = []
        cursor = None
        
        while True:
            try:
                params = {
                    "limit": limit,
                    "startTime": self.start_time  # Filter by days
                }
                if cursor:
                    params["cursor"] = cursor
                    
                response = self.session.get_withdrawal_records(**params)
                
                if response["retCode"] == 0:
                    data = response["result"]
                    if data["rows"]:
                        withdrawals.extend(data["rows"])
                        print(f"  Found {len(data['rows'])} withdrawals")
                    
                    cursor = data.get("nextPageCursor")
                    if not cursor:
                        break
                else:
                    break
                    
                time.sleep(0.1)
                
            except:
                break
                
        return withdrawals
    
    def calculate_slippage(self, order_price, exec_price, side):
        """Calculate slippage between order and execution price"""
        try:
            order_p = float(order_price) if order_price else 0
            exec_p = float(exec_price) if exec_price else 0
            
            if order_p == 0 or exec_p == 0:
                return 0
            
            if side.lower() == "buy":
                # For buy orders, negative slippage means paying more
                slippage = (exec_p - order_p) / order_p * 100
            else:  # sell
                # For sell orders, negative slippage means receiving less
                slippage = (order_p - exec_p) / order_p * 100
            
            return round(slippage, 4)
        except:
            return 0
    
    def export_all_transactions(self):
        """Export all transactions from different categories"""
        print(f"\n=== Fetching Transactions (Last {self.days} days) ===")
        
        # Trading executions
        print("\n1. Trade Executions:")
        spot_executions = self.get_executions("spot")
        linear_executions = self.get_executions("linear")
        inverse_executions = self.get_executions("inverse")
        option_executions = self.get_executions("option")
        
        # P&L records
        print("\n2. Closed P&L:")
        linear_pnl = self.get_closed_pnl("linear")
        inverse_pnl = self.get_closed_pnl("inverse")
        
        # Deposits and withdrawals
        print("\n3. Deposits/Withdrawals:")
        deposits = self.get_deposit_records()
        withdrawals = self.get_withdrawal_records()
        
        # Process executions
        for exec in spot_executions + linear_executions + inverse_executions + option_executions:
            # Calculate slippage if order price is available
            order_price = exec.get("orderPrice", "0")
            exec_price = exec.get("execPrice", "0")
            side = exec.get("side", "")
            slippage = self.calculate_slippage(order_price, exec_price, side)
            
            # Get mark price for spread calculation (if available)
            mark_price = exec.get("markPrice", "")
            spread = 0
            if mark_price and exec_price:
                try:
                    spread = abs(float(exec_price) - float(mark_price))
                except:
                    spread = 0
            
            self.transactions.append({
                "timestamp": datetime.fromtimestamp(int(exec["execTime"])/1000).isoformat(),
                "type": "trade",
                "category": exec.get("category", "unknown"),
                "symbol": exec["symbol"],
                "side": exec["side"],
                "price": exec["execPrice"],
                "qty": exec.get("execQty", exec.get("orderQty", "0")),
                "value": exec.get("execValue", "0"),
                "fee": exec.get("execFee", "0"),
                "fee_currency": exec.get("feeCurrency", ""),
                "order_id": exec["orderId"],
                "exec_id": exec["execId"],
                "is_maker": exec.get("isMaker", False),
                "pnl": "",
                "slippage": slippage,
                "spread": spread,
                "coin": "",
                "tx_id": "",
                "status": "completed"
            })
        
        # Process P&L records
        for pnl in linear_pnl + inverse_pnl:
            self.transactions.append({
                "timestamp": datetime.fromtimestamp(int(pnl["updatedTime"])/1000).isoformat(),
                "type": "pnl",
                "category": pnl.get("category", "unknown"),
                "symbol": pnl["symbol"],
                "side": pnl["side"],
                "price": pnl.get("avgExitPrice", "0"),
                "qty": pnl.get("qty", "0"),
                "value": "",
                "fee": pnl.get("totalFee", "0"),
                "fee_currency": "",
                "order_id": pnl.get("orderId", ""),
                "exec_id": "",
                "is_maker": "",
                "pnl": pnl.get("closedPnl", "0"),
                "slippage": 0,
                "spread": 0,
                "coin": "",
                "tx_id": "",
                "status": "completed"
            })
        
        # Process deposits
        for dep in deposits:
            self.transactions.append({
                "timestamp": datetime.fromtimestamp(int(dep["successAt"])/1000).isoformat() if dep.get("successAt") else "",
                "type": "deposit",
                "category": "wallet",
                "symbol": "",
                "side": "",
                "price": "",
                "qty": dep.get("amount", "0"),
                "value": "",
                "fee": "",
                "fee_currency": "",
                "order_id": "",
                "exec_id": "",
                "is_maker": "",
                "pnl": "",
                "slippage": 0,
                "spread": 0,
                "coin": dep.get("coin", ""),
                "tx_id": dep.get("txID", ""),
                "status": dep.get("status", "")
            })
        
        # Process withdrawals
        for wit in withdrawals:
            self.transactions.append({
                "timestamp": datetime.fromtimestamp(int(wit["updatedTime"])/1000).isoformat(),
                "type": "withdrawal",
                "category": "wallet",
                "symbol": "",
                "side": "",
                "price": "",
                "qty": wit.get("amount", "0"),
                "value": "",
                "fee": wit.get("withdrawFee", "0"),
                "fee_currency": wit.get("coin", ""),
                "order_id": "",
                "exec_id": "",
                "is_maker": "",
                "pnl": "",
                "slippage": 0,
                "spread": 0,
                "coin": wit.get("coin", ""),
                "tx_id": wit.get("txID", ""),
                "status": wit.get("status", "")
            })
        
        # Sort by timestamp
        self.transactions.sort(key=lambda x: x["timestamp"] if x["timestamp"] else "0", reverse=True)
        
        return self.transactions
    
    def save_to_csv(self, filename=None):
        """Save transactions to CSV file"""
        if not self.transactions:
            print("\n✗ No transactions to export")
            return None
        
        # Default filename with days info
        if filename is None:
            filename = f"bybit_transactions_{self.days}days.csv"
        
        # Write to CSV
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            fieldnames = [
                "timestamp", "type", "category", "symbol", "side", 
                "price", "qty", "value", "fee", "fee_currency",
                "order_id", "exec_id", "is_maker", "pnl",
                "slippage", "spread",
                "coin", "tx_id", "status"
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.transactions)
        
        print(f"\n✓ Exported {len(self.transactions)} transactions to {filename}")
        
        # Create DataFrame for analysis
        df = pd.DataFrame(self.transactions)
        return df

def main():
    print("=== Bybit Transaction Exporter ===")
    print(f"Mode: {'TESTNET' if USE_TESTNET else 'MAINNET'}")
    print(f"Fetching last {DAYS_TO_FETCH} days of transactions")
    
    # Check API credentials
    if not API_KEY or not API_SECRET:
        print("\n✗ Error: API credentials not found in .env file")
        print("Please add to your .env file:")
        if USE_TESTNET:
            print("TESTNET_BYBIT_API_KEY=your_testnet_key")
            print("TESTNET_BYBIT_API_SECRET=your_testnet_secret")
        else:
            print("BYBIT_API_KEY=your_api_key")
            print("BYBIT_API_SECRET=your_api_secret")
        return
    
    try:
        # Initialize exporter with DAYS_TO_FETCH
        exporter = BybitExporter(API_KEY, API_SECRET, testnet=USE_TESTNET, days=DAYS_TO_FETCH)
        
        # Export all transactions
        transactions = exporter.export_all_transactions()
        
        # Save to CSV (will use default filename with days info)
        df = exporter.save_to_csv()
        
        # Print summary
        if df is not None and not df.empty:
            print("\n=== Export Summary ===")
            print(f"Total transactions: {len(df)}")
            
            # Filter out empty timestamps
            df_with_time = df[df['timestamp'] != '']
            if not df_with_time.empty:
                print(f"Date range: {df_with_time['timestamp'].min()} to {df_with_time['timestamp'].max()}")
            
            print(f"\nTransaction types:")
            print(df['type'].value_counts())
            
            if 'category' in df.columns:
                categories = df[df['category'] != '']['category'].value_counts()
                if not categories.empty:
                    print(f"\nCategories:")
                    print(categories)
                    
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Check your API key has the correct permissions:")
        print("   - Spot Trade")
        print("   - Contract Trade") 
        print("   - Exchange")
        print("2. Make sure you're using the correct environment (testnet vs mainnet)")
        print("3. Verify your .env file has the correct variable names")

if __name__ == "__main__":
    main()