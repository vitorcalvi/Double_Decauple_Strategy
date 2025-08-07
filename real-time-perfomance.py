#!/usr/bin/env python3
"""
Real-Time Trading Bot Performance Monitor
Track the impact of fixes on your bot performance
"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
import pandas as pd

class PerformanceMonitor:
    def __init__(self, log_dir="logs"):
        self.log_dir = Path(log_dir)
        self.baseline = {}  # Store baseline metrics
        self.current = {}   # Current metrics
        
    def parse_recent_logs(self, hours=24):
        """Parse logs from the last N hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        bot_metrics = defaultdict(lambda: {
            'trades': [],
            'fees': 0,
            'gross_pnl': 0,
            'net_pnl': 0,
            'market_orders': 0,
            'limit_orders': 0,
            'wins': 0,
            'losses': 0
        })
        
        for log_file in self.log_dir.glob("*.log"):
            bot_name = log_file.stem
            
            with open(log_file, 'r') as f:
                for line in f:
                    try:
                        trade = json.loads(line.strip())
                        
                        # Parse timestamp
                        ts = datetime.fromisoformat(trade['ts'].replace('Z', '+00:00'))
                        if ts < cutoff_time:
                            continue
                        
                        # Track metrics
                        if trade.get('action') == 'OPEN':
                            bot_metrics[bot_name]['trades'].append(ts)
                            
                            # Check order type (from info field)
                            if 'Market' in str(trade):
                                bot_metrics[bot_name]['market_orders'] += 1
                            else:
                                bot_metrics[bot_name]['limit_orders'] += 1
                        
                        elif trade.get('action') == 'CLOSE':
                            # Track P&L
                            gross = trade.get('gross_pnl', 0)
                            net = trade.get('net_pnl', 0)
                            
                            bot_metrics[bot_name]['gross_pnl'] += gross
                            bot_metrics[bot_name]['net_pnl'] += net
                            
                            # Calculate fees
                            if 'total_fees' in trade:
                                bot_metrics[bot_name]['fees'] += abs(trade['total_fees'])
                            elif 'fee_rebates' in trade:
                                rebates = trade['fee_rebates'].get('total', 0)
                                bot_metrics[bot_name]['fees'] += abs(rebates)
                            else:
                                bot_metrics[bot_name]['fees'] += (gross - net)
                            
                            # Track wins/losses
                            if net > 0:
                                bot_metrics[bot_name]['wins'] += 1
                            else:
                                bot_metrics[bot_name]['losses'] += 1
                    
                    except:
                        continue
        
        return bot_metrics
    
    def calculate_metrics(self, bot_data):
        """Calculate performance metrics"""
        metrics = {}
        
        for bot_name, data in bot_data.items():
            total_trades = len(data['trades'])
            
            if total_trades == 0:
                continue
            
            # Calculate trade frequency
            if data['trades']:
                first_trade = min(data['trades'])
                last_trade = max(data['trades'])
                hours_active = max((last_trade - first_trade).total_seconds() / 3600, 1)
                trades_per_hour = total_trades / hours_active
            else:
                trades_per_hour = 0
            
            # Calculate fee ratio
            fee_ratio = (data['fees'] / abs(data['gross_pnl']) * 100) if data['gross_pnl'] != 0 else 0
            
            # Calculate win rate
            total_closed = data['wins'] + data['losses']
            win_rate = (data['wins'] / total_closed * 100) if total_closed > 0 else 0
            
            # Order type ratio
            total_orders = data['market_orders'] + data['limit_orders']
            market_ratio = (data['market_orders'] / total_orders * 100) if total_orders > 0 else 0
            
            metrics[bot_name] = {
                'total_trades': total_trades,
                'trades_per_hour': round(trades_per_hour, 2),
                'gross_pnl': round(data['gross_pnl'], 2),
                'net_pnl': round(data['net_pnl'], 2),
                'total_fees': round(data['fees'], 2),
                'fee_ratio': round(fee_ratio, 2),
                'win_rate': round(win_rate, 2),
                'market_order_ratio': round(market_ratio, 2),
                'wins': data['wins'],
                'losses': data['losses']
            }
        
        return metrics
    
    def save_baseline(self):
        """Save current metrics as baseline"""
        bot_data = self.parse_recent_logs(24)
        self.baseline = self.calculate_metrics(bot_data)
        
        # Save to file
        with open('baseline_metrics.json', 'w') as f:
            json.dump(self.baseline, f, indent=2)
        
        print("✅ Baseline metrics saved")
        return self.baseline
    
    def load_baseline(self):
        """Load baseline metrics"""
        try:
            with open('baseline_metrics.json', 'r') as f:
                self.baseline = json.load(f)
            return True
        except:
            return False
    
    def compare_performance(self):
        """Compare current performance to baseline"""
        bot_data = self.parse_recent_logs(24)
        self.current = self.calculate_metrics(bot_data)
        
        comparison = {}
        
        for bot_name in set(list(self.baseline.keys()) + list(self.current.keys())):
            base = self.baseline.get(bot_name, {})
            curr = self.current.get(bot_name, {})
            
            if not base or not curr:
                continue
            
            comparison[bot_name] = {
                'fee_ratio_change': curr.get('fee_ratio', 0) - base.get('fee_ratio', 0),
                'win_rate_change': curr.get('win_rate', 0) - base.get('win_rate', 0),
                'trades_change': curr.get('trades_per_hour', 0) - base.get('trades_per_hour', 0),
                'pnl_change': curr.get('net_pnl', 0) - base.get('net_pnl', 0),
                'market_orders_change': curr.get('market_order_ratio', 0) - base.get('market_order_ratio', 0)
            }
        
        return comparison
    
    def display_dashboard(self):
        """Display performance dashboard"""
        print("\n" + "="*80)
        print(" TRADING BOT PERFORMANCE MONITOR ".center(80))
        print("="*80)
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Get current metrics
        bot_data = self.parse_recent_logs(1)  # Last hour
        hourly = self.calculate_metrics(bot_data)
        
        bot_data_24h = self.parse_recent_logs(24)  # Last 24 hours
        daily = self.calculate_metrics(bot_data_24h)
        
        # Display current performance
        print("\n📊 LAST 24 HOURS PERFORMANCE")
        print("-" * 80)
        
        if daily:
            df_data = []
            for bot_name, metrics in sorted(daily.items()):
                status = "🟢" if metrics['net_pnl'] > 0 else "🔴"
                fee_warning = "⚠️" if metrics['fee_ratio'] > 30 else ""
                
                df_data.append({
                    'Bot': bot_name[:15],
                    'Status': status,
                    'Net P&L': f"${metrics['net_pnl']:.2f}",
                    'Fees': f"${metrics['total_fees']:.2f}",
                    'Fee%': f"{metrics['fee_ratio']:.1f}%{fee_warning}",
                    'Trades/Hr': metrics['trades_per_hour'],
                    'Win%': f"{metrics['win_rate']:.1f}%",
                    'Market%': f"{metrics['market_order_ratio']:.0f}%"
                })
            
            df = pd.DataFrame(df_data)
            print(df.to_string(index=False))
        else:
            print("No trades in the last 24 hours")
        
        # Show comparison if baseline exists
        if self.baseline:
            print("\n📈 IMPROVEMENT SINCE FIXES")
            print("-" * 80)
            
            comparison = self.compare_performance()
            
            improvements = []
            for bot_name, changes in comparison.items():
                improvements.append({
                    'Bot': bot_name[:15],
                    'Fee% Change': f"{changes['fee_ratio_change']:+.1f}%",
                    'Win% Change': f"{changes['win_rate_change']:+.1f}%",
                    'Trades/Hr Change': f"{changes['trades_change']:+.1f}",
                    'P&L Change': f"${changes['pnl_change']:+.2f}",
                    'Market Orders': f"{changes['market_orders_change']:+.0f}%"
                })
            
            if improvements:
                df_imp = pd.DataFrame(improvements)
                print(df_imp.to_string(index=False))
        
        # Summary statistics
        print("\n📊 SUMMARY STATISTICS")
        print("-" * 80)
        
        if daily:
            total_pnl = sum(m['net_pnl'] for m in daily.values())
            total_fees = sum(m['total_fees'] for m in daily.values())
            avg_fee_ratio = sum(m['fee_ratio'] for m in daily.values()) / len(daily)
            avg_win_rate = sum(m['win_rate'] for m in daily.values()) / len(daily)
            total_trades = sum(m['total_trades'] for m in daily.values())
            
            print(f"Total Net P&L: ${total_pnl:.2f}")
            print(f"Total Fees Paid: ${total_fees:.2f}")
            print(f"Average Fee Ratio: {avg_fee_ratio:.2f}%")
            print(f"Average Win Rate: {avg_win_rate:.2f}%")
            print(f"Total Trades (24h): {total_trades}")
            
            # Warnings
            if avg_fee_ratio > 30:
                print("\n⚠️ WARNING: High fee ratio detected! Check for Market orders.")
            
            if total_trades > 1000:
                print("\n⚠️ WARNING: Over-trading detected! Add cooldowns.")
            
            if total_pnl < 0:
                print("\n🔴 WARNING: Negative P&L! Review and fix bots immediately.")
        
        # Recommendations
        print("\n💡 RECOMMENDATIONS")
        print("-" * 80)
        
        high_fee_bots = [b for b, m in daily.items() if m.get('fee_ratio', 0) > 30]
        if high_fee_bots:
            print(f"High fee bots (fix immediately): {', '.join(high_fee_bots[:5])}")
        
        market_order_bots = [b for b, m in daily.items() if m.get('market_order_ratio', 0) > 20]
        if market_order_bots:
            print(f"Still using Market orders: {', '.join(market_order_bots[:5])}")
        
        overtrading_bots = [b for b, m in daily.items() if m.get('trades_per_hour', 0) > 5]
        if overtrading_bots:
            print(f"Over-trading (>5 trades/hr): {', '.join(overtrading_bots[:5])}")
    
    def continuous_monitor(self, refresh_seconds=30):
        """Run continuous monitoring"""
        print("Starting continuous monitoring... (Press Ctrl+C to stop)")
        
        # Check for baseline
        if not self.load_baseline():
            print("\n⚠️ No baseline found. Creating baseline from last 24 hours...")
            self.save_baseline()
        
        try:
            while True:
                # Clear screen (works on Unix/Linux/Mac)
                print("\033[2J\033[H", end="")
                
                # Display dashboard
                self.display_dashboard()
                
                # Wait before refresh
                print(f"\n🔄 Refreshing in {refresh_seconds} seconds...")
                time.sleep(refresh_seconds)
                
        except KeyboardInterrupt:
            print("\n\n✋ Monitoring stopped")

def quick_check():
    """Quick performance check"""
    monitor = PerformanceMonitor()
    
    print("\n🔍 QUICK PERFORMANCE CHECK")
    print("="*60)
    
    # Get last hour metrics
    bot_data = monitor.parse_recent_logs(1)
    metrics = monitor.calculate_metrics(bot_data)
    
    if not metrics:
        print("No trades in the last hour")
        return
    
    # Check for critical issues
    issues = []
    
    for bot, m in metrics.items():
        if m['fee_ratio'] > 40:
            issues.append(f"🔴 {bot}: Fee ratio {m['fee_ratio']:.1f}% (>40%)")
        
        if m['market_order_ratio'] > 50:
            issues.append(f"🔴 {bot}: {m['market_order_ratio']:.0f}% Market orders")
        
        if m['trades_per_hour'] > 10:
            issues.append(f"⚠️ {bot}: {m['trades_per_hour']:.1f} trades/hr (over-trading)")
        
        if m['net_pnl'] < -10:
            issues.append(f"🔴 {bot}: Lost ${abs(m['net_pnl']):.2f}")
    
    if issues:
        print("\n⚠️ CRITICAL ISSUES FOUND:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("\n✅ No critical issues found")
    
    # Show summary
    total_pnl = sum(m['net_pnl'] for m in metrics.values())
    print(f"\n📊 Last Hour Total P&L: ${total_pnl:.2f}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'baseline':
            # Save baseline metrics
            monitor = PerformanceMonitor()
            monitor.save_baseline()
            print("Baseline saved. Run without arguments to start monitoring.")
        elif sys.argv[1] == 'quick':
            # Quick check
            quick_check()
    else:
        # Start continuous monitoring
        monitor = PerformanceMonitor()
        monitor.continuous_monitor(refresh_seconds=30)