import os
import re
from pathlib import Path
import ast

class BotIssueIdentifier:
    def __init__(self, bot_directory="."):
        self.bot_dir = Path(bot_directory)
        self.issues_found = {}
        
    def scan_bot_file(self, filepath):
        """Scan a bot file for common issues"""
        with open(filepath, 'r') as f:
            content = f.read()
        
        bot_name = filepath.stem
        issues = []
        
        # 1. FEE ISSUES
        fee_issues = self.check_fee_issues(content)
        if fee_issues:
            issues.extend(fee_issues)
        
        # 2. OVER-TRADING ISSUES
        trading_issues = self.check_overtrading_issues(content)
        if trading_issues:
            issues.extend(trading_issues)
        
        # 3. SLIPPAGE ISSUES
        slippage_issues = self.check_slippage_issues(content)
        if slippage_issues:
            issues.extend(slippage_issues)
        
        # 4. RISK MANAGEMENT ISSUES
        risk_issues = self.check_risk_issues(content)
        if risk_issues:
            issues.extend(risk_issues)
        
        if issues:
            self.issues_found[bot_name] = issues
        
        return issues
    
    def check_fee_issues(self, content):
        """Check for fee-related problems"""
        issues = []
        
        # Check if using market orders (high fees)
        if re.search(r'orderType\s*=\s*["\']Market["\']', content):
            issues.append({
                'type': 'FEE_DRAG',
                'severity': 'HIGH',
                'issue': 'Using Market orders (0.06% taker fee)',
                'fix': 'Use Limit orders with PostOnly flag for -0.01% maker rebate',
                'code_fix': '''
# Replace:
orderType="Market"

# With:
orderType="Limit",
timeInForce="PostOnly"  # Get maker rebate instead of paying taker fee
'''
            })
        
        # Check for missing PostOnly flag
        if 'orderType="Limit"' in content and 'PostOnly' not in content:
            issues.append({
                'type': 'FEE_DRAG',
                'severity': 'MEDIUM',
                'issue': 'Limit orders without PostOnly flag may become taker orders',
                'fix': 'Add timeInForce="PostOnly" to ensure maker rebate',
                'code_fix': '''
# Add to all limit orders:
timeInForce="PostOnly"
'''
            })
        
        # Check for frequent small trades
        if re.search(r'position_size\s*=\s*\d+', content):
            matches = re.findall(r'position_size\s*=\s*(\d+)', content)
            for match in matches:
                if int(match) < 20:
                    issues.append({
                        'type': 'FEE_DRAG',
                        'severity': 'HIGH',
                        'issue': f'Fixed position size ${match} too small - fees will dominate',
                        'fix': 'Use risk-based position sizing (1-2% of account)',
                        'code_fix': '''
# Replace fixed position:
position_size = 10

# With risk-based sizing:
risk_pct = 1.0  # Risk 1% of account
risk_amount = account_balance * (risk_pct / 100)
position_size = risk_amount / stop_distance
'''
                    })
                    break
        
        return issues
    
    def check_overtrading_issues(self, content):
        """Check for over-trading patterns"""
        issues = []
        
        # Check for missing cooldown between trades
        if 'execute_trade' in content and 'cooldown' not in content.lower():
            issues.append({
                'type': 'OVER_TRADING',
                'severity': 'HIGH',
                'issue': 'No trade cooldown mechanism',
                'fix': 'Add minimum time between trades',
                'code_fix': '''
# Add to class init:
self.last_trade_time = 0
self.trade_cooldown = 30  # 30 seconds minimum

# Add to execute_trade():
if time.time() - self.last_trade_time < self.trade_cooldown:
    return
self.last_trade_time = time.time()
'''
            })
        
        # Check for missing position check before trading
        if 'execute_trade' in content and not re.search(r'if.*position.*:\s*return', content):
            issues.append({
                'type': 'OVER_TRADING',
                'severity': 'HIGH',
                'issue': 'Not checking for existing position before trading',
                'fix': 'Add position check to prevent duplicate trades',
                'code_fix': '''
# Add to execute_trade():
if self.position:
    print("Position already exists")
    return
'''
            })
        
        # Check for aggressive signal thresholds
        rsi_matches = re.findall(r'rsi.*?[<>]\s*(\d+)', content, re.IGNORECASE)
        for match in rsi_matches:
            threshold = int(match)
            if 40 <= threshold <= 60:
                issues.append({
                    'type': 'OVER_TRADING',
                    'severity': 'MEDIUM',
                    'issue': f'Weak RSI threshold {threshold} generates too many signals',
                    'fix': 'Use stronger thresholds: <30 for oversold, >70 for overbought',
                    'code_fix': '''
# Use stronger RSI thresholds:
rsi_oversold = 30   # Only buy when strongly oversold
rsi_overbought = 70  # Only sell when strongly overbought
'''
                })
                break
        
        return issues
    
    def check_slippage_issues(self, content):
        """Check for slippage handling problems"""
        issues = []
        
        # Check if slippage is being modeled
        if 'slippage' not in content.lower():
            issues.append({
                'type': 'SLIPPAGE',
                'severity': 'HIGH',
                'issue': 'No slippage modeling in execution',
                'fix': 'Add slippage calculation to execution prices',
                'code_fix': '''
def apply_slippage(self, price, side, slippage_pct=0.02):
    """Apply realistic slippage to execution"""
    if side == "BUY":
        return price * (1 + slippage_pct/100)
    else:
        return price * (1 - slippage_pct/100)
'''
            })
        
        # Check for large position sizes without slippage adjustment
        if re.search(r'qty.*?>\s*100', content) or re.search(r'position.*?>\s*1000', content):
            issues.append({
                'type': 'SLIPPAGE',
                'severity': 'MEDIUM',
                'issue': 'Large positions without dynamic slippage adjustment',
                'fix': 'Scale slippage with position size',
                'code_fix': '''
# Scale slippage with position size:
base_slippage = 0.02
size_factor = min(position_value / 10000, 2.0)
total_slippage = base_slippage * (1 + size_factor * 0.5)
'''
            })
        
        return issues
    
    def check_risk_issues(self, content):
        """Check for risk management problems"""
        issues = []
        
        # Check for missing stop loss
        if 'stop_loss' not in content.lower():
            issues.append({
                'type': 'RISK',
                'severity': 'CRITICAL',
                'issue': 'No stop loss implementation',
                'fix': 'Add stop loss to all positions',
                'code_fix': '''
# Add stop loss check:
if profit_pct <= -self.config['stop_loss']:
    return True, "stop_loss"
'''
            })
        
        # Check for fixed position sizing
        if re.search(r'qty\s*=\s*\d+[^.]', content):
            issues.append({
                'type': 'RISK',
                'severity': 'HIGH',
                'issue': 'Using fixed quantity instead of risk-based sizing',
                'fix': 'Implement percentage-based risk management',
                'code_fix': '''
# Calculate position size based on risk:
def calculate_position_size(self, price, stop_loss_price):
    risk_amount = self.account_balance * 0.01  # 1% risk
    stop_distance = abs(price - stop_loss_price)
    return risk_amount / stop_distance
'''
            })
        
        return issues
    
    def generate_fix_script(self, bot_name):
        """Generate a fix script for specific bot"""
        if bot_name not in self.issues_found:
            return None
        
        fixes = []
        fixes.append(f"# Fixes for {bot_name}")
        fixes.append("# " + "="*50)
        
        for issue in self.issues_found[bot_name]:
            fixes.append(f"\n# {issue['severity']}: {issue['issue']}")
            fixes.append(f"# Fix: {issue['fix']}")
            if 'code_fix' in issue:
                fixes.append(issue['code_fix'])
        
        return "\n".join(fixes)
    
    def generate_report(self):
        """Generate comprehensive issue report"""
        print("\n" + "="*100)
        print(" BOT CODE ISSUE ANALYSIS ".center(100))
        print("="*100)
        
        # Count issues by type
        issue_counts = {
            'FEE_DRAG': 0,
            'OVER_TRADING': 0,
            'SLIPPAGE': 0,
            'RISK': 0
        }
        
        severity_counts = {
            'CRITICAL': 0,
            'HIGH': 0,
            'MEDIUM': 0,
            'LOW': 0
        }
        
        for bot, issues in self.issues_found.items():
            for issue in issues:
                issue_counts[issue['type']] = issue_counts.get(issue['type'], 0) + 1
                severity_counts[issue['severity']] = severity_counts.get(issue['severity'], 0) + 1
        
        # Summary
        print("\n📊 ISSUE SUMMARY")
        print("-" * 50)
        print(f"Total Bots Analyzed: {len(self.issues_found)}")
        print(f"Total Issues Found: {sum(issue_counts.values())}")
        print("\nBy Type:")
        for issue_type, count in issue_counts.items():
            print(f"  • {issue_type}: {count}")
        print("\nBy Severity:")
        for severity, count in severity_counts.items():
            if count > 0:
                emoji = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}
                print(f"  {emoji.get(severity, '')} {severity}: {count}")
        
        # Detailed issues per bot
        print("\n" + "="*100)
        print(" DETAILED ISSUES BY BOT ".center(100))
        print("="*100)
        
        for bot_name, issues in sorted(self.issues_found.items()):
            print(f"\n📁 {bot_name}")
            print("-" * 50)
            
            # Group by severity
            critical = [i for i in issues if i['severity'] == 'CRITICAL']
            high = [i for i in issues if i['severity'] == 'HIGH']
            medium = [i for i in issues if i['severity'] == 'MEDIUM']
            
            if critical:
                print("\n🔴 CRITICAL Issues:")
                for issue in critical:
                    print(f"  • [{issue['type']}] {issue['issue']}")
                    print(f"    Fix: {issue['fix']}")
            
            if high:
                print("\n🟠 HIGH Priority Issues:")
                for issue in high:
                    print(f"  • [{issue['type']}] {issue['issue']}")
                    print(f"    Fix: {issue['fix']}")
            
            if medium:
                print("\n🟡 MEDIUM Priority Issues:")
                for issue in medium:
                    print(f"  • [{issue['type']}] {issue['issue']}")
                    print(f"    Fix: {issue['fix']}")
        
        # Top recommendations
        print("\n" + "="*100)
        print(" TOP 5 CRITICAL FIXES ".center(100))
        print("="*100)
        
        print("""
1. 🔴 REPLACE ALL MARKET ORDERS WITH POSTONLY LIMIT ORDERS
   - Change: orderType="Market" → orderType="Limit", timeInForce="PostOnly"
   - Impact: Save 0.07% per trade (0.06% taker fee + 0.01% maker rebate)

2. 🔴 IMPLEMENT TRADE COOLDOWNS
   - Add: 30-second minimum between trades
   - Impact: Reduce over-trading by 50-70%

3. 🔴 USE RISK-BASED POSITION SIZING
   - Change: Fixed $10-100 positions → 1-2% of account balance
   - Impact: Proper risk management, scalable profits

4. 🟠 ADD SLIPPAGE MODELING
   - Add: 0.02-0.05% slippage to all execution prices
   - Impact: Realistic P&L calculations

5. 🟠 STRENGTHEN SIGNAL THRESHOLDS
   - Change: RSI 40/60 → RSI 30/70
   - Impact: Reduce false signals by 60%
""")
    
    def scan_all_bots(self):
        """Scan all bot files in directory"""
        for py_file in self.bot_dir.glob("*.py"):
            if 'FEES' in py_file.name or 'bot' in py_file.name.lower():
                print(f"Scanning {py_file.name}...")
                self.scan_bot_file(py_file)

# Usage
if __name__ == "__main__":
    scanner = BotIssueIdentifier(".")
    scanner.scan_all_bots()
    scanner.generate_report()
    
    # Generate fix scripts for each bot
    print("\n" + "="*100)
    print(" GENERATING FIX SCRIPTS ".center(100))
    print("="*100)
    
    for bot_name in scanner.issues_found:
        fix_script = scanner.generate_fix_script(bot_name)
        if fix_script:
            fix_filename = f"fixes_{bot_name}.py"
            with open(fix_filename, 'w') as f:
                f.write(fix_script)
            print(f"✅ Generated {fix_filename}")