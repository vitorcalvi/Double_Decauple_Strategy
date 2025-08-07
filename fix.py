#!/usr/bin/env python3
"""
Automated Trading Bot Fix Implementation
Applies critical fixes to all bot files
"""

import os
import re
import shutil
from pathlib import Path
from datetime import datetime

class BotAutoFixer:
    def __init__(self, bot_directory="."):
        self.bot_dir = Path(bot_directory)
        self.backup_dir = Path("backups") / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.fixes_applied = {}
        
    def backup_file(self, filepath):
        """Create backup before modifying"""
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        backup_path = self.backup_dir / filepath.name
        shutil.copy2(filepath, backup_path)
        print(f"✅ Backed up {filepath.name} to {backup_path}")
        
    def fix_market_orders(self, content, bot_name):
        """Fix #1: Replace Market orders with PostOnly Limit orders"""
        fixes = []
        
        # Fix close_position function that uses Market orders
        pattern = r'(async def close_position.*?orderType\s*=\s*["\'])Market(["\'])'
        if re.search(pattern, content, re.DOTALL):
            # Replace Market with Limit in close_position
            new_content = re.sub(
                r'(orderType\s*=\s*["\'])Market(["\'])',
                r'\1Limit\2',
                content
            )
            
            # Add PostOnly flag if not present
            if 'timeInForce="PostOnly"' not in new_content:
                new_content = re.sub(
                    r'(orderType\s*=\s*["\']Limit["\'])',
                    r'\1,\n                timeInForce="PostOnly"',
                    new_content
                )
            
            # Update offset calculation for maker orders
            new_content = re.sub(
                r'# Use market order for quick exit.*?\n',
                '# Use limit order with PostOnly for maker rebate\n',
                new_content
            )
            
            fixes.append("Replaced Market orders with PostOnly Limit orders")
            content = new_content
            
        return content, fixes
    
    def add_trade_cooldown(self, content, bot_name):
        """Fix #2: Add trade cooldown mechanism"""
        fixes = []
        
        # Check if cooldown already exists
        if 'trade_cooldown' in content:
            return content, []
        
        # Add cooldown variables to __init__
        init_pattern = r'(def __init__\(self.*?\):.*?)(self\.symbol\s*=)'
        if re.search(init_pattern, content, re.DOTALL):
            new_init = r'\1\n        # Trade cooldown mechanism\n        self.last_trade_time = 0\n        self.trade_cooldown = 30  # 30 seconds between trades\n        \n        \2'
            content = re.sub(init_pattern, new_init, content, flags=re.DOTALL)
            fixes.append("Added trade cooldown variables to __init__")
        
        # Add cooldown check to execute_trade
        exec_pattern = r'(async def execute_trade\(self, signal\):)(.*?)(\n.*?if.*?position.*?:.*?return)'
        if re.search(exec_pattern, content, re.DOTALL):
            cooldown_check = r'''\1\2
        
        # Check trade cooldown
        import time
        if time.time() - self.last_trade_time < self.trade_cooldown:
            remaining = self.trade_cooldown - (time.time() - self.last_trade_time)
            print(f"⏰ Trade cooldown: wait {remaining:.0f}s")
            return\3'''
            content = re.sub(exec_pattern, cooldown_check, content, flags=re.DOTALL)
            
            # Add time update after order placement
            order_pattern = r'(if order\.get\(["\']retCode["\']\)\s*==\s*0:)'
            if re.search(order_pattern, content):
                update_time = r'''\1
                self.last_trade_time = time.time()  # Update last trade time'''
                content = re.sub(order_pattern, update_time, content, count=1)
            
            fixes.append("Added trade cooldown check to execute_trade")
        
        return content, fixes
    
    def fix_rsi_thresholds(self, content, bot_name):
        """Fix #3: Strengthen RSI thresholds"""
        fixes = []
        
        # Find weak RSI thresholds
        patterns = [
            (r"'rsi_oversold'\s*:\s*(\d+)", 30),
            (r"'rsi_overbought'\s*:\s*(\d+)", 70),
            (r"'rsi_long_threshold'\s*:\s*(\d+)", 30),
            (r"'rsi_short_threshold'\s*:\s*(\d+)", 70),
        ]
        
        for pattern, new_value in patterns:
            match = re.search(pattern, content)
            if match:
                old_value = int(match.group(1))
                # Fix weak thresholds
                if pattern.endswith("oversold'\\s*:\\s*(\\d+)") and old_value > 30:
                    content = re.sub(pattern, f"'rsi_oversold': {new_value}", content)
                    fixes.append(f"Fixed RSI oversold: {old_value} → {new_value}")
                elif pattern.endswith("overbought'\\s*:\\s*(\\d+)") and old_value < 70:
                    content = re.sub(pattern, f"'rsi_overbought': {new_value}", content)
                    fixes.append(f"Fixed RSI overbought: {old_value} → {new_value}")
        
        # Fix inline RSI comparisons
        weak_patterns = [
            (r'rsi\s*<\s*4[0-5]', 'rsi < 30'),
            (r'rsi\s*>\s*[56][05]', 'rsi > 70'),
        ]
        
        for old_pattern, new_pattern in weak_patterns:
            if re.search(old_pattern, content):
                content = re.sub(old_pattern, new_pattern, content)
                fixes.append(f"Fixed weak RSI comparison: {old_pattern} → {new_pattern}")
        
        return content, fixes
    
    def add_emergency_stop(self, content, bot_name):
        """Fix #4: Add emergency stop logic"""
        fixes = []
        
        # Check if emergency stop exists
        if 'emergency_stop' in content:
            return content, []
        
        # Add daily P&L tracking to __init__
        if 'daily_pnl' not in content:
            init_pattern = r'(def __init__\(self.*?\):.*?)(self\.symbol\s*=)'
            if re.search(init_pattern, content, re.DOTALL):
                new_vars = r'''\1
        # Emergency stop tracking
        self.daily_pnl = 0
        self.consecutive_losses = 0
        self.max_daily_loss = 50  # $50 max daily loss
        
        \2'''
                content = re.sub(init_pattern, new_vars, content, flags=re.DOTALL)
                fixes.append("Added emergency stop variables")
        
        # Add emergency check to run_cycle
        cycle_pattern = r'(async def run_cycle\(self\):)(.*?)(\n\s+if not await self\.get_market_data)'
        if re.search(cycle_pattern, content, re.DOTALL):
            emergency_check = r'''\1\2
        
        # Emergency stop check
        if self.daily_pnl < -self.max_daily_loss:
            print(f"🔴 EMERGENCY STOP: Daily loss ${abs(self.daily_pnl):.2f} exceeded limit")
            if self.position:
                await self.close_position("emergency_stop")
            return\3'''
            content = re.sub(cycle_pattern, emergency_check, content, flags=re.DOTALL)
            fixes.append("Added emergency stop check")
        
        return content, fixes
    
    def add_import_time(self, content):
        """Ensure time module is imported"""
        if 'import time' not in content:
            # Add after other imports
            import_pattern = r'(import.*?\n)(from|class|\n\n)'
            if re.search(import_pattern, content):
                content = re.sub(import_pattern, r'\1import time\n\2', content, count=1)
            else:
                content = 'import time\n' + content
        return content
    
    def apply_fixes(self, filepath):
        """Apply all fixes to a bot file"""
        with open(filepath, 'r') as f:
            content = f.read()
        
        bot_name = filepath.stem
        all_fixes = []
        
        # Apply each fix
        content, fixes = self.fix_market_orders(content, bot_name)
        all_fixes.extend(fixes)
        
        content, fixes = self.add_trade_cooldown(content, bot_name)
        all_fixes.extend(fixes)
        
        content, fixes = self.fix_rsi_thresholds(content, bot_name)
        all_fixes.extend(fixes)
        
        content, fixes = self.add_emergency_stop(content, bot_name)
        all_fixes.extend(fixes)
        
        # Ensure time is imported if we added cooldowns
        if 'trade_cooldown' in content:
            content = self.add_import_time(content)
        
        if all_fixes:
            # Backup original
            self.backup_file(filepath)
            
            # Write fixed content
            with open(filepath, 'w') as f:
                f.write(content)
            
            self.fixes_applied[bot_name] = all_fixes
            print(f"✅ Fixed {bot_name}: {len(all_fixes)} fixes applied")
        else:
            print(f"ℹ️ {bot_name}: No fixes needed")
        
        return all_fixes
    
    def fix_all_bots(self):
        """Apply fixes to all bot files"""
        print("\n" + "="*60)
        print(" AUTOMATED BOT FIX IMPLEMENTATION ".center(60))
        print("="*60)
        
        # Find all bot files
        bot_files = []
        for py_file in self.bot_dir.glob("*_FEES_*.py"):
            bot_files.append(py_file)
        
        print(f"\nFound {len(bot_files)} bot files to fix")
        print(f"Backup directory: {self.backup_dir}\n")
        
        # Apply fixes to each bot
        for bot_file in sorted(bot_files):
            self.apply_fixes(bot_file)
        
        # Summary report
        print("\n" + "="*60)
        print(" FIX SUMMARY ".center(60))
        print("="*60)
        
        total_fixes = sum(len(fixes) for fixes in self.fixes_applied.values())
        print(f"\nTotal fixes applied: {total_fixes}")
        
        if self.fixes_applied:
            print("\nDetailed fixes by bot:")
            for bot_name, fixes in sorted(self.fixes_applied.items()):
                print(f"\n📁 {bot_name}:")
                for fix in fixes:
                    print(f"  ✅ {fix}")
        
        # Calculate estimated savings
        market_order_fixes = sum(1 for fixes in self.fixes_applied.values() 
                                for fix in fixes if 'Market order' in fix)
        
        if market_order_fixes:
            print("\n" + "="*60)
            print(" ESTIMATED IMPACT ".center(60))
            print("="*60)
            
            savings_per_trade = 0.07  # 0.06% taker fee + 0.01% maker rebate
            print(f"\n💰 Market → PostOnly fixes: {market_order_fixes} bots")
            print(f"   Savings: {savings_per_trade}% per trade")
            print(f"   Daily savings (100 trades): ${100 * 1000 * savings_per_trade / 100:.2f}")
            print(f"   Monthly savings: ${100 * 1000 * savings_per_trade / 100 * 30:.2f}")
        
        print("\n🎯 Next steps:")
        print("1. Review the changes in each file")
        print("2. Test on demo account first")
        print("3. Monitor for 24 hours before full deployment")
        print("4. Check logs for improved fee ratios")
        
        return self.fixes_applied

# Quick verification script
def verify_fixes():
    """Verify that fixes were applied correctly"""
    print("\n" + "="*60)
    print(" VERIFICATION ".center(60))
    print("="*60)
    
    issues = []
    
    for py_file in Path(".").glob("*_FEES_*.py"):
        with open(py_file, 'r') as f:
            content = f.read()
        
        # Check for remaining issues
        if 'orderType="Market"' in content and 'reduceOnly' not in content:
            issues.append(f"{py_file.name}: Still has Market orders")
        
        if 'execute_trade' in content and 'trade_cooldown' not in content:
            issues.append(f"{py_file.name}: Missing trade cooldown")
        
        if "'rsi_oversold': 40" in content or "'rsi_oversold': 45" in content:
            issues.append(f"{py_file.name}: Weak RSI thresholds")
    
    if issues:
        print("\n⚠️ Remaining issues:")
        for issue in issues:
            print(f"  • {issue}")
    else:
        print("\n✅ All critical issues fixed!")
    
    return len(issues) == 0

if __name__ == "__main__":
    # Run the auto-fixer
    fixer = BotAutoFixer(".")
    fixes = fixer.fix_all_bots()
    
    # Verify fixes
    print("\nRunning verification...")
    if verify_fixes():
        print("\n🎉 SUCCESS! All bots have been fixed.")
        print("Remember to test on demo before deploying to live!")
    else:
        print("\n⚠️ Some issues remain. Please review manually.")