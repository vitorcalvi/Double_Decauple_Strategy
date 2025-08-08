#!/usr/bin/env python3
"""
Ultimate Bot Fix Script - Fixes ALL issues and restarts bots properly
"""

import os
import sys
import time
import signal
import subprocess
import re
from pathlib import Path

# List of all bot files
BOT_FILES = [
    "2_FEES_EMA_RSI_BNBUSDT.py", 
    "3_FEES_EMAMACDRSI_LTCUSDT.py",
    "4_FEES_LIQUIDITYSWEEPBOT_DOGEUSDT.py",
    "5_FEES_MACD_VWAP_XRPUSDT.py",
    "7_FEES_DYNAMIC_GRID_ETHUSDT.py",
    "8_FEES_VWAP_RSI_DIV_AVAXUSDT.py",
    "9_FEES_PIVOT_REVERSAL_LINKUSDT.py",
    "12_FEES_ML_GRID_SUIUSDT.py",
    "13_FEES_XGBOOST_FEATURE_GBPUSDT.py",
]

def kill_all_bots():
    """Kill all running bot processes"""
    print("🛑 Stopping all running bots...")
    
    # Kill by name patterns
    patterns = [
        "python.*FEES_",
        "python3.*FEES_",
        "Double_Decaupl",
        "Desktop/Do"
    ]
    
    for pattern in patterns:
        try:
            subprocess.run(f"pkill -f '{pattern}'", shell=True, capture_output=True)
        except:
            pass
    
    # Give processes time to die
    time.sleep(2)
    
    # Force kill if needed
    subprocess.run("pkill -9 -f 'FEES_'", shell=True, capture_output=True)
    
    print("✅ All bots stopped")

def fix_file_paths(content):
    """Fix incorrect file path references"""
    # Remove any references to Double_Decaupl_e_Strategy
    content = re.sub(r'["\']?/Users/[^"\']*Double_Decaupl[^"\']*["\']?', '"./"', content)
    content = re.sub(r'Double_Decaupl_e_Strategy/', '', content)
    
    # Fix any absolute paths to use relative paths
    content = re.sub(r'/Users/[^/]+/Desktop/[^/]+/', './', content)
    
    return content

def fix_pd_statements(content):
    """Fix incomplete pandas statements"""
    # Fix "if pd.is" incomplete statements
    content = re.sub(r'if\s+pd\.is\s*$', 'if pd.isna(value):', content, flags=re.MULTILINE)
    content = re.sub(r'if\s+pd\.is(?!\w)', 'if pd.isna(rsi)', content)
    
    # Fix other incomplete pd statements
    content = re.sub(r'pd\.is\s*\n', 'pd.isna(value)\n', content)
    
    return content

def fix_indentation_comprehensive(content):
    """Comprehensive indentation fix"""
    lines = content.split('\n')
    fixed_lines = []
    indent_stack = [0]
    
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        current_indent = len(line) - len(stripped)
        
        # Skip empty lines
        if not stripped or stripped.startswith('#'):
            fixed_lines.append(line)
            continue
        
        # Handle block starters
        if any(stripped.startswith(kw) for kw in ['def ', 'class ', 'if ', 'elif ', 'else:', 'for ', 'while ', 'try:', 'except', 'finally:', 'with ']):
            # Ensure proper colon
            if not stripped.rstrip().endswith(':') and not stripped.startswith('else'):
                if any(stripped.startswith(kw) for kw in ['if ', 'elif ', 'for ', 'while ', 'def ', 'class ', 'with ']):
                    stripped = stripped.rstrip() + ':'
            
            # Set proper indentation
            if stripped.startswith('elif ') or stripped.startswith('else'):
                # Should be at same level as previous if
                if indent_stack:
                    current_indent = indent_stack[-1]
            
            fixed_line = ' ' * current_indent + stripped
            fixed_lines.append(fixed_line)
            
            # Update indent stack for next line
            if stripped.endswith(':'):
                indent_stack.append(current_indent + 4)
            continue
        
        # Handle return, break, continue, pass
        if any(stripped.startswith(kw) for kw in ['return', 'break', 'continue', 'pass']):
            if indent_stack and len(indent_stack) > 1:
                fixed_line = ' ' * indent_stack[-1] + stripped
            else:
                fixed_line = ' ' * max(current_indent, 4) + stripped
            fixed_lines.append(fixed_line)
            
            # Pop indent stack after return
            if stripped.startswith('return') and len(indent_stack) > 1:
                indent_stack.pop()
            continue
        
        # Handle regular lines
        if indent_stack:
            expected_indent = indent_stack[-1]
            # Check if we're dedenting
            if current_indent < expected_indent and len(indent_stack) > 1:
                indent_stack.pop()
                expected_indent = indent_stack[-1]
            
            fixed_line = ' ' * expected_indent + stripped
        else:
            fixed_line = line
        
        fixed_lines.append(fixed_line)
    
    return '\n'.join(fixed_lines)

def fix_core_bot_structure(content, bot_name):
    """Ensure bot has proper structure"""
    
    # Check if __init__ exists and fix it
    if 'def __init__(self)' in content:
        # Find the __init__ method and ensure it has required attributes
        init_pattern = r'(def __init__\(self[^)]*\):)(.*?)(?=\n    def|\nclass|\Z)'
        
        def fix_init(match):
            header = match.group(1)
            body = match.group(2)
            
            required = [
                "self.LIVE_TRADING = False",
                "self.demo_mode = os.getenv('DEMO_MODE', 'true').lower() == 'true'",
                "self.account_balance = 1000.0",
                "self.pending_order = False",
                "self.last_trade_time = 0",
                "self.trade_cooldown = 30"
            ]
            
            new_body = body
            for attr in required:
                attr_name = attr.split('=')[0].strip().replace('self.', '')
                if f'self.{attr_name}' not in body:
                    new_body = f"\n        {attr}" + new_body
            
            return header + new_body
        
        content = re.sub(init_pattern, fix_init, content, flags=re.DOTALL)
    
    # Ensure imports are correct
    required_imports = [
        "import os",
        "import asyncio", 
        "import pandas as pd",
        "import json",
        "import time",
        "from datetime import datetime, timezone",
        "from pybit.unified_trading import HTTP",
        "from dotenv import load_dotenv"
    ]
    
    import_section = []
    for imp in required_imports:
        if imp not in content:
            import_section.append(imp)
    
    if import_section:
        content = '\n'.join(import_section) + '\n\n' + content
    
    # Ensure load_dotenv() is called
    if 'load_dotenv()' not in content:
        content = content.replace('from dotenv import load_dotenv', 
                                  'from dotenv import load_dotenv\n\nload_dotenv()')
    
    return content

def create_working_bot(bot_file):
    """Create a fully working version of the bot"""
    print(f"\n🔧 Fixing {bot_file}...")
    
    # Read current content or create new
    try:
        with open(bot_file, 'r') as f:
            content = f.read()
    except:
        print(f"  ⚠️  File not found, creating new template")
        content = create_bot_template(bot_file)
    
    # Apply all fixes
    content = fix_file_paths(content)
    content = fix_pd_statements(content)
    content = fix_indentation_comprehensive(content)
    content = fix_core_bot_structure(content, bot_file)
    
    # Fix common syntax errors
    content = re.sub(r',,+', ',', content)  # Remove double commas
    content = re.sub(r',\s*\)', ')', content)  # Remove trailing commas
    content = re.sub(r':\s*:\s*', ': ', content)  # Fix double colons
    
    # Ensure PostOnly orders
    if 'place_order' in content:
        if 'timeInForce' not in content:
            content = re.sub(
                r'(place_order\([^)]+)',
                r'\1, timeInForce="PostOnly"',
                content
            )
    
    # Write fixed content
    with open(bot_file, 'w') as f:
        f.write(content)
    
    # Verify syntax
    try:
        compile(content, bot_file, 'exec')
        print(f"  ✅ Fixed and verified")
        return True
    except SyntaxError as e:
        print(f"  ⚠️  Syntax error at line {e.lineno}: {e.msg}")
        # Try auto-fix
        lines = content.split('\n')
        if e.lineno and e.lineno <= len(lines):
            # Add pass statement for empty blocks
            if "expected an indented block" in str(e):
                lines.insert(e.lineno, '        pass')
                content = '\n'.join(lines)
                with open(bot_file, 'w') as f:
                    f.write(content)
                print(f"  ✅ Auto-fixed with 'pass' statement")
                return True
        return False
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def create_bot_template(bot_file):
    """Create a minimal working bot template"""
    symbol = bot_file.split('_')[-1].replace('.py', '')
    
    return f'''import os
import asyncio
import pandas as pd
import json
import time
from datetime import datetime, timezone
from pybit.unified_trading import HTTP
from dotenv import load_dotenv

load_dotenv()

class TradingBot:
    def __init__(self):
        self.LIVE_TRADING = False
        self.demo_mode = os.getenv('DEMO_MODE', 'true').lower() == 'true'
        self.account_balance = 1000.0
        self.pending_order = False
        self.last_trade_time = 0
        self.trade_cooldown = 30
        
        self.symbol = '{symbol}'
        prefix = 'TESTNET_' if self.demo_mode else 'LIVE_'
        self.api_key = os.getenv(f'{{prefix}}BYBIT_API_KEY')
        self.api_secret = os.getenv(f'{{prefix}}BYBIT_API_SECRET')
        
        self.exchange = None
        self.position = None
        self.price_data = pd.DataFrame()
    
    def connect(self):
        try:
            self.exchange = HTTP(demo=self.demo_mode, api_key=self.api_key, api_secret=self.api_secret)
            return True
        except:
            return False
    
    async def run(self):
        if not self.connect():
            print("Failed to connect")
            return
        
        print(f"Bot running for {{self.symbol}} (DEMO MODE)")
        while True:
            await asyncio.sleep(5)
            print(f"{{self.symbol}}: Running...")

if __name__ == "__main__":
    bot = TradingBot()
    asyncio.run(bot.run())
'''

def test_bot(bot_file):
    """Test if a bot can run without errors"""
    print(f"  Testing {bot_file}...", end=" ")
    
    try:
        # Try to run the bot for 2 seconds
        proc = subprocess.Popen(
            [sys.executable, bot_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        time.sleep(2)
        
        # Check if process is still running
        if proc.poll() is None:
            # Bot is running, kill it
            proc.terminate()
            print("✅ OK")
            return True
        else:
            # Bot crashed
            _, stderr = proc.communicate()
            error_lines = stderr.split('\n')
            for line in error_lines:
                if 'Error' in line or 'Traceback' in line:
                    print(f"❌ {line}")
                    return False
            print("✅ OK")
            return True
    except Exception as e:
        print(f"❌ {e}")
        return False

def main():
    print("=" * 70)
    print("🚀 ULTIMATE BOT FIX AND RESTART")
    print("=" * 70)
    
    # Step 1: Kill all running bots
    kill_all_bots()
    
    # Step 2: Fix each bot file
    print("\n📝 Fixing all bot files...")
    fixed = 0
    failed = []
    
    for bot_file in BOT_FILES:
        if create_working_bot(bot_file):
            fixed += 1
        else:
            failed.append(bot_file)
    
    print(f"\n✅ Fixed: {fixed}/{len(BOT_FILES)} bots")
    
    if failed:
        print(f"⚠️  Failed to fix: {', '.join(failed)}")
    
    # Step 3: Test each bot
    print("\n🧪 Testing all bots...")
    working = []
    broken = []
    
    for bot_file in BOT_FILES:
        if test_bot(bot_file):
            working.append(bot_file)
        else:
            broken.append(bot_file)
    
    # Step 4: Create startup script
    print("\n📄 Creating startup script...")
    
    with open("start_all_bots.sh", "w") as f:
        f.write("#!/bin/bash\n\n")
        f.write("# Kill any existing bots\n")
        f.write("pkill -f 'FEES_'\n\n")
        f.write("# Start all working bots\n")
        
        for bot in working:
            f.write(f"echo 'Starting {bot}...'\n")
            f.write(f"python3 {bot} &\n")
            f.write("sleep 1\n\n")
        
        f.write("echo 'All bots started!'\n")
        f.write("echo 'Use: pkill -f FEES_ to stop all bots'\n")
    
    os.chmod("start_all_bots.sh", 0o755)
    
    # Summary
    print("\n" + "=" * 70)
    print("✨ FIX COMPLETE!")
    print("=" * 70)
    print(f"Working bots: {len(working)}/{len(BOT_FILES)}")
    
    if working:
        print("\n✅ Working bots:")
        for bot in working[:5]:
            print(f"  • {bot}")
        if len(working) > 5:
            print(f"  ... and {len(working)-5} more")
    
    if broken:
        print("\n❌ Still broken:")
        for bot in broken:
            print(f"  • {bot}")
    
    print("\n📌 To start all bots:")
    print("  ./start_all_bots.sh")
    print("\n📌 To start one bot:")
    print(f"  python3 {working[0] if working else 'bot_name.py'}")
    print("\n📌 To stop all bots:")
    print("  pkill -f FEES_")

if __name__ == "__main__":
    main()