#!/usr/bin/env python3
"""
Ultimate fix for all bot syntax errors
Usage: python3 ultimate_fix.py
"""

import os
import re

def fix_bot(filename):
    """Fix a specific bot file with targeted fixes"""
    
    if not os.path.exists(filename):
        print(f"⚠️  {filename} not found")
        return False
    
    with open(filename, 'r') as f:
        content = f.read()
    
    original = content
    
    # Specific fixes based on the file
    if "2_FEES_EMA_RSI" in filename or "9_FEES_PIVOT" in filename:
        # Fix else: alignment issue (line 88/89)
        lines = content.split('\n')
        for i in range(len(lines)):
            if i > 85 and i < 92:  # Around line 88-89
                if lines[i].strip() == 'else:':
                    lines[i] = '        else:'
        content = '\n'.join(lines)
    
    elif "3_FEES_EMAMACDRSI" in filename:
        # Fix line 455 - duplicate timeInForce
        content = re.sub(
            r'timeInForce="PostOnly"\),\s*timeInForce="PostOnly"',
            'timeInForce="PostOnly"',
            content
        )
    
    elif "4_FEES_LIQUIDITYSWEEP" in filename:
        # Fix indentation after function definition
        lines = content.split('\n')
        for i in range(20, 30):  # Around line 24
            if i < len(lines):
                if '"""Execute limit order' in lines[i]:
                    if not lines[i].startswith('    '):
                        lines[i] = '    ' + lines[i].strip()
        content = '\n'.join(lines)
    
    elif any(x in filename for x in ["5_FEES_MACD", "7_FEES_DYNAMIC", "12_FEES_ML", "13_FEES_XGBOOST"]):
        # Fix unmatched parenthesis in place_order
        content = re.sub(r'\)\)', ')', content)
        content = re.sub(r',\)', ')', content)
        content = re.sub(
            r'timeInForce="PostOnly"\),\s*timeInForce="PostOnly"',
            'timeInForce="PostOnly"',
            content
        )
    
    # Common fixes for all files
    content = re.sub(r'\)\)', ')', content)  # Double parentheses
    content = re.sub(r',\)', ')', content)   # Trailing comma
    content = re.sub(          # Duplicate timeInForce
        r'timeInForce="PostOnly"\),\s*timeInForce="PostOnly"',
        'timeInForce="PostOnly"',
        content
    )
    
    # Fix missing comma in place_order calls
    content = re.sub(
        r'(qty=self\.format_qty\([^)]+\))\s+(timeInForce="PostOnly")',
        r'\1,\n            \2',
        content
    )
    
    # Save if changed
    if content != original:
        # Backup original
        with open(f"{filename}.backup", 'w') as f:
            f.write(original)
        
        # Write fixed
        with open(filename, 'w') as f:
            f.write(content)
        
        # Test syntax
        try:
            compile(content, filename, 'exec')
            print(f"✅ {filename} - Fixed")
            return True
        except SyntaxError as e:
            print(f"⚠️  {filename} - Line {e.lineno}: {e.msg}")
            # Restore original
            with open(filename, 'w') as f:
                f.write(original)
            return False
    else:
        print(f"ℹ️  {filename} - No changes needed")
        return True

def main():
    print("🔧 ULTIMATE BOT FIX")
    print("=" * 50)
    
    bots = [
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
    
    fixed = 0
    for bot in bots:
        if fix_bot(bot):
            fixed += 1
    
    print("=" * 50)
    print(f"Result: {fixed}/{len(bots)} files fixed")
    
    if fixed == len(bots):
        print("\n✅ All files are now working!")
        print("\nTo run a bot:")
        print("  python3 2_FEES_EMA_RSI_BNBUSDT.py")
        
        # Clean up backups
        for bot in bots:
            backup = f"{bot}.backup"
            if os.path.exists(backup):
                os.remove(backup)
    else:
        print("\n⚠️  Some files need manual fixing")
        print("Backups saved as *.backup")

if __name__ == "__main__":
    main()