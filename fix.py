#!/usr/bin/env python3
"""
Fix syntax errors in trading bot files
Main issues:
1. Incorrect place_order calls with bad string formatting
2. Duplicated timeInForce parameters
"""

import os
import re
import glob

def fix_place_order_syntax(content):
    """Fix broken place_order calls with incorrect syntax"""
    
    # Pattern 1: Fix price=str(limit_price, timeInForce="PostOnly")
    # Should be: price=str(limit_price)
    pattern1 = r'price=str\([^,)]+,\s*timeInForce="[^"]+"\)'
    fixed_content = re.sub(
        pattern1,
        lambda m: 'price=str(' + re.search(r'str\(([^,)]+)', m.group()).group(1) + ')',
        content
    )
    
    # Pattern 2: Remove duplicated timeInForce parameters
    # Find place_order calls and ensure only one timeInForce
    lines = fixed_content.split('\n')
    new_lines = []
    
    for line in lines:
        if 'place_order' in line and line.count('timeInForce') > 1:
            # Keep only the last timeInForce parameter
            parts = line.split(',')
            seen_time_in_force = False
            filtered_parts = []
            
            for i in range(len(parts) - 1, -1, -1):
                if 'timeInForce' in parts[i] and not seen_time_in_force:
                    filtered_parts.insert(0, parts[i])
                    seen_time_in_force = True
                elif 'timeInForce' not in parts[i]:
                    filtered_parts.insert(0, parts[i])
            
            line = ','.join(filtered_parts)
        
        new_lines.append(line)
    
    return '\n'.join(new_lines)

def fix_postonly_close_issue(content):
    """Fix PostOnly usage in close_position methods"""
    
    # Find close_position methods and check for PostOnly usage
    lines = content.split('\n')
    in_close_method = False
    new_lines = []
    indent_level = 0
    
    for line in lines:
        if 'def close_position' in line:
            in_close_method = True
            indent_level = len(line) - len(line.lstrip())
        elif in_close_method and line.strip() and not line.strip().startswith('#'):
            current_indent = len(line) - len(line.lstrip())
            if current_indent <= indent_level and 'def ' in line:
                in_close_method = False
        
        # If we're in close_position and see PostOnly, add a comment warning
        if in_close_method and 'timeInForce="PostOnly"' in line and 'place_order' in line:
            # Change PostOnly to IOC for closes (Immediate or Cancel)
            line = line.replace('timeInForce="PostOnly"', 'timeInForce="IOC"')
            # Add comment
            new_lines.append(line + '  # Changed from PostOnly to IOC for reliable closes')
        else:
            new_lines.append(line)
    
    return '\n'.join(new_lines)

def fix_symbol_mismatch(filepath, content):
    """Fix symbol mismatches between filename and code"""
    
    filename = os.path.basename(filepath)
    
    # Extract symbol from filename (e.g., LTCUSDT from 3_FEES_EMAMACDRSI_LTCUSDT.py)
    match = re.search(r'([A-Z]+USDT)\.py$', filename)
    if match:
        expected_symbol = match.group(1)
        
        # Check if class initialization has different symbol
        if 'LTCUSDT' in filename and 'BNBUSDT' in content:
            content = content.replace('"BNBUSDT"', f'"{expected_symbol}"')
            content = content.replace("'BNBUSDT'", f"'{expected_symbol}'")
            print(f"  Fixed symbol mismatch in {filename}: BNBUSDT -> {expected_symbol}")
    
    return content

def process_bot_file(filepath):
    """Process a single bot file and fix issues"""
    
    print(f"\nProcessing {filepath}...")
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        original_content = content
        
        # Apply fixes
        content = fix_place_order_syntax(content)
        content = fix_postonly_close_issue(content)
        content = fix_symbol_mismatch(filepath, content)
        
        # Write back if changed
        if content != original_content:
            # Create backup
            backup_path = filepath + '.backup'
            with open(backup_path, 'w') as f:
                f.write(original_content)
            print(f"  Created backup: {backup_path}")
            
            # Write fixed content
            with open(filepath, 'w') as f:
                f.write(content)
            print(f"  ✅ Fixed and saved: {filepath}")
        else:
            print(f"  ℹ️ No changes needed: {filepath}")
            
    except FileNotFoundError:
        print(f"  ❌ File not found: {filepath}")
    except Exception as e:
        print(f"  ❌ Error processing {filepath}: {e}")

def main():
    """Main function to fix all bot files"""
    
    # List of bot files to fix (based on the review)
    bot_files = [
        "1_FEES_EMA_BB_SOLUSDT.py",
        "2_FEES_EMA_RSI_BNBUSDT.py", 
        "3_FEES_EMAMACDRSI_LTCUSDT.py",
        "4_FEES_LIQUIDITYSWEEPBOT_DOGEUSDT.py",
        "5_FEES_MACD_VWAP_XRPUSDT.py",
        "6_FEES_MLFiltered_ARBUSDT.py",
        "7_FEES_DYNAMIC_GRID_ETHUSDT.py",
        "8_FEES_VWAP_RSI_DIV_AVAXUSDT.py",
        "9_FEES_PIVOT_REVERSAL_LINKUSDT.py",
        "10_FEES_RMI_SUPERTREND_ADAUSDT.py"
    ]
    
    print("🔧 Starting bot fixes...")
    print("=" * 50)
    
    for bot_file in bot_files:
        # Try current directory first
        if os.path.exists(bot_file):
            process_bot_file(bot_file)
        else:
            # Try to find it with glob
            matches = glob.glob(f"**/{bot_file}", recursive=True)
            if matches:
                process_bot_file(matches[0])
            else:
                print(f"\n⚠️ Skipping {bot_file} - not found")
    
    print("\n" + "=" * 50)
    print("✅ Fix script completed!")
    print("\nNote: Backup files created with .backup extension")
    print("Test each bot after fixes to ensure proper operation")

if __name__ == "__main__":
    main()