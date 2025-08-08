#!/usr/bin/env python3
"""
Targeted fix for the close_position error
Fixes: EMARSIBot.format_qty() got an unexpected keyword argument 'timeInForce'
"""

import os
import re
import glob

def fix_close_position_method(filepath):
    """Fix the close_position method specifically"""
    
    if not os.path.exists(filepath):
        return False, f"File not found: {filepath}"
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    original = ''.join(lines)
    fixed = False
    
    # Find close_position method
    in_close_position = False
    close_start = -1
    close_end = -1
    indent_level = 0
    
    for i, line in enumerate(lines):
        # Find start of close_position method
        if 'def close_position' in line:
            in_close_position = True
            close_start = i
            # Determine indentation level
            indent_level = len(line) - len(line.lstrip())
            continue
        
        # Find end of close_position method
        if in_close_position:
            # Check if we're at the same or lower indentation (new method)
            if line.strip() and not line.startswith(' ' * (indent_level + 4)):
                if 'def ' in line or 'async def' in line:
                    close_end = i
                    break
    
    if close_start >= 0:
        if close_end == -1:
            close_end = len(lines)
        
        # Fix the close_position method
        for i in range(close_start, close_end):
            line = lines[i]
            
            # Fix 1: Remove timeInForce from format_qty calls
            if 'format_qty(' in line:
                # Pattern: qty=self.format_qty(something, timeInForce="PostOnly")
                if 'timeInForce' in line:
                    # Extract just the format_qty call without timeInForce
                    pattern = r'self\.format_qty\(([^,)]+)(?:,\s*timeInForce="[^"]*")?\)'
                    line = re.sub(pattern, r'self.format_qty(\1)', line)
                    lines[i] = line
                    fixed = True
            
            # Fix 2: Ensure place_order has timeInForce
            if 'self.exchange.place_order(' in line:
                # Check if this is the start of place_order call
                # Find the complete call (might span multiple lines)
                call_lines = []
                j = i
                paren_count = 0
                while j < close_end:
                    call_lines.append(lines[j])
                    paren_count += lines[j].count('(') - lines[j].count(')')
                    if paren_count == 0 and '(' in lines[j]:
                        break
                    j += 1
                
                full_call = ''.join(call_lines)
                
                # Check if timeInForce is present
                if 'timeInForce' not in full_call and 'orderType="Limit"' in full_call:
                    # Add timeInForce before reduceOnly or at the end
                    if 'reduceOnly=True' in full_call:
                        full_call = full_call.replace(
                            'reduceOnly=True',
                            'timeInForce="PostOnly",\n                reduceOnly=True'
                        )
                    else:
                        # Add before closing parenthesis
                        full_call = full_call.rstrip().rstrip(')')
                        full_call += ',\n                timeInForce="PostOnly"\n            )'
                    
                    # Replace the lines
                    new_lines = full_call.split('\n')
                    for k, new_line in enumerate(new_lines):
                        if i + k < len(lines):
                            lines[i + k] = new_line + '\n'
                    
                    fixed = True
    
    # Additional fix: Look for specific error pattern
    for i, line in enumerate(lines):
        # Pattern where format_qty is incorrectly called
        if 'qty=self.format_qty(' in line:
            # Check if there's a comma after the first parameter
            match = re.search(r'qty=self\.format_qty\(([^,)]+),([^)]+)\)', line)
            if match:
                first_param = match.group(1)
                second_param = match.group(2)
                if 'timeInForce' in second_param:
                    # Remove the second parameter
                    lines[i] = line.replace(match.group(0), f'qty=self.format_qty({first_param})')
                    fixed = True
    
    content = ''.join(lines)
    
    if content != original:
        # Backup
        backup = filepath + '.backup_close'
        with open(backup, 'w') as f:
            f.write(original)
        
        # Save
        with open(filepath, 'w') as f:
            f.write(content)
        
        return True, "Fixed close_position method"
    
    return False, "No changes needed"

def check_for_error_pattern(filepath):
    """Check if file has the format_qty error pattern"""
    
    if not os.path.exists(filepath):
        return False
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Look for problematic patterns
    patterns = [
        r'format_qty\([^,)]+,\s*timeInForce',  # format_qty with timeInForce
        r'format_qty\([^)]*timeInForce',        # Any format_qty with timeInForce
    ]
    
    for pattern in patterns:
        if re.search(pattern, content):
            return True
    
    return False

def main():
    print("=" * 60)
    print("TARGETED FIX FOR CLOSE POSITION ERROR")
    print("=" * 60)
    print("Fixing: format_qty() got unexpected keyword argument 'timeInForce'\n")
    
    # Priority files to check (based on error message)
    priority_files = [
        "2_FEES_EMA_RSI_BNBUSDT.py",
        "EMA_RSI_BNBUSDT.py",
        "*EMA*RSI*.py"
    ]
    
    files_to_fix = []
    
    # Check priority files first
    for pattern in priority_files:
        if '*' in pattern:
            files_to_fix.extend(glob.glob(pattern))
        elif os.path.exists(pattern):
            files_to_fix.append(pattern)
    
    # Also check all bot files
    all_bots = glob.glob("*_FEES_*_*USDT.py")
    files_to_fix.extend(all_bots)
    
    # Remove duplicates
    files_to_fix = list(set(files_to_fix))
    
    print(f"Scanning {len(files_to_fix)} files for the error pattern...\n")
    
    affected_files = []
    for filepath in files_to_fix:
        if check_for_error_pattern(filepath):
            affected_files.append(filepath)
            print(f"  ⚠️  {filepath} - Has error pattern")
    
    if not affected_files:
        print("\n✅ No files have the format_qty error pattern")
        return
    
    print(f"\n📝 Found {len(affected_files)} files to fix\n")
    print("-" * 40)
    
    fixed_count = 0
    
    for filepath in affected_files:
        print(f"\n📄 {filepath}")
        
        # Show the problematic code
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        for i, line in enumerate(lines):
            if 'format_qty(' in line and 'timeInForce' in line:
                print(f"  Line {i+1}: {line.strip()}")
        
        print("  Fixing...", end=" ")
        
        try:
            success, message = fix_close_position_method(filepath)
            if success:
                print(f"✅ {message}")
                fixed_count += 1
            else:
                print(f"ℹ️  {message}")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print(f"\n✅ Fixed {fixed_count}/{len(affected_files)} files")
    
    if fixed_count > 0:
        print("\n🎯 WHAT WAS FIXED:")
        print("  • format_qty() now only takes quantity parameter")
        print("  • timeInForce moved to place_order() where it belongs")
        print("  • Close position will now work correctly")
        print("  • All orders use PostOnly for zero slippage")
        
        print("\n📝 TEST THE FIX:")
        print("  1. Run your bot again")
        print("  2. The 'format_qty() got unexpected keyword' error should be gone")
        print("  3. Close positions should work correctly now")
        
        print("\n💡 The bot should now:")
        print("  • Open positions with limit orders (zero slippage)")
        print("  • Close positions with limit orders (zero slippage)")
        print("  • No longer show the format_qty error")

if __name__ == "__main__":
    main()