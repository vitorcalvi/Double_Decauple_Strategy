#!/usr/bin/env python3
"""
Script to fix missing qty and price parameters in close orders for Bybit trading bots
"""

import os
import re
import glob
from pathlib import Path

def find_close_order_patterns(content):
    """Find patterns where close orders might be missing qty/price"""
    patterns = [
        # Pattern 1: reduceOnly without qty/price
        (r'({\s*["\']?category["\']?\s*:\s*["\']linear["\'][^}]*["\']?reduceOnly["\']?\s*:\s*true[^}]*})',
         'close_order_dict'),
        # Pattern 2: order creation for reduce/close
        (r'(order\s*=\s*{\s*[^}]*reduceOnly["\']?\s*:\s*true[^}]*})', 
         'order_creation'),
        # Pattern 3: closeOrder variable assignment
        (r'((?:const|let|var)?\s*closeOrder\s*=\s*{[^}]*})',
         'close_order_var'),
        # Pattern 4: Python dict for close orders
        (r'(close_order\s*=\s*{[^}]*["\']?reduceOnly["\']?\s*:\s*True[^}]*})',
         'python_close_order'),
        # Pattern 5: API call with reduceOnly
        (r'(api\.(?:place_order|create_order|submit_order)\([^)]*reduceOnly[^)]*\))',
         'api_call'),
    ]
    
    issues = []
    for pattern, pattern_type in patterns:
        matches = re.finditer(pattern, content, re.MULTILINE | re.DOTALL)
        for match in matches:
            order_text = match.group(1)
            # Check if qty and price are missing
            has_qty = any(x in order_text.lower() for x in ['qty', 'quantity', 'size', 'amount'])
            has_price = 'price' in order_text.lower()
            
            if not has_qty or not has_price:
                issues.append({
                    'type': pattern_type,
                    'text': order_text,
                    'start': match.start(),
                    'end': match.end(),
                    'missing_qty': not has_qty,
                    'missing_price': not has_price
                })
    
    return issues

def fix_close_order(content, issue):
    """Fix a close order by adding missing qty and price"""
    order_text = issue['text']
    fixed_text = order_text
    
    # Detect if it's JavaScript/TypeScript or Python
    is_python = 'True' in order_text or '.py' in str(issue.get('filename', ''))
    
    if is_python:
        # Python fix
        if issue['missing_qty'] and issue['missing_price']:
            # Add both qty and price
            if 'reduceOnly' in fixed_text:
                insertion_point = fixed_text.find('reduceOnly')
                fixed_text = fixed_text[:insertion_point] + \
                    '"qty": str(position_qty),\n    "price": str(current_price),\n    ' + \
                    fixed_text[insertion_point:]
        elif issue['missing_qty']:
            # Add only qty
            if 'reduceOnly' in fixed_text:
                insertion_point = fixed_text.find('reduceOnly')
                fixed_text = fixed_text[:insertion_point] + \
                    '"qty": str(position_qty),\n    ' + \
                    fixed_text[insertion_point:]
        elif issue['missing_price']:
            # Add only price
            if 'reduceOnly' in fixed_text:
                insertion_point = fixed_text.find('reduceOnly')
                fixed_text = fixed_text[:insertion_point] + \
                    '"price": str(current_price),\n    ' + \
                    fixed_text[insertion_point:]
    else:
        # JavaScript/TypeScript fix
        if issue['missing_qty'] and issue['missing_price']:
            # Add both qty and price
            if 'reduceOnly' in fixed_text:
                insertion_point = fixed_text.find('reduceOnly')
                fixed_text = fixed_text[:insertion_point] + \
                    'qty: positionQty.toString(),\n    price: currentPrice.toString(),\n    ' + \
                    fixed_text[insertion_point:]
        elif issue['missing_qty']:
            # Add only qty
            if 'reduceOnly' in fixed_text:
                insertion_point = fixed_text.find('reduceOnly')
                fixed_text = fixed_text[:insertion_point] + \
                    'qty: positionQty.toString(),\n    ' + \
                    fixed_text[insertion_point:]
        elif issue['missing_price']:
            # Add only price
            if 'reduceOnly' in fixed_text:
                insertion_point = fixed_text.find('reduceOnly')
                fixed_text = fixed_text[:insertion_point] + \
                    'price: currentPrice.toString(),\n    ' + \
                    fixed_text[insertion_point:]
    
    # Replace in content
    new_content = content[:issue['start']] + fixed_text + content[issue['end']:]
    return new_content

def process_file(filepath):
    """Process a single file to fix close order issues"""
    print(f"\n📄 Processing: {filepath}")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        issues = find_close_order_patterns(content)
        
        if not issues:
            print(f"  ✅ No issues found")
            return False
        
        print(f"  ⚠️  Found {len(issues)} potential issues")
        
        # Add filename to issues for context
        for issue in issues:
            issue['filename'] = filepath
        
        # Fix issues (process in reverse order to maintain positions)
        fixed_content = content
        for issue in reversed(issues):
            print(f"    🔧 Fixing {issue['type']}: missing", end='')
            if issue['missing_qty'] and issue['missing_price']:
                print(" qty and price")
            elif issue['missing_qty']:
                print(" qty")
            elif issue['missing_price']:
                print(" price")
            
            fixed_content = fix_close_order(fixed_content, issue)
        
        # Create backup
        backup_path = f"{filepath}.backup"
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  💾 Backup saved: {backup_path}")
        
        # Write fixed content
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        print(f"  ✅ File updated successfully")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error processing file: {e}")
        return False

def main():
    """Main function to process all bot files"""
    print("🚀 Starting Bybit Close Order Fix Script")
    print("=" * 50)
    
    # Get current directory
    current_dir = Path.cwd()
    
    # Find all Python bot files
    bot_patterns = [
        '*_FEES_*.py',
        '*bot*.py',
        '*Bot*.py',
        'bybit_*.py'
    ]
    
    files_to_process = []
    for pattern in bot_patterns:
        files_to_process.extend(glob.glob(str(current_dir / pattern)))
    
    # Remove duplicates
    files_to_process = list(set(files_to_process))
    
    if not files_to_process:
        print("⚠️  No bot files found to process")
        print("Make sure you run this script in the directory containing your bot files")
        return
    
    print(f"📂 Found {len(files_to_process)} files to check")
    
    fixed_count = 0
    for filepath in sorted(files_to_process):
        if process_file(filepath):
            fixed_count += 1
    
    print("\n" + "=" * 50)
    print(f"✅ Complete! Fixed {fixed_count}/{len(files_to_process)} files")
    
    if fixed_count > 0:
        print("\n📝 Important Notes:")
        print("  1. Backup files created with .backup extension")
        print("  2. Review the changes to ensure correctness")
        print("  3. Make sure position_qty and current_price variables are available in your code")
        print("  4. You may need to adjust variable names based on your implementation")

if __name__ == "__main__":
    main()