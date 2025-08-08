import os
import re

def update_bot_files():
    # Find all bot files
    bot_files = [f for f in os.listdir('.') if f.endswith('USDT.py') and 'FEES' in f]
    
    for bot_file in bot_files:
        with open(bot_file, 'r') as f:
            content = f.read()
        
        # Pattern to find order creation (Bybit)
        patterns = [
            # Pattern 1: place_order
            (r'(place_order\([^)]*)',
             r'\1, time_in_force="PostOnly"'),
            
            # Pattern 2: create_order
            (r'(create_order\([^)]*type\s*=\s*["\']Limit["\'][^)]*)',
             r'\1, time_in_force="PostOnly"'),
            
            # Pattern 3: submit_order
            (r'(submit_order\([^)]*orderType\s*=\s*["\']Limit["\'][^)]*)',
             r'\1, timeInForce="PostOnly"')
        ]
        
        modified = False
        for pattern, replacement in patterns:
            if 'time_in_force' not in content and 'timeInForce' not in content:
                new_content = re.sub(pattern, replacement, content)
                if new_content != content:
                    content = new_content
                    modified = True
        
        # If already has timeInForce, update it
        if 'timeInForce="GTC"' in content or 'time_in_force="GTC"' in content:
            content = content.replace('timeInForce="GTC"', 'timeInForce="PostOnly"')
            content = content.replace('time_in_force="GTC"', 'time_in_force="PostOnly"')
            modified = True
        
        if modified:
            # Backup original
            os.rename(bot_file, f"{bot_file}.backup")
            
            # Write updated file
            with open(bot_file, 'w') as f:
                f.write(content)
            
            print(f"✓ Updated: {bot_file}")
        else:
            print(f"⚠ Check manually: {bot_file}")

if __name__ == "__main__":
    update_bot_files()
    print("\n✅ Complete! Backups created as .backup files")