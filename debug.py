#!/usr/bin/env python3
"""
Test if fixed bots now generate signals
"""

import subprocess
import time
import os
import signal
import sys

def test_bot_signals(bot_num, duration=30):
    """Test if bot generates signals"""
    
    # Find bot file
    bot_file = None
    for f in os.listdir('.'):
        if f.startswith(f"{bot_num}_FEES_") and f.endswith('.py'):
            bot_file = f
            break
    
    if not bot_file:
        return None, "File not found"
    
    symbol = bot_file.split('_')[-1].replace('.py', '')
    log_file = f"logs/{bot_num}_FEES_{symbol}.log"
    
    print(f"\n{'='*50}")
    print(f"Testing Bot #{bot_num}: {bot_file}")
    print(f"Symbol: {symbol}")
    print(f"{'='*50}")
    
    # Check initial log state
    initial_size = 0
    if os.path.exists(log_file):
        initial_size = os.path.getsize(log_file)
        print(f"📄 Log exists: {initial_size} bytes")
    else:
        print(f"📄 No log file yet")
    
    # Run bot for X seconds
    print(f"🏃 Running bot for {duration} seconds...")
    
    try:
        proc = subprocess.Popen(
            [sys.executable, bot_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait and capture output
        time.sleep(duration)
        
        # Terminate bot
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except:
            proc.kill()
        
        # Check results
        if os.path.exists(log_file):
            new_size = os.path.getsize(log_file)
            if new_size > initial_size:
                print(f"✅ LOG UPDATED: +{new_size - initial_size} bytes")
                
                # Show last lines of log
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        print("\n📊 Recent log entries:")
                        for line in lines[-5:]:
                            print(f"   {line.strip()}")
                
                return True, "Generating logs"
            else:
                print(f"⚠️ Log not updated")
                return False, "No new logs"
        else:
            print(f"❌ No log created")
            return False, "No logs"
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False, str(e)

def quick_test_all():
    """Quick test all non-working bots"""
    
    print("="*60)
    print("QUICK SIGNAL TEST - 30 seconds per bot")
    print("="*60)
    
    # Test previously non-working bots
    test_bots = [2, 5, 7, 8, 9, 10, 11, 12, 13, 14]
    
    results = {}
    
    for bot_num in test_bots[:5]:  # Test first 5
        success, message = test_bot_signals(bot_num, duration=30)
        results[bot_num] = (success, message)
        
        if bot_num < test_bots[4]:
            response = input("\nContinue? (y/n/all): ").lower()
            if response == 'n':
                break
            elif response == 'all':
                # Test remaining without asking
                for remaining_bot in test_bots[5:]:
                    success, message = test_bot_signals(remaining_bot, duration=30)
                    results[remaining_bot] = (success, message)
                break
    
    # Summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    working_now = []
    still_not_working = []
    
    for bot_num, (success, message) in results.items():
        if success:
            working_now.append(bot_num)
            print(f"✅ Bot #{bot_num}: {message}")
        else:
            still_not_working.append(bot_num)
            print(f"❌ Bot #{bot_num}: {message}")
    
    print(f"\n📊 Statistics:")
    print(f"   Previously working: [1, 3, 4, 6]")
    print(f"   Now working: {working_now}")
    print(f"   Still not working: {still_not_working}")
    
    if still_not_working:
        print(f"\n💡 For bots {still_not_working}:")
        print("   - They may need specific market conditions")
        print("   - Check if symbols exist on Bybit")
        print("   - May need more time to generate signals")
        print("   - Run individually to see errors:")
        print(f"     python3 {still_not_working[0]}_FEES_*.py")

def main():
    print("="*60)
    print("TEST FIXED BOTS")
    print("="*60)
    
    print("\nOptions:")
    print("1. Quick test all bots (30 sec each)")
    print("2. Test specific bot")
    print("3. Check logs only")
    
    choice = input("\nChoice (1/2/3): ")
    
    if choice == '1':
        quick_test_all()
    elif choice == '2':
        bot_num = int(input("Bot number: "))
        duration = int(input("Duration (seconds): ") or 30)
        test_bot_signals(bot_num, duration)
    elif choice == '3':
        print("\n📄 Checking all logs...")
        log_files = []
        for f in os.listdir('logs'):
            if f.endswith('.log'):
                size = os.path.getsize(f'logs/{f}')
                if size > 0:
                    log_files.append((f, size))
        
        if log_files:
            print(f"\nLogs with content:")
            for log, size in sorted(log_files):
                bot_num = log.split('_')[0]
                print(f"   Bot #{bot_num}: {log} ({size} bytes)")
        else:
            print("No logs with content found")

if __name__ == "__main__":
    main()