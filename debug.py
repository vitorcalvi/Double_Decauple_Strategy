#!/usr/bin/env python3
"""
Quick fixes to make trades fire more frequently
"""

import os
import shutil
from datetime import datetime

def apply_signal_fixes():
    """Apply fixes to both bots to make signals fire more easily"""
    
    fixes_applied = []
    
    # Fix Bot 5 - XRP
    print("\n📄 Fixing Bot 5 (XRP) signal thresholds...")
    
    xrp_file = "5_FEES_MACD_VWAP_XRPUSDT.py"
    if os.path.exists(xrp_file):
        # Backup
        backup = f"{xrp_file}.backup_signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy(xrp_file, backup)
        print(f"   ✅ Backup: {backup}")
        
        with open(xrp_file, 'r') as f:
            content = f.read()
        
        # Original conservative thresholds
        old_config = """        self.config = {
            'ema_fast': 10,
            'ema_slow': 21,
            'rsi_period': 14,
            'rsi_long_threshold': 45,   # matches logic below
            'rsi_short_threshold': 55,  # matches logic below"""
        
        # More aggressive thresholds for more signals
        new_config = """        self.config = {
            'ema_fast': 10,
            'ema_slow': 21,
            'rsi_period': 14,
            'rsi_long_threshold': 50,   # RELAXED: was 45
            'rsi_short_threshold': 50,  # RELAXED: was 55"""
        
        if old_config in content:
            content = content.replace(old_config, new_config)
            fixes_applied.append("XRP: Relaxed RSI thresholds (45/55 → 50/50)")
        
        # Also reduce EMA periods for faster crossovers
        content = content.replace("'ema_fast': 10,", "'ema_fast': 8,")
        content = content.replace("'ema_slow': 21,", "'ema_slow': 13,")
        fixes_applied.append("XRP: Faster EMAs (10/21 → 8/13)")
        
        # Reduce cooldown
        content = content.replace("self.trade_cooldown = 30", "self.trade_cooldown = 10")
        fixes_applied.append("XRP: Reduced cooldown (30s → 10s)")
        
        with open(xrp_file, 'w') as f:
            f.write(content)
        print("   ✅ XRP bot fixed")
    
    # Fix Bot 12 - SUI
    print("\n📄 Fixing Bot 12 (SUI) ML thresholds...")
    
    sui_file = "12_FEES_ML_GRID_SUIUSDT.py"
    if os.path.exists(sui_file):
        # Backup
        backup = f"{sui_file}.backup_signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy(sui_file, backup)
        print(f"   ✅ Backup: {backup}")
        
        with open(sui_file, 'r') as f:
            content = f.read()
        
        # Reduce ML threshold
        content = content.replace("'ml_threshold': 0.60,", "'ml_threshold': 0.45,")
        fixes_applied.append("SUI: Lowered ML threshold (0.60 → 0.45)")
        
        # Increase grid proximity threshold
        content = content.replace("if distance_pct < 0.25", "if distance_pct < 0.50")
        fixes_applied.append("SUI: Increased grid proximity (0.25% → 0.50%)")
        
        # Reduce cooldown
        content = content.replace("self.trade_cooldown = 30", "self.trade_cooldown = 10")
        fixes_applied.append("SUI: Reduced cooldown (30s → 10s)")
        
        # Use shorter timeframe
        content = content.replace("'timeframe': '5',", "'timeframe': '1',")
        fixes_applied.append("SUI: Shorter timeframe (5m → 1m)")
        
        with open(sui_file, 'w') as f:
            f.write(content)
        print("   ✅ SUI bot fixed")
    
    return fixes_applied


def create_test_config():
    """Create a test configuration for immediate signals"""
    
    test_config = """# TEST CONFIGURATION - AGGRESSIVE SIGNALS
# Add this to your bot files for testing

TEST_MODE = True  # Enable test mode

if TEST_MODE:
    # XRP Bot Test Config
    XRP_TEST = {
        'ema_fast': 5,           # Very fast
        'ema_slow': 10,          # Fast
        'rsi_period': 7,         # Shorter period
        'rsi_long_threshold': 60,   # Easy to trigger
        'rsi_short_threshold': 40,  # Easy to trigger
        'trade_cooldown': 5,     # Minimal cooldown
        'risk_per_trade_pct': 0.5,  # Small risk for testing
    }
    
    # SUI Bot Test Config
    SUI_TEST = {
        'timeframe': '1',        # 1-minute for more signals
        'ml_threshold': 0.30,    # Very low threshold
        'grid_levels': 10,       # More levels
        'base_grid_spacing': 0.2,  # Tighter grid
        'trade_cooldown': 5,     # Minimal cooldown
        'risk_per_trade': 0.5,   # Small risk for testing
    }
"""
    
    with open("test_config.py", 'w') as f:
        f.write(test_config)
    
    print("\n✅ Created test_config.py with aggressive settings")
    return True


def main():
    print("\n" + "="*60)
    print("SIGNAL GENERATION FIX")
    print("Making trades fire more frequently")
    print("="*60)
    
    print("\n🎯 CURRENT ISSUES:")
    print("• XRP: RSI thresholds too strict (45/55)")
    print("• XRP: EMA periods too slow (10/21)")
    print("• SUI: ML confidence threshold too high (0.60)")
    print("• SUI: Grid proximity too tight (0.25%)")
    print("• Both: Trade cooldown too long (30s)")
    
    print("\n🔧 APPLYING FIXES...")
    
    fixes = apply_signal_fixes()
    
    if fixes:
        print("\n✅ FIXES APPLIED:")
        for fix in fixes:
            print(f"   • {fix}")
    
    create_test_config()
    
    print("\n" + "="*60)
    print("✅ SIGNAL FIXES COMPLETE!")
    print("="*60)
    
    print("\n📊 WHAT CHANGED:")
    print("\nXRP Bot:")
    print("• RSI thresholds: 45/55 → 50/50 (more signals)")
    print("• EMA periods: 10/21 → 8/13 (faster crossovers)")
    print("• Cooldown: 30s → 10s (more frequent trades)")
    
    print("\nSUI Bot:")
    print("• ML threshold: 0.60 → 0.45 (easier signals)")
    print("• Grid proximity: 0.25% → 0.50% (wider trigger zone)")
    print("• Timeframe: 5m → 1m (more data points)")
    print("• Cooldown: 30s → 10s (more frequent trades)")
    
    print("\n🚀 NEXT STEPS:")
    print("\n1. Run the signal debugger to see real-time conditions:")
    print("   python signal_debugger.py")
    
    print("\n2. Run the bots with new settings:")
    print("   python 5_FEES_MACD_VWAP_XRPUSDT.py")
    print("   python 12_FEES_ML_GRID_SUIUSDT.py")
    
    print("\n3. Monitor for signals (should fire within 5-10 minutes)")
    
    print("\n⚠️  IMPORTANT:")
    print("• These are aggressive settings for testing")
    print("• Expect more trades but potentially lower quality")
    print("• Revert to original settings for production")
    print("• Backups created with '_signals_' timestamp")


if __name__ == "__main__":
    main()