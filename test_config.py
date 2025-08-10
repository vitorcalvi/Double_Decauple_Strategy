# TEST CONFIGURATION - AGGRESSIVE SIGNALS
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
