#!/usr/bin/env python3
"""
Specific fixes for the exact issues shown in your bot outputs
"""

def diagnose_and_fix():
    print("🔍 SPECIFIC BOT ISSUES DIAGNOSIS")
    print("=" * 50)
    
    issues = {
        "BNB Bot": {
            "problem": "RSI showing 0.0, no signals despite oversold",
            "symptoms": ["RSI: 0.0", "NO SIGNAL", "Trend: DOWN"],
            "root_cause": "RSI calculation returning 0 when price unchanged",
            "fix": """
# In your BNB bot, replace RSI calculation:
def calculate_rsi(prices, period=14):
    import numpy as np
    if len(prices) < period + 2:
        return 50.0  # Not enough data
    
    # Ensure we have price changes
    prices = np.array(prices, dtype=float)
    if np.std(prices) < 0.0001:  # No price movement
        return 50.0  # Neutral RSI
    
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    # Use EMA for RSI
    alpha = 1.0 / period
    gain_ema = gains[-1]
    loss_ema = losses[-1]
    
    for i in range(len(gains) - 2, -1, -1):
        gain_ema = alpha * gains[i] + (1 - alpha) * gain_ema
        loss_ema = alpha * losses[i] + (1 - alpha) * loss_ema
    
    if loss_ema < 0.00001:
        return 100 if gain_ema > 0 else 50
    
    rs = gain_ema / loss_ema
    return 100 - (100 / (1 + rs))

# Also fix signal generation:
if rsi < 30:  # Oversold
    signal = "BUY"
elif rsi > 70:  # Overbought
    signal = "SELL"
"""
        },
        
        "DOGE Liquidity Bot": {
            "problem": "Stuck with open SHORT position, P&L ~$50",
            "symptoms": ["Sell: 21331 DOGE @ $0.2232", "PnL: $50+"],
            "root_cause": "Position not closing despite profit",
            "fix": """
# Force close profitable positions:
if current_pnl > 20:  # $20 profit
    print(f"✅ Closing position with profit: ${current_pnl:.2f}")
    close_position()
elif hold_time > 1800:  # 30 minutes
    print(f"⏰ Timeout close after {hold_time}s")
    close_position()
"""
        },
        
        "ETH Grid Bot": {
            "problem": "Balance: $0.00",
            "symptoms": ["Balance: $0.00", "Waiting for optimal grid signals"],
            "root_cause": "Balance not initialized from config",
            "fix": """
# At the start of your ETH bot:
import json
import os

# Load or create config
config_file = 'config/FEES_GRID_ETHUSDT.json'
if not os.path.exists(config_file):
    config = {'initial_balance': 1000.0, 'symbol': 'ETHUSDT'}
    os.makedirs('config', exist_ok=True)
    with open(config_file, 'w') as f:
        json.dump(config, f)
else:
    with open(config_file) as f:
        config = json.load(f)

balance = config.get('initial_balance', 1000.0)
if balance <= 0:
    balance = 1000.0
    print(f"⚠️ Reset balance to ${balance}")
"""
        },
        
        "XRP Bot": {
            "problem": "RSI < 20 but no trades",
            "symptoms": ["RSI: 9.5", "RSI: 12.0", "No trades"],
            "root_cause": "Extreme oversold not triggering buys",
            "fix": """
# Add emergency buy on extreme RSI:
if rsi < 20:
    print(f"🚨 EXTREME OVERSOLD RSI: {rsi}")
    # Emergency buy with tight stop
    execute_buy(
        qty=calculate_position_size(balance * 0.02),  # 2% risk
        stop_loss=price * 0.99,  # 1% stop
        take_profit=price * 1.02  # 2% profit
    )
"""
        },
        
        "SUI ML Grid Bot": {
            "problem": "High confidence 0.92-0.97 but no trades",
            "symptoms": ["ML Confidence: 0.94", "Balance: $101141.86"],
            "root_cause": "Trade execution disabled or conditions not met",
            "fix": """
# Enable trades on high confidence:
if ml_confidence > 0.90:
    # Calculate position size (1% of balance)
    position_size = (balance * 0.01) / price
    
    # Check grid levels
    if price <= next_buy_grid * 1.001:
        print(f"🎯 ML BUY: Confidence {ml_confidence:.2f}")
        execute_buy(position_size)
    elif price >= next_sell_grid * 0.999:
        print(f"🎯 ML SELL: Confidence {ml_confidence:.2f}")
        execute_sell(position_size)
"""
        },
        
        "General Issues": {
            "problem": "Bots not executing trades despite signals",
            "symptoms": ["Various indicators showing but no trades"],
            "root_cause": "Trade execution disabled or dry-run mode",
            "fix": """
# Add this to ALL bots:
LIVE_TRADING = True  # Set to True to execute real trades

def execute_trade(side, qty, symbol):
    if LIVE_TRADING:
        # Real trade execution
        order = place_order(side, qty, symbol)
        print(f"🎯 LIVE TRADE: {side} {qty} {symbol}")
        return order
    else:
        # Simulation only
        print(f"📝 SIMULATED: {side} {qty} {symbol}")
        return None
"""
        }
    }
    
    # Print diagnosis and fixes
    for bot, info in issues.items():
        print(f"\n🤖 {bot}")
        print(f"   Problem: {info['problem']}")
        print(f"   Root Cause: {info['root_cause']}")
        print(f"   Fix to apply:")
        print("-" * 40)
        print(info['fix'])
        print("-" * 40)
    
    print("\n" + "=" * 50)
    print("🛠️ QUICK FIX COMMANDS")
    print("=" * 50)
    
    commands = """
# 1. Stop all bots
pkill -f "python3"

# 2. Create default configs for all bots
python3 -c "
import json
import os
os.makedirs('config', exist_ok=True)
bots = ['BNBUSDT', 'DOGEUSDT', 'ETHUSDT', 'XRPUSDT', 'LINKUSDT', 'SUIUSDT', 'AVAXUSDT']
for bot in bots:
    config = {
        'initial_balance': 1000.0,
        'risk_percent': 2.0,
        'enabled': True,
        'live_trading': False,  # Start in simulation
        'min_rsi': 30,
        'max_rsi': 70
    }
    with open(f'config/FEES_{bot}.json', 'w') as f:
        json.dump(config, f, indent=2)
    print(f'Created config for {bot}')
"

# 3. Test one bot with debug output
python3 -c "
# Test BNB bot RSI calculation
prices = [785.3, 785.4, 785.5, 785.6, 785.5, 785.4, 785.3, 785.4, 785.5, 785.6, 785.7, 785.8, 785.9, 786.0]
import numpy as np
deltas = np.diff(prices)
print(f'Price changes: {deltas}')
gains = np.where(deltas > 0, deltas, 0)
losses = np.where(deltas < 0, -deltas, 0)
print(f'Gains: {gains}')
print(f'Losses: {losses}')
avg_gain = np.mean(gains)
avg_loss = np.mean(losses)
if avg_loss > 0:
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    print(f'RSI: {rsi:.1f}')
else:
    print('RSI: 50.0 (no losses)')
"

# 4. Start bots with forced balance
for bot in FEES_*.py; do
    echo "Starting $bot with $1000 balance..."
    python3 -c "
import sys
sys.path.insert(0, '.')
balance = 1000.0  # Force balance
exec(open('$bot').read())
" > logs/${bot%.py}.log 2>&1 &
done

# 5. Monitor output
tail -f logs/*.log | grep -E "(TRADE|SIGNAL|ERROR|RSI)"
"""
    
    print(commands)

if __name__ == "__main__":
    diagnose_and_fix()