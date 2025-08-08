#!/usr/bin/env python3
"""
PnL Guard - Universal Profit Protection Layer for Bybit
Monitors all open positions and protects profits using dynamic thresholds
"""

import json
import time
import asyncio
from datetime import datetime
import numpy as np
from pybit.unified_trading import HTTP

class PnLGuard:
    def __init__(self, api_key, api_secret, demo=True, maker_fee=0.001):
        self.exchange = HTTP(demo=demo, api_key=api_key, api_secret=api_secret)
        self.maker_fee = maker_fee
        self.positions = {}
        self.peak_profits = {}
        self.atr_cache = {}
        self.stats = {}
        
        self.config = {
            'atr_period': 14,
            'atr_multiplier': 0.5,
            'min_profit_to_guard': 0.002,
            'max_drawdown_pct': 0.3,
            'time_decay_factor': 0.1,
            'check_interval': 1,
            'force_close_after_minutes': 30
        }
    
    def calculate_atr(self, symbol, candles):
        """Calculate ATR for dynamic threshold"""
        if len(candles) < self.config['atr_period']:
            return None
        
        h = np.array([float(c[2]) for c in candles])  # high
        l = np.array([float(c[3]) for c in candles])  # low
        c = np.array([float(c[4]) for c in candles])  # close
        
        tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
        return np.mean(tr[-self.config['atr_period']:])
    
    def get_dynamic_threshold(self, symbol, position):
        """Calculate dynamic profit protection threshold"""
        now = time.time()
        
        # Update ATR cache if needed
        if symbol not in self.atr_cache or now - self.atr_cache[symbol][1] > 60:
            try:
                resp = self.exchange.get_kline(
                    category="linear",
                    symbol=symbol,
                    interval="1",
                    limit=20
                )
                if resp.get("retCode") == 0:
                    candles = resp["result"]["list"]
                    atr = self.calculate_atr(symbol, candles)
                    if atr:
                        self.atr_cache[symbol] = (atr, now)
            except:
                pass
        
        # Calculate threshold
        threshold = self.config['max_drawdown_pct']
        if symbol in self.atr_cache:
            atr_threshold = (self.atr_cache[symbol][0] * self.config['atr_multiplier']) / position['current_price']
            threshold = min(atr_threshold, threshold)
        
        # Apply time decay
        mins = (now - position['profitable_since']) / 60
        time_factor = max(0.5, 1 - (self.config['time_decay_factor'] * mins / 60))
        
        return threshold * time_factor
    
    def calculate_pnl(self, pos):
        """Calculate current PnL including fees"""
        entry_val = pos['entry_price'] * pos['size']
        current_val = pos['current_price'] * pos['size']
        
        gross = (current_val - entry_val) if pos['side'] == 'Buy' else (entry_val - current_val)
        fees = entry_val * self.maker_fee * 2
        
        return {
            'gross_pnl': gross,
            'net_pnl': gross - fees,
            'pnl_percentage': ((gross - fees) / entry_val) * 100,
            'fees': fees
        }
    
    def should_close_position(self, pid, pos):
        """Determine if position should be closed"""
        pnl = self.calculate_pnl(pos)
        
        if pnl['net_pnl'] <= 0:
            return False, None
        
        # Track peak profit
        if pid not in self.peak_profits:
            pos['profitable_since'] = time.time()
            self.peak_profits[pid] = pnl['net_pnl']
            self._update_stats(pos['symbol'], 'positions_guarded', 1)
            return False, None
        
        self.peak_profits[pid] = max(self.peak_profits[pid], pnl['net_pnl'])
        drawdown = (self.peak_profits[pid] - pnl['net_pnl']) / self.peak_profits[pid]
        
        # Check close conditions
        reasons = []
        threshold = self.get_dynamic_threshold(pos['symbol'], pos)
        
        if drawdown > threshold:
            reasons.append(f"Drawdown {drawdown:.1%} > {threshold:.1%}")
        
        mins = (time.time() - pos['profitable_since']) / 60
        if mins > self.config['force_close_after_minutes']:
            reasons.append(f"Profitable for {mins:.0f} minutes")
        
        if 'last_pnl' in pos and pnl['net_pnl'] - pos['last_pnl'] < -self.peak_profits[pid] * 0.1:
            reasons.append(f"Rapid loss: ${pnl['net_pnl'] - pos['last_pnl']:.2f}")
        
        pos['last_pnl'] = pnl['net_pnl']
        
        return bool(reasons), {
            'reason': ', '.join(reasons),
            'peak_profit': self.peak_profits[pid],
            'current_profit': pnl['net_pnl'],
            'saved_profit': pnl['net_pnl'],
            'drawdown': drawdown
        } if reasons else None
    
    def close_position(self, pid, pos, reason):
        """Force close a position"""
        try:
            side = 'Sell' if pos['side'] == 'Buy' else 'Buy'
            
            resp = self.exchange.place_order(
                category="linear",
                symbol=pos['symbol'],
                side=side,
                orderType="Market",
                qty=str(pos['size']),
                reduceOnly=True
            )
            
            if resp.get("retCode") != 0:
                print(f"❌ Failed to close: {resp.get('retMsg')}")
                return False
            
            # Log
            with open('pnl_guard_log.jsonl', 'a') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'action': 'PNL_GUARD_CLOSE',
                    'position_id': pid,
                    'symbol': pos['symbol'],
                    'reason': reason['reason'],
                    'peak_profit': reason['peak_profit'],
                    'closed_profit': reason['current_profit'],
                    'drawdown_at_close': reason['drawdown'],
                    'order_id': resp.get('result', {}).get('orderId')
                }, f)
                f.write('\n')
            
            # Update stats
            self._update_stats(pos['symbol'], 'positions_closed', 1)
            self._update_stats(pos['symbol'], 'profit_saved', reason['saved_profit'])
            self._update_stats(pos['symbol'], 'peak_profits_lost', 
                             reason['peak_profit'] - reason['current_profit'])
            
            del self.positions[pid]
            del self.peak_profits[pid]
            
            print(f"✅ PnL Guard closed {pos['symbol']}: {reason['reason']}")
            print(f"   Saved: ${reason['saved_profit']:.2f}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to close {pid}: {e}")
            return False
    
    def _update_stats(self, symbol, key, value):
        """Update statistics"""
        if symbol not in self.stats:
            self.stats[symbol] = {
                'positions_guarded': 0,
                'positions_closed': 0,
                'profit_saved': 0,
                'peak_profits_lost': 0
            }
        self.stats[symbol][key] += value
    
    def update_positions(self):
        """Fetch current positions from exchange"""
        try:
            resp = self.exchange.get_positions(category="linear", settleCoin="USDT")
            if resp.get("retCode") != 0:
                return
            
            current_ids = set()
            
            for p in resp["result"]["list"]:
                if float(p.get("size", 0)) > 0:
                    symbol = p["symbol"]
                    side = p["side"]
                    pid = f"{symbol}_{side}"
                    current_ids.add(pid)
                    
                    if pid not in self.positions:
                        self.positions[pid] = {
                            'symbol': symbol,
                            'side': side,
                            'size': float(p["size"]),
                            'entry_price': float(p["avgPrice"]),
                            'opened_at': time.time()
                        }
                    
                    self.positions[pid]['current_price'] = float(p["markPrice"])
            
            # Remove closed
            for pid in set(self.positions.keys()) - current_ids:
                self.positions.pop(pid, None)
                self.peak_profits.pop(pid, None)
                
        except Exception as e:
            print(f"Error updating positions: {e}")
    
    async def guard_loop(self):
        """Main monitoring loop"""
        print(f"🛡️ PnL Guard Started (Bybit)\nConfig: {json.dumps(self.config, indent=2)}")
        
        while True:
            try:
                self.update_positions()
                
                # Check positions
                to_close = []
                for pid, pos in self.positions.items():
                    should, reason = self.should_close_position(pid, pos)
                    if should:
                        to_close.append((pid, pos, reason))
                
                # Close if needed
                for pid, pos, reason in to_close:
                    self.close_position(pid, pos, reason)
                
                # Stats every minute
                if int(time.time()) % 60 == 0:
                    self.display_stats()
                
                await asyncio.sleep(self.config['check_interval'])
                
            except Exception as e:
                print(f"Guard loop error: {e}")
                await asyncio.sleep(5)
    
    def display_stats(self):
        """Display current statistics"""
        if not self.stats:
            return
            
        print("\n📊 PnL Guard Statistics\n" + "-" * 50)
        
        total_saved = sum(s['profit_saved'] for s in self.stats.values())
        total_closed = sum(s['positions_closed'] for s in self.stats.values())
        
        for symbol, s in self.stats.items():
            if s['positions_closed']:
                print(f"{symbol}:\n  Guarded: {s['positions_guarded']}\n  "
                      f"Closed: {s['positions_closed']}\n  "
                      f"Saved: ${s['profit_saved']:.2f}\n  "
                      f"Lost from peak: ${s['peak_profits_lost']:.2f}")
        
        if total_closed:
            print(f"\nTotal saved: ${total_saved:.2f}\nTotal closed: {total_closed}")
        
        print(f"Monitoring: {len(self.positions)} positions\n" + "-" * 50)


class PnLGuardManager:
    """Manage PnL Guard for multiple accounts"""
    
    def __init__(self):
        self.guards = {}
    
    def add_account(self, name, api_key, api_secret, demo=True, config=None):
        """Add account to monitor"""
        guard = PnLGuard(api_key, api_secret, demo)
        if config:
            guard.config.update(config)
        self.guards[name] = guard
    
    async def start_all(self):
        """Start monitoring all accounts"""
        tasks = [asyncio.create_task(g.guard_loop()) 
                for name, g in self.guards.items()]
        await asyncio.gather(*tasks)


async def main():
    """Run PnL Guard standalone"""
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Use demo/testnet by default
    demo = os.getenv("DEMO_MODE", "true").lower() == "true"
    prefix = "TESTNET_" if demo else "LIVE_"
    
    api_key = os.getenv(f"{prefix}BYBIT_API_KEY", "")
    api_secret = os.getenv(f"{prefix}BYBIT_API_SECRET", "")
    
    if not api_key or not api_secret:
        print("❌ API credentials not found in .env")
        return
    
    config = {
        'atr_multiplier': 0.75,
        'min_profit_to_guard': 0.001,
        'max_drawdown_pct': 0.25,
        'force_close_after_minutes': 20,
        'check_interval': 2
    }
    
    guard = PnLGuard(api_key, api_secret, demo, maker_fee=0.001)
    guard.config.update(config)
    
    try:
        await guard.guard_loop()
    except KeyboardInterrupt:
        print("\n🛑 PnL Guard stopped")


if __name__ == "__main__":
    asyncio.run(main())