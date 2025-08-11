#!/usr/bin/perl
use strict;
use warnings;
use File::Copy;

# List of files to process
my @files = (
    'OK_1_FEES_EMA_BB_SOLUSDT.py',
    'OK_2_FEES_EMA_RSI_BNBUSDT.py',
    'OK_3_FEES_EMAMACDRSI_LTCUSDT.py',
    'OK_4_FEES_LIQUIDITYSWEEPBOT_DOGEUSDT.py',
    'OK_5_FEES_MACD_VWAP_XRPUSDT.py',
    'OK_6_Ref_FEES_MLFiltered_ARBUSDT.py',
    'OK_7_FEES_DYNAMIC_GRID_ETHUSDT.py',
    'OK_8_FEES_VWAP_RSI_DIV_AVAXUSDT.py',
    'OK_9_FEES_PIVOT_REVERSAL_LINKUSDT.py',
    'OK_10_FEES_RMI_SUPERTREND_ADAUSDT.py',
    'OK_11_FEES_RANGE_REGRESSION_DOTUSDT.py',
    'OK_12_FEES_ML_GRID_SUIUSDT.py',
    'OK_13_FEES_FEATURE_BTCUSDT.py'
);

# New TradeLogger implementation with all fixes
my $new_tradelogger = q{# Utilities
_iso = lambda: datetime.now(timezone.utc).isoformat()
_bps = lambda d,b: (float(d)/float(b))*1e4 if b else None
_float = lambda v,d=0: float(v) if v and str(v).strip() else d

def _safe_float(v, default=0):
    """Safely convert to float"""
    try:
        return float(v) if v is not None else default
    except (ValueError, TypeError):
        return default

def _quantize(value, step):
    """Quantize value to step size"""
    if step <= 0:
        return value
    return round(value / step) * step

def _tags(tags):
    if not tags: return {}
    if isinstance(tags, dict): return tags
    out = {}
    if isinstance(tags, str):
        for kv in tags.split("|"):
            if not kv: continue
            if ":" in kv:
                k,v = kv.split(":",1)
                k,v = k.strip(), v.strip()
                try: out[k] = float(v)
                except: out[k] = True if v.lower()=="true" else False if v.lower()=="false" else v
            else: out[kv.strip()] = True
    return out

class TradeLogger:
    def __init__(self, bot_name, symbol, log_file=None, tf=None):
        self.ver = 1
        self.cls = "TradeLogger.v1"
        self.sess = f"s{int(time.time()*1000):x}"
        self.env = "testnet" if os.getenv("DEMO_MODE","true").lower()=="true" else "live"
        self.bot = bot_name
        self.sym = symbol
        self.tf = tf
        self.ccy = "USDT"
        self.id_seq = 1000
        self.open = {}
        self.pnl_day = 0.0
        self.daily_pnl = 0.0  # Alias for compatibility
        self.streak_loss = 0
        self.consecutive_losses = 0  # Alias for compatibility
        self.max_daily_loss = 1000.0  # Default max daily loss
        os.makedirs("logs", exist_ok=True)
        self.log_file = log_file or f"logs/{self.bot}_{self.sym}_{datetime.now(timezone.utc):%Y%m%d}.jsonl"

    def _id(self): 
        self.id_seq += 1
        return self.id_seq
    
    def generate_trade_id(self):
        """Alias for backward compatibility"""
        return self._id()
    
    def _w(self, o): 
        with open(self.log_file, "a") as f:
            f.write(json.dumps(o, separators=(",",":"), ensure_ascii=False) + "\n")
    
    def _bars(self, dur_s):
        if not self.tf: return None
        try: return max(1, (dur_s + int(self.tf)*60 - 1) // (int(self.tf)*60))
        except: return None

    def log_open(self, side_long_short, expected_px, actual_px, qty, stop_loss_px, take_profit_px, balance_usd, tags=None):
        tid = self._id()
        actual_px = _float(actual_px)
        expected_px = _float(expected_px)
        stop_loss_px = _float(stop_loss_px)
        take_profit_px = _float(take_profit_px)
        qty = _float(qty)
        
        slip_bps = _bps(actual_px - expected_px, expected_px)
        stop_move = abs(actual_px - stop_loss_px)
        tp_move = abs(take_profit_px - actual_px)
        risk = stop_move * qty if stop_move > 0 else 0.0
        rr = (tp_move / stop_move) if stop_move > 0 else None
        tg = _tags(tags)
        
        rec = {
            "ver": self.ver, "cls": self.cls, "sess": self.sess, "env": self.env,
            "bot": self.bot, "sym": self.sym, "tf": self.tf,
            "t": "O", "id": tid, "sd": side_long_short, "ccy": self.ccy,
            "px": round(actual_px, 6), "exp": round(expected_px, 6),
            "slip_bps": round(slip_bps, 2) if slip_bps is not None else None,
            "qty": round(qty, 6), "sl": round(stop_loss_px, 6), "tp": round(take_profit_px, 6),
            "risk_usd": round(risk, 4), "rr_plan": round(rr, 4) if rr else None,
            "bal": round(_float(balance_usd, 1000), 2), "tags": tg, "ts": _iso()
        }
        self._w(rec)
        self.open[tid] = {
            "tso": datetime.now(timezone.utc), "entry": actual_px, 
            "sd": side_long_short, "qty": qty, "risk": risk, "tags": tg
        }
        return tid
    
    def log_trade_open(self, side, expected_px, actual_px, qty, stop_loss_px, take_profit_px, info=None):
        """Backward compatibility wrapper"""
        side_ls = "L" if side == "BUY" else "S"
        return self.log_open(side_ls, expected_px, actual_px, qty, stop_loss_px, take_profit_px, 1000.0, tags=info)

    def log_close(self, trade_id, expected_exit, actual_exit, exit_reason, in_bps, out_bps, extra=None):
        st = self.open.get(trade_id)
        if not st: return None
        
        actual_exit = _float(actual_exit)
        expected_exit = _float(expected_exit)
        dur = max(0, int((datetime.now(timezone.utc) - st["tso"]).total_seconds()))
        edge_bps = _bps(actual_exit - expected_exit, expected_exit)
        
        qty = st["qty"]
        sd = st["sd"]
        entry = st["entry"]
        gross = (actual_exit - entry) * qty if sd == "L" else (entry - actual_exit) * qty
        
        fe_in = (abs(_float(in_bps))/1e4 * entry * qty) if in_bps else 0.0
        fe_out = (abs(_float(out_bps))/1e4 * actual_exit * qty) if out_bps else 0.0
        fees = fe_in + fe_out
        net = gross - fees
        R = (net / st["risk"]) if st["risk"] > 0 else None
        
        self.pnl_day += net
        self.daily_pnl = self.pnl_day  # Keep alias updated
        self.streak_loss = self.streak_loss + 1 if net < 0 else 0
        self.consecutive_losses = self.streak_loss  # Keep alias updated
        
        rec = {
            "ver": self.ver, "cls": self.cls, "sess": self.sess, "env": self.env,
            "bot": self.bot, "sym": self.sym, "tf": self.tf,
            "t": "C", "ref": trade_id, "ccy": self.ccy,
            "px": round(actual_exit, 6), "ref_px": round(expected_exit, 6),
            "edge_bps": round(edge_bps, 2) if edge_bps else None,
            "dur_s": dur, "bars_held": self._bars(dur),
            "qty": round(qty, 6), "gross": round(gross, 4),
            "fees_in_bps": round(_float(in_bps), 2) if in_bps else None,
            "fees_out_bps": round(_float(out_bps), 2) if out_bps else None,
            "fees_total": round(fees, 4),
            "net": round(net, 4), "R": round(R, 4) if R else None,
            "exit": exit_reason, "pnl_day": round(self.pnl_day, 4),
            "streak_loss": int(self.streak_loss),
            "tags": st["tags"], "extra": extra or {}, "ts": _iso()
        }
        self._w(rec)
        del self.open[trade_id]
        return rec
    
    def log_trade_close(self, trade_id, expected_exit, actual_exit, exit_reason, fees_entry=0.0004, fees_exit=0.0004):
        """Backward compatibility wrapper"""
        # Convert percentage fees to basis points (0.0004 = 4 bps)
        in_bps = fees_entry * 10000 if fees_entry < 1 else fees_entry
        out_bps = fees_exit * 10000 if fees_exit < 1 else fees_exit
        return self.log_close(trade_id, expected_exit, actual_exit, exit_reason, in_bps, out_bps)

    def log_close_unknown(self, trade_id, reason="unknown", extra=None):
        st = self.open.get(trade_id)
        if not st: return None
        
        dur = max(0, int((datetime.now(timezone.utc) - st["tso"]).total_seconds()))
        rec = {
            "ver": self.ver, "cls": self.cls, "sess": self.sess, "env": self.env,
            "bot": self.bot, "sym": self.sym, "tf": self.tf,
            "t": "C", "ref": trade_id, "ccy": self.ccy,
            "px": None, "ref_px": None, "edge_bps": None,
            "dur_s": dur, "bars_held": self._bars(dur),
            "qty": round(st["qty"], 6),
            "gross": None, "fees_in_bps": None, "fees_out_bps": None, "fees_total": None,
            "net": None, "R": None, "exit": reason,
            "pnl_day": round(self.pnl_day, 4), "streak_loss": int(self.streak_loss),
            "tags": st["tags"], "extra": extra or {"note": "external close"}, "ts": _iso()
        }
        self._w(rec)
        del self.open[trade_id]
        return rec};

print "TradeLogger Fix Script v4.0 (Final)\n";
print "====================================\n\n";

my $total_fixes = 0;

foreach my $file (@files) {
    if (!-f $file) {
        print "⚠️  File not found: $file\n";
        next;
    }
    
    print "Processing: $file\n";
    
    # Backup original
    my $backup = $file . ".backup_final";
    copy($file, $backup) or die "Cannot backup $file: $!";
    print "  ✓ Backup created: $backup\n";
    
    # Read file
    open(my $fh, '<', $file) or die "Cannot open $file: $!";
    my $content = do { local $/; <$fh> };
    close($fh);
    
    my @fixes_applied = ();
    
    # Fix 1: datetime.utcnow() -> datetime.now(timezone.utc)
    my $utc_fixes = ($content =~ s/datetime\.utcnow\(\)/datetime.now(timezone.utc)/g);
    push @fixes_applied, "Fixed $utc_fixes datetime.utcnow() calls" if $utc_fixes;
    
    # Fix 2: Ensure timezone import
    if ($content =~ /from datetime import/ && $content !~ /from datetime import.*timezone/) {
        $content =~ s/(from datetime import .*?)(\n)/$1, timezone$2/;
        push @fixes_applied, "Added timezone import";
    }
    
    # Fix 3: Fix duplicate tf parameters (BEFORE replacing TradeLogger)
    my $dup_tf_fixes = ($content =~ s/TradeLogger\(([^,]+),\s*([^,]+),\s*tf="(\d+)",\s*tf="\3"\)/TradeLogger($1, $2, tf="$3")/g);
    push @fixes_applied, "Fixed $dup_tf_fixes duplicate tf parameters" if $dup_tf_fixes;
    
    # Extract timeframe if present
    my $tf = '';
    if ($content =~ /'timeframe'\s*:\s*'(\d+)'/) {
        $tf = $1;
    } elsif ($content =~ /TIMEFRAME\s*=\s*'(\d+)'/) {
        $tf = $1;
    }
    
    # Fix 4: Replace TradeLogger class
    my $class_replaced = 0;
    
    # Find where TradeLogger class starts
    if ($content =~ /class\s+TradeLogger:/) {
        # Simple replacement - find class and replace until next major section
        my $before = '';
        my $after = '';
        
        if ($content =~ /(.*?)((?:# Utilities.*?)?class\s+TradeLogger:.*?)(\n(?:class\s+|async\s+def\s+|if\s+__name__|# =====|# -----).*)$/s) {
            $before = $1;
            $after = $3;
            $content = $before . $new_tradelogger . $after;
            $class_replaced = 1;
        } elsif ($content =~ /(.*?)((?:# Utilities.*?)?class\s+TradeLogger:.*)$/s) {
            # TradeLogger is at end of file
            $before = $1;
            $content = $before . $new_tradelogger;
            $class_replaced = 1;
        }
    }
    
    if ($class_replaced) {
        push @fixes_applied, "Replaced TradeLogger class";
    } else {
        print "  ⚠️  TradeLogger class not found - skipping replacement\n";
    }
    
    # Fix 5: Update TradeLogger initialization (only if not already has tf)
    if ($tf && $class_replaced) {
        # Only add tf if not already present
        my $init_fixes = 0;
        while ($content =~ /self\.logger\s*=\s*TradeLogger\(([^)]+)\)/g) {
            my $params = $1;
            if ($params !~ /tf=/) {
                $content =~ s/(self\.logger\s*=\s*TradeLogger\($params)(\))/$1, tf="$tf"$2/;
                $init_fixes++;
            }
        }
        push @fixes_applied, "Added tf to $init_fixes TradeLogger inits" if $init_fixes;
    }
    
    # Fix 6: Update references (only outside TradeLogger class)
    # These are safe global replacements
    my $ref_fixes = 0;
    
    # Fix method references
    $ref_fixes += ($content =~ s/self\.logger\.daily_pnl/self.logger.pnl_day/g);
    $ref_fixes += ($content =~ s/self\.logger\.consecutive_losses/self.logger.streak_loss/g);
    
    push @fixes_applied, "Updated $ref_fixes references" if $ref_fixes;
    
    # Write updated file
    open(my $out, '>', $file) or die "Cannot write $file: $!";
    print $out $content;
    close($out);
    
    if (@fixes_applied) {
        print "  ✅ Applied fixes:\n";
        foreach my $fix (@fixes_applied) {
            print "     • $fix\n";
        }
        $total_fixes++;
    } else {
        print "  ℹ️  No changes needed\n";
    }
    print "\n";
}

print "=" x 50 . "\n";
print "✅ Processed all files ($total_fixes files modified)\n\n";
print "Summary of fixes:\n";
print "  • datetime.utcnow() → datetime.now(timezone.utc)\n";
print "  • Added timezone imports where needed\n";
print "  • Fixed duplicate tf parameters\n";
print "  • Added missing utility functions (_safe_float, _quantize)\n";
print "  • Added max_daily_loss attribute\n";
print "  • Added backward compatibility wrappers for old methods\n";
print "  • Updated pnl_day and streak_loss references\n";
print "\nBackup files created with .backup_final extension\n";
print "\nNext steps:\n";
print "1. Run: bash vai.sh\n";
print "2. If you see file not found errors, check file names\n";
print "3. Delete backups when everything works: rm *.backup_final\n";