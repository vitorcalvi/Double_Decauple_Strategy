#!/usr/bin/env python3
"""
Fix syntax errors and consistency in trading bot files.

Fixes:
1) Bad place_order args like: price=str(limit_price, timeInForce="PostOnly")
   → price=str(limit_price)
2) Duplicate timeInForce params inside place_order(...) (keeps the LAST one)
3) Change PostOnly → IOC inside close_position() calls (reliable closes)
4) Symbol mismatch between filename and code (e.g., file ..._LTCUSDT.py sets BNBUSDT)
5) Log filename standardization → logs/<filename-without-py>.log
"""

import os
import re
import glob
from pathlib import Path

# ---------- Helpers ----------

def fix_place_order_price_arg(content: str) -> str:
    """
    Fix pattern:
      price=str(<expr>, timeInForce="Something")
    to:
      price=str(<expr>)
    """
    pattern = r'price=str\(\s*([^,\)]+)\s*,\s*timeInForce="[^"]+"\s*\)'
    return re.sub(pattern, r'price=str(\1)', content)


def _dedupe_timeinforce_in_call(call_src: str) -> str:
    """
    Given the full text of a place_order(...) call, keep only the LAST timeInForce=...
    """
    # Split on commas that are not inside parentheses (simple-ish split)
    # Fallback approach: scan tokens and track last timeInForce.
    parts = []
    depth = 0
    token = []
    for ch in call_src:
        if ch == '(':
            depth += 1
            token.append(ch)
        elif ch == ')':
            depth -= 1
            token.append(ch)
        elif ch == ',' and depth == 1:  # top-level args inside this call
            parts.append(''.join(token))
            token = []
        else:
            token.append(ch)
    if token:
        parts.append(''.join(token))

    # parts now contain: ['place_order(', 'arg', 'arg', ..., ')'] in a rough sense.
    # Clean arg list between first '(' and last ')'
    # Extract prefix "place_order(" and suffix ")"
    if not parts:
        return call_src

    # Reconstruct argument string in a simpler way:
    m = re.search(r'place_order\s*\((.*)\)\s*$', call_src, flags=re.DOTALL)
    if not m:
        return call_src
    args_src = m.group(1)

    # Split args_src by commas at depth 0
    args = []
    depth = 0
    buf = []
    for ch in args_src:
        if ch == '(':
            depth += 1
            buf.append(ch)
        elif ch == ')':
            depth -= 1
            buf.append(ch)
        elif ch == ',' and depth == 0:
            args.append(''.join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    if buf:
        args.append(''.join(buf).strip())

    # Keep only last timeInForce
    last_idx = None
    for i, a in enumerate(args):
        if re.search(r'\btimeInForce\s*=', a):
            last_idx = i
    if last_idx is None:
        return call_src  # nothing to do

    new_args = []
    for i, a in enumerate(args):
        if re.search(r'\btimeInForce\s*=', a) and i != last_idx:
            continue
        new_args.append(a)

    return f"place_order({', '.join(new_args)})"


def fix_duplicate_timeinforce(content: str) -> str:
    """
    For any place_order(...) call (multi-line safe), keep only the LAST timeInForce.
    """
    def repl(m):
        call_src = m.group(0)
        return _dedupe_timeinforce_in_call(call_src)

    pattern = r'place_order\s*\((?:[^)(]+|\((?:[^)(]+|\([^)(]*\))*\))*\)'
    return re.sub(pattern, repl, content, flags=re.DOTALL)


def fix_postonly_in_close(content: str) -> str:
    """
    Inside def close_position...: replace PostOnly with IOC for place_order(...timeInForce=...)
    and annotate.
    """
    lines = content.splitlines()
    out = []
    in_close = False
    base_indent = None

    for i, line in enumerate(lines):
        stripped = line.lstrip()
        indent = len(line) - len(stripped)

        if stripped.startswith("def close_position"):
            in_close = True
            base_indent = indent

        elif in_close and stripped.startswith("def ") and indent <= base_indent:
            in_close = False

        if in_close and 'place_order' in line and 'timeInForce="PostOnly"' in line:
            line = line.replace('timeInForce="PostOnly"', 'timeInForce="IOC"') + '  # changed from PostOnly to IOC for reliable closes'

        out.append(line)

    return "\n".join(out)


def expected_symbol_from_filename(filepath: str) -> str | None:
    """
    Extract <SYMBOL>USDT from filename like: 3_FEES_EMAMACDRSI_LTCUSDT.py
    """
    name = os.path.basename(filepath)
    m = re.search(r'([A-Z]+USDT)\.py$', name)
    return m.group(1) if m else None


def fix_symbol_mismatch(filepath: str, content: str) -> str:
    """
    If filename contains XUSDT, ensure code uses that symbol in common places.
    Only touches obvious constant usages (string literals assigned to symbol/self.symbol).
    """
    expected = expected_symbol_from_filename(filepath)
    if not expected:
        return content

    # Replace in assignments like:
    #   self.symbol = "BNBUSDT"  or  symbol = 'BNBUSDT'
    def repl_assign(m):
        return f'{m.group(1)}{expected}{m.group(3)}'

    content = re.sub(
        r'(\bself\.symbol\s*=\s*["\'])([A-Z]+USDT)(["\'])',
        repl_assign,
        content
    )
    content = re.sub(
        r'(\bsymbol\s*=\s*["\'])([A-Z]+USDT)(["\'])',
        repl_assign,
        content
    )

    # Also replace obvious API init params like symbol="BNBUSDT"
    def repl_kw(m):
        return f'{m.group(1)}{expected}{m.group(3)}'

    content = re.sub(
        r'(\bsymbol\s*=\s*["\'])([A-Z]+USDT)(["\'])',
        repl_kw,
        content
    )

    return content


def fix_log_filename(filepath: str, content: str) -> str:
    """
    Force log filename to be logs/<base>.log where base is the python filename without .py
    """
    base = Path(filepath).stem
    patterns = [
        r'self\.log_file\s*=\s*f?["\']logs/[^"\']+\.log["\']',
    ]
    replacement = f'self.log_file = f"logs/{base}.log"'
    for pat in patterns:
        content, n = re.subn(pat, replacement, content)
        if n:
            # only need to hit once; most bots set this once
            break
    return content


def process_bot_file(filepath: str) -> None:
    print(f"\nProcessing {filepath}...")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            original = f.read()

        content = original
        content = fix_place_order_price_arg(content)
        content = fix_duplicate_timeinforce(content)
        content = fix_postonly_in_close(content)
        content = fix_symbol_mismatch(filepath, content)
        content = fix_log_filename(filepath, content)

        if content != original:
            backup = filepath + ".backup"
            with open(backup, 'w', encoding='utf-8') as b:
                b.write(original)
            with open(filepath, 'w', encoding='utf-8') as w:
                w.write(content)
            print(f"  Created backup: {backup}")
            print(f"  ✅ Fixed and saved: {filepath}")
        else:
            print(f"  ℹ️ No changes needed: {filepath}")

    except FileNotFoundError:
        print(f"  ❌ File not found: {filepath}")
    except Exception as e:
        print(f"  ❌ Error processing {filepath}: {e}")


def main():
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
        "10_FEES_RMI_SUPERTREND_ADAUSDT.py",
    ]

    print("🔧 Starting bot fixes...")
    print("=" * 50)

    for fname in bot_files:
        if os.path.exists(fname):
            process_bot_file(fname)
            continue
        matches = glob.glob(f"**/{fname}", recursive=True)
        if matches:
            process_bot_file(matches[0])
        else:
            print(f"\n⚠️ Skipping {fname} - not found")

    print("\n" + "=" * 50)
    print("✅ Fix script completed!")
    print("\nNote: Backups saved with .backup extension. Re-run tests for each bot.")


if __name__ == "__main__":
    main()
