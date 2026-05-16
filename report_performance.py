"""
report_performance.py — CLI performance report from trades.db.

Usage:
    python report_performance.py
    ssh root@134.209.197.173 "cd /opt/btcbot && .venv/bin/python report_performance.py"
"""
from __future__ import annotations

from pathlib import Path
from dotenv import load_dotenv

_vps_env = Path("/opt/btcbot/.env")
load_dotenv(dotenv_path=_vps_env if _vps_env.exists() else Path(".env"))

import performance as perf  # noqa: E402

SEP = "═" * 58
DIV = "─" * 58


def _pnl_str(v: float) -> str:
    return f"{v:+.4f}"


n = perf.total_trades()
if n == 0:
    print(f"\n  No trades recorded yet. Run the bot first to generate data.\n")
    raise SystemExit(0)

print(f"\n{SEP}")
print(f"  Performance Report  ({n} total trades)")
print(SEP)

# ── General ───────────────────────────────────────────────────────────────────
print(f"\n  GENERAL")
print(f"  {DIV}")
print(f"  Total trades       : {n}")
print(f"  Total PnL          : {_pnl_str(perf.total_pnl())} USDT")
print(f"  Win rate           : {perf.win_rate():.1%}")
print(f"  Profit factor      : {perf.profit_factor():.2f}")
print(f"  Expectancy         : {_pnl_str(perf.expectancy())} USDT / trade")
print(f"  Average win        : {_pnl_str(perf.average_win())} USDT")
print(f"  Average loss       : {_pnl_str(perf.average_loss())} USDT")
print(f"  Max drawdown       : {perf.max_drawdown_pct():.2f}%")
print(f"  Avg trade duration : {perf.avg_duration_minutes():.0f} min")
print(f"  Sharpe-like ratio  : {perf.sharpe_like_ratio():.3f}")
print(f"  Max consec. wins   : {perf.consecutive_wins()}")
print(f"  Max consec. losses : {perf.consecutive_losses()}")

# ── By symbol ─────────────────────────────────────────────────────────────────
sym_data = perf.pnl_by_symbol()
if sym_data:
    print(f"\n  BY SYMBOL")
    print(f"  {DIV}")
    print(f"  {'Symbol':<10} {'Trades':>6}  {'Win%':>6}  {'PnL (USDT)':>12}  {'Avg PnL':>10}")
    print(f"  {DIV}")
    for sym, d in sym_data.items():
        base = sym.replace("USDT", "")
        print(
            f"  {base:<10} {d['trades']:>6}  {d['win_rate']:>6.1%}  "
            f"{d['total_pnl']:>+12.4f}  {d['avg_pnl']:>+10.4f}"
        )

# ── By strategy ───────────────────────────────────────────────────────────────
strat_data = perf.pnl_by_strategy()
if strat_data:
    print(f"\n  BY STRATEGY")
    print(f"  {DIV}")
    print(f"  {'Strategy':<14} {'Trades':>6}  {'Win%':>6}  {'PnL (USDT)':>12}  {'Avg PnL':>10}")
    print(f"  {DIV}")
    for strat, d in strat_data.items():
        print(
            f"  {strat:<14} {d['trades']:>6}  {d['win_rate']:>6.1%}  "
            f"{d['total_pnl']:>+12.4f}  {d['avg_pnl']:>+10.4f}"
        )

# ── Recent trades ─────────────────────────────────────────────────────────────
recent = perf.recent_trades(10)
if recent:
    print(f"\n  RECENT TRADES  (last {len(recent)})")
    print(f"  {DIV}")
    print(
        f"  {'Symbol':<10} {'Side':<4}  {'PnL':>10}  {'PnL%':>7}  "
        f"{'Reason':<20}  {'Dur(min)':>8}"
    )
    print(f"  {DIV}")
    for t in recent:
        base = t["symbol"].replace("USDT", "")
        print(
            f"  {base:<10} {t['side']:<4}  "
            f"{t['pnl_usdt']:>+10.4f}  {t['pnl_pct']:>+7.3f}%  "
            f"{t['close_reason']:<20}  {t['duration_minutes']:>8.0f}"
        )

# ── Weekly summary ────────────────────────────────────────────────────────────
weekly = perf.weekly_summary()
if weekly:
    print(f"\n  WEEKLY SUMMARY  (last {len(weekly)} weeks)")
    print(f"  {DIV}")
    print(f"  {'Week':<12} {'Trades':>6}  {'Wins':>5}  {'PnL (USDT)':>12}")
    print(f"  {DIV}")
    for w in weekly:
        print(
            f"  {w['week']:<12} {w['n']:>6}  {w['wins']:>5}  {w['pnl']:>+12.4f}"
        )

print(f"\n{SEP}\n")
