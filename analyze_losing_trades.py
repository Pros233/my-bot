#!/usr/bin/env python3
"""
analyze_losing_trades.py — Detect patterns in losing trades from trades.db.

Usage:
  python3 analyze_losing_trades.py
  python3 analyze_losing_trades.py --min-trades 5
  python3 analyze_losing_trades.py --output losing_report.txt

Outputs a structured report covering:
  - Losing trade count and total drawdown
  - Worst sessions, regimes, hours, weekdays
  - ATR/ADX patterns in losing trades vs winners
  - Score quality distribution
  - Consecutive loss clusters
  - Actionable filter recommendations
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path


DB_PATHS = [Path("/opt/btcbot/trades.db"), Path("trades.db")]


def _get_db() -> Path:
    for p in DB_PATHS:
        if p.exists():
            return p
    sys.exit("trades.db not found in /opt/btcbot/ or current directory.")


def _rows(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> list[dict]:
    conn.row_factory = sqlite3.Row
    return [dict(r) for r in conn.execute(sql, params).fetchall()]


def _hour_to_session(hour: int) -> str:
    if 0 <= hour < 8:    return "Asia"
    elif 8 <= hour < 13: return "London"
    elif 13 <= hour < 16: return "NY/London"
    elif 16 <= hour < 21: return "New York"
    else:                 return "Off-hours"


def _atr_bucket(atr_pct: float) -> str:
    if atr_pct < 0.3:    return "<0.3% low"
    elif atr_pct < 0.6:  return "0.3-0.6% medium"
    elif atr_pct < 1.0:  return "0.6-1.0% elevated"
    else:                return ">1.0% high"


def _adx_bucket(adx: float) -> str:
    if adx < 20:    return "<20 weak"
    elif adx < 30:  return "20-30 moderate"
    elif adx < 40:  return "30-40 strong"
    else:           return ">40 very strong"


def _score_bucket(score_pct: float) -> str:
    if score_pct < 40:    return "<40% poor"
    elif score_pct < 60:  return "40-60% weak"
    elif score_pct < 80:  return "60-80% good"
    else:                 return ">80% strong"


def _pct_change(a: float, b: float) -> str:
    if b == 0:
        return "N/A"
    return f"{(a - b) / abs(b) * 100:+.1f}%"


def _group(rows: list[dict], key_fn) -> dict[str, dict]:
    """Group by key, compute n, pnl, wins."""
    buckets: dict[str, dict] = {}
    for r in rows:
        k = key_fn(r)
        if k not in buckets:
            buckets[k] = {"n": 0, "pnl": 0.0, "wins": 0}
        buckets[k]["n"]   += 1
        buckets[k]["pnl"] += r.get("pnl_usdt", 0.0)
        if r.get("pnl_usdt", 0) > 0:
            buckets[k]["wins"] += 1
    return buckets


def analyze(min_trades: int = 2) -> str:
    db_path = _get_db()
    lines   = []
    sep     = "─" * 60

    with sqlite3.connect(str(db_path)) as conn:
        all_trades = _rows(conn,
            "SELECT * FROM trades ORDER BY id ASC")
        losers = [t for t in all_trades if t.get("pnl_usdt", 0) < 0]
        winners = [t for t in all_trades if t.get("pnl_usdt", 0) > 0]

    total  = len(all_trades)
    n_loss = len(losers)
    n_win  = len(winners)

    if total == 0:
        return "No trades recorded yet."

    total_loss_usdt = sum(t["pnl_usdt"] for t in losers)
    total_win_usdt  = sum(t["pnl_usdt"] for t in winners)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines += [
        "",
        "=" * 60,
        "  BTC Bot — Losing Trade Pattern Analysis",
        f"  Generated: {now}",
        "=" * 60,
        "",
        "── Overview ──────────────────────────────────────────────",
        f"  Total trades:   {total}",
        f"  Winners:        {n_win} ({n_win/total*100:.1f}%)",
        f"  Losers:         {n_loss} ({n_loss/total*100:.1f}%)",
        f"  Total loss:     ${total_loss_usdt:.4f} USDT",
        f"  Total profit:   +${total_win_usdt:.4f} USDT",
        f"  Profit factor:  {abs(total_win_usdt/total_loss_usdt):.2f}" if total_loss_usdt < 0 else "  Profit factor:  ∞ (no losses)",
        "",
    ]

    if not losers:
        lines.append("  No losing trades found — nothing to analyze.")
        return "\n".join(lines)

    # ── Session breakdown ─────────────────────────────────────────────────────
    def _session(t: dict) -> str:
        try:
            return _hour_to_session(int(t["opened_at_utc"][11:13]))
        except Exception:
            return "Unknown"

    loss_sessions = _group(losers, _session)
    win_sessions  = _group(winners, _session)

    lines += ["── Session Patterns ─────────────────────────────────────", ""]
    for sess, d in sorted(loss_sessions.items(), key=lambda x: x[1]["n"], reverse=True):
        w_d   = win_sessions.get(sess, {"n": 0})
        total_sess = d["n"] + w_d["n"]
        loss_rate  = d["n"] / total_sess * 100 if total_sess > 0 else 0
        lines.append(f"  {sess:<15} {d['n']:>3} losses / {total_sess:>3} trades "
                     f"({loss_rate:.0f}% loss rate)  avg=${d['pnl']/d['n']:.4f}")
    lines.append("")

    # ── Regime breakdown ──────────────────────────────────────────────────────
    loss_regimes = _group(losers, lambda t: t.get("regime") or "Unknown")
    win_regimes  = _group(winners, lambda t: t.get("regime") or "Unknown")

    lines += ["── Regime Patterns ──────────────────────────────────────", ""]
    for reg, d in sorted(loss_regimes.items(), key=lambda x: x[1]["n"], reverse=True):
        w_d   = win_regimes.get(reg, {"n": 0})
        total_reg = d["n"] + w_d["n"]
        loss_rate  = d["n"] / total_reg * 100 if total_reg > 0 else 0
        lines.append(f"  {reg:<30} {d['n']:>2} losses / {total_reg:>2} "
                     f"({loss_rate:.0f}%)  avg=${d['pnl']/d['n']:.4f}")
    lines.append("")

    # ── Hour breakdown ────────────────────────────────────────────────────────
    loss_hours = _group(losers, lambda t: f"{int(t['opened_at_utc'][11:13]):02d}:00")
    lines += ["── Worst Hours (UTC) ────────────────────────────────────", ""]
    for hour, d in sorted(loss_hours.items(), key=lambda x: x[1]["n"], reverse=True)[:6]:
        lines.append(f"  {hour}  {d['n']} losses  avg=${d['pnl']/d['n']:.4f}")
    lines.append("")

    # ── ATR analysis ──────────────────────────────────────────────────────────
    loss_atr = _group(losers, lambda t: _atr_bucket(float(t.get("atr_pct") or 0)))
    win_atr  = _group(winners, lambda t: _atr_bucket(float(t.get("atr_pct") or 0)))

    lines += ["── ATR Bucket Analysis ──────────────────────────────────", ""]
    for bkt, d in sorted(loss_atr.items(), key=lambda x: x[1]["n"], reverse=True):
        w_d   = win_atr.get(bkt, {"n": 0})
        total_bkt = d["n"] + w_d["n"]
        loss_rate  = d["n"] / total_bkt * 100 if total_bkt > 0 else 0
        lines.append(f"  {bkt:<20} {d['n']:>2} losses / {total_bkt:>2} ({loss_rate:.0f}%)")
    lines.append("")

    # ── ADX analysis ──────────────────────────────────────────────────────────
    loss_adx = _group(losers, lambda t: _adx_bucket(float(t.get("adx") or 0)))
    win_adx  = _group(winners, lambda t: _adx_bucket(float(t.get("adx") or 0)))

    lines += ["── ADX Bucket Analysis ──────────────────────────────────", ""]
    for bkt, d in sorted(loss_adx.items(), key=lambda x: x[1]["n"], reverse=True):
        w_d   = win_adx.get(bkt, {"n": 0})
        total_bkt = d["n"] + w_d["n"]
        loss_rate  = d["n"] / total_bkt * 100 if total_bkt > 0 else 0
        lines.append(f"  {bkt:<20} {d['n']:>2} losses / {total_bkt:>2} ({loss_rate:.0f}%)")
    lines.append("")

    # ── Score quality ─────────────────────────────────────────────────────────
    avg_score_losers  = (sum(float(t.get("score_pct") or 0) for t in losers) / n_loss) if n_loss else 0
    avg_score_winners = (sum(float(t.get("score_pct") or 0) for t in winners) / n_win) if n_win else 0

    lines += [
        "── Signal Score Quality ─────────────────────────────────", "",
        f"  Avg score — winners:  {avg_score_winners:.1f}%",
        f"  Avg score — losers:   {avg_score_losers:.1f}%",
        f"  Difference:           {_pct_change(avg_score_losers, avg_score_winners)}",
        "",
    ]

    # ── Consecutive loss clusters ──────────────────────────────────────────────
    max_streak = streak = 0
    streak_starts = []
    cur_start = None
    for t in all_trades:
        if t.get("pnl_usdt", 0) < 0:
            streak += 1
            if streak == 1:
                cur_start = t.get("opened_at_utc", "")
            if streak > max_streak:
                max_streak = streak
        else:
            if streak >= 2:
                streak_starts.append((cur_start, streak))
            streak = 0
            cur_start = None
    if streak >= 2:
        streak_starts.append((cur_start, streak))

    lines += [
        "── Consecutive Loss Clusters ────────────────────────────", "",
        f"  Max consecutive losses: {max_streak}",
    ]
    for start, length in sorted(streak_starts, key=lambda x: x[1], reverse=True)[:5]:
        lines.append(f"  {length} losses starting {start[:16]}")
    lines.append("")

    # ── Recommendations ────────────────────────────────────────────────────────
    lines += ["── Filter Recommendations ───────────────────────────────", ""]

    # Bad session?
    worst_sess = sorted(loss_sessions.items(), key=lambda x: x[1]["n"], reverse=True)
    if worst_sess:
        sess_name = worst_sess[0][0]
        sess_d    = worst_sess[0][1]
        w_d       = win_sessions.get(sess_name, {"n": 0})
        total_s   = sess_d["n"] + w_d["n"]
        if total_s >= min_trades and sess_d["n"] / total_s > 0.6:
            lines.append(f"  ⚠ Session '{sess_name}' has >60% loss rate — consider blocking or requiring A+ only")

    # Bad ATR?
    for bkt, d in loss_atr.items():
        w_d = win_atr.get(bkt, {"n": 0})
        total_bkt = d["n"] + w_d["n"]
        if total_bkt >= min_trades and d["n"] / total_bkt > 0.65:
            lines.append(f"  ⚠ ATR bucket '{bkt}' has >65% loss rate — consider candle extension / volatility filter")

    # Low score correlation?
    if avg_score_winners > 0 and (avg_score_winners - avg_score_losers) > 5:
        lines.append(f"  ✓ Score quality correlates with wins (winner avg={avg_score_winners:.0f}% vs loser={avg_score_losers:.0f}%)")
        lines.append("    → Raising MIN_SCORE threshold or requiring A grade could improve selectivity")

    if max_streak >= 3:
        lines.append(f"  ⚠ Max streak of {max_streak} consecutive losses — adaptive filter (ENABLE_ADAPTIVE_FILTERS=true) may help")

    lines += ["", "=" * 60, ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Analyze losing trade patterns")
    parser.add_argument("--min-trades", type=int, default=2,
                        help="Minimum trades in bucket to report (default 2)")
    parser.add_argument("--output", type=str, default=None,
                        help="Write report to file instead of stdout")
    args = parser.parse_args()

    report = analyze(min_trades=args.min_trades)

    if args.output:
        Path(args.output).write_text(report)
        print(f"Report written to {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
