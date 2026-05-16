"""
report_trends.py — CLI analytics for trend_watchlist.db.

Sections:
  1. Recent signals (grade, MTF, score, momentum)
  2. Outcome summary by horizon
  3. Leaderboard — top continuation coins & worst trend traps
  4. Adaptive learning — grade, volume bucket, MTF combo, failing symbols

Usage:
    python report_trends.py                 # last 50 signals + all sections
    python report_trends.py --all           # all signals
    python report_trends.py --symbol BTCUSDT
    python report_trends.py --leaderboard   # leaderboard + adaptive only
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

DB_PATH = Path("trend_watchlist.db")
SEP = "═" * 72


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _g(row, key, default=None):
    """Safe row access that returns default if column missing."""
    try:
        return row[key]
    except (IndexError, KeyError):
        return default


# ── Section 1: signal table ───────────────────────────────────────────────────

def _print_signals(rows) -> None:
    hdr = (
        f"{'Detected (UTC)':<20} {'Symbol':<12} {'Grd':>4} {'Score':>5} "
        f"{'1h%':>7} {'4h%':>7} {'Vol×':>6}  {'15m':<8}{'1h':<8}{'4h':<8}"
    )
    print(f"\n{hdr}")
    print("─" * 72)
    for r in rows:
        grade = _g(r, "grade", "?")
        print(
            f"{str(_g(r, 'detected_at_utc', ''))[:19]:<20} "
            f"{_g(r, 'symbol', '?'):<12} "
            f"{grade:>4} "
            f"{(_g(r, 'score', 0) or 0):>5.0f} "
            f"{(_g(r, 'price_change_1h', 0) or 0):>+7.2f} "
            f"{(_g(r, 'price_change_4h', 0) or 0):>+7.2f} "
            f"{(_g(r, 'volume_spike', 0) or 0):>6.2f}  "
            f"{str(_g(r, 'mtf_15m', '?')):<8}"
            f"{str(_g(r, 'mtf_1h', '?')):<8}"
            f"{str(_g(r, 'mtf_4h', '?')):<8}"
        )


# ── Section 2: outcome summary ────────────────────────────────────────────────

def _outcome_summary(conn: sqlite3.Connection) -> None:
    rows = conn.execute("""
        SELECT o.horizon,
               COUNT(*)  AS n,
               AVG(o.return_pct) AS avg_ret,
               SUM(CASE WHEN o.return_pct > 0 THEN 1 ELSE 0 END) AS wins
        FROM trend_outcomes o
        GROUP BY o.horizon
        ORDER BY CASE o.horizon WHEN '1h' THEN 1 WHEN '4h' THEN 2 ELSE 3 END
    """).fetchall()

    if not rows:
        print("\n  No outcome data yet (signals need 1–24h to mature).")
        return

    print(f"\n{'Horizon':<8} {'Signals':>7} {'Avg Return':>11} {'Win Rate':>9}")
    print("─" * 40)
    for r in rows:
        wr = r["wins"] / r["n"] * 100 if r["n"] > 0 else 0
        print(
            f"{r['horizon']:<8} {r['n']:>7} "
            f"{r['avg_ret']:>+10.2f}%  {wr:>8.1f}%"
        )


# ── Section 3: leaderboard ────────────────────────────────────────────────────

def _leaderboard(conn: sqlite3.Connection) -> None:

    # Top continuation coins
    top = conn.execute("""
        SELECT s.symbol,
               COUNT(*)   AS signals,
               AVG(s.continuation_score) AS avg_cont,
               AVG(s.max_move_pct)       AS avg_max,
               AVG(CASE WHEN o.horizon='4h'  THEN o.return_pct END) AS avg_4h,
               AVG(CASE WHEN o.horizon='24h' THEN o.return_pct END) AS avg_24h
        FROM trend_signals s
        LEFT JOIN trend_outcomes o ON o.signal_id = s.id
        WHERE s.metrics_computed = 1
        GROUP BY s.symbol
        HAVING COUNT(*) >= 2
        ORDER BY avg_cont DESC, avg_4h DESC
        LIMIT 10
    """).fetchall()

    print(
        f"\n{'Symbol':<14} {'Sigs':>5} {'Avg Cont':>9} "
        f"{'Max Move':>9} {'Avg 4h':>8} {'Avg 24h':>8}"
    )
    print("─" * 60)
    if top:
        for r in top:
            print(
                f"{r['symbol']:<14} {r['signals']:>5} "
                f"{(r['avg_cont'] or 0):>9.2f} "
                f"{(r['avg_max'] or 0):>+9.2f}% "
                f"{(r['avg_4h'] or 0):>+8.2f}% "
                f"{(r['avg_24h'] or 0):>+8.2f}%"
            )
    else:
        print("  Insufficient data (need ≥ 2 completed signals per symbol).")

    # Worst trend traps
    worst = conn.execute("""
        SELECT s.symbol,
               COUNT(*) AS signals,
               AVG(s.reversal_score) AS avg_rev,
               SUM(CASE WHEN s.reversal_score >= 1.0 THEN 1 ELSE 0 END) AS full_reversals,
               AVG(CASE WHEN o.horizon='4h' THEN o.return_pct END) AS avg_4h
        FROM trend_signals s
        LEFT JOIN trend_outcomes o ON o.signal_id = s.id
        WHERE s.metrics_computed = 1
        GROUP BY s.symbol
        HAVING COUNT(*) >= 2
        ORDER BY avg_rev DESC
        LIMIT 5
    """).fetchall()

    print(f"\n  Worst Trend Traps (highest reversal rate)")
    print(f"{'Symbol':<14} {'Sigs':>5} {'Avg Rev':>8} {'Reversals':>10} {'Avg 4h':>8}")
    print("─" * 52)
    if worst:
        for r in worst:
            print(
                f"{r['symbol']:<14} {r['signals']:>5} "
                f"{(r['avg_rev'] or 0):>8.2f} "
                f"{r['full_reversals']:>10} "
                f"{(r['avg_4h'] or 0):>+8.2f}%"
            )
    else:
        print("  Insufficient data.")


# ── Section 4: adaptive learning ─────────────────────────────────────────────

def _adaptive_learning(conn: sqlite3.Connection) -> None:

    # Grade performance
    grade_rows = conn.execute("""
        SELECT s.grade,
               COUNT(*) AS signals,
               AVG(s.continuation_score) AS avg_cont,
               AVG(s.reversal_score)     AS avg_rev,
               AVG(CASE WHEN o.horizon='4h'  THEN o.return_pct END) AS avg_4h,
               AVG(CASE WHEN o.horizon='24h' THEN o.return_pct END) AS avg_24h
        FROM trend_signals s
        LEFT JOIN trend_outcomes o ON o.signal_id = s.id
        WHERE s.metrics_computed = 1
        GROUP BY s.grade
        ORDER BY CASE s.grade
            WHEN 'A+' THEN 0 WHEN 'A' THEN 1
            WHEN 'B'  THEN 2 WHEN 'C' THEN 3 ELSE 4 END
    """).fetchall()

    print(f"\n  Grade Performance")
    print(
        f"{'Grade':<6} {'Sigs':>5} {'Avg Cont':>9} "
        f"{'Avg Rev':>8} {'Avg 4h':>8} {'Avg 24h':>8}"
    )
    print("─" * 52)
    if grade_rows:
        for r in grade_rows:
            print(
                f"{r['grade']:<6} {r['signals']:>5} "
                f"{(r['avg_cont'] or 0):>9.2f} "
                f"{(r['avg_rev'] or 0):>8.2f} "
                f"{(r['avg_4h'] or 0):>+8.2f}% "
                f"{(r['avg_24h'] or 0):>+8.2f}%"
            )
    else:
        print("  No completed signals yet.")

    # Volume spike bucket performance
    vol_rows = conn.execute("""
        SELECT
            CASE
                WHEN volume_spike < 2.5 THEN '1.8–2.5×'
                WHEN volume_spike < 4.0 THEN '2.5–4.0×'
                ELSE '4.0×+'
            END AS bucket,
            COUNT(*) AS signals,
            AVG(continuation_score) AS avg_cont,
            AVG(reversal_score)     AS avg_rev
        FROM trend_signals
        WHERE metrics_computed = 1
        GROUP BY bucket
        ORDER BY bucket
    """).fetchall()

    print(f"\n  Volume Spike Bucket Performance")
    print(f"{'Bucket':<10} {'Sigs':>5} {'Avg Cont':>9} {'Avg Rev':>8}")
    print("─" * 36)
    if vol_rows:
        for r in vol_rows:
            print(
                f"{r['bucket']:<10} {r['signals']:>5} "
                f"{(r['avg_cont'] or 0):>9.2f} "
                f"{(r['avg_rev'] or 0):>8.2f}"
            )
    else:
        print("  No completed signals yet.")

    # MTF combination effectiveness
    mtf_rows = conn.execute("""
        SELECT mtf_15m || '/' || mtf_1h || '/' || mtf_4h AS combo,
               COUNT(*) AS signals,
               AVG(continuation_score) AS avg_cont,
               AVG(reversal_score)     AS avg_rev
        FROM trend_signals
        WHERE metrics_computed = 1
        GROUP BY combo
        HAVING COUNT(*) >= 2
        ORDER BY avg_cont DESC
        LIMIT 8
    """).fetchall()

    print(f"\n  MTF Combination Effectiveness  (15m / 1h / 4h)")
    print(f"{'Combo':<30} {'Sigs':>5} {'Avg Cont':>9} {'Avg Rev':>8}")
    print("─" * 56)
    if mtf_rows:
        for r in mtf_rows:
            print(
                f"{r['combo']:<30} {r['signals']:>5} "
                f"{(r['avg_cont'] or 0):>9.2f} "
                f"{(r['avg_rev'] or 0):>8.2f}"
            )
    else:
        print("  Insufficient data (need ≥ 2 signals per combo).")

    # Consistently failing symbols
    fail_rows = conn.execute("""
        SELECT symbol, COUNT(*) AS signals,
               AVG(reversal_score)     AS avg_rev,
               AVG(continuation_score) AS avg_cont
        FROM trend_signals
        WHERE metrics_computed = 1
        GROUP BY symbol
        HAVING COUNT(*) >= 3 AND AVG(reversal_score) >= 0.5
        ORDER BY avg_rev DESC
        LIMIT 5
    """).fetchall()

    print(f"\n  Consistently Failing Symbols  (≥ 3 signals, avg reversal ≥ 0.5)")
    print(f"{'Symbol':<14} {'Sigs':>5} {'Avg Rev':>8} {'Avg Cont':>9}")
    print("─" * 40)
    if fail_rows:
        for r in fail_rows:
            print(
                f"{r['symbol']:<14} {r['signals']:>5} "
                f"{r['avg_rev']:>8.2f} {r['avg_cont']:>9.2f}"
            )
    else:
        print("  No consistently failing symbols detected.")


# ── Section 5: frequency table ────────────────────────────────────────────────

def _top_symbols(conn: sqlite3.Connection) -> None:
    rows = conn.execute("""
        SELECT symbol,
               COUNT(*) AS appearances,
               AVG(score) AS avg_score,
               MAX(grade) AS best_grade,
               SUM(CASE WHEN alert_sent THEN 1 ELSE 0 END) AS alerts_sent
        FROM trend_signals
        GROUP BY symbol
        ORDER BY appearances DESC
        LIMIT 10
    """).fetchall()

    if not rows:
        return

    print(f"\n{'Symbol':<14} {'Appearances':>12} {'Avg Score':>10} {'Best':>5} {'Alerts':>7}")
    print("─" * 52)
    for r in rows:
        print(
            f"{r['symbol']:<14} {r['appearances']:>12} "
            f"{r['avg_score']:>10.1f} {r['best_grade']:>5} "
            f"{r['alerts_sent']:>7}"
        )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Trend scanner analytics")
    parser.add_argument("--all",         action="store_true", help="Show all signals")
    parser.add_argument("--symbol",      help="Filter by symbol, e.g. BTCUSDT")
    parser.add_argument("--leaderboard", action="store_true",
                        help="Show leaderboard + adaptive learning only")
    args = parser.parse_args()

    if not DB_PATH.exists():
        print(f"\n  {DB_PATH} not found — run bot with ENABLE_TREND_SCANNER=true first.")
        return

    conn = _conn()

    if args.leaderboard:
        print(f"\n{SEP}")
        print("  Trend Leaderboard — Top Continuation Coins")
        print(SEP)
        _leaderboard(conn)
        print(f"\n{SEP}")
        print("  Adaptive Learning Metrics")
        print(SEP)
        _adaptive_learning(conn)
        print(f"\n{SEP}\n")
        conn.close()
        return

    # ── Signals ───────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  Trend Scanner Analytics")
    print(SEP)

    query  = "SELECT * FROM trend_signals"
    params: list = []
    if args.symbol:
        query += " WHERE symbol = ?"
        params.append(args.symbol.upper())
    query += " ORDER BY detected_at_utc DESC"
    if not args.all:
        query += " LIMIT 50"

    rows  = conn.execute(query, params).fetchall()
    total = conn.execute("SELECT COUNT(*) FROM trend_signals").fetchone()[0]

    grade_counts = {
        r["grade"]: r["cnt"]
        for r in conn.execute(
            "SELECT grade, COUNT(*) AS cnt FROM trend_signals GROUP BY grade"
        ).fetchall()
    }
    gc_str = "  ".join(
        f"{g}:{grade_counts.get(g, 0)}" for g in ("A+", "A", "B", "C")
    )
    print(f"\n  Signals shown: {len(rows)} / {total} total  |  Grades: {gc_str}")
    _print_signals(rows)

    # ── Outcome summary ───────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  Outcome Summary  (actual returns after detection)")
    print(SEP)
    _outcome_summary(conn)

    # ── Leaderboard ───────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  Leaderboard — Top Continuation Coins")
    print(SEP)
    _leaderboard(conn)

    # ── Adaptive learning ─────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  Adaptive Learning Metrics")
    print(SEP)
    _adaptive_learning(conn)

    # ── Frequency ─────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  Most Frequently Trending Symbols")
    print(SEP)
    _top_symbols(conn)

    print(f"\n{SEP}\n")
    conn.close()


if __name__ == "__main__":
    main()
