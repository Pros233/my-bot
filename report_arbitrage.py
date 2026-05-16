#!/usr/bin/env python3
"""
report_arbitrage.py — Print arbitrage signal stats from arbitrage_watchlist.db.

Usage:
    python report_arbitrage.py
    python report_arbitrage.py --days 7
"""
from __future__ import annotations

import argparse
import sqlite3
from datetime import datetime, timezone, timedelta

_DB_PATH = "arbitrage_watchlist.db"


def _connect():
    try:
        return sqlite3.connect(_DB_PATH)
    except Exception as exc:
        print(f"[error] Cannot open {_DB_PATH}: {exc}")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Arbitrage signal report")
    parser.add_argument(
        "--days", type=int, default=30,
        help="Number of days of history to include (default: 30)"
    )
    args = parser.parse_args()

    since = (datetime.now(timezone.utc) - timedelta(days=args.days)).isoformat()

    try:
        conn = _connect()
    except Exception:
        return

    with conn:
        # ── Table existence check ─────────────────────────────────────────────
        exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='arbitrage_signals'"
        ).fetchone()
        if not exists:
            print("[report_arbitrage] No data yet — arbitrage_watchlist.db has no signals table.")
            return

        # ── Totals ────────────────────────────────────────────────────────────
        row = conn.execute(
            "SELECT COUNT(*), AVG(net_profit_pct), AVG(gross_profit_pct) "
            "FROM arbitrage_signals WHERE created_at >= ?", (since,)
        ).fetchone()
        total, avg_net, avg_gross = row
        avg_net   = avg_net   or 0.0
        avg_gross = avg_gross or 0.0

        alerted = conn.execute(
            "SELECT COUNT(*) FROM arbitrage_signals "
            "WHERE alert_sent=1 AND created_at >= ?", (since,)
        ).fetchone()[0]

        # ── By type ───────────────────────────────────────────────────────────
        by_type = conn.execute(
            "SELECT arb_type, COUNT(*) FROM arbitrage_signals "
            "WHERE created_at >= ? GROUP BY arb_type", (since,)
        ).fetchall()

        # ── Best route ────────────────────────────────────────────────────────
        best = conn.execute(
            "SELECT route, arb_type, net_profit_pct, gross_profit_pct, "
            "liquidity_score, detected_at_utc "
            "FROM arbitrage_signals WHERE created_at >= ? "
            "ORDER BY net_profit_pct DESC LIMIT 1", (since,)
        ).fetchone()

        # ── Worst (lowest net, but still saved) ───────────────────────────────
        worst = conn.execute(
            "SELECT route, arb_type, net_profit_pct, gross_profit_pct, "
            "liquidity_score, detected_at_utc "
            "FROM arbitrage_signals WHERE created_at >= ? "
            "ORDER BY net_profit_pct ASC LIMIT 1", (since,)
        ).fetchone()

        # ── Ephemeral: routes that appeared only once in the window (alerted) ─
        ephemeral = conn.execute(
            "SELECT route, COUNT(*) as cnt "
            "FROM arbitrage_signals "
            "WHERE alert_sent=1 AND created_at >= ? "
            "GROUP BY route HAVING cnt=1 "
            "ORDER BY cnt", (since,)
        ).fetchall()

        # ── Recent 20 signals ─────────────────────────────────────────────────
        recent = conn.execute(
            "SELECT detected_at_utc, arb_type, route, gross_profit_pct, "
            "net_profit_pct, liquidity_score, alert_sent "
            "FROM arbitrage_signals "
            "ORDER BY id DESC LIMIT 20"
        ).fetchall()

    # ── Print ─────────────────────────────────────────────────────────────────
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"\n{'='*64}")
    print(f"  Arbitrage Signal Report — last {args.days} day(s)  [{now_str}]")
    print(f"{'='*64}")
    print(f"  Total signals saved    : {total}")
    print(f"  Alerts sent (>=thresh) : {alerted}")
    print(f"  Avg gross profit       : {avg_gross:+.3f}%")
    print(f"  Avg net profit         : {avg_net:+.3f}%")

    if by_type:
        print(f"\n  By type:")
        for arb_type, cnt in by_type:
            print(f"    {arb_type:<20} {cnt:>6} signals")

    if best:
        print(f"\n  Best route:")
        print(f"    [{best[1]}] {best[0]}")
        print(f"    Net: {best[2]:+.3f}%  Gross: {best[3]:+.3f}%  "
              f"Liq: {best[4]}  At: {best[5][:16]}")

    if worst:
        print(f"\n  Worst route (lowest net saved):")
        print(f"    [{worst[1]}] {worst[0]}")
        print(f"    Net: {worst[2]:+.3f}%  Gross: {worst[3]:+.3f}%  "
              f"Liq: {worst[4]}  At: {worst[5][:16]}")

    if ephemeral:
        print(f"\n  Ephemeral (alerted once only — likely flash):")
        for route, cnt in ephemeral[:10]:
            print(f"    {route}")
    else:
        print(f"\n  Ephemeral signals: none (all alerted routes recurred)")

    print(f"\n  Recent 20 signals:")
    print(f"  {'Time':<20} {'Type':<14} {'Gross':>7} {'Net':>7} {'Liq':<5} {'Alert':<6} Route")
    print(f"  {'-'*90}")
    for ts, arb_type, route, gross, net, liq, sent in recent:
        ts_short = ts[:16] if ts else "—"
        alert_flag = "YES" if sent else "—"
        print(
            f"  {ts_short:<20} {arb_type:<14} {gross:>+6.2f}% {net:>+6.2f}% "
            f"{liq:<5} {alert_flag:<6} {route}"
        )

    print(f"\n{'='*64}\n")


if __name__ == "__main__":
    main()
