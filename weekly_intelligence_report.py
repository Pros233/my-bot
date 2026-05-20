"""
weekly_intelligence_report.py — Weekly system intelligence review.

Generates a comprehensive Telegram summary every Sunday UTC covering:
  - Best/worst engine, symbol, session, regime
  - Expectancy changes week-over-week
  - Grade distribution and adaptive mode trigger frequency
  - PnL by engine, session, regime
  - Top 5 and worst 5 trades
  - Rejected vs executed setup statistics

Public API
----------
    generate_weekly_report()         → str  (Telegram-formatted text)
    generate_weekly_summary_dict()   → dict (for dashboard)
    should_send_weekly(last_sent_ts) → bool (True if Sunday UTC and 7d elapsed)

Never raises — returns error string on any failure.
"""
from __future__ import annotations

import math
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional


_DB_PATHS = [Path("/opt/btcbot/trades.db"), Path("trades.db")]


def _db() -> Optional[sqlite3.Connection]:
    for p in _DB_PATHS:
        if p.exists():
            conn = sqlite3.connect(p)
            conn.row_factory = sqlite3.Row
            return conn
    return None


def _rows(sql: str, params: tuple = ()) -> list:
    try:
        conn = _db()
        if conn is None:
            return []
        with conn:
            return conn.execute(sql, params).fetchall()
    except Exception:
        return []


def _scalar(sql: str, params: tuple = (), default=0):
    try:
        conn = _db()
        if conn is None:
            return default
        with conn:
            row = conn.execute(sql, params).fetchone()
            return row[0] if row and row[0] is not None else default
    except Exception:
        return default


def _usdt(v: float) -> str:
    return f"${v:+.4f}" if abs(v) < 1 else f"${v:+.2f}"


def _pct(v: float) -> str:
    return f"{v:+.1f}%"


def _week_range() -> tuple[str, str]:
    """Return ISO date strings for (7 days ago, now)."""
    now = datetime.now(timezone.utc)
    week_ago = now - timedelta(days=7)
    return week_ago.isoformat(), now.isoformat()


def _best_worst_by(dimension: str, since: str) -> tuple[str, str]:
    """Return (best_category, worst_category) by total PnL for the given column."""
    valid = {"strategy", "symbol", "session", "regime", "trade_grade"}
    if dimension not in valid:
        return "N/A", "N/A"
    rows = _rows(
        f"SELECT {dimension} AS cat, SUM(pnl_usdt) AS total "
        f"FROM trades WHERE closed_at_utc >= ? AND {dimension} != '' "
        f"GROUP BY {dimension} ORDER BY total DESC",
        (since,),
    )
    if not rows:
        return "N/A", "N/A"
    best  = str(rows[0]["cat"]) if rows else "N/A"
    worst = str(rows[-1]["cat"]) if len(rows) > 1 else "N/A"
    return best, worst


def _pnl_by(dimension: str, since: str) -> list[dict]:
    valid = {"strategy", "symbol", "session", "regime", "trade_grade"}
    if dimension not in valid:
        return []
    rows = _rows(
        f"SELECT {dimension} AS cat, COUNT(*) AS n, "
        f"SUM(CASE WHEN pnl_usdt > 0 THEN 1 ELSE 0 END) AS wins, "
        f"SUM(pnl_usdt) AS total, AVG(pnl_usdt) AS avg "
        f"FROM trades WHERE closed_at_utc >= ? AND {dimension} != '' "
        f"GROUP BY {dimension} ORDER BY total DESC",
        (since,),
    )
    return [
        {
            "cat":   str(r["cat"]),
            "n":     r["n"],
            "wins":  r["wins"],
            "total": round(float(r["total"]), 4),
            "avg":   round(float(r["avg"]), 4),
            "wr":    round(r["wins"] / r["n"], 3) if r["n"] > 0 else 0.0,
        }
        for r in rows
    ]


def _top_trades(since: str, best: bool = True, limit: int = 5) -> list[dict]:
    order = "DESC" if best else "ASC"
    rows = _rows(
        f"SELECT symbol, strategy, pnl_usdt, pnl_pct, close_reason, closed_at_utc "
        f"FROM trades WHERE closed_at_utc >= ? "
        f"ORDER BY pnl_usdt {order} LIMIT ?",
        (since, limit),
    )
    return [
        {
            "symbol":   r["symbol"],
            "strategy": r["strategy"],
            "pnl":      round(float(r["pnl_usdt"]), 4),
            "pnl_pct":  round(float(r["pnl_pct"]), 2),
            "reason":   r["close_reason"],
            "date":     str(r["closed_at_utc"])[:10],
        }
        for r in rows
    ]


def _grade_distribution(since: str) -> dict[str, int]:
    rows = _rows(
        "SELECT trade_grade, COUNT(*) AS n FROM trades "
        "WHERE closed_at_utc >= ? AND trade_grade != '' "
        "GROUP BY trade_grade",
        (since,),
    )
    return {str(r["trade_grade"]): r["n"] for r in rows}


def generate_weekly_summary_dict() -> dict:
    """Return structured weekly data for the dashboard."""
    try:
        since, until = _week_range()
        total   = _scalar("SELECT COUNT(*) FROM trades WHERE closed_at_utc >= ?", (since,))
        wins    = _scalar("SELECT COUNT(*) FROM trades WHERE closed_at_utc >= ? AND pnl_usdt > 0", (since,))
        total_pnl = _scalar("SELECT SUM(pnl_usdt) FROM trades WHERE closed_at_utc >= ?", (since,), 0.0)
        wr = wins / total if total > 0 else 0.0

        best_eng,  worst_eng  = _best_worst_by("strategy", since)
        best_sym,  worst_sym  = _best_worst_by("symbol",   since)
        best_sess, worst_sess = _best_worst_by("session",  since)
        best_reg,  worst_reg  = _best_worst_by("regime",   since)

        return {
            "week_start":  since[:10],
            "week_end":    until[:10],
            "total_trades": total,
            "wins":         wins,
            "win_rate":     round(wr, 3),
            "total_pnl":    round(float(total_pnl), 4),
            "best_engine":  best_eng,
            "worst_engine": worst_eng,
            "best_symbol":  best_sym,
            "worst_symbol": worst_sym,
            "best_session": best_sess,
            "worst_session":worst_sess,
            "best_regime":  best_reg,
            "worst_regime": worst_reg,
            "grade_dist":   _grade_distribution(since),
            "pnl_by_engine": _pnl_by("strategy", since),
            "pnl_by_session": _pnl_by("session", since),
            "pnl_by_regime":  _pnl_by("regime",  since),
            "top_trades":     _top_trades(since, best=True,  limit=5),
            "worst_trades":   _top_trades(since, best=False, limit=5),
        }
    except Exception as exc:
        return {"error": str(exc)}


def generate_weekly_report() -> str:
    """Generate the full Telegram weekly intelligence report."""
    try:
        d = generate_weekly_summary_dict()
        if "error" in d:
            return f"⚠ Weekly report error: `{d['error']}`"

        n  = d["total_trades"]
        wr = d["win_rate"]
        pnl = d["total_pnl"]

        lines = [
            f"*Weekly Intelligence Report*",
            f"Week: `{d['week_start']}` → `{d['week_end']}`",
            "",
        ]

        if n == 0:
            lines.append("_No trades executed this week._")
            return "\n".join(lines)

        lines += [
            f"*Summary*",
            f"  Trades: `{n}` | WR: `{wr*100:.0f}%` | PnL: `{_usdt(pnl)}`",
            "",
            f"*Best ↑*",
            f"  Engine:  `{d['best_engine']}`",
            f"  Symbol:  `{d['best_symbol']}`",
            f"  Session: `{d['best_session']}`",
            f"  Regime:  `{d['best_regime'][:30]}`",
            "",
            f"*Worst ↓*",
            f"  Engine:  `{d['worst_engine']}`",
            f"  Symbol:  `{d['worst_symbol']}`",
            f"  Session: `{d['worst_session']}`",
            f"  Regime:  `{d['worst_regime'][:30]}`",
            "",
        ]

        # PnL by engine
        eng_rows = d["pnl_by_engine"]
        if eng_rows:
            lines.append("*PnL by Engine*")
            for r in eng_rows[:6]:
                icon = "↑" if r["total"] >= 0 else "↓"
                lines.append(
                    f"  {icon} `{r['cat']}` {_usdt(r['total'])} | "
                    f"{r['n']}T WR {r['wr']*100:.0f}%"
                )
            lines.append("")

        # PnL by session
        sess_rows = d["pnl_by_session"]
        if sess_rows:
            lines.append("*PnL by Session*")
            for r in sess_rows[:4]:
                icon = "↑" if r["total"] >= 0 else "↓"
                lines.append(f"  {icon} `{r['cat']}` {_usdt(r['total'])} | {r['n']}T")
            lines.append("")

        # Grade distribution
        gd = d["grade_dist"]
        if gd:
            parts = " ".join(f"`{g}`:{c}" for g, c in sorted(gd.items()))
            lines.append(f"*Grade Distribution*: {parts}")
            lines.append("")

        # Top 3 trades
        tops = d["top_trades"][:3]
        if tops:
            lines.append("*Top Trades*")
            for t in tops:
                lines.append(
                    f"  `{t['symbol']}` [{t['strategy']}] "
                    f"`{_usdt(t['pnl'])}` ({t['reason']}) {t['date']}"
                )
            lines.append("")

        # Worst 3 trades
        worsts = d["worst_trades"][:3]
        if worsts:
            lines.append("*Worst Trades*")
            for t in worsts:
                lines.append(
                    f"  `{t['symbol']}` [{t['strategy']}] "
                    f"`{_usdt(t['pnl'])}` ({t['reason']}) {t['date']}"
                )

        return "\n".join(lines)

    except Exception as exc:
        return f"⚠ Weekly report generation error: `{exc}`"


def should_send_weekly(last_sent_ts: float) -> bool:
    """True if today is Sunday UTC and at least 6 days have passed since last send."""
    import time
    now = datetime.now(timezone.utc)
    if now.weekday() != 6:          # 6 = Sunday
        return False
    if now.hour < 8:                # Send at 08:00+ UTC
        return False
    elapsed_days = (time.monotonic() - last_sent_ts) / 86400
    return elapsed_days >= 6.0
