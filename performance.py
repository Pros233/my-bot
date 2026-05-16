"""
performance.py — Trade analytics from trades.db.

All functions return 0 / empty on a missing or empty database.
No function raises — all DB errors are swallowed and return defaults.
"""
from __future__ import annotations

import math
import sqlite3
from pathlib import Path
from typing import Optional

from trade_journal import DB_PATH


# ── DB helpers ────────────────────────────────────────────────────────────────

def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _rows(sql: str, params: tuple = ()) -> list[sqlite3.Row]:
    try:
        with _conn() as conn:
            return conn.execute(sql, params).fetchall()
    except Exception:
        return []


def _scalar(sql: str, params: tuple = (), default=0):
    try:
        with _conn() as conn:
            row = conn.execute(sql, params).fetchone()
            return row[0] if row and row[0] is not None else default
    except Exception:
        return default


# ── Core metrics ──────────────────────────────────────────────────────────────

def total_trades() -> int:
    return _scalar("SELECT COUNT(*) FROM trades")


def win_rate() -> float:
    n = total_trades()
    if n == 0:
        return 0.0
    wins = _scalar("SELECT COUNT(*) FROM trades WHERE pnl_usdt > 0")
    return wins / n


def total_pnl() -> float:
    return _scalar("SELECT SUM(pnl_usdt) FROM trades", default=0.0)


def average_win() -> float:
    return _scalar(
        "SELECT AVG(pnl_usdt) FROM trades WHERE pnl_usdt > 0", default=0.0
    )


def average_loss() -> float:
    return _scalar(
        "SELECT AVG(pnl_usdt) FROM trades WHERE pnl_usdt < 0", default=0.0
    )


def profit_factor() -> float:
    gross_win = _scalar(
        "SELECT SUM(pnl_usdt) FROM trades WHERE pnl_usdt > 0", default=0.0
    )
    gross_loss = abs(
        _scalar("SELECT SUM(pnl_usdt) FROM trades WHERE pnl_usdt < 0", default=0.0)
    )
    if gross_loss == 0:
        return float("inf") if gross_win > 0 else 0.0
    return gross_win / gross_loss


def expectancy() -> float:
    """Average PnL per trade in USDT."""
    return _scalar("SELECT AVG(pnl_usdt) FROM trades", default=0.0)


def avg_duration_minutes() -> float:
    return _scalar("SELECT AVG(duration_minutes) FROM trades", default=0.0)


def max_drawdown_pct() -> float:
    """Peak-to-trough drawdown on cumulative PnL equity curve (%)."""
    rows = _rows("SELECT pnl_usdt FROM trades ORDER BY id ASC")
    if not rows:
        return 0.0
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for row in rows:
        equity += row[0]
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    return max_dd * 100.0


def sharpe_like_ratio() -> float:
    """Mean trade PnL / std dev of trade PnL (trade-level Sharpe proxy)."""
    rows = _rows("SELECT pnl_usdt FROM trades")
    if len(rows) < 2:
        return 0.0
    values = [r[0] for r in rows]
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    std = math.sqrt(variance)
    return mean / std if std > 0 else 0.0


# ── Streak metrics ────────────────────────────────────────────────────────────

def consecutive_wins() -> int:
    """Longest streak of back-to-back winning trades."""
    rows = _rows("SELECT pnl_usdt FROM trades ORDER BY id ASC")
    max_streak = streak = 0
    for row in rows:
        if row[0] > 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    return max_streak


def consecutive_losses() -> int:
    """Longest streak of back-to-back losing trades."""
    rows = _rows("SELECT pnl_usdt FROM trades ORDER BY id ASC")
    max_streak = streak = 0
    for row in rows:
        if row[0] < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    return max_streak


# ── Breakdown metrics ─────────────────────────────────────────────────────────

def pnl_by_symbol() -> dict[str, dict]:
    rows = _rows(
        "SELECT symbol, COUNT(*) AS n, "
        "SUM(CASE WHEN pnl_usdt > 0 THEN 1 ELSE 0 END) AS wins, "
        "SUM(pnl_usdt) AS total_pnl, AVG(pnl_usdt) AS avg_pnl "
        "FROM trades GROUP BY symbol ORDER BY total_pnl DESC"
    )
    result = {}
    for row in rows:
        n = row["n"]
        result[row["symbol"]] = {
            "trades": n,
            "wins": row["wins"],
            "win_rate": row["wins"] / n if n > 0 else 0.0,
            "total_pnl": row["total_pnl"] or 0.0,
            "avg_pnl": row["avg_pnl"] or 0.0,
        }
    return result


def pnl_by_strategy() -> dict[str, dict]:
    rows = _rows(
        "SELECT strategy, COUNT(*) AS n, "
        "SUM(CASE WHEN pnl_usdt > 0 THEN 1 ELSE 0 END) AS wins, "
        "SUM(pnl_usdt) AS total_pnl, AVG(pnl_usdt) AS avg_pnl "
        "FROM trades GROUP BY strategy ORDER BY total_pnl DESC"
    )
    result = {}
    for row in rows:
        n = row["n"]
        result[row["strategy"]] = {
            "trades": n,
            "wins": row["wins"],
            "win_rate": row["wins"] / n if n > 0 else 0.0,
            "total_pnl": row["total_pnl"] or 0.0,
            "avg_pnl": row["avg_pnl"] or 0.0,
        }
    return result


def best_symbol() -> Optional[str]:
    rows = _rows(
        "SELECT symbol, SUM(pnl_usdt) AS total_pnl FROM trades "
        "GROUP BY symbol ORDER BY total_pnl DESC LIMIT 1"
    )
    return rows[0]["symbol"] if rows else None


def worst_symbol() -> Optional[str]:
    rows = _rows(
        "SELECT symbol, SUM(pnl_usdt) AS total_pnl FROM trades "
        "GROUP BY symbol ORDER BY total_pnl ASC LIMIT 1"
    )
    return rows[0]["symbol"] if rows else None


# ── Time-bucketed summaries ───────────────────────────────────────────────────

def daily_summary() -> list[dict]:
    rows = _rows(
        "SELECT DATE(closed_at_utc) AS day, COUNT(*) AS n, "
        "SUM(CASE WHEN pnl_usdt > 0 THEN 1 ELSE 0 END) AS wins, "
        "SUM(pnl_usdt) AS pnl "
        "FROM trades GROUP BY day ORDER BY day DESC LIMIT 30"
    )
    return [dict(row) for row in rows]


def weekly_summary() -> list[dict]:
    rows = _rows(
        "SELECT strftime('%Y-W%W', closed_at_utc) AS week, COUNT(*) AS n, "
        "SUM(CASE WHEN pnl_usdt > 0 THEN 1 ELSE 0 END) AS wins, "
        "SUM(pnl_usdt) AS pnl "
        "FROM trades GROUP BY week ORDER BY week DESC LIMIT 12"
    )
    return [dict(row) for row in rows]


def daily_pnl(date_str: str) -> float:
    """PnL for a single UTC date string like '2026-05-15'."""
    return _scalar(
        "SELECT SUM(pnl_usdt) FROM trades WHERE DATE(closed_at_utc) = ?",
        (date_str,),
        default=0.0,
    )


def weekly_pnl(week_str: str) -> float:
    """PnL for a week string like '2026-W20' (strftime format)."""
    return _scalar(
        "SELECT SUM(pnl_usdt) FROM trades "
        "WHERE strftime('%Y-W%W', closed_at_utc) = ?",
        (week_str,),
        default=0.0,
    )


def recent_trades(limit: int = 10) -> list[dict]:
    rows = _rows(
        "SELECT symbol, side, entry_price, exit_price, pnl_usdt, pnl_pct, "
        "close_reason, duration_minutes, strategy, opened_at_utc, closed_at_utc "
        "FROM trades ORDER BY id DESC LIMIT ?",
        (limit,),
    )
    return [dict(row) for row in rows]
