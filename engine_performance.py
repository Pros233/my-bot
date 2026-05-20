"""
engine_performance.py — Per-engine live performance analytics.

Reads directly from trades.db (strategy column) so no separate recording
step is needed — every closed trade is already there.

Public API
----------
    get_engine_stats(engine, days)     → single-engine stat dict
    get_all_stats(days)                → all engines
    get_recent_stats(engine, days)     → recent N-day slice only
    get_breakdown(engine, dimension)   → stats sliced by symbol/regime/session/etc.
    ENGINE_NAMES                       → canonical list of engine names

Stat dict keys:
    trades, wins, losses, win_rate, total_pnl, expectancy,
    avg_win, avg_loss, profit_factor, max_drawdown_pct,
    sharpe_ratio, avg_duration_min, consecutive_losses,
    consecutive_losses_current

Never raises — returns empty/default dicts on any DB error.
"""
from __future__ import annotations

import math
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

# ── Constants ─────────────────────────────────────────────────────────────────

ENGINE_NAMES: list[str] = [
    "RMR", "CONSENSUS", "PULLBACK", "BREAKOUT", "NY_MOMENTUM", "MICRO_MR",
]

_DB_PATHS = [Path("/opt/btcbot/trades.db"), Path("trades.db")]


def _db_path() -> Path:
    for p in _DB_PATHS:
        if p.exists():
            return p
    return _DB_PATHS[-1]


# ── DB helpers ────────────────────────────────────────────────────────────────

def _rows(sql: str, params: tuple = ()) -> list:
    try:
        with sqlite3.connect(_db_path()) as conn:
            conn.row_factory = sqlite3.Row
            return conn.execute(sql, params).fetchall()
    except Exception:
        return []


def _scalar(sql: str, params: tuple = (), default=0):
    try:
        with sqlite3.connect(_db_path()) as conn:
            row = conn.execute(sql, params).fetchone()
            return row[0] if row and row[0] is not None else default
    except Exception:
        return default


# ── Core stat builder ─────────────────────────────────────────────────────────

def _build_stats(rows: list) -> dict:
    """Build full stat dict from a list of sqlite Rows with pnl_usdt, duration_minutes."""
    if not rows:
        return _empty_stats()

    pnls = [float(r["pnl_usdt"]) for r in rows]
    durations = [float(r["duration_minutes"]) for r in rows]

    n = len(pnls)
    wins  = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    total_pnl   = sum(pnls)
    avg_win     = sum(wins) / len(wins)   if wins   else 0.0
    avg_loss    = sum(losses) / len(losses) if losses else 0.0
    gross_win   = sum(wins)
    gross_loss  = abs(sum(losses))
    pf          = gross_win / gross_loss if gross_loss > 0 else (float("inf") if gross_win > 0 else 0.0)

    # Max drawdown on this engine's equity curve
    equity = peak = max_dd = 0.0
    for p in pnls:
        equity += p
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd

    # Sharpe-like ratio
    mean = total_pnl / n
    variance = sum((p - mean) ** 2 for p in pnls) / n
    std = math.sqrt(variance)
    sharpe = mean / std if std > 0 else 0.0

    # Consecutive losses (current streak from tail)
    consec_losses_current = 0
    for p in reversed(pnls):
        if p < 0:
            consec_losses_current += 1
        else:
            break

    # Max consecutive losses (all-time)
    max_streak = streak = 0
    for p in pnls:
        if p < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0

    return {
        "trades":                    n,
        "wins":                      len(wins),
        "losses":                    len(losses),
        "win_rate":                  round(len(wins) / n, 4),
        "total_pnl":                 round(total_pnl, 4),
        "expectancy":                round(mean, 4),
        "avg_win":                   round(avg_win, 4),
        "avg_loss":                  round(avg_loss, 4),
        "profit_factor":             round(pf, 3) if pf != float("inf") else 999.0,
        "max_drawdown_pct":          round(max_dd * 100, 2),
        "sharpe_ratio":              round(sharpe, 3),
        "avg_duration_min":          round(sum(durations) / n, 1),
        "consecutive_losses":        max_streak,
        "consecutive_losses_current": consec_losses_current,
    }


def _empty_stats() -> dict:
    return {
        "trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0,
        "total_pnl": 0.0, "expectancy": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
        "profit_factor": 0.0, "max_drawdown_pct": 0.0, "sharpe_ratio": 0.0,
        "avg_duration_min": 0.0, "consecutive_losses": 0,
        "consecutive_losses_current": 0,
    }


# ── Public API ────────────────────────────────────────────────────────────────

def get_engine_stats(engine: str, days: int = 90) -> dict:
    """Return performance stats for one engine over last N days."""
    try:
        since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        rows = _rows(
            "SELECT pnl_usdt, duration_minutes FROM trades "
            "WHERE strategy = ? AND closed_at_utc >= ? ORDER BY id ASC",
            (engine, since),
        )
        return _build_stats(rows)
    except Exception:
        return _empty_stats()


def get_all_stats(days: int = 90) -> dict[str, dict]:
    """Return stats dict keyed by engine name for all known engines."""
    return {eng: get_engine_stats(eng, days=days) for eng in ENGINE_NAMES}


def get_recent_stats(engine: str, days: int = 14) -> dict:
    """Stats for the most recent N days only (short-window performance)."""
    return get_engine_stats(engine, days=days)


def get_breakdown(engine: str, dimension: str, days: int = 90) -> list[dict]:
    """
    Stats sliced by a dimension column.

    dimension: 'symbol' | 'regime' | 'session' | 'trade_grade'

    Returns list of dicts: [{category, trades, win_rate, expectancy, total_pnl}]
    """
    valid_dims = {"symbol", "regime", "session", "trade_grade"}
    if dimension not in valid_dims:
        return []
    try:
        since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        rows = _rows(
            f"SELECT {dimension} AS cat, pnl_usdt, duration_minutes FROM trades "
            f"WHERE strategy = ? AND closed_at_utc >= ? ORDER BY id ASC",
            (engine, since),
        )
        # Group by category
        groups: dict[str, list] = {}
        for r in rows:
            cat = str(r["cat"] or "unknown")
            groups.setdefault(cat, []).append(r)

        result = []
        for cat, cat_rows in sorted(groups.items()):
            st = _build_stats(cat_rows)
            result.append({
                "category":   cat,
                "trades":     st["trades"],
                "win_rate":   st["win_rate"],
                "expectancy": st["expectancy"],
                "total_pnl":  st["total_pnl"],
            })
        return sorted(result, key=lambda x: -x["total_pnl"])
    except Exception:
        return []


def get_all_breakdown(dimension: str, days: int = 90) -> dict[str, list[dict]]:
    """get_breakdown for all engines keyed by engine name."""
    return {eng: get_breakdown(eng, dimension, days=days) for eng in ENGINE_NAMES}
