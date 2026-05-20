"""
learning_memory.py — Pattern memory for engine × regime × session performance.

Maintains a rolling map of (engine, regime, session) → performance modifier,
derived from trades.db.  Updated once per hour; used to apply a small
rank_score boost or penalty during opportunity scoring.

Public API
----------
    memory_modifier(engine, regime, session) → float  (-10 to +10 rank_score delta)
    get_strongest_pairs(n)                   → list[dict]
    get_weakest_pairs(n)                     → list[dict]
    get_memory_summary()                     → dict
    refresh_memory()                         → None  (force rebuild from DB)

Never raises.
"""
from __future__ import annotations

import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import logger

# ── Config ────────────────────────────────────────────────────────────────────

_DB_PATHS = [Path("/opt/btcbot/trades.db"), Path("trades.db")]
_REFRESH_INTERVAL_S = 3600   # rebuild from DB every hour
_MIN_SAMPLE         = 8      # minimum trades for a pair to count
_MAX_MODIFIER       = 10.0   # max rank_score delta

# ── State ─────────────────────────────────────────────────────────────────────

_lock         = threading.Lock()
_memory: dict = {}          # (engine, regime, session) → {expectancy, trades, modifier}
_last_refresh: float = 0.0


# ── DB helpers ────────────────────────────────────────────────────────────────

def _db() -> Optional[sqlite3.Connection]:
    for p in _DB_PATHS:
        if p.exists():
            conn = sqlite3.connect(p)
            conn.row_factory = sqlite3.Row
            return conn
    return None


def _build_memory() -> dict:
    """
    Query trades.db and compute expectancy per (strategy, regime, session) group.
    Returns new memory dict.
    """
    try:
        conn = _db()
        if conn is None:
            return {}

        sql = """
            SELECT
                strategy,
                COALESCE(regime,  '') AS regime,
                COALESCE(session, '') AS session,
                COUNT(*) AS n,
                SUM(CASE WHEN pnl_usdt > 0 THEN 1 ELSE 0 END) AS wins,
                AVG(pnl_usdt) AS avg_pnl,
                AVG(CASE WHEN pnl_usdt > 0 THEN pnl_usdt ELSE NULL END) AS avg_win,
                AVG(CASE WHEN pnl_usdt <= 0 THEN pnl_usdt ELSE NULL END) AS avg_loss
            FROM trades
            WHERE strategy != '' AND closed_at_utc >= datetime('now', '-90 days')
            GROUP BY strategy, regime, session
            HAVING COUNT(*) >= ?
        """
        with conn:
            rows = conn.execute(sql, (_MIN_SAMPLE,)).fetchall()

        mem: dict = {}
        for r in rows:
            n        = r["n"]
            wins     = r["wins"]
            avg_win  = float(r["avg_win"]  or 0.0)
            avg_loss = float(r["avg_loss"] or 0.0)
            wr       = wins / n if n > 0 else 0.0
            exp      = wr * avg_win + (1 - wr) * avg_loss

            key = (str(r["strategy"]), str(r["regime"]), str(r["session"]))
            mem[key] = {
                "engine":     r["strategy"],
                "regime":     r["regime"],
                "session":    r["session"],
                "trades":     n,
                "win_rate":   round(wr, 3),
                "expectancy": round(exp, 4),
                "avg_pnl":    round(float(r["avg_pnl"]), 4),
            }
        return mem
    except Exception as exc:
        logger.log_warning(f"learning_memory._build_memory error: {exc}")
        return {}


def _maybe_refresh() -> None:
    global _last_refresh
    with _lock:
        age = time.monotonic() - _last_refresh
    if age >= _REFRESH_INTERVAL_S:
        new_mem = _build_memory()
        with _lock:
            _memory.clear()
            _memory.update(new_mem)
            _last_refresh = time.monotonic()
        logger.log_info(f"LEARNING_MEMORY | rebuilt {len(new_mem)} patterns")


def _exp_to_modifier(expectancy: float, trades: int) -> float:
    """
    Map expectancy to a rank_score modifier.
    Scales by sample confidence (capped at 50 trades).
    exp=+0.01 USDT → roughly +5 modifier at full confidence.
    """
    # Confidence scaling: 8 trades = 30%, 20 trades = 70%, 50+ = 100%
    confidence = min(1.0, (trades - _MIN_SAMPLE) / (50 - _MIN_SAMPLE))
    raw = expectancy * 500.0   # 0.02 → 10, -0.02 → -10
    return round(max(-_MAX_MODIFIER, min(_MAX_MODIFIER, raw * confidence)), 2)


# ── Public API ────────────────────────────────────────────────────────────────

def memory_modifier(engine: str, regime: str, session: str) -> float:
    """
    Return rank_score delta for (engine, regime, session).
    Returns 0.0 if insufficient data or on any error.
    """
    try:
        _maybe_refresh()
        with _lock:
            entry = _memory.get((engine, regime, session))
        if entry is None:
            return 0.0
        return _exp_to_modifier(entry["expectancy"], entry["trades"])
    except Exception:
        return 0.0


def get_strongest_pairs(n: int = 5) -> list[dict]:
    """Return top-n (engine, regime, session) combinations by expectancy."""
    try:
        _maybe_refresh()
        with _lock:
            items = list(_memory.values())
        sorted_items = sorted(items, key=lambda x: x["expectancy"], reverse=True)
        result = []
        for item in sorted_items[:n]:
            result.append({
                **item,
                "modifier": _exp_to_modifier(item["expectancy"], item["trades"]),
            })
        return result
    except Exception:
        return []


def get_weakest_pairs(n: int = 5) -> list[dict]:
    """Return bottom-n (engine, regime, session) combinations by expectancy."""
    try:
        _maybe_refresh()
        with _lock:
            items = list(_memory.values())
        sorted_items = sorted(items, key=lambda x: x["expectancy"])
        result = []
        for item in sorted_items[:n]:
            result.append({
                **item,
                "modifier": _exp_to_modifier(item["expectancy"], item["trades"]),
            })
        return result
    except Exception:
        return []


def get_memory_summary() -> dict:
    """Return full memory summary for dashboard / Telegram."""
    try:
        _maybe_refresh()
        with _lock:
            total_patterns = len(_memory)
            age_s = time.monotonic() - _last_refresh

        strongest = get_strongest_pairs(5)
        weakest   = get_weakest_pairs(5)

        # Engine-level aggregates
        engine_totals: dict[str, list] = {}
        with _lock:
            for key, entry in _memory.items():
                eng = entry["engine"]
                if eng not in engine_totals:
                    engine_totals[eng] = []
                engine_totals[eng].append(entry["expectancy"])

        engine_avg = {
            eng: round(sum(vals) / len(vals), 4)
            for eng, vals in engine_totals.items()
        }

        return {
            "total_patterns":  total_patterns,
            "cache_age_s":     round(age_s, 0),
            "refresh_interval_s": _REFRESH_INTERVAL_S,
            "min_sample":      _MIN_SAMPLE,
            "strongest_pairs": strongest,
            "weakest_pairs":   weakest,
            "engine_avg_expectancy": engine_avg,
        }
    except Exception as exc:
        logger.log_warning(f"learning_memory.get_memory_summary error: {exc}")
        return {"total_patterns": 0, "error": str(exc)}


def refresh_memory() -> None:
    """Force a memory rebuild from trades.db."""
    global _last_refresh
    with _lock:
        _last_refresh = 0.0
    _maybe_refresh()
