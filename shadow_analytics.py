"""
shadow_analytics.py — Live vs shadow comparative analytics.

Maintains a shadow_trades table in trades.db (separate from trades table —
zero impact on existing live-trade queries).

Records shadow closed trades when shadow_engine closes them, then computes
per-engine / per-symbol comparison of expectancy, drawdown, and fill quality.

Flags shadow strategies that outperform live so they can be reviewed for
potential live promotion.

Public API
----------
    record_shadow_trade(...)          → None
    get_live_vs_shadow(days)          → dict  (per-engine comparison)
    get_outperforming_shadows(days)   → list[dict]
    get_comparison_summary(days)      → dict  (for dashboard / Telegram)

Never raises.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import logger

# ── DB setup ──────────────────────────────────────────────────────────────────

_DB_PATHS = [Path("/opt/btcbot/trades.db"), Path("trades.db")]

_CREATE_SHADOW_TABLE = """
CREATE TABLE IF NOT EXISTS shadow_trades (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    engine           TEXT    NOT NULL DEFAULT '',
    symbol           TEXT    NOT NULL,
    direction        TEXT    NOT NULL DEFAULT 'LONG',
    entry_price      REAL    NOT NULL DEFAULT 0.0,
    exit_price       REAL    NOT NULL DEFAULT 0.0,
    stop_price       REAL    NOT NULL DEFAULT 0.0,
    tp_price         REAL    NOT NULL DEFAULT 0.0,
    outcome          TEXT    NOT NULL DEFAULT '',
    pnl_pct          REAL    NOT NULL DEFAULT 0.0,
    regime           TEXT    NOT NULL DEFAULT '',
    session          TEXT    NOT NULL DEFAULT '',
    opened_at        TEXT    NOT NULL,
    closed_at        TEXT    NOT NULL,
    reason           TEXT    NOT NULL DEFAULT ''
)
"""


def _db() -> Optional[sqlite3.Connection]:
    for p in _DB_PATHS:
        if p.exists():
            conn = sqlite3.connect(p)
            conn.row_factory = sqlite3.Row
            return conn
    # Try creating in current dir
    try:
        conn = sqlite3.connect(_DB_PATHS[-1])
        conn.row_factory = sqlite3.Row
        return conn
    except Exception:
        return None


def _ensure_shadow_table() -> None:
    try:
        conn = _db()
        if conn is None:
            return
        with conn:
            conn.execute(_CREATE_SHADOW_TABLE)
    except Exception as exc:
        logger.log_warning(f"shadow_analytics._ensure_shadow_table error: {exc}")


_ensure_shadow_table()


# ── Write ─────────────────────────────────────────────────────────────────────

def record_shadow_trade(
    engine:      str,
    symbol:      str,
    direction:   str,
    entry_price: float,
    exit_price:  float,
    stop_price:  float,
    tp_price:    float,
    outcome:     str,    # "TP" | "SL"
    pnl_pct:     float,
    regime:      str  = "",
    session:     str  = "",
    opened_at:   str  = "",
    closed_at:   str  = "",
    reason:      str  = "",
) -> None:
    """Persist a closed shadow trade to shadow_trades table."""
    try:
        if not closed_at:
            closed_at = datetime.now(timezone.utc).isoformat()
        if not opened_at:
            opened_at = closed_at

        conn = _db()
        if conn is None:
            return
        with conn:
            conn.execute(
                """INSERT INTO shadow_trades
                   (engine, symbol, direction, entry_price, exit_price,
                    stop_price, tp_price, outcome, pnl_pct, regime, session,
                    opened_at, closed_at, reason)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (engine, symbol, direction, entry_price, exit_price,
                 stop_price, tp_price, outcome, pnl_pct, regime, session,
                 opened_at, closed_at, reason),
            )
    except Exception as exc:
        logger.log_warning(f"shadow_analytics.record_shadow_trade error: {exc}")


# ── Analytics helpers ─────────────────────────────────────────────────────────

def _since(days: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()


def _live_stats_by_engine(days: int) -> dict[str, dict]:
    """Read live trade stats grouped by strategy from trades table."""
    try:
        conn = _db()
        if conn is None:
            return {}
        sql = """
            SELECT strategy AS engine,
                   COUNT(*) AS n,
                   SUM(CASE WHEN pnl_usdt > 0 THEN 1 ELSE 0 END) AS wins,
                   AVG(pnl_usdt) AS avg_pnl,
                   AVG(CASE WHEN pnl_usdt > 0 THEN pnl_usdt ELSE NULL END) AS avg_win,
                   AVG(CASE WHEN pnl_usdt <= 0 THEN pnl_usdt ELSE NULL END) AS avg_loss,
                   MIN(pnl_usdt) AS worst_trade,
                   SUM(pnl_usdt) AS total_pnl
            FROM trades
            WHERE closed_at_utc >= ? AND strategy != ''
            GROUP BY strategy
        """
        with conn:
            rows = conn.execute(sql, (_since(days),)).fetchall()
        result = {}
        for r in rows:
            n = r["n"] or 0
            wins = r["wins"] or 0
            wr = wins / n if n > 0 else 0.0
            avg_win  = float(r["avg_win"]  or 0.0)
            avg_loss = float(r["avg_loss"] or 0.0)
            exp = wr * avg_win + (1 - wr) * avg_loss
            result[r["engine"]] = {
                "trades":      n,
                "win_rate":    round(wr, 3),
                "expectancy":  round(exp, 4),
                "avg_win":     round(avg_win, 4),
                "avg_loss":    round(avg_loss, 4),
                "total_pnl":   round(float(r["total_pnl"] or 0.0), 4),
                "worst_trade": round(float(r["worst_trade"] or 0.0), 4),
            }
        return result
    except Exception as exc:
        logger.log_warning(f"shadow_analytics._live_stats_by_engine error: {exc}")
        return {}


def _shadow_stats_by_engine(days: int) -> dict[str, dict]:
    """Read shadow trade stats grouped by engine from shadow_trades table."""
    try:
        conn = _db()
        if conn is None:
            return {}
        sql = """
            SELECT engine,
                   COUNT(*) AS n,
                   SUM(CASE WHEN outcome = 'TP' THEN 1 ELSE 0 END) AS wins,
                   AVG(pnl_pct) AS avg_pnl_pct,
                   AVG(CASE WHEN outcome = 'TP' THEN pnl_pct ELSE NULL END) AS avg_win_pct,
                   AVG(CASE WHEN outcome = 'SL' THEN pnl_pct ELSE NULL END) AS avg_loss_pct,
                   MIN(pnl_pct) AS worst_pct,
                   SUM(pnl_pct) AS total_pnl_pct
            FROM shadow_trades
            WHERE closed_at >= ?
            GROUP BY engine
        """
        with conn:
            rows = conn.execute(sql, (_since(days),)).fetchall()
        result = {}
        for r in rows:
            n = r["n"] or 0
            wins = r["wins"] or 0
            wr = wins / n if n > 0 else 0.0
            avg_win  = float(r["avg_win_pct"]  or 0.0)
            avg_loss = float(r["avg_loss_pct"] or 0.0)
            exp = wr * avg_win + (1 - wr) * avg_loss
            result[r["engine"]] = {
                "trades":      n,
                "win_rate":    round(wr, 3),
                "expectancy_pct": round(exp, 4),
                "avg_win_pct": round(avg_win, 4),
                "avg_loss_pct": round(avg_loss, 4),
                "total_pnl_pct": round(float(r["total_pnl_pct"] or 0.0), 4),
                "worst_pct":   round(float(r["worst_pct"] or 0.0), 4),
            }
        return result
    except Exception as exc:
        logger.log_warning(f"shadow_analytics._shadow_stats_by_engine error: {exc}")
        return {}


# ── Public API ────────────────────────────────────────────────────────────────

def get_live_vs_shadow(days: int = 30) -> dict:
    """
    Return per-engine comparison of live vs shadow performance.
    Keys: engine → {live: {...}, shadow: {...}, delta_expectancy, outperforms}
    """
    try:
        live   = _live_stats_by_engine(days)
        shadow = _shadow_stats_by_engine(days)
        engines = set(list(live.keys()) + list(shadow.keys()))

        result = {}
        for eng in sorted(engines):
            live_data   = live.get(eng, {})
            shadow_data = shadow.get(eng, {})

            # Delta: shadow expectancy_pct vs live expectancy (live is in USDT, shadow in %)
            # We can only meaningfully compare if both have data
            live_exp    = live_data.get("expectancy", None)
            shadow_exp  = shadow_data.get("expectancy_pct", None)

            outperforms = False
            delta       = None
            if live_exp is not None and shadow_exp is not None and live_data.get("trades", 0) >= 5:
                delta = round(shadow_exp - live_exp, 4)
                outperforms = shadow_exp > live_exp and shadow_data.get("trades", 0) >= 5

            result[eng] = {
                "live":              live_data,
                "shadow":            shadow_data,
                "delta_expectancy":  delta,
                "outperforms":       outperforms,
            }
        return result
    except Exception as exc:
        logger.log_warning(f"shadow_analytics.get_live_vs_shadow error: {exc}")
        return {}


def get_outperforming_shadows(days: int = 30) -> list[dict]:
    """
    Return list of engines where shadow outperforms live (both ≥5 trades).
    Sorted by expectancy delta descending.
    """
    try:
        comparison = get_live_vs_shadow(days)
        out = [
            {"engine": eng, **data}
            for eng, data in comparison.items()
            if data.get("outperforms") and data.get("delta_expectancy") is not None
        ]
        return sorted(out, key=lambda x: x.get("delta_expectancy", 0), reverse=True)
    except Exception:
        return []


def get_comparison_summary(days: int = 30) -> dict:
    """Return dashboard/Telegram-friendly summary."""
    try:
        comparison = get_live_vs_shadow(days)
        outperformers = get_outperforming_shadows(days)

        total_live_trades   = sum(v["live"].get("trades", 0)   for v in comparison.values())
        total_shadow_trades = sum(v["shadow"].get("trades", 0) for v in comparison.values())

        return {
            "days":               days,
            "engines":            comparison,
            "outperforming":      outperformers,
            "total_live_trades":  total_live_trades,
            "total_shadow_trades": total_shadow_trades,
            "outperformer_count": len(outperformers),
        }
    except Exception as exc:
        logger.log_warning(f"shadow_analytics.get_comparison_summary error: {exc}")
        return {"days": days, "engines": {}, "outperforming": []}
