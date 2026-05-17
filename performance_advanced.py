"""
performance_advanced.py — Extended trade analytics by category.

All functions query trades.db and return lists of dicts or dicts.
All functions are fail-safe: return [] or {} on any error.
All data is read-only — no writes.

Categories tracked:
  - Session (Asia / London / NY/London / New York / Off-hours)
  - Regime (TRENDING+NORMAL, RANGING+NORMAL, etc.)
  - Hour (0-23 UTC)
  - Weekday (Mon-Sun)
  - Signal score bucket (<40% / 40-60% / 60-80% / >80%)
  - ADX bucket (<20 / 20-30 / 30-40 / >40)
  - ATR bucket (<0.3% / 0.3-0.6% / 0.6-1.0% / >1.0%)
  - Trade grade (A+ / A / B / C)
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

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


# ── Session helper ─────────────────────────────────────────────────────────────

def _hour_to_session(hour: int) -> str:
    if 0 <= hour < 8:
        return "Asia"
    elif 8 <= hour < 13:
        return "London"
    elif 13 <= hour < 16:
        return "NY/London"
    elif 16 <= hour < 21:
        return "New York"
    else:
        return "Off-hours"


def _adx_bucket(adx: float) -> str:
    if adx < 20:
        return "<20 (weak)"
    elif adx < 30:
        return "20-30 (moderate)"
    elif adx < 40:
        return "30-40 (strong)"
    else:
        return ">40 (very strong)"


def _atr_bucket(atr_pct: float) -> str:
    if atr_pct < 0.3:
        return "<0.3% (low)"
    elif atr_pct < 0.6:
        return "0.3-0.6% (medium)"
    elif atr_pct < 1.0:
        return "0.6-1.0% (elevated)"
    else:
        return ">1.0% (high)"


def _score_bucket(score_pct: float) -> str:
    if score_pct < 40:
        return "<40%"
    elif score_pct < 60:
        return "40-60%"
    elif score_pct < 80:
        return "60-80%"
    else:
        return ">80%"


def _weekday_name(wd: int) -> str:
    """SQLite strftime %w: 0=Sunday."""
    names = {0: "Sunday", 1: "Monday", 2: "Tuesday", 3: "Wednesday",
             4: "Thursday", 5: "Friday", 6: "Saturday"}
    return names.get(wd, str(wd))


# ── Aggregation helper ────────────────────────────────────────────────────────

def _aggregate(rows: list, key_fn) -> list[dict]:
    """Group rows by key_fn, compute n/wins/pnl/win_rate/avg_pnl."""
    buckets: dict[str, dict] = {}
    for row in rows:
        key = key_fn(row)
        if key not in buckets:
            buckets[key] = {"n": 0, "wins": 0, "pnl": 0.0}
        buckets[key]["n"]   += 1
        buckets[key]["pnl"] += float(row["pnl_usdt"] or 0)
        if float(row["pnl_usdt"] or 0) > 0:
            buckets[key]["wins"] += 1

    result = []
    for k, d in buckets.items():
        n = d["n"]
        result.append({
            "category": k,
            "trades":   n,
            "wins":     d["wins"],
            "win_rate": round(d["wins"] / n * 100, 1) if n > 0 else 0.0,
            "total_pnl": round(d["pnl"], 4),
            "avg_pnl":  round(d["pnl"] / n, 4) if n > 0 else 0.0,
        })
    return sorted(result, key=lambda x: x["total_pnl"], reverse=True)


# ── Public analytics functions ─────────────────────────────────────────────────

def pnl_by_session() -> list[dict]:
    """PnL breakdown by trading session derived from opened_at_utc hour."""
    try:
        rows = _rows("SELECT opened_at_utc, pnl_usdt FROM trades WHERE pnl_usdt IS NOT NULL")
        for row in rows:
            # attach session to each row for aggregation
            pass  # can't mutate sqlite3.Row — use dict

        data = []
        for row in rows:
            try:
                hour = int(row["opened_at_utc"][11:13])
            except Exception:
                hour = 0
            data.append({"session": _hour_to_session(hour), "pnl_usdt": float(row["pnl_usdt"] or 0)})

        buckets: dict[str, dict] = {}
        for d in data:
            s = d["session"]
            if s not in buckets:
                buckets[s] = {"n": 0, "wins": 0, "pnl": 0.0}
            buckets[s]["n"]   += 1
            buckets[s]["pnl"] += d["pnl_usdt"]
            if d["pnl_usdt"] > 0:
                buckets[s]["wins"] += 1

        result = []
        for s, d in buckets.items():
            n = d["n"]
            result.append({
                "category": s, "trades": n, "wins": d["wins"],
                "win_rate": round(d["wins"] / n * 100, 1) if n > 0 else 0.0,
                "total_pnl": round(d["pnl"], 4),
                "avg_pnl":   round(d["pnl"] / n, 4) if n > 0 else 0.0,
            })
        return sorted(result, key=lambda x: x["total_pnl"], reverse=True)
    except Exception:
        return []


def pnl_by_regime() -> list[dict]:
    """PnL breakdown by regime string stored in trades.db."""
    try:
        rows = _rows("SELECT regime, pnl_usdt FROM trades WHERE pnl_usdt IS NOT NULL AND regime != ''")
        return _aggregate(rows, lambda r: r["regime"] or "Unknown")
    except Exception:
        return []


def pnl_by_hour() -> list[dict]:
    """PnL breakdown by UTC hour of entry (0-23)."""
    try:
        rows = _rows("SELECT opened_at_utc, pnl_usdt FROM trades WHERE pnl_usdt IS NOT NULL")
        data = []
        for row in rows:
            try:
                hour = int(row["opened_at_utc"][11:13])
            except Exception:
                hour = 0
            data.append({"hour": hour, "pnl_usdt": float(row["pnl_usdt"] or 0)})

        buckets: dict[int, dict] = {}
        for d in data:
            h = d["hour"]
            if h not in buckets:
                buckets[h] = {"n": 0, "wins": 0, "pnl": 0.0}
            buckets[h]["n"]   += 1
            buckets[h]["pnl"] += d["pnl_usdt"]
            if d["pnl_usdt"] > 0:
                buckets[h]["wins"] += 1

        result = []
        for h in sorted(buckets):
            d = buckets[h]
            n = d["n"]
            result.append({
                "category": f"{h:02d}:00 UTC",
                "trades": n, "wins": d["wins"],
                "win_rate": round(d["wins"] / n * 100, 1) if n > 0 else 0.0,
                "total_pnl": round(d["pnl"], 4),
                "avg_pnl":   round(d["pnl"] / n, 4) if n > 0 else 0.0,
            })
        return result
    except Exception:
        return []


def pnl_by_weekday() -> list[dict]:
    """PnL breakdown by weekday of entry."""
    try:
        rows = _rows(
            "SELECT CAST(strftime('%w', opened_at_utc) AS INTEGER) AS wd, pnl_usdt "
            "FROM trades WHERE pnl_usdt IS NOT NULL"
        )
        result = _aggregate(rows, lambda r: _weekday_name(int(r["wd"])))
        return result
    except Exception:
        return []


def pnl_by_score_bucket() -> list[dict]:
    """PnL breakdown by consensus score bucket."""
    try:
        rows = _rows("SELECT score_pct, pnl_usdt FROM trades WHERE pnl_usdt IS NOT NULL")
        return _aggregate(rows, lambda r: _score_bucket(float(r["score_pct"] or 0)))
    except Exception:
        return []


def pnl_by_adx_bucket() -> list[dict]:
    """PnL breakdown by ADX range at entry."""
    try:
        rows = _rows("SELECT adx, pnl_usdt FROM trades WHERE pnl_usdt IS NOT NULL")
        return _aggregate(rows, lambda r: _adx_bucket(float(r["adx"] or 0)))
    except Exception:
        return []


def pnl_by_atr_bucket() -> list[dict]:
    """PnL breakdown by ATR% range at entry."""
    try:
        rows = _rows("SELECT atr_pct, pnl_usdt FROM trades WHERE pnl_usdt IS NOT NULL")
        return _aggregate(rows, lambda r: _atr_bucket(float(r["atr_pct"] or 0)))
    except Exception:
        return []


def pnl_by_grade() -> list[dict]:
    """PnL breakdown by trade grade (A+/A/B/C/unknown)."""
    try:
        rows = _rows("SELECT trade_grade, pnl_usdt FROM trades WHERE pnl_usdt IS NOT NULL")
        return _aggregate(rows, lambda r: r["trade_grade"] or "ungraded")
    except Exception:
        return []


def pnl_by_symbol() -> list[dict]:
    """PnL breakdown by symbol."""
    try:
        rows = _rows("SELECT symbol, pnl_usdt FROM trades WHERE pnl_usdt IS NOT NULL")
        return _aggregate(rows, lambda r: r["symbol"])
    except Exception:
        return []


def pnl_by_strategy() -> list[dict]:
    """PnL breakdown by strategy."""
    try:
        rows = _rows("SELECT strategy, pnl_usdt FROM trades WHERE pnl_usdt IS NOT NULL")
        return _aggregate(rows, lambda r: r["strategy"] or "UNKNOWN")
    except Exception:
        return []


# ── Best / worst conditions ────────────────────────────────────────────────────

def best_market_conditions(top_n: int = 3) -> dict:
    """
    Return the top-N best conditions across all dimensions.
    'Best' = highest average PnL per trade with at least 2 trades.
    """
    try:
        results: dict[str, list] = {}
        dims = {
            "session":    pnl_by_session,
            "regime":     pnl_by_regime,
            "hour":       pnl_by_hour,
            "weekday":    pnl_by_weekday,
            "score":      pnl_by_score_bucket,
            "adx":        pnl_by_adx_bucket,
            "atr":        pnl_by_atr_bucket,
            "grade":      pnl_by_grade,
        }
        for dim, fn in dims.items():
            rows = [r for r in fn() if r["trades"] >= 2]
            top = sorted(rows, key=lambda x: x["avg_pnl"], reverse=True)[:top_n]
            results[dim] = top
        return results
    except Exception:
        return {}


def worst_market_conditions(top_n: int = 3) -> dict:
    """
    Return the worst-N conditions across all dimensions.
    'Worst' = lowest average PnL per trade with at least 2 trades.
    """
    try:
        results: dict[str, list] = {}
        dims = {
            "session":    pnl_by_session,
            "regime":     pnl_by_regime,
            "hour":       pnl_by_hour,
            "weekday":    pnl_by_weekday,
            "score":      pnl_by_score_bucket,
            "adx":        pnl_by_adx_bucket,
            "atr":        pnl_by_atr_bucket,
            "grade":      pnl_by_grade,
        }
        for dim, fn in dims.items():
            rows = [r for r in fn() if r["trades"] >= 2]
            worst = sorted(rows, key=lambda x: x["avg_pnl"])[:top_n]
            results[dim] = worst
        return results
    except Exception:
        return {}


def grade_distribution() -> dict[str, int]:
    """Count of trades by grade."""
    try:
        rows = _rows("SELECT trade_grade, COUNT(*) AS n FROM trades GROUP BY trade_grade")
        return {row["trade_grade"] or "ungraded": row["n"] for row in rows}
    except Exception:
        return {}


def summary_report() -> dict:
    """Combined summary for Telegram /conditions command."""
    try:
        best  = best_market_conditions(top_n=1)
        worst = worst_market_conditions(top_n=1)

        def _fmt(dim_data: list[dict]) -> str:
            if not dim_data:
                return "N/A (no data)"
            r = dim_data[0]
            return f"{r['category']} (WR={r['win_rate']:.0f}%, avg={r['avg_pnl']:+.4f})"

        return {
            "best_session":  _fmt(best.get("session", [])),
            "best_regime":   _fmt(best.get("regime", [])),
            "best_hour":     _fmt(best.get("hour", [])),
            "best_weekday":  _fmt(best.get("weekday", [])),
            "best_grade":    _fmt(best.get("grade", [])),
            "worst_session": _fmt(worst.get("session", [])),
            "worst_regime":  _fmt(worst.get("regime", [])),
            "worst_hour":    _fmt(worst.get("hour", [])),
            "worst_weekday": _fmt(worst.get("weekday", [])),
            "grade_dist":    grade_distribution(),
        }
    except Exception:
        return {}
