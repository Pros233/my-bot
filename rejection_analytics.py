"""
rejection_analytics.py — Setup funnel tracking and rejection analytics.

Tracks every candle cycle: how many setups were scanned, why they were
rejected, which filters fired, and what grade the surviving candidates
received. State persists across bot restarts via a JSON sidecar file.

Public API
----------
    record_scan(...)          — call once per symbol per candle cycle
    get_summary()             — dict with all top-level stats
    get_funnel()              — dict with funnel counts (scanned → executed)
    get_daily_series(days)    — per-day counts for charting
"""
from __future__ import annotations

import json
import threading
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


# ── Persistence paths ─────────────────────────────────────────────────────────

_SAVE_PATHS = [
    Path("/opt/btcbot/rejection_analytics.json"),
    Path("rejection_analytics.json"),
]


def _save_path() -> Path:
    for p in _SAVE_PATHS:
        if p.parent.exists():
            return p
    return _SAVE_PATHS[-1]


# ── In-memory state ───────────────────────────────────────────────────────────

_lock = threading.Lock()

_state: dict = {
    "total_scanned":      0,
    "total_rejected":     0,
    "total_executed":     0,
    "rejection_reasons":  {},   # reason_str  → count
    "grade_distribution": {},   # grade        → count
    "symbol_rejected":    {},   # symbol       → count
    "session_rejected":   {},   # session_name → count
    "filter_hits":        {},   # filter_name  → count
    "daily_scanned":      {},   # date_str     → count
    "daily_rejected":     {},   # date_str     → count
    "daily_executed":     {},   # date_str     → count
    "by_strategy":        {},   # strategy_name → {"scanned": N, "executed": N, "rejected": N}
}


def _inc(d: dict, key: str, amount: int = 1) -> None:
    """Increment dict[key] in-place (no lock — caller must hold _lock)."""
    d[key] = d.get(key, 0) + amount


# ── Persistence ───────────────────────────────────────────────────────────────

def _load() -> None:
    """Load persisted state from disk on startup. Silent on error."""
    p = _save_path()
    if not p.exists():
        return
    try:
        data = json.loads(p.read_text())
        with _lock:
            for k, v in data.items():
                if k in _state and isinstance(v, dict):
                    _state[k] = v
                elif k in _state and isinstance(_state[k], int):
                    _state[k] = int(v)
    except Exception:
        pass


def _save() -> None:
    """Atomically write state to disk. Silent on error."""
    p = _save_path()
    try:
        with _lock:
            snapshot = json.dumps(_state, indent=2)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(snapshot)
        tmp.replace(p)
    except Exception:
        pass


# ── Public API ────────────────────────────────────────────────────────────────

def record_scan(
    symbol: str,
    session: str,
    rejected: bool,
    reject_reason: str = "",
    grade: str = "",
    filter_hits: list[str] | None = None,
    executed: bool = False,
    strategy: str = "",
) -> None:
    """
    Record one scanned setup.

    Parameters
    ----------
    symbol        : Trading symbol, e.g. "BTCUSDT"
    session       : Session name from trade_filters.hour_to_session()
    rejected      : True if the setup was NOT executed
    reject_reason : Human-readable reason for rejection
    grade         : Trade grade ("A+", "A", "B", "C", "REJECT"), if computed
    filter_hits   : Names of filters that did not pass
    executed      : True if an order was successfully placed
    strategy      : Strategy/engine name ("RMR", "PULLBACK", "BREAKOUT", etc.)
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    with _lock:
        _inc(_state, "total_scanned")
        _inc(_state["daily_scanned"], today)

        if executed:
            _inc(_state, "total_executed")
            _inc(_state["daily_executed"], today)

        if rejected:
            _inc(_state, "total_rejected")
            _inc(_state["daily_rejected"], today)
            if reject_reason:
                _inc(_state["rejection_reasons"], reject_reason)
            _inc(_state["symbol_rejected"],  symbol)
            _inc(_state["session_rejected"], session)

        if grade:
            _inc(_state["grade_distribution"], grade)

        if filter_hits:
            for f in filter_hits:
                _inc(_state["filter_hits"], f)

        if strategy:
            if strategy not in _state["by_strategy"]:
                _state["by_strategy"][strategy] = {"scanned": 0, "executed": 0, "rejected": 0}
            _state["by_strategy"][strategy]["scanned"] += 1
            if executed:
                _state["by_strategy"][strategy]["executed"] += 1
            if rejected:
                _state["by_strategy"][strategy]["rejected"] += 1

    _save()


def get_summary() -> dict:
    """Return a full summary dict (safe snapshot)."""
    with _lock:
        total    = _state["total_scanned"]
        rejected = _state["total_rejected"]
        executed = _state["total_executed"]

        top_reasons = sorted(
            _state["rejection_reasons"].items(), key=lambda x: -x[1]
        )[:10]

        top_symbols = sorted(
            _state["symbol_rejected"].items(), key=lambda x: -x[1]
        )[:5]

        top_sessions = sorted(
            _state["session_rejected"].items(), key=lambda x: -x[1]
        )[:5]

        top_filters = sorted(
            _state["filter_hits"].items(), key=lambda x: -x[1]
        )[:8]

        grade_dist = dict(_state["grade_distribution"])

    return {
        "total_scanned":         total,
        "total_rejected":        rejected,
        "total_executed":        executed,
        "rejection_rate_pct":    round(rejected / total * 100, 1) if total > 0 else 0.0,
        "top_reasons":           top_reasons,
        "top_symbols_rejected":  top_symbols,
        "top_sessions_rejected": top_sessions,
        "top_filters_hit":       top_filters,
        "grade_distribution":    grade_dist,
    }


def get_funnel() -> dict:
    """Return setup funnel breakdown counts."""
    with _lock:
        total    = _state["total_scanned"]
        rejected = _state["total_rejected"]
        executed = _state["total_executed"]
        grade_dist = dict(_state["grade_distribution"])

    return {
        "scanned":      total,
        "rejected":     rejected,
        "passed":       max(0, total - rejected),
        "executed":     executed,
        "grade_Aplus":  grade_dist.get("A+", 0),
        "grade_A":      grade_dist.get("A",  0),
        "grade_B":      grade_dist.get("B",  0),
        "grade_C":      grade_dist.get("C",  0),
        "grade_REJECT": grade_dist.get("REJECT", 0),
    }


def get_frequency_stats() -> dict:
    """Return per-strategy scan/execute/reject breakdowns."""
    with _lock:
        by_strategy = {k: dict(v) for k, v in _state["by_strategy"].items()}
        total_scanned  = _state["total_scanned"]
        total_executed = _state["total_executed"]

        # Setups per day (last 7 days average)
        daily_scanned  = dict(_state["daily_scanned"])
        daily_executed = dict(_state["daily_executed"])

    days_7 = sorted(daily_scanned.keys())[-7:]
    avg_scanned_per_day  = (
        sum(daily_scanned.get(d, 0) for d in days_7) / len(days_7)
        if days_7 else 0.0
    )
    avg_executed_per_day = (
        sum(daily_executed.get(d, 0) for d in days_7) / len(days_7)
        if days_7 else 0.0
    )

    return {
        "total_scanned":          total_scanned,
        "total_executed":         total_executed,
        "avg_scanned_per_day_7d": round(avg_scanned_per_day, 1),
        "avg_executed_per_day_7d": round(avg_executed_per_day, 1),
        "by_strategy":            by_strategy,
    }


def get_daily_series(days: int = 14) -> dict:
    """Return per-day scanned/rejected/executed counts for charting."""
    with _lock:
        scanned  = dict(_state["daily_scanned"])
        rejected = dict(_state["daily_rejected"])
        executed = dict(_state["daily_executed"])

    all_dates = sorted(set(list(scanned) + list(rejected) + list(executed)))
    if days > 0:
        all_dates = all_dates[-days:]

    return {
        "dates":    all_dates,
        "scanned":  [scanned.get(d, 0)  for d in all_dates],
        "rejected": [rejected.get(d, 0) for d in all_dates],
        "executed": [executed.get(d, 0) for d in all_dates],
    }


# ── Load persisted state on module import ─────────────────────────────────────
_load()
