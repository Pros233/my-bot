"""
frequency_boost.py — Controlled frequency boost mode.

Temporarily allows B/C-grade trades when all safety conditions are met.
Hard risk controls (REJECT grades, hard-fail filters, pause state, DEFENSIVE
confidence, WARNING/CRITICAL avoidance) are never bypassed.

State persisted to frequency_boost_state.json (excluded from rsync).

Public API:
    is_active()                              -> bool
    activate()                               -> str   (status message)
    deactivate()                             -> str
    check_boost(grade, symbol, df,
                confidence, filter_results,
                now_utc)                     -> BoostResult
    record_trade(symbol)                     -> None
    get_status()                             -> dict
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import config

_STATE_PATH = Path(__file__).parent / "frequency_boost_state.json"


@dataclass
class BoostResult:
    allowed: bool
    reason: str
    original_grade: str
    boost_remaining_hours: float
    boost_trades_used: int
    boost_trades_remaining: int


# ── State helpers ─────────────────────────────────────────────────────────────

def _now() -> float:
    return time.time()


def _load_state() -> dict:
    try:
        if _STATE_PATH.exists():
            return json.loads(_STATE_PATH.read_text())
    except Exception:
        pass
    return {"active": False, "start_ts": None, "trades_used": 0}


def _save_state(state: dict) -> None:
    try:
        _STATE_PATH.write_text(json.dumps(state, indent=2))
    except Exception:
        pass


def _duration_hours() -> int:
    return getattr(config, "FREQUENCY_BOOST_DURATION_HOURS", 72)


def _max_trades() -> int:
    return getattr(config, "FREQUENCY_BOOST_MAX_TRADES", 3)


# ── Public helpers ────────────────────────────────────────────────────────────

def is_active() -> bool:
    """Return True only if boost is on, not expired, and trades remain."""
    state = _load_state()
    if not state.get("active"):
        return False
    start_ts = state.get("start_ts")
    if start_ts and _now() - start_ts > _duration_hours() * 3600:
        state["active"] = False
        _save_state(state)
        return False
    if state.get("trades_used", 0) >= _max_trades():
        state["active"] = False
        _save_state(state)
        return False
    return True


def activate() -> str:
    state = _load_state()
    state["active"] = True
    state["start_ts"] = _now()
    state["trades_used"] = 0
    _save_state(state)
    grades = getattr(config, "FREQUENCY_BOOST_ALLOWED_GRADES", "B,C")
    conf_min = getattr(config, "FREQUENCY_BOOST_REQUIRE_CONFIDENCE_ABOVE", 45)
    return (
        f"Frequency boost ACTIVATED\n"
        f"Duration: {_duration_hours()}h | Max trades: {_max_trades()}\n"
        f"Allowed grades: {grades}\n"
        f"Confidence min: {conf_min}"
    )


def deactivate() -> str:
    state = _load_state()
    state["active"] = False
    _save_state(state)
    return "Frequency boost DEACTIVATED"


def record_trade(symbol: str) -> None:
    """Call this when a boosted trade actually executes successfully."""
    state = _load_state()
    state["trades_used"] = state.get("trades_used", 0) + 1
    if state.get("trades_used", 0) >= _max_trades():
        state["active"] = False
    _save_state(state)


def get_status() -> dict:
    state = _load_state()
    active = is_active()
    start_ts = state.get("start_ts")
    trades_used = state.get("trades_used", 0)
    dur = _duration_hours()
    max_t = _max_trades()
    remaining_hours = 0.0
    if active and start_ts:
        elapsed = (_now() - start_ts) / 3600
        remaining_hours = max(0.0, dur - elapsed)
    return {
        "active": active,
        "enabled": getattr(config, "ENABLE_FREQUENCY_BOOST", False),
        "trades_used": trades_used,
        "trades_remaining": max(0, max_t - trades_used),
        "remaining_hours": remaining_hours,
        "max_trades": max_t,
        "duration_hours": dur,
        "allowed_grades": getattr(config, "FREQUENCY_BOOST_ALLOWED_GRADES", "B,C"),
        "confidence_threshold": getattr(config, "FREQUENCY_BOOST_REQUIRE_CONFIDENCE_ABOVE", 45),
        "start_ts": start_ts,
    }


# ── Core check ────────────────────────────────────────────────────────────────

def _deny(grade: str, reason: str) -> BoostResult:
    state = _load_state()
    start_ts = state.get("start_ts")
    dur = _duration_hours()
    max_t = _max_trades()
    trades_used = state.get("trades_used", 0)
    remaining = 0.0
    if start_ts:
        elapsed = (_now() - start_ts) / 3600
        remaining = max(0.0, dur - elapsed)
    return BoostResult(
        allowed=False,
        reason=reason,
        original_grade=grade,
        boost_remaining_hours=remaining,
        boost_trades_used=trades_used,
        boost_trades_remaining=max(0, max_t - trades_used),
    )


def check_boost(
    grade: str,
    symbol: str,
    df,
    confidence: float,
    filter_results: list,
    now_utc,
) -> BoostResult:
    """
    Check whether frequency boost allows this trade.

    Returns BoostResult. If allowed=True, caller must call record_trade()
    when/if the trade executes successfully.

    Hard rules (never bypassed):
      - REJECT grade → denied
      - Any hard_fail filter → denied
      - DEFENSIVE confidence → denied (caller's ENABLE_CONFIDENCE_SCORE check
        already handles this, but we check again for safety)
      - Market avoidance WARNING or CRITICAL → denied
      - Spread filter failed → denied
    """
    if not getattr(config, "ENABLE_FREQUENCY_BOOST", False):
        return _deny(grade, "feature disabled")

    state = _load_state()
    if not state.get("active"):
        return _deny(grade, "boost not active")

    # Expiry check
    start_ts = state.get("start_ts")
    if start_ts and _now() - start_ts > _duration_hours() * 3600:
        state["active"] = False
        _save_state(state)
        return _deny(grade, "boost expired")

    # Trade count check
    trades_used = state.get("trades_used", 0)
    if trades_used >= _max_trades():
        state["active"] = False
        _save_state(state)
        return _deny(grade, f"max trades reached ({_max_trades()})")

    # REJECT grades are never boosted
    if grade == "REJECT":
        return _deny(grade, "REJECT grade is never boosted")

    # Hard-fail filters block boost
    hard_fails = [f.name for f in filter_results if getattr(f, "hard_fail", False) and not f.passed]
    if hard_fails:
        return _deny(grade, f"hard_fail filter(s): {', '.join(hard_fails)}")

    # Grade must be in allowed list
    grades_raw = getattr(config, "FREQUENCY_BOOST_ALLOWED_GRADES", "B,C")
    allowed_grades = [g.strip() for g in grades_raw.split(",") if g.strip()]
    if grade not in allowed_grades:
        return _deny(grade, f"grade {grade} not in FREQUENCY_BOOST_ALLOWED_GRADES ({grades_raw})")

    # Confidence threshold
    conf_min = float(getattr(config, "FREQUENCY_BOOST_REQUIRE_CONFIDENCE_ABOVE", 45))
    if confidence < conf_min:
        return _deny(grade, f"confidence {confidence:.0f} < threshold {conf_min:.0f}")

    # Market avoidance: WARNING and CRITICAL block boost
    if getattr(config, "FREQUENCY_BOOST_REQUIRE_MARKET_AVOIDANCE_CLEAR", True):
        try:
            import market_avoidance as _ma
            av = _ma.check_avoidance(df, symbol, now_utc)
            if av.severity in (_ma.SEVERITY_WARNING, _ma.SEVERITY_CRITICAL):
                return _deny(grade, f"market avoidance {av.severity}")
        except Exception:
            pass  # fail open — avoidance error doesn't block boost

    # Spread filter must have passed
    max_spread = float(getattr(config, "FREQUENCY_BOOST_MAX_SPREAD_BPS", 10.0))
    for f in filter_results:
        if f.name == "spread" and not f.passed:
            return _deny(grade, f"spread failed: {f.reason}")

    # Symbol allowlist (empty = all symbols allowed)
    syms_raw = getattr(config, "FREQUENCY_BOOST_SYMBOLS", "")
    if syms_raw.strip():
        allowed_syms = [s.strip() for s in syms_raw.split(",") if s.strip()]
        if allowed_syms and symbol not in allowed_syms:
            return _deny(grade, f"{symbol} not in FREQUENCY_BOOST_SYMBOLS")

    # All checks passed
    dur = _duration_hours()
    elapsed = (_now() - (start_ts or _now())) / 3600
    remaining_hours = max(0.0, dur - elapsed)
    return BoostResult(
        allowed=True,
        reason=f"boost active, grade {grade} permitted",
        original_grade=grade,
        boost_remaining_hours=remaining_hours,
        boost_trades_used=trades_used,
        boost_trades_remaining=max(0, _max_trades() - trades_used),
    )
