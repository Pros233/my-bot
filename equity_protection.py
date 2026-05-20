"""
equity_protection.py — Equity curve protection layer.

Monitors the equity curve and tightens grade requirements when the
curve weakens.  Returns to normal operation automatically when the
equity curve recovers.

Protection states:
  normal     — standard operation; use config MIN_TRADE_GRADE
  selective  — minor drawdown; tighten by one grade level
  defensive  — significant drawdown; require A or better

Config:
  ENABLE_EQUITY_PROTECTION=true  (default: false)

Rules:
  * NEVER increases risk
  * NEVER enables martingale or averaging down
  * NEVER bypasses pause protection or hard fails
  * Fail-open: any error → returns "normal"

Public API
----------
    get_state()                              → "normal" | "selective" | "defensive"
    effective_min_grade(base_grade)          → adjusted grade string
    get_summary()                            → dict for dashboard/Telegram
"""
from __future__ import annotations

import config
import logger


# ── Thresholds ────────────────────────────────────────────────────────────────

# Selective mode triggers
_SELECTIVE_CONSEC_LOSSES = 3       # ≥ N consecutive losses
_SELECTIVE_DAILY_LOSS_PCT = -0.40  # daily PnL < -0.40% of balance
_SELECTIVE_DD_PCT = 3.0            # max drawdown > 3%

# Defensive mode triggers (stricter)
_DEFENSIVE_CONSEC_LOSSES = 5
_DEFENSIVE_DAILY_LOSS_PCT = -0.75
_DEFENSIVE_DD_PCT = 6.0

# Recovery: return to normal when
_RECOVERY_CONSEC_WINS = 2
_RECOVERY_WEEKLY_PNL_PCT = 0.20   # weekly PnL > +0.20% of balance


# ── Grade ordering ────────────────────────────────────────────────────────────

_GRADE_RANK  = {"A+": 0, "A": 1, "B": 2, "C": 3}
_RANK_GRADE  = {v: k for k, v in _GRADE_RANK.items()}


def _tighten(grade: str, levels: int) -> str:
    rank = _GRADE_RANK.get(grade, 2)
    return _RANK_GRADE.get(max(0, rank - levels), "A+")


# ── State calculation ─────────────────────────────────────────────────────────

def get_state(balance: float = 0.0) -> str:
    """
    Return the current equity protection state.
    Reads live performance metrics from performance.py.

    Returns "normal" | "selective" | "defensive"
    """
    if not getattr(config, "ENABLE_EQUITY_PROTECTION", False):
        return "normal"

    try:
        import performance
        from datetime import datetime, timezone

        consec_losses = performance.consecutive_losses()
        max_dd        = performance.max_drawdown_pct()

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        iso   = datetime.now(timezone.utc).isocalendar()
        week  = f"{iso[0]}-W{iso[1]:02d}"
        daily_pnl  = performance.daily_pnl(today)
        weekly_pnl = performance.weekly_pnl(week)
        daily_pct  = (daily_pnl / balance * 100) if balance > 0 else 0.0
        weekly_pct = (weekly_pnl / balance * 100) if balance > 0 else 0.0

        consec_wins = performance.consecutive_wins()

        # ── Recovery check (override everything) ──────────────────────────────
        if (consec_wins >= _RECOVERY_CONSEC_WINS
                and weekly_pct >= _RECOVERY_WEEKLY_PNL_PCT):
            return "normal"

        # ── Defensive ──────────────────────────────────────────────────────────
        if (consec_losses >= _DEFENSIVE_CONSEC_LOSSES
                or daily_pct <= _DEFENSIVE_DAILY_LOSS_PCT
                or max_dd >= _DEFENSIVE_DD_PCT):
            return "defensive"

        # ── Selective ──────────────────────────────────────────────────────────
        if (consec_losses >= _SELECTIVE_CONSEC_LOSSES
                or daily_pct <= _SELECTIVE_DAILY_LOSS_PCT
                or max_dd >= _SELECTIVE_DD_PCT):
            return "selective"

        return "normal"

    except Exception as exc:
        logger.log_warning(f"equity_protection.get_state error (fail-safe normal): {exc}")
        return "normal"


def effective_min_grade(base_grade: str, balance: float = 0.0) -> str:
    """
    Return effective minimum grade after applying equity protection.
    Never loosens below base_grade.
    """
    state = get_state(balance)
    if state == "defensive":
        # Tighten by 2 levels (e.g. B → A+)
        return _tighten(base_grade, 2)
    elif state == "selective":
        # Tighten by 1 level (e.g. B → A)
        return _tighten(base_grade, 1)
    return base_grade


def get_summary(balance: float = 0.0) -> dict:
    """Return a summary dict for dashboard / Telegram."""
    try:
        import performance
        from datetime import datetime, timezone

        state = get_state(balance)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        iso   = datetime.now(timezone.utc).isocalendar()
        week  = f"{iso[0]}-W{iso[1]:02d}"
        base  = getattr(config, "MIN_TRADE_GRADE", "A")

        return {
            "state":              state,
            "enabled":            getattr(config, "ENABLE_EQUITY_PROTECTION", False),
            "consecutive_losses": performance.consecutive_losses(),
            "consecutive_wins":   performance.consecutive_wins(),
            "max_drawdown_pct":   performance.max_drawdown_pct(),
            "daily_pnl":          performance.daily_pnl(today),
            "weekly_pnl":         performance.weekly_pnl(week),
            "base_grade":         base,
            "effective_grade":    effective_min_grade(base, balance),
            "tightening_active":  state != "normal",
        }
    except Exception as exc:
        logger.log_warning(f"equity_protection.get_summary error: {exc}")
        return {
            "state": "normal", "enabled": False,
            "tightening_active": False,
            "base_grade": getattr(config, "MIN_TRADE_GRADE", "A"),
            "effective_grade": getattr(config, "MIN_TRADE_GRADE", "A"),
        }
