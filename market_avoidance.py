"""
market_avoidance.py — Dangerous environment detection.

Detects market conditions where trading is inadvisable and returns grade
floor adjustments.  Never blocks trading entirely — only tightens grades.

Danger conditions detected:
  DEAD_LIQUIDITY     — extremely low volume (< 20% of 20-bar average)
  VOLATILITY_CHAOS   — ATR spike > 3.5× recent average (untradeable whipsaw)
  POST_NEWS_CHOP     — sharp move followed by choppy bars (range collapses)
  EXHAUSTED_TREND    — strong trend with weakening momentum (reversal risk)
  WEEKEND_THIN       — UTC Saturday/Sunday 22:00–08:00 (thin order books)
  SPREAD_EXPLOSION   — detected via unusually large bar wicks (proxy)

Grade floor adjustments:
  CRITICAL → A+ required (2 tighter levels)
  WARNING  → A  required (1 tighter level)
  CAUTION  → no change   (logged only)

Public API
----------
    check_avoidance(df, symbol, now_utc)  → AvoidanceResult
    grade_floor_from_avoidance(result, base_grade) → str
    get_avoidance_summary(df, symbol, now_utc)     → dict

Never raises.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import logger

try:
    import pandas as pd
    import numpy as np
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False


# ── Result types ──────────────────────────────────────────────────────────────

SEVERITY_NONE     = "NONE"
SEVERITY_CAUTION  = "CAUTION"
SEVERITY_WARNING  = "WARNING"
SEVERITY_CRITICAL = "CRITICAL"


@dataclass
class AvoidanceResult:
    severity:    str          = SEVERITY_NONE
    conditions:  list[str]    = field(default_factory=list)
    reasons:     list[str]    = field(default_factory=list)
    grade_delta: int          = 0      # how many grade levels to tighten (0-2)


# ── Detection helpers ─────────────────────────────────────────────────────────

def _check_dead_liquidity(df) -> Optional[tuple[str, str]]:
    """Return (severity, reason) or None."""
    try:
        if "volume" not in df.columns or len(df) < 21:
            return None
        vol_now = float(df["volume"].iloc[-1])
        vol_avg = float(df["volume"].iloc[-21:-1].mean())
        if vol_avg <= 0:
            return None
        ratio = vol_now / vol_avg
        if ratio < 0.20:
            return SEVERITY_CRITICAL, f"dead liquidity vol={ratio:.2f}× avg"
        if ratio < 0.40:
            return SEVERITY_WARNING, f"low liquidity vol={ratio:.2f}× avg"
    except Exception:
        pass
    return None


def _check_volatility_chaos(df, atr_col: str = "atr") -> Optional[tuple[str, str]]:
    """Detect extreme ATR spikes."""
    try:
        if len(df) < 21:
            return None

        if atr_col in df.columns:
            atr_now = float(df[atr_col].iloc[-1])
            atr_avg = float(df[atr_col].iloc[-21:-1].mean())
        else:
            # Estimate from bar range
            highs  = df["high"].astype(float)
            lows   = df["low"].astype(float)
            ranges = highs - lows
            atr_now = float(ranges.iloc[-1])
            atr_avg = float(ranges.iloc[-21:-1].mean())

        if atr_avg <= 0:
            return None
        ratio = atr_now / atr_avg
        if ratio > 3.5:
            return SEVERITY_CRITICAL, f"volatility chaos ATR={ratio:.1f}× avg"
        if ratio > 2.5:
            return SEVERITY_WARNING, f"high volatility ATR={ratio:.1f}× avg"
    except Exception:
        pass
    return None


def _check_post_news_chop(df) -> Optional[tuple[str, str]]:
    """
    Detect a sharp move (>2.5× ATR) in last 3 bars followed by
    indecisive bars (bar range < 0.5× ATR).  Indicates post-news chop.
    """
    try:
        if len(df) < 8:
            return None

        highs  = df["high"].astype(float)
        lows   = df["low"].astype(float)
        ranges = highs - lows
        atr    = float(ranges.iloc[-21:-1].mean()) if len(df) >= 22 else float(ranges.mean())
        if atr <= 0:
            return None

        recent_max_range = float(ranges.iloc[-5:-2].max())
        last_range       = float(ranges.iloc[-1])

        if recent_max_range > 2.5 * atr and last_range < 0.5 * atr:
            return SEVERITY_WARNING, "post-news chop detected"
    except Exception:
        pass
    return None


def _check_exhausted_trend(df) -> Optional[tuple[str, str]]:
    """
    Detect: strong trend (5+ bars same direction) + declining volume.
    Signals trend exhaustion / reversal risk.
    """
    try:
        if len(df) < 8 or "close" not in df.columns:
            return None

        closes = df["close"].astype(float)
        # Count consecutive bars in one direction
        diffs  = closes.diff().iloc[-6:]
        signs  = diffs.apply(lambda x: 1 if x > 0 else -1)
        consec = 0
        last   = signs.iloc[-1]
        for s in reversed(signs.tolist()):
            if s == last:
                consec += 1
            else:
                break

        if consec < 5:
            return None

        # Volume declining?
        if "volume" not in df.columns or len(df) < 6:
            return None
        vols = df["volume"].astype(float).iloc[-5:]
        if float(vols.iloc[-1]) < float(vols.iloc[0]) * 0.6:
            return SEVERITY_CAUTION, f"exhausted trend ({consec} bars, declining vol)"
    except Exception:
        pass
    return None


def _check_weekend_thin(now_utc: datetime) -> Optional[tuple[str, str]]:
    """Detect weekend thin liquidity window."""
    try:
        wd   = now_utc.weekday()   # 5=Sat, 6=Sun
        hour = now_utc.hour
        # Sat 22:00 → Sun 08:00 and Sun 22:00 → Mon 08:00
        if (wd == 5 and hour >= 22) or \
           (wd == 6 and (hour < 8 or hour >= 22)) or \
           (wd == 0 and hour < 8):
            return SEVERITY_CAUTION, "weekend thin liquidity window"
    except Exception:
        pass
    return None


def _check_spread_explosion(df) -> Optional[tuple[str, str]]:
    """
    Use wick ratio as a spread proxy.  If wicks consistently > 60% of
    bar range → spread explosion / noisy order book.
    """
    try:
        if len(df) < 4 or not all(c in df.columns for c in ("open","high","low","close")):
            return None

        opens  = df["open"].astype(float).iloc[-4:]
        highs  = df["high"].astype(float).iloc[-4:]
        lows   = df["low"].astype(float).iloc[-4:]
        closes = df["close"].astype(float).iloc[-4:]

        wick_ratios = []
        for i in range(len(opens)):
            bar_range = float(highs.iloc[i] - lows.iloc[i])
            if bar_range <= 0:
                continue
            body = abs(float(closes.iloc[i] - opens.iloc[i]))
            wick = bar_range - body
            wick_ratios.append(wick / bar_range)

        if not wick_ratios:
            return None
        avg_wick_ratio = sum(wick_ratios) / len(wick_ratios)
        if avg_wick_ratio > 0.75:
            return SEVERITY_WARNING, f"spread explosion (avg wick {avg_wick_ratio*100:.0f}% of range)"
        if avg_wick_ratio > 0.60:
            return SEVERITY_CAUTION, f"elevated spreads (avg wick {avg_wick_ratio*100:.0f}% of range)"
    except Exception:
        pass
    return None


# ── Public API ────────────────────────────────────────────────────────────────

_SEVERITY_RANK = {
    SEVERITY_NONE:     0,
    SEVERITY_CAUTION:  1,
    SEVERITY_WARNING:  2,
    SEVERITY_CRITICAL: 3,
}
_GRADE_DELTA = {
    SEVERITY_NONE:     0,
    SEVERITY_CAUTION:  0,
    SEVERITY_WARNING:  1,
    SEVERITY_CRITICAL: 2,
}


def check_avoidance(
    df,
    symbol: str = "",
    now_utc: Optional[datetime] = None,
) -> AvoidanceResult:
    """
    Run all avoidance checks and return combined AvoidanceResult.
    Fail-open: returns SEVERITY_NONE on any error.
    """
    result = AvoidanceResult()
    try:
        if now_utc is None:
            now_utc = datetime.now(timezone.utc)

        checks = [
            _check_dead_liquidity(df),
            _check_volatility_chaos(df),
            _check_post_news_chop(df),
            _check_exhausted_trend(df),
            _check_weekend_thin(now_utc),
            _check_spread_explosion(df),
        ]

        worst_rank = 0
        for check in checks:
            if check is None:
                continue
            sev, reason = check
            rank = _SEVERITY_RANK.get(sev, 0)
            result.conditions.append(sev)
            result.reasons.append(reason)
            if rank > worst_rank:
                worst_rank = rank
                result.severity = sev

        result.grade_delta = _GRADE_DELTA.get(result.severity, 0)

        if result.severity != SEVERITY_NONE:
            logger.log_info(
                f"AVOIDANCE | {symbol} | {result.severity} | "
                + "; ".join(result.reasons)
            )

    except Exception as exc:
        logger.log_warning(f"market_avoidance.check_avoidance error: {exc}")

    return result


def grade_floor_from_avoidance(result: AvoidanceResult, base_grade: str = "B") -> str:
    """
    Tighten grade floor based on avoidance severity.
    CRITICAL → 2 levels tighter, WARNING → 1 level tighter.
    """
    try:
        _grade_rank = {"A+": 0, "A": 1, "B": 2, "C": 3}
        _rank_grade = {v: k for k, v in _grade_rank.items()}
        base_rank   = _grade_rank.get(base_grade, 2)
        new_rank    = max(0, base_rank - result.grade_delta)
        return _rank_grade.get(new_rank, base_grade)
    except Exception:
        return base_grade


def get_avoidance_summary(
    df,
    symbol: str = "",
    now_utc: Optional[datetime] = None,
) -> dict:
    """Return dict for dashboard / Telegram."""
    try:
        result = check_avoidance(df, symbol, now_utc)
        return {
            "severity":    result.severity,
            "conditions":  result.conditions,
            "reasons":     result.reasons,
            "grade_delta": result.grade_delta,
            "trading_ok":  result.severity != SEVERITY_CRITICAL,
        }
    except Exception as exc:
        return {"severity": SEVERITY_NONE, "error": str(exc)}
