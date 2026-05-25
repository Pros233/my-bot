"""
trade_grader.py — Trade quality grading system.

Generates a grade (A+/A/B/C/REJECT) for every candidate trade before
execution, based on regime quality, trend alignment, volatility, session,
signal score, and filter results.

Grades:
  A+  — exceptional setup, all conditions aligned
  A   — good setup, minor weaknesses acceptable
  B   — below average, marginal edge (skipped by default)
  C   — poor setup, likely marginal or no edge
  REJECT — hard failure: news window, BTC bearish, extreme candle, etc.

Config:
  MIN_TRADE_GRADE = B   → A+, A, and B grades execute (default)

Grade is logged with full reasoning for every candidate, whether executed or not.
Never raises — returns ("REJECT", [...]) on any internal error.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

import config
import logger
from trade_filters import FilterResult, has_hard_fail, total_grade_penalty, hour_to_session

if TYPE_CHECKING:
    pass


# ── Grade ordering ─────────────────────────────────────────────────────────────

_GRADE_RANK: dict[str, int] = {
    "A+":     0,
    "A":      1,
    "B":      2,
    "C":      3,
    "REJECT": 4,
}

_SCORE_TO_GRADE = [
    (13, "A+"),
    (9,  "A"),
    (5,  "B"),
    (2,  "C"),
]


def grade_rank(grade: str) -> int:
    """Lower is better. A+=0, A=1, B=2, C=3, REJECT=4."""
    return _GRADE_RANK.get(grade, 4)


def grade_passes_minimum(grade: str) -> bool:
    """True if grade meets or beats MIN_TRADE_GRADE config."""
    min_grade = getattr(config, "MIN_TRADE_GRADE", "A").strip()
    return grade_rank(grade) <= grade_rank(min_grade)


# ── Scoring components ────────────────────────────────────────────────────────

def _score_regime(trend: str, vol: str) -> tuple[int, str]:
    """Return (points, reason) for regime quality."""
    if trend == "TRENDING" and vol == "NORMAL":
        return 4, "TRENDING+NORMAL (best)"
    if trend == "TRENDING":
        return 2, f"TRENDING+{vol}"
    if trend == "RANGING" and vol in ("NORMAL", "LOW_VOLATILITY"):
        return 3, "RANGING+NORMAL (RMR ideal)"
    if trend == "RANGING":
        return 1, f"RANGING+{vol}"
    if vol == "HIGH_VOLATILITY":
        return -1, "HIGH_VOLATILITY (caution)"
    return 0, f"{trend}+{vol}"


def _score_adx(adx: float) -> tuple[int, str]:
    """Return (points, reason) for ADX quality."""
    if 25 <= adx < 40:
        return 3, f"ADX={adx:.1f} (strong trend)"
    if 20 <= adx < 25:
        return 2, f"ADX={adx:.1f} (moderate trend)"
    if 40 <= adx < 55:
        return 1, f"ADX={adx:.1f} (very strong, risk of reversal)"
    if adx >= 55:
        return -1, f"ADX={adx:.1f} (extreme — reversal risk)"
    # ADX < 20 → pure ranging market, ideal for RMR setups (neutral, not a penalty)
    return 0, f"ADX={adx:.1f} (ranging — RMR ideal)"


def _score_signal(score_pct: float) -> tuple[int, str]:
    """Return (points, reason) for consensus score quality."""
    if score_pct >= 80:
        return 3, f"score={score_pct:.0f}% (strong)"
    if score_pct >= 65:
        return 2, f"score={score_pct:.0f}% (good)"
    if score_pct >= 50:
        return 1, f"score={score_pct:.0f}% (moderate)"
    if score_pct > 0:
        return 0, f"score={score_pct:.0f}% (weak)"
    return 0, "score=N/A (RMR setup)"


def _score_session(now_utc: datetime) -> tuple[int, str]:
    """Return (points, reason) for trading session quality."""
    session = hour_to_session(now_utc.hour)
    weights = {
        "NY/London": 3,
        "London":    2,
        "New York":  2,
        "Asia":      1,    # softened: was 0 — Asia can produce valid RMR setups
        "Off-hours": -1,
    }
    pts = weights.get(session, 0)
    return pts, f"session={session}"


def _score_bb_expansion(filter_results: list[FilterResult]) -> tuple[int, str]:
    """Bonus points if BB compression filter detected squeeze breakout."""
    for r in filter_results:
        if r.name == "bb_compression" and r.passed and "squeeze breakout" in r.reason:
            return 2, "squeeze breakout (BB expanding)"
    return 0, ""


def _score_atr(atr_pct: float) -> tuple[int, str]:
    """Return (points, reason) for ATR quality."""
    if 0.3 <= atr_pct < 0.7:
        return 2, f"ATR={atr_pct:.2f}% (ideal volatility)"
    if 0.7 <= atr_pct < 1.2:
        return 1, f"ATR={atr_pct:.2f}% (moderate)"
    if atr_pct >= 1.2:
        return -1, f"ATR={atr_pct:.2f}% (high volatility)"
    return 0, f"ATR={atr_pct:.2f}% (low)"


# ── Main grader ───────────────────────────────────────────────────────────────

def grade_trade(
    symbol: str,
    trend: str,
    vol: str,
    adx: float,
    atr_pct: float,
    score_pct: float,
    now_utc: datetime,
    filter_results: list[FilterResult],
) -> tuple[str, int, list[str]]:
    """
    Compute trade quality grade.

    Returns:
        grade:   "A+" | "A" | "B" | "C" | "REJECT"
        score:   raw integer score (informational)
        reasons: list of score components (for logging / Telegram)

    Never raises — returns ("REJECT", 0, [reason]) on error.
    """
    try:
        reasons: list[str] = []
        total_score = 0

        # ── Hard fail from any filter → immediate REJECT ──────────────────────
        if has_hard_fail(filter_results):
            hard = [r for r in filter_results if r.hard_fail]
            hard_reasons = [r.reason for r in hard]
            return "REJECT", -99, [f"HARD FAIL: {r}" for r in hard_reasons]

        # ── Point-based scoring ───────────────────────────────────────────────
        pts, reason = _score_regime(trend, vol)
        total_score += pts
        reasons.append(f"regime={pts:+d} ({reason})")

        pts, reason = _score_adx(adx)
        total_score += pts
        reasons.append(f"adx={pts:+d} ({reason})")

        pts, reason = _score_signal(score_pct)
        total_score += pts
        reasons.append(f"signal={pts:+d} ({reason})")

        pts, reason = _score_session(now_utc)
        total_score += pts
        reasons.append(f"session={pts:+d} ({reason})")

        pts, reason = _score_atr(atr_pct)
        total_score += pts
        reasons.append(f"atr={pts:+d} ({reason})")

        pts, reason = _score_bb_expansion(filter_results)
        if pts:
            total_score += pts
            reasons.append(f"bb={pts:+d} ({reason})")

        # ── Subtract filter penalties ─────────────────────────────────────────
        penalty = total_grade_penalty(filter_results)
        if penalty:
            total_score -= penalty
            reasons.append(f"filter_penalty=-{penalty}")

        # ── Map score to grade ────────────────────────────────────────────────
        grade = "C"
        for threshold, g in _SCORE_TO_GRADE:
            if total_score >= threshold:
                grade = g
                break

        logger.log_info(
            f"GRADE | {symbol} | {grade} | score={total_score} | "
            + " | ".join(reasons)
        )
        return grade, total_score, reasons

    except Exception as exc:
        # Fail-open: grader error must never block a trade
        logger.log_warning(f"trade_grader error for {symbol} (non-critical, allowing trade): {exc}")
        return "A", 0, [f"grader error (fail-open): {exc}"]


# ── Adaptive grade floor ───────────────────────────────────────────────────────

def adaptive_min_grade(
    consecutive_losses: int,
    daily_loss_pct: float,
    consecutive_wins: int,
    weekly_pnl: float,
) -> str:
    """
    Compute the effective minimum grade, tightening after drawdowns and
    relaxing after stable performance. Never increases risk — only affects
    which grades are allowed to trade.

    Config base: MIN_TRADE_GRADE (default "A")
    """
    if not getattr(config, "ENABLE_ADAPTIVE_GRADES", False):
        return getattr(config, "MIN_TRADE_GRADE", "A")

    base = getattr(config, "MIN_TRADE_GRADE", "A")
    base_rank = grade_rank(base)

    # Tighten conditions
    if consecutive_losses >= 4 or daily_loss_pct <= -0.75:
        # Severe drawdown → only A+
        effective_rank = max(0, base_rank - 2)
    elif consecutive_losses >= 2 or daily_loss_pct <= -0.50:
        # Moderate drawdown → tighten by one
        effective_rank = max(0, base_rank - 1)
    # Relax conditions (never below config base)
    elif consecutive_wins >= 3 and weekly_pnl > 0:
        effective_rank = min(base_rank + 1, 3)   # relax to B at most
    else:
        effective_rank = base_rank

    # Rank → grade string
    rank_to_grade = {v: k for k, v in _GRADE_RANK.items() if k != "REJECT"}
    result = rank_to_grade.get(effective_rank, base)

    if result != base:
        logger.log_info(
            f"ADAPTIVE | grade floor adjusted: {base} → {result} "
            f"(consec_losses={consecutive_losses}, daily_pnl={daily_loss_pct:.1f}%, "
            f"consec_wins={consecutive_wins})"
        )

    return result
