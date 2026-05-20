"""
confidence_score.py — Daily system confidence scoring (0-100).

Aggregates multiple system health signals into one actionable score that
adjusts trading behavior dynamically.

Components (weights sum to 1.0):
  engine_alignment    0.25 — avg engine health score from engine_ranker
  regime_clarity      0.20 — how unambiguous the current market state is
  portfolio_health    0.20 — from portfolio_brain
  volatility_quality  0.15 — ATR in the tradeable "sweet spot" (not too high/low)
  historical_exp      0.20 — recent expectancy across live engines

Behavior thresholds:
  NORMAL    score ≥ 70  — full operation
  CAUTIOUS  score 40-69 — risk_scale 0.75, tighten grade 1 level
  DEFENSIVE score < 40  — pause new entries

Public API
----------
    compute_confidence(market_states, balance, open_positions) → float
    get_confidence_state(score)  → "NORMAL" | "CAUTIOUS" | "DEFENSIVE"
    risk_scale_factor(score)     → float  (0.5 – 1.0)
    effective_min_grade(base_grade, score) → str
    get_confidence_summary(market_states, balance, open_positions) → dict

Never raises.
"""
from __future__ import annotations

import math
from typing import Optional

import logger

# ── Thresholds ─────────────────────────────────────────────────────────────────

NORMAL    = "NORMAL"
CAUTIOUS  = "CAUTIOUS"
DEFENSIVE = "DEFENSIVE"

_NORMAL_THRESHOLD    = 70.0
_CAUTIOUS_THRESHOLD  = 40.0   # below this → DEFENSIVE


# ── Component scorers ─────────────────────────────────────────────────────────

def _engine_alignment_score() -> float:
    """0-100 from average engine health scores."""
    try:
        import engine_ranker as er
        ranked = er.rank_engines(days=30)
        if not ranked:
            return 50.0
        scores = [r.get("score", 50.0) for r in ranked if r.get("trades", 0) >= 3]
        if not scores:
            return 50.0
        return round(sum(scores) / len(scores), 1)
    except Exception:
        return 50.0


def _regime_clarity_score(market_states: list) -> float:
    """
    0-100: penalise ambiguous / risk-off states.
    Uses list of market_state.MarketState objects from the current scan cycle.
    """
    try:
        if not market_states:
            return 50.0

        # States that are clear and tradeable → high score
        _CLARITY = {
            "RANGING":               90.0,
            "MEAN_REV_FAVORABLE":    85.0,
            "WEAK_TREND":            70.0,
            "STRONG_TREND":          65.0,
            "MOMENTUM_EXPANSION":    60.0,
            "VOL_COMPRESSION":       55.0,
            "VOL_EXPANSION":         35.0,
            "LOW_LIQUIDITY":         25.0,
            "RISK_OFF":              15.0,
            "UNKNOWN":               40.0,
        }
        states = [getattr(ms, "state", "UNKNOWN") for ms in market_states if ms is not None]
        if not states:
            return 50.0
        scores = [_CLARITY.get(s, 40.0) for s in states]
        return round(sum(scores) / len(scores), 1)
    except Exception:
        return 50.0


def _portfolio_health_score(balance: float, open_positions: list) -> float:
    """0-100 from portfolio_brain health score."""
    try:
        import portfolio_brain as pb
        return pb.compute_health_score(open_positions, balance)
    except Exception:
        return 50.0


def _volatility_quality_score(market_states: list) -> float:
    """
    0-100: reward moderate volatility (ATR% in 0.3-1.5%), penalise extremes.
    """
    try:
        atr_pcts = [getattr(ms, "atr_pct", None) for ms in (market_states or []) if ms]
        atr_pcts = [v for v in atr_pcts if v is not None]
        if not atr_pcts:
            return 50.0
        avg_atr = sum(atr_pcts) / len(atr_pcts)
        # Sweet spot: 0.3 – 1.5%
        if 0.3 <= avg_atr <= 1.5:
            return 100.0
        elif avg_atr < 0.1:
            return 20.0  # dead market
        elif avg_atr > 4.0:
            return 10.0  # chaotic
        elif avg_atr < 0.3:
            return 40.0 + (avg_atr - 0.1) / 0.2 * 60.0
        else:  # 1.5 - 4.0
            return max(10.0, 100.0 - (avg_atr - 1.5) / 2.5 * 90.0)
    except Exception:
        return 50.0


def _historical_expectancy_score() -> float:
    """0-100 from recent (30d) live expectancy across all engines."""
    try:
        import engine_performance as ep
        stats_list = [ep.get_engine_stats(e, days=30) for e in ep.ENGINE_NAMES]
        valid = [s for s in stats_list if s.get("trades", 0) >= 5]
        if not valid:
            return 50.0
        avg_exp = sum(s["expectancy"] for s in valid) / len(valid)
        # exp=+0.02 → 100, 0 → 50, -0.01 → 0
        score = 50.0 + avg_exp * 2500.0
        return round(max(0.0, min(100.0, score)), 1)
    except Exception:
        return 50.0


# ── Public API ────────────────────────────────────────────────────────────────

def compute_confidence(
    market_states:  list  = None,
    balance:        float = 0.0,
    open_positions: list  = None,
) -> float:
    """
    Compute system confidence 0-100.
    All components are fail-open (return 50 on error).
    """
    try:
        if market_states   is None: market_states   = []
        if open_positions  is None: open_positions  = []

        eng   = _engine_alignment_score()
        reg   = _regime_clarity_score(market_states)
        port  = _portfolio_health_score(balance, open_positions)
        vol   = _volatility_quality_score(market_states)
        hist  = _historical_expectancy_score()

        score = (
            0.25 * eng  +
            0.20 * reg  +
            0.20 * port +
            0.15 * vol  +
            0.20 * hist
        )
        return round(max(0.0, min(100.0, score)), 1)
    except Exception as exc:
        logger.log_warning(f"confidence_score.compute_confidence error: {exc}")
        return 50.0


def get_confidence_state(score: float) -> str:
    """Map score to named state."""
    if score >= _NORMAL_THRESHOLD:
        return NORMAL
    if score >= _CAUTIOUS_THRESHOLD:
        return CAUTIOUS
    return DEFENSIVE


def risk_scale_factor(score: float) -> float:
    """
    Return risk scaling multiplier.
    NORMAL=1.0, CAUTIOUS=0.75, DEFENSIVE=0.50
    """
    state = get_confidence_state(score)
    return {NORMAL: 1.0, CAUTIOUS: 0.75, DEFENSIVE: 0.50}.get(state, 1.0)


def effective_min_grade(base_grade: str, score: float) -> str:
    """Tighten grade floor based on confidence state."""
    _grade_rank = {"A+": 0, "A": 1, "B": 2, "C": 3}
    _rank_grade = {v: k for k, v in _grade_rank.items()}
    state = get_confidence_state(score)
    base_rank = _grade_rank.get(base_grade, 2)
    if state == CAUTIOUS:
        new_rank = max(0, base_rank - 1)
    elif state == DEFENSIVE:
        new_rank = max(0, base_rank - 2)
    else:
        new_rank = base_rank
    return _rank_grade.get(new_rank, base_grade)


def get_confidence_summary(
    market_states:  list  = None,
    balance:        float = 0.0,
    open_positions: list  = None,
) -> dict:
    """Return full summary for dashboard / Telegram."""
    try:
        if market_states   is None: market_states   = []
        if open_positions  is None: open_positions  = []

        eng  = _engine_alignment_score()
        reg  = _regime_clarity_score(market_states)
        port = _portfolio_health_score(balance, open_positions)
        vol  = _volatility_quality_score(market_states)
        hist = _historical_expectancy_score()

        score = round(max(0.0, min(100.0,
            0.25*eng + 0.20*reg + 0.20*port + 0.15*vol + 0.20*hist
        )), 1)
        state = get_confidence_state(score)

        return {
            "score":       score,
            "state":       state,
            "risk_scale":  risk_scale_factor(score),
            "components": {
                "engine_alignment":   round(eng,  1),
                "regime_clarity":     round(reg,  1),
                "portfolio_health":   round(port, 1),
                "volatility_quality": round(vol,  1),
                "historical_exp":     round(hist, 1),
            },
            "thresholds": {
                "normal":    _NORMAL_THRESHOLD,
                "cautious":  _CAUTIOUS_THRESHOLD,
            },
        }
    except Exception as exc:
        logger.log_warning(f"confidence_score.get_confidence_summary error: {exc}")
        return {"score": 50.0, "state": NORMAL, "risk_scale": 1.0, "components": {}}
