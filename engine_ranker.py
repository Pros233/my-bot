"""
engine_ranker.py — Dynamic engine ranking and adaptive grade weighting.

Scores each engine on a 0-100 scale using expectancy, profit factor,
drawdown, stability, and sample size.  Used to:
  1. Log engine leaderboard each cycle (informational)
  2. Provide adaptive minimum grade requirements per engine
  3. Temporarily suppress engines with strongly negative expectancy

Config flags (all default off):
  ENABLE_ADAPTIVE_ENGINE_WEIGHTING — influence selection priority by score
  ENABLE_AUTO_DISABLE_ENGINES      — disable engines below threshold

Never raises.  Returns safe defaults on any error.
"""
from __future__ import annotations

import time
from typing import Optional

import config
import engine_performance as ep
import logger

# ── Scoring weights ───────────────────────────────────────────────────────────

_MIN_SAMPLE          = 5     # trades required before scoring (below → neutral score)
_AUTO_DISABLE_SAMPLE = 10    # trades required before auto-disable fires
_AUTO_DISABLE_EXPIRY = 6.0   # hours before auto-disabled engine is re-enabled
_NEGATIVE_EXPIRY_THRESHOLD = -0.0050   # expectancy < -0.5% USDT → candidate for disable

# ── Auto-disable state (in-memory; resets on bot restart) ─────────────────────

_disabled_until: dict[str, float] = {}   # engine → monotonic timestamp


def _is_auto_disabled(engine: str) -> bool:
    """True if engine is within its auto-disable window."""
    until = _disabled_until.get(engine, 0.0)
    return time.monotonic() < until


def _auto_disable(engine: str) -> None:
    """Disable engine for _AUTO_DISABLE_EXPIRY hours."""
    _disabled_until[engine] = time.monotonic() + _AUTO_DISABLE_EXPIRY * 3600
    logger.log_info(
        f"ENGINE_RANKER | auto-disabled {engine} for {_AUTO_DISABLE_EXPIRY:.0f}h "
        f"(negative expectancy, sample ≥ {_AUTO_DISABLE_SAMPLE})"
    )


# ── Health score ──────────────────────────────────────────────────────────────

def engine_health_score(stats: dict) -> float:
    """
    0-100 health score for a single engine stat dict.

    Weights:
      expectancy    35% — primary profitability signal
      profit factor 25% — consistency of wins vs losses
      max drawdown  20% — risk-adjusted stability
      sharpe        10% — volatility-adjusted return
      sample bonus  10% — confidence / significance
    """
    n = stats.get("trades", 0)
    if n < _MIN_SAMPLE:
        return 50.0   # neutral — not enough data to judge

    # ── Expectancy score (0-35) ──
    exp = stats.get("expectancy", 0.0)
    # Map expectancy linearly: -0.01 USDT → 0, 0 → 17.5, +0.02 USDT → 35
    exp_score = max(0.0, min(35.0, (exp + 0.01) / 0.03 * 35.0))

    # ── Profit factor score (0-25) ──
    pf = min(stats.get("profit_factor", 0.0), 4.0)   # cap at 4
    pf_score = max(0.0, min(25.0, (pf - 0.5) / 3.5 * 25.0))

    # ── Drawdown score (0-20) ── lower drawdown = better
    dd = stats.get("max_drawdown_pct", 100.0)
    dd_score = max(0.0, min(20.0, (1 - dd / 30.0) * 20.0))   # 0% dd → 20, 30%+ dd → 0

    # ── Sharpe score (0-10) ──
    sh = max(-2.0, min(2.0, stats.get("sharpe_ratio", 0.0)))
    sh_score = (sh + 2.0) / 4.0 * 10.0

    # ── Sample bonus (0-10) ──
    sample_score = min(10.0, n / 20.0 * 10.0)   # full bonus at 20+ trades

    return round(exp_score + pf_score + dd_score + sh_score + sample_score, 1)


# ── Ranking ───────────────────────────────────────────────────────────────────

def rank_engines(days: int = 60) -> list[dict]:
    """
    Return engines sorted by health score descending.

    Each item: {engine, score, trades, expectancy, profit_factor,
                win_rate, total_pnl, max_drawdown_pct, disabled}
    """
    try:
        all_stats = ep.get_all_stats(days=days)
        ranked = []
        for engine, stats in all_stats.items():
            score = engine_health_score(stats)
            disabled = _is_auto_disabled(engine)
            ranked.append({
                "engine":          engine,
                "score":           score,
                "trades":          stats["trades"],
                "expectancy":      stats["expectancy"],
                "profit_factor":   stats["profit_factor"],
                "win_rate":        stats["win_rate"],
                "total_pnl":       stats["total_pnl"],
                "max_drawdown_pct": stats["max_drawdown_pct"],
                "sharpe_ratio":    stats["sharpe_ratio"],
                "disabled":        disabled,
            })
        return sorted(ranked, key=lambda x: (-x["score"], x["engine"]))
    except Exception as exc:
        logger.log_warning(f"engine_ranker.rank_engines error (non-critical): {exc}")
        return []


def best_engine(days: int = 60) -> Optional[str]:
    """Return name of highest-scoring engine with sufficient sample."""
    ranked = rank_engines(days=days)
    for r in ranked:
        if r["trades"] >= _MIN_SAMPLE and not r["disabled"]:
            return r["engine"]
    return None


def worst_engine(days: int = 60) -> Optional[str]:
    """Return name of lowest-scoring engine with sufficient sample."""
    ranked = rank_engines(days=days)
    for r in reversed(ranked):
        if r["trades"] >= _MIN_SAMPLE and not r["disabled"]:
            return r["engine"]
    return None


# ── Adaptive grade requirement ────────────────────────────────────────────────

def effective_min_grade_for_engine(engine: str, days: int = 60) -> str:
    """
    Return the minimum trade grade this engine should require based on its
    recent performance.  Never loosens below config MIN_TRADE_GRADE.

    Rules (ENABLE_ADAPTIVE_ENGINE_WEIGHTING must be true):
      score ≥ 70  → allow B (strong engine)
      score 50-69 → use config default
      score 30-49 → require A  (weak engine, tighten)
      score < 30  → require A+ (poor engine, maximally selective)
    """
    base = getattr(config, "MIN_TRADE_GRADE", "A")
    if not getattr(config, "ENABLE_ADAPTIVE_ENGINE_WEIGHTING", False):
        return base

    try:
        stats = ep.get_engine_stats(engine, days=days)
        score = engine_health_score(stats)

        if stats["trades"] < _MIN_SAMPLE:
            return base  # no data — use default

        if score >= 70:
            # strong engine — can accept B (but never looser than config allows)
            _grade_rank = {"A+": 0, "A": 1, "B": 2, "C": 3}
            _rank_grade = {v: k for k, v in _grade_rank.items()}
            base_rank = _grade_rank.get(base, 2)
            return _rank_grade.get(min(base_rank + 1, 2), base)  # max relax to B
        elif score >= 50:
            return base
        elif score >= 30:
            return "A" if base in ("B", "C") else base
        else:
            return "A+"
    except Exception:
        return base


# ── Auto-disable check ────────────────────────────────────────────────────────

def check_auto_disable(days: int = 30) -> None:
    """
    Check each engine for strongly negative expectancy and auto-disable
    if ENABLE_AUTO_DISABLE_ENGINES=true and sample size is sufficient.
    Called once per candle cycle.
    """
    if not getattr(config, "ENABLE_AUTO_DISABLE_ENGINES", False):
        return
    try:
        all_stats = ep.get_all_stats(days=days)
        for engine, stats in all_stats.items():
            if stats["trades"] < _AUTO_DISABLE_SAMPLE:
                continue
            if _is_auto_disabled(engine):
                continue
            if stats["expectancy"] < _NEGATIVE_EXPIRY_THRESHOLD:
                _auto_disable(engine)
    except Exception as exc:
        logger.log_warning(f"engine_ranker.check_auto_disable error (non-critical): {exc}")


def is_engine_allowed(engine: str) -> bool:
    """
    False if engine has been auto-disabled.
    Always returns True if ENABLE_AUTO_DISABLE_ENGINES=false.
    """
    if not getattr(config, "ENABLE_AUTO_DISABLE_ENGINES", False):
        return True
    return not _is_auto_disabled(engine)


def rank_score_multiplier(engine: str, days: int = 60) -> float:
    """
    A 0.5-1.5 multiplier applied to rank_score when adaptive weighting is on.
    Strong engines are prioritised; weak engines are de-prioritised.
    Never removes an engine entirely — that's left to grade requirements.
    """
    if not getattr(config, "ENABLE_ADAPTIVE_ENGINE_WEIGHTING", False):
        return 1.0
    try:
        stats = ep.get_engine_stats(engine, days=days)
        if stats["trades"] < _MIN_SAMPLE:
            return 1.0
        score = engine_health_score(stats)
        # Linear map: score 0 → 0.6, score 50 → 1.0, score 100 → 1.4
        return round(0.6 + (score / 100.0) * 0.8, 2)
    except Exception:
        return 1.0
