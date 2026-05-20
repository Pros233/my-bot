"""
engine_governor.py — Engine promotion / demotion tier system.

Tiers:
  TRUSTED   — strong expectancy, low drawdown, stable WR, ≥ 15 trades
               → higher priority, can execute B-grade setups, 1.3× rank multiplier
  NEUTRAL   — default / insufficient data
               → standard config grade floor, 1.0× rank multiplier
  PROBATION — poor expectancy, excessive drawdown, or ≥ 5 consecutive losses
               → requires A/A+ only, 0.7× rank multiplier

Thresholds (configurable via constants):
  TRUSTED:   score ≥ 68, trades ≥ 15, max_dd < 8%, consec_losses_current < 3
  PROBATION: score < 35 OR max_dd ≥ 12% OR consec_losses_current ≥ 5

Tier history is persisted to JSON so it survives bot restarts.
Telegram notifications are returned as strings for main.py to send.

Public API
----------
    get_tier(engine)             → "TRUSTED" | "NEUTRAL" | "PROBATION"
    check_all_tiers()            → list[str]  (notification messages for tier changes)
    grade_floor_for_tier(tier)   → str  (min grade string)
    rank_multiplier_for_tier(tier) → float
    get_tier_summary()           → dict  (all engines with tiers + history)

Never raises.
"""
from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import engine_performance as ep
import engine_ranker as er
import logger

# ── Tier constants ────────────────────────────────────────────────────────────

TRUSTED   = "TRUSTED"
NEUTRAL   = "NEUTRAL"
PROBATION = "PROBATION"

_TRUSTED_MIN_SCORE        = 68.0
_TRUSTED_MIN_TRADES       = 15
_TRUSTED_MAX_DD           = 8.0
_TRUSTED_MAX_CONSEC_LOSS  = 2

_PROBATION_MAX_SCORE      = 35.0
_PROBATION_MAX_DD         = 12.0
_PROBATION_MIN_CONSEC_LOSS = 5

_RANK_MULT = {TRUSTED: 1.3, NEUTRAL: 1.0, PROBATION: 0.7}

# ── Persistence ───────────────────────────────────────────────────────────────

_SAVE_PATHS = [
    Path("/opt/btcbot/engine_governor.json"),
    Path("engine_governor.json"),
]

_lock = threading.Lock()

# State: {engine: {tier, since, history: [{tier, ts, reason}]}}
_state: dict = {}


def _save_path() -> Path:
    for p in _SAVE_PATHS:
        if p.parent.exists():
            return p
    return _SAVE_PATHS[-1]


def _load() -> None:
    p = _save_path()
    if not p.exists():
        return
    try:
        data = json.loads(p.read_text())
        with _lock:
            _state.update(data)
    except Exception:
        pass


def _save() -> None:
    p = _save_path()
    try:
        with _lock:
            snap = json.dumps(_state, indent=2)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(snap)
        tmp.replace(p)
    except Exception:
        pass


_load()

# ── Tier calculation ──────────────────────────────────────────────────────────

def _compute_tier(engine: str, stats: dict, score: float) -> str:
    n   = stats.get("trades", 0)
    dd  = stats.get("max_drawdown_pct", 100.0)
    cl  = stats.get("consecutive_losses_current", 0)

    # PROBATION takes priority — protect capital first
    if (score < _PROBATION_MAX_SCORE and n >= 10) or \
       dd >= _PROBATION_MAX_DD or \
       cl >= _PROBATION_MIN_CONSEC_LOSS:
        return PROBATION

    # TRUSTED requires all criteria met
    if (score >= _TRUSTED_MIN_SCORE and
            n >= _TRUSTED_MIN_TRADES and
            dd < _TRUSTED_MAX_DD and
            cl < _TRUSTED_MAX_CONSEC_LOSS):
        return TRUSTED

    return NEUTRAL


# ── Public API ────────────────────────────────────────────────────────────────

def get_tier(engine: str) -> str:
    """Return current tier for an engine.  NEUTRAL if no data."""
    with _lock:
        return _state.get(engine, {}).get("tier", NEUTRAL)


def check_all_tiers(days: int = 60) -> list[str]:
    """
    Recompute tiers for all engines.  Returns list of notification messages
    for any tier changes (for Telegram).  Safe to call each cycle.
    """
    notifications: list[str] = []
    try:
        all_stats = ep.get_all_stats(days=days)
        ts = datetime.now(timezone.utc).isoformat()

        for engine in ep.ENGINE_NAMES:
            stats  = all_stats[engine]
            score  = er.engine_health_score(stats)
            new_tier = _compute_tier(engine, stats, score)

            with _lock:
                existing = _state.get(engine, {})
                old_tier = existing.get("tier", NEUTRAL)

                if new_tier != old_tier:
                    reason = (
                        f"score={score:.0f} trades={stats['trades']} "
                        f"dd={stats['max_drawdown_pct']:.1f}% "
                        f"consec_loss_cur={stats['consecutive_losses_current']}"
                    )
                    history = existing.get("history", [])
                    history.append({"tier": new_tier, "ts": ts, "reason": reason})
                    _state[engine] = {
                        "tier":    new_tier,
                        "since":   ts,
                        "score":   round(score, 1),
                        "history": history[-20:],   # keep last 20 events
                    }
                    arrow = "↑" if (
                        (new_tier == TRUSTED) or
                        (new_tier == NEUTRAL and old_tier == PROBATION)
                    ) else "↓"
                    msg = (
                        f"*ENGINE GOVERNOR* {arrow}\n"
                        f"`{engine}`: {old_tier} → *{new_tier}*\n"
                        f"_{reason}_"
                    )
                    notifications.append(msg)
                    logger.log_info(
                        f"GOVERNOR | {engine} | {old_tier} → {new_tier} | {reason}"
                    )
                else:
                    # Update score even when tier unchanged
                    _state[engine] = {
                        "tier":    new_tier,
                        "since":   existing.get("since", ts),
                        "score":   round(score, 1),
                        "history": existing.get("history", []),
                    }

        _save()
    except Exception as exc:
        logger.log_warning(f"engine_governor.check_all_tiers error (non-critical): {exc}")

    return notifications


def grade_floor_for_tier(tier: str, base_grade: str = "B") -> str:
    """Return minimum grade for this tier.  TRUSTED can accept B; PROBATION requires A."""
    _grade_rank = {"A+": 0, "A": 1, "B": 2, "C": 3}
    _rank_grade = {v: k for k, v in _grade_rank.items()}

    base_rank = _grade_rank.get(base_grade, 2)

    if tier == TRUSTED:
        # Relax by 1 level (e.g. A → B) but never looser than B
        return _rank_grade.get(min(base_rank + 1, 2), base_grade)
    elif tier == PROBATION:
        # Tighten by 2 levels (e.g. B → A+)
        return _rank_grade.get(max(0, base_rank - 2), "A+")
    return base_grade


def rank_multiplier_for_tier(engine: str) -> float:
    """Return rank_score multiplier for engine's current tier."""
    tier = get_tier(engine)
    return _RANK_MULT.get(tier, 1.0)


def get_tier_summary() -> dict:
    """Return full tier state for dashboard / Telegram."""
    try:
        with _lock:
            snap = dict(_state)
        return {
            "engines": {
                eng: {
                    "tier":    snap.get(eng, {}).get("tier",  NEUTRAL),
                    "score":   snap.get(eng, {}).get("score", 0.0),
                    "since":   snap.get(eng, {}).get("since", ""),
                    "history": snap.get(eng, {}).get("history", [])[-5:],
                }
                for eng in ep.ENGINE_NAMES
            },
            "tier_counts": {
                TRUSTED:   sum(1 for e in ep.ENGINE_NAMES if snap.get(e, {}).get("tier") == TRUSTED),
                NEUTRAL:   sum(1 for e in ep.ENGINE_NAMES if snap.get(e, {}).get("tier", NEUTRAL) == NEUTRAL),
                PROBATION: sum(1 for e in ep.ENGINE_NAMES if snap.get(e, {}).get("tier") == PROBATION),
            },
        }
    except Exception as exc:
        logger.log_warning(f"engine_governor.get_tier_summary error: {exc}")
        return {"engines": {}, "tier_counts": {}}
