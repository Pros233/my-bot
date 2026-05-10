"""
consensus.py — Weighted voting engine.

Aggregates strategy signals according to active regime groups,
then compares normalised score against the configurable threshold.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

import config
import regime as reg
from strategies import (
    ema_signal,
    macd_signal,
    rsi_signal,
    stochastic_signal,
    bollinger_signal,
    volume_signal,
    vwap_signal,
)

# Decision constants
BUY = "BUY"
SELL = "SELL"
HOLD = "HOLD"


@dataclass
class StrategyResult:
    name: str
    group: str
    weight: float
    signal: int        # raw: +1, 0, -1
    contribution: float = field(init=False)

    def __post_init__(self) -> None:
        self.contribution = self.signal * self.weight


@dataclass
class ConsensusResult:
    decision: str
    score: float
    max_possible: float
    ratio: float
    threshold: float
    breakdown: list[StrategyResult]
    active_groups: set[str]

    @property
    def active_names(self) -> list[str]:
        return [s.name for s in self.breakdown]


# ── Strategy registry ─────────────────────────────────────────────────────────
# Each entry: (name, group, weight, signal_fn)
_REGISTRY: list[tuple[str, str, float, Callable[[pd.DataFrame], int]]] = [
    ("EMA Cross",   "trend",       2.5, ema_signal),
    ("MACD",        "trend",       2.5, macd_signal),
    ("RSI",         "oscillator",  1.0, rsi_signal),
    ("Stochastic",  "oscillator",  0.0, stochastic_signal),  # reporting only — no vote
    ("Bollinger",   "universal",   1.0, bollinger_signal),
    ("Volume",      "universal",   1.0, volume_signal),
    ("VWAP",        "universal",   1.0, vwap_signal),
]


def compute(
    df: pd.DataFrame,
    trend: str,
    vol: str,
    threshold: float = config.CONSENSUS_THRESHOLD,
) -> ConsensusResult:
    """
    Run all active strategies for the given regime and return a ConsensusResult.

    Parameters
    ----------
    df        : OHLCV DataFrame with DatetimeIndex.
    trend     : "TRENDING" | "RANGING"
    vol       : "HIGH_VOLATILITY" | "NORMAL"
    threshold : normalised score cutoff (default from config).
    """
    groups = reg.active_groups(trend, vol)
    breakdown: list[StrategyResult] = []
    score = 0.0
    max_possible = 0.0

    for name, group, weight, fn in _REGISTRY:
        if group not in groups:
            continue

        signal = fn(df)
        result = StrategyResult(name=name, group=group, weight=weight, signal=signal)
        breakdown.append(result)
        score += result.contribution
        max_possible += weight

    if max_possible == 0:
        return ConsensusResult(
            decision=HOLD,
            score=0.0,
            max_possible=0.0,
            ratio=0.0,
            threshold=threshold,
            breakdown=breakdown,
            active_groups=groups,
        )

    ratio = score / max_possible

    if ratio >= threshold:
        decision = BUY
    elif ratio <= -threshold:
        decision = SELL
    else:
        decision = HOLD

    return ConsensusResult(
        decision=decision,
        score=score,
        max_possible=max_possible,
        ratio=ratio,
        threshold=threshold,
        breakdown=breakdown,
        active_groups=groups,
    )
