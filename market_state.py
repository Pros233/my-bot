"""
market_state.py — Deterministic live market state classification.

Uses ADX, ATR, Bollinger Bands, volume, and regime labels to produce
a named market state summary.  Also provides engine ↔ market-state
affinity scores so the bot can favour engines that match the current
environment.

Public API
----------
    classify(df, adx, atr_pct, trend, vol)  → MarketState
    engine_affinity(state, engine)           → float  (0.0 = poor fit, 1.0 = ideal)
    best_engines_for_state(state)            → list[str] sorted by affinity
    describe(state)                          → human-readable string

Never raises — returns a safe UNKNOWN state on any error.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import engine_performance as ep   # noqa: F401 (imported for ENGINE_NAMES)


# ── Market state names ────────────────────────────────────────────────────────

STRONG_TREND         = "Strong Trend"
WEAK_TREND           = "Weak Trend"
RANGING              = "Ranging"
VOL_EXPANSION        = "Volatility Expansion"
VOL_COMPRESSION      = "Volatility Compression"
LOW_LIQUIDITY        = "Low Liquidity"
MOMENTUM_EXPANSION   = "Momentum Expansion"
MEAN_REV_FAVORABLE   = "Mean Reversion Favorable"
RISK_OFF             = "Risk-Off Market"
UNKNOWN_STATE        = "Unknown"


# ── MarketState dataclass ─────────────────────────────────────────────────────

@dataclass
class MarketState:
    state: str                  # primary label from constants above
    trend_quality: str          # "strong" | "moderate" | "weak" | "none"
    vol_state: str              # "high" | "expanding" | "normal" | "compressing" | "low"
    liquidity: str              # "good" | "thin"
    momentum: str               # "expanding" | "fading" | "neutral"
    adx: float
    atr_pct: float
    bb_width_pctile: float      # 0-1, where 0 = tightest squeeze
    vol_ratio: float            # current volume / 20-bar avg


# ── Classification ────────────────────────────────────────────────────────────

def classify(
    df,
    adx: float,
    atr_pct: float,
    trend: str,
    vol: str,
) -> MarketState:
    """
    Classify the current market state from df + regime metadata.

    Parameters
    ----------
    df      : 1H OHLCV DataFrame (pandas)
    adx     : Current ADX value
    atr_pct : ATR as % of close
    trend   : regime string ("TRENDING" | "RANGING" | ...)
    vol     : volatility string ("NORMAL" | "HIGH_VOLATILITY" | "LOW_VOLATILITY")
    """
    try:
        import pandas as pd

        # ── BB width percentile ──
        period = 20
        close = df["close"]
        sma   = close.rolling(period).mean()
        std   = close.rolling(period).std()
        bb_upper = sma + 2 * std
        bb_lower = sma - 2 * std
        bb_width = (bb_upper - bb_lower) / sma

        w_now = float(bb_width.iloc[-1])
        w_20  = bb_width.iloc[-20:]
        w_min, w_max = float(w_20.min()), float(w_20.max())
        bb_pctile = (w_now - w_min) / (w_max - w_min + 1e-9)

        # ── BB expansion vs contraction ──
        w_prev = float(bb_width.iloc[-4:-1].mean()) if len(bb_width) >= 4 else w_now
        bb_expanding = w_now > w_prev * 1.05

        # ── Volume ratio ──
        vol_ma  = df["volume"].rolling(20).mean()
        vol_now = float(df["volume"].iloc[-1])
        vol_avg = float(vol_ma.iloc[-1])
        vol_ratio = vol_now / vol_avg if vol_avg > 0 else 1.0

        # ── Momentum: consecutive up-bars ──
        last3_close  = [float(close.iloc[i]) for i in (-3, -2, -1)]
        last3_open   = [float(df["open"].iloc[i]) for i in (-3, -2, -1)]
        up_bars      = sum(1 for c, o in zip(last3_close, last3_open) if c > o)
        momentum = "expanding" if up_bars >= 2 else "fading" if up_bars == 0 else "neutral"

        # ── Liquidity proxy: spread-like measure (high-low vs close) ──
        hl_range = (float(df["high"].iloc[-1]) - float(df["low"].iloc[-1]))
        hl_pct   = hl_range / float(close.iloc[-1]) * 100
        liquidity = "thin" if (hl_pct > 3.0 or vol_ratio < 0.4) else "good"

        # ── Trend quality ──
        if adx >= 35:
            trend_quality = "strong"
        elif adx >= 25:
            trend_quality = "moderate"
        elif adx >= 18:
            trend_quality = "weak"
        else:
            trend_quality = "none"

        # ── Vol state ──
        if vol == "HIGH_VOLATILITY" or atr_pct >= 1.2:
            vol_state = "high"
        elif bb_expanding and atr_pct >= 0.5:
            vol_state = "expanding"
        elif bb_pctile <= 0.20:
            vol_state = "compressing"
        elif atr_pct < 0.25:
            vol_state = "low"
        else:
            vol_state = "normal"

        # ── Primary state classification ──
        if liquidity == "thin" and vol_ratio < 0.5:
            state = LOW_LIQUIDITY
        elif vol == "HIGH_VOLATILITY" and trend_quality == "none":
            state = RISK_OFF
        elif trend_quality == "strong" and trend == "TRENDING":
            state = STRONG_TREND
        elif trend == "TRENDING" and trend_quality in ("moderate", "weak"):
            state = WEAK_TREND
        elif vol_state == "expanding" and up_bars >= 2 and vol_ratio >= 1.5:
            state = MOMENTUM_EXPANSION
        elif bb_pctile <= 0.20 and bb_expanding:
            state = VOL_EXPANSION
        elif bb_pctile <= 0.20:
            state = VOL_COMPRESSION
        elif trend_quality in ("none", "weak") and vol_state in ("normal", "low"):
            state = MEAN_REV_FAVORABLE
        elif trend == "RANGING":
            state = RANGING
        else:
            state = UNKNOWN_STATE

        return MarketState(
            state=state,
            trend_quality=trend_quality,
            vol_state=vol_state,
            liquidity=liquidity,
            momentum=momentum,
            adx=round(adx, 1),
            atr_pct=round(atr_pct, 2),
            bb_width_pctile=round(bb_pctile, 3),
            vol_ratio=round(vol_ratio, 2),
        )

    except Exception:
        return MarketState(
            state=UNKNOWN_STATE,
            trend_quality="none", vol_state="normal",
            liquidity="good", momentum="neutral",
            adx=adx, atr_pct=atr_pct,
            bb_width_pctile=0.5, vol_ratio=1.0,
        )


# ── Engine ↔ market-state affinity ───────────────────────────────────────────

# Affinity table: (state, engine) → 0.0-1.0
# 1.0 = ideal match, 0.5 = neutral, 0.2 = poor fit
_AFFINITY: dict[tuple[str, str], float] = {
    # ── RMR (range mean-reversion, 2H) ──
    (RANGING,             "RMR"): 1.0,
    (MEAN_REV_FAVORABLE,  "RMR"): 0.9,
    (VOL_COMPRESSION,     "RMR"): 0.8,
    (WEAK_TREND,          "RMR"): 0.5,
    (STRONG_TREND,        "RMR"): 0.2,
    (VOL_EXPANSION,       "RMR"): 0.3,
    (MOMENTUM_EXPANSION,  "RMR"): 0.3,
    (RISK_OFF,            "RMR"): 0.2,
    (LOW_LIQUIDITY,       "RMR"): 0.3,

    # ── PULLBACK (EMA21 trend continuation) ──
    (STRONG_TREND,        "PULLBACK"): 1.0,
    (WEAK_TREND,          "PULLBACK"): 0.8,
    (MOMENTUM_EXPANSION,  "PULLBACK"): 0.7,
    (RANGING,             "PULLBACK"): 0.3,
    (MEAN_REV_FAVORABLE,  "PULLBACK"): 0.3,
    (VOL_COMPRESSION,     "PULLBACK"): 0.4,
    (VOL_EXPANSION,       "PULLBACK"): 0.6,
    (RISK_OFF,            "PULLBACK"): 0.3,
    (LOW_LIQUIDITY,       "PULLBACK"): 0.3,

    # ── BREAKOUT (BB squeeze → expansion) ──
    (VOL_EXPANSION,       "BREAKOUT"): 1.0,
    (VOL_COMPRESSION,     "BREAKOUT"): 0.8,   # setup forming
    (MOMENTUM_EXPANSION,  "BREAKOUT"): 0.8,
    (STRONG_TREND,        "BREAKOUT"): 0.6,
    (WEAK_TREND,          "BREAKOUT"): 0.5,
    (RANGING,             "BREAKOUT"): 0.4,
    (RISK_OFF,            "BREAKOUT"): 0.2,
    (LOW_LIQUIDITY,       "BREAKOUT"): 0.2,
    (MEAN_REV_FAVORABLE,  "BREAKOUT"): 0.4,

    # ── NY_MOMENTUM (13-16 UTC consecutive up-bars) ──
    (MOMENTUM_EXPANSION,  "NY_MOMENTUM"): 1.0,
    (STRONG_TREND,        "NY_MOMENTUM"): 0.9,
    (VOL_EXPANSION,       "NY_MOMENTUM"): 0.8,
    (WEAK_TREND,          "NY_MOMENTUM"): 0.6,
    (RANGING,             "NY_MOMENTUM"): 0.3,
    (MEAN_REV_FAVORABLE,  "NY_MOMENTUM"): 0.3,
    (VOL_COMPRESSION,     "NY_MOMENTUM"): 0.3,
    (RISK_OFF,            "NY_MOMENTUM"): 0.2,
    (LOW_LIQUIDITY,       "NY_MOMENTUM"): 0.2,

    # ── MICRO_MR (1H mean reversion, ADX < 22) ──
    (MEAN_REV_FAVORABLE,  "MICRO_MR"): 1.0,
    (RANGING,             "MICRO_MR"): 0.9,
    (VOL_COMPRESSION,     "MICRO_MR"): 0.8,
    (WEAK_TREND,          "MICRO_MR"): 0.5,
    (STRONG_TREND,        "MICRO_MR"): 0.2,
    (VOL_EXPANSION,       "MICRO_MR"): 0.3,
    (MOMENTUM_EXPANSION,  "MICRO_MR"): 0.3,
    (RISK_OFF,            "MICRO_MR"): 0.2,
    (LOW_LIQUIDITY,       "MICRO_MR"): 0.3,

    # ── CONSENSUS (MACD+VWAP, legacy) ──
    (STRONG_TREND,        "CONSENSUS"): 0.7,
    (WEAK_TREND,          "CONSENSUS"): 0.6,
    (MOMENTUM_EXPANSION,  "CONSENSUS"): 0.7,
    (RANGING,             "CONSENSUS"): 0.5,
    (MEAN_REV_FAVORABLE,  "CONSENSUS"): 0.5,
    (VOL_EXPANSION,       "CONSENSUS"): 0.6,
    (VOL_COMPRESSION,     "CONSENSUS"): 0.4,
    (RISK_OFF,            "CONSENSUS"): 0.3,
    (LOW_LIQUIDITY,       "CONSENSUS"): 0.3,
}

_DEFAULT_AFFINITY = 0.5


def engine_affinity(state: MarketState, engine: str) -> float:
    """Return 0.0-1.0 affinity for (state, engine) pair."""
    return _AFFINITY.get((state.state, engine), _DEFAULT_AFFINITY)


def best_engines_for_state(state: MarketState) -> list[str]:
    """Return engine names sorted by affinity for the given state, descending."""
    from engine_performance import ENGINE_NAMES
    return sorted(
        ENGINE_NAMES,
        key=lambda e: -engine_affinity(state, e),
    )


def affinity_rank_boost(state: MarketState, engine: str) -> float:
    """
    Convert affinity score to a rank_score boost/penalty (-30 to +30).
    Applied on top of the base rank_score before candidate selection.
    """
    aff = engine_affinity(state, engine)
    # aff 0.0 → -30, aff 0.5 → 0, aff 1.0 → +30
    return round((aff - 0.5) * 60.0, 1)


def describe(state: MarketState) -> str:
    """Human-readable one-liner."""
    return (
        f"{state.state} | trend={state.trend_quality} "
        f"vol={state.vol_state} adx={state.adx:.1f} "
        f"atr={state.atr_pct:.2f}% vol_ratio={state.vol_ratio:.1f}x"
    )
