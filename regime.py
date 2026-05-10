"""
regime.py — Market regime classifier.

Returns (trend_regime, vol_regime) tuple:
  trend_regime : "TRENDING" | "RANGING"
  vol_regime   : "HIGH_VOLATILITY" | "NORMAL"

Routing:
  TRENDING + NORMAL     → all strategies, full position
  TRENDING + HIGH_VOL   → trend strategies only, halved position
  RANGING  + NORMAL     → oscillator strategies only, full position
  RANGING  + HIGH_VOL   → NO TRADE
"""
from __future__ import annotations

import pandas as pd
import pandas_ta as ta  # noqa: F401 — registers .ta accessor

import config

# Public constants so consensus.py can check regime without string literals
TRENDING = "TRENDING"
RANGING = "RANGING"
HIGH_VOLATILITY = "HIGH_VOLATILITY"
NORMAL = "NORMAL"
NO_TRADE = "NO_TRADE"


def classify(df: pd.DataFrame) -> tuple[str, str]:
    """
    Classify the current market regime from *df*.

    *df* must have columns: open, high, low, close, volume
    and a DatetimeIndex.  The last row is treated as the current candle.

    Returns
    -------
    (trend_regime, vol_regime)
    """
    if len(df) < config.ADX_PERIOD + 5:
        # Not enough data — conservative fallback: ranging + high vol → no trade
        return RANGING, HIGH_VOLATILITY

    adx_df = df.ta.adx(length=config.ADX_PERIOD)
    atr_series = df.ta.atr(length=config.ATR_PERIOD)

    adx_col = f"ADX_{config.ADX_PERIOD}"

    adx_val: float = float(adx_df[adx_col].iloc[-1])
    atr_val: float = float(atr_series.iloc[-1])
    close_val: float = float(df["close"].iloc[-1])

    atr_pct: float = (atr_val / close_val) * 100.0

    trend_regime = TRENDING if adx_val > config.ADX_TREND_THRESHOLD else RANGING
    vol_regime = HIGH_VOLATILITY if atr_pct > config.ATR_HIGH_VOL_THRESHOLD_PCT else NORMAL

    return trend_regime, vol_regime


def regime_allows_trade(trend: str, vol: str) -> bool:
    """Return False when regime is RANGING + HIGH_VOL (no-trade zone)."""
    return not (trend == RANGING and vol == HIGH_VOLATILITY)


def should_halve_position(trend: str, vol: str) -> bool:
    """Return True when position size must be halved (TRENDING + HIGH_VOL)."""
    return trend == TRENDING and vol == HIGH_VOLATILITY


def active_groups(trend: str, vol: str) -> set[str]:
    """
    Return the set of strategy groups active for the given regime.

    Groups: "trend", "oscillator", "universal"
    """
    if trend == TRENDING:
        return {"trend", "universal"}
    else:  # RANGING
        return {"oscillator", "universal"}


def regime_label(trend: str, vol: str) -> str:
    """Human-readable label for the dashboard."""
    return f"{trend} + {vol}"
