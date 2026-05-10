"""
strategies/rsi.py — RSI mean-reversion strategy.

Group  : oscillator
Weight : 1.5

Signal logic:
  +1 if RSI < 30 AND RSI is rising  AND close > EMA200
  -1 if RSI > 70 AND RSI is falling AND close < EMA200
   0 otherwise
"""
from __future__ import annotations

import pandas as pd
import pandas_ta as ta  # noqa: F401

import config

GROUP = "oscillator"
WEIGHT = 1.0
NAME = "RSI"


def get_signal(df: pd.DataFrame) -> int:
    """Return +1, 0, or -1 for the most recent completed candle."""
    min_len = max(config.RSI_PERIOD + 2, 202)
    if len(df) < min_len:
        return 0

    rsi    = df.ta.rsi(length=config.RSI_PERIOD)
    ema200 = df.ta.ema(length=200)

    if rsi is None or ema200 is None:
        return 0

    curr        = float(rsi.iloc[-1])
    prev        = float(rsi.iloc[-2])
    curr_ema200 = float(ema200.iloc[-1])
    curr_close  = float(df["close"].iloc[-1])

    if any(v != v for v in (curr, prev, curr_ema200, curr_close)):  # NaN check
        return 0

    if curr < 30 and curr > prev and curr_close > curr_ema200:   # oversold, rising, above EMA200
        return 1

    if curr > 70 and curr < prev and curr_close < curr_ema200:   # overbought, falling, below EMA200
        return -1

    return 0
