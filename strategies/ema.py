"""
strategies/ema.py — EMA Crossover strategy.

Group  : trend
Weight : 2

Signal logic:
  +1 if EMA9 crosses ABOVE EMA21 AND close > EMA200
  -1 if EMA9 crosses BELOW EMA21 AND close < EMA200
   0 otherwise  (continuation, or against EMA200 bias)
"""
from __future__ import annotations

import pandas as pd
import pandas_ta as ta  # noqa: F401

import config

GROUP = "trend"
WEIGHT = 2.5
NAME = "EMA Cross"


def get_signal(df: pd.DataFrame) -> int:
    """Return +1, 0, or -1 for the most recent completed candle."""
    min_len = 202  # EMA200 needs 200 candles + 2 for crossover comparison
    if len(df) < min_len:
        return 0

    ema_fast = df.ta.ema(length=config.EMA_FAST)
    ema_slow = df.ta.ema(length=config.EMA_SLOW)
    ema200   = df.ta.ema(length=200)

    if ema_fast is None or ema_slow is None or ema200 is None:
        return 0

    curr_fast  = float(ema_fast.iloc[-1])
    prev_fast  = float(ema_fast.iloc[-2])
    curr_slow  = float(ema_slow.iloc[-1])
    prev_slow  = float(ema_slow.iloc[-2])
    curr_ema200 = float(ema200.iloc[-1])
    curr_close  = float(df["close"].iloc[-1])

    # Guard against NaN
    if any(v != v for v in (curr_fast, prev_fast, curr_slow, prev_slow, curr_ema200, curr_close)):
        return 0

    # Bullish crossover with price above EMA200
    if prev_fast <= prev_slow and curr_fast > curr_slow and curr_close > curr_ema200:
        return 1

    # Bearish crossover with price below EMA200
    if prev_fast >= prev_slow and curr_fast < curr_slow and curr_close < curr_ema200:
        return -1

    return 0
