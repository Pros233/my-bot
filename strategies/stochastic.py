"""
strategies/stochastic.py — Stochastic oscillator strategy.

Group  : oscillator
Weight : 1.5

Signal logic:
  +1 if %K crosses above %D AND %K < 25 AND close > EMA200
  -1 if %K crosses below %D AND %K > 75 AND close < EMA200
   0 otherwise
"""
from __future__ import annotations

import pandas as pd
import pandas_ta as ta  # noqa: F401

import config

GROUP = "oscillator"
WEIGHT = 1.0
NAME = "Stochastic"


def get_signal(df: pd.DataFrame) -> int:
    """Return +1, 0, or -1 for the most recent completed candle."""
    min_len = max(config.STOCH_K + config.STOCH_D + 2, 202)
    if len(df) < min_len:
        return 0

    stoch_df = df.ta.stoch(
        k=config.STOCH_K,
        d=config.STOCH_D,
        smooth_k=config.STOCH_SMOOTH_K,
    )
    ema200 = df.ta.ema(length=200)

    if stoch_df is None or stoch_df.empty or ema200 is None:
        return 0

    k_col = f"STOCHk_{config.STOCH_K}_{config.STOCH_D}_{config.STOCH_SMOOTH_K}"
    d_col = f"STOCHd_{config.STOCH_K}_{config.STOCH_D}_{config.STOCH_SMOOTH_K}"

    if k_col not in stoch_df.columns or d_col not in stoch_df.columns:
        return 0

    curr_k      = float(stoch_df[k_col].iloc[-1])
    prev_k      = float(stoch_df[k_col].iloc[-2])
    curr_d      = float(stoch_df[d_col].iloc[-1])
    prev_d      = float(stoch_df[d_col].iloc[-2])
    curr_ema200 = float(ema200.iloc[-1])
    curr_close  = float(df["close"].iloc[-1])

    if any(v != v for v in (curr_k, prev_k, curr_d, prev_d, curr_ema200, curr_close)):
        return 0

    # Bullish crossover in oversold zone, price above EMA200
    if prev_k <= prev_d and curr_k > curr_d and curr_k < 25 and curr_close > curr_ema200:
        return 1

    # Bearish crossover in overbought zone, price below EMA200
    if prev_k >= prev_d and curr_k < curr_d and curr_k > 75 and curr_close < curr_ema200:
        return -1

    return 0
