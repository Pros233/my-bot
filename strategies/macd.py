"""
strategies/macd.py — MACD crossover strategy.

Group  : trend
Weight : 2

Signal logic:
  +1 if MACD line crosses ABOVE signal line AND close > EMA200
  -1 if MACD line crosses BELOW signal line AND close < EMA200
   0 otherwise
"""
from __future__ import annotations

import pandas as pd
import pandas_ta as ta  # noqa: F401

import config

GROUP = "trend"
WEIGHT = 2.5
NAME = "MACD"


def get_signal(df: pd.DataFrame) -> int:
    """Return +1, 0, or -1 for the most recent completed candle."""
    min_len = max(config.MACD_SLOW + config.MACD_SIGNAL + 2, 202)
    if len(df) < min_len:
        return 0

    macd_df = df.ta.macd(
        fast=config.MACD_FAST,
        slow=config.MACD_SLOW,
        signal=config.MACD_SIGNAL,
    )
    ema200 = df.ta.ema(length=200)

    if macd_df is None or macd_df.empty or ema200 is None:
        return 0

    macd_col = f"MACD_{config.MACD_FAST}_{config.MACD_SLOW}_{config.MACD_SIGNAL}"
    sig_col  = f"MACDs_{config.MACD_FAST}_{config.MACD_SLOW}_{config.MACD_SIGNAL}"

    if macd_col not in macd_df.columns or sig_col not in macd_df.columns:
        return 0

    curr_macd   = float(macd_df[macd_col].iloc[-1])
    prev_macd   = float(macd_df[macd_col].iloc[-2])
    curr_sig    = float(macd_df[sig_col].iloc[-1])
    prev_sig    = float(macd_df[sig_col].iloc[-2])
    curr_ema200 = float(ema200.iloc[-1])
    curr_close  = float(df["close"].iloc[-1])

    if any(v != v for v in (curr_macd, prev_macd, curr_sig, prev_sig, curr_ema200, curr_close)):
        return 0

    # Bullish crossover with price above EMA200
    if prev_macd <= prev_sig and curr_macd > curr_sig and curr_close > curr_ema200:
        return 1

    # Bearish crossover with price below EMA200
    if prev_macd >= prev_sig and curr_macd < curr_sig and curr_close < curr_ema200:
        return -1

    return 0
