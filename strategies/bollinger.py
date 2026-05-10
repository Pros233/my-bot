"""
strategies/bollinger.py — Bollinger Bands mean-reversion strategy.

Group  : universal
Weight : 1.0

Signal logic:
  +1 if close < lower band AND close is rising  (price bouncing up from lower band)
  -1 if close > upper band AND close is falling (price rolling over from upper band)
   0 otherwise
"""
from __future__ import annotations

import pandas as pd
import pandas_ta as ta  # noqa: F401

import config

GROUP = "universal"
WEIGHT = 1.0
NAME = "Bollinger"


def get_signal(df: pd.DataFrame) -> int:
    """Return +1, 0, or -1 for the most recent completed candle."""
    min_len = config.BB_PERIOD + 2
    if len(df) < min_len:
        return 0

    bb_df = df.ta.bbands(length=config.BB_PERIOD, std=config.BB_STD)
    if bb_df is None or bb_df.empty:
        return 0

    # Column name format varies across pandas-ta versions (e.g. BBL_20_2.0 vs
    # BBL_20_2.0_2.0).  Locate by prefix to be version-agnostic.
    lower_col = next(
        (c for c in bb_df.columns if c.startswith(f"BBL_{config.BB_PERIOD}_")), None
    )
    upper_col = next(
        (c for c in bb_df.columns if c.startswith(f"BBU_{config.BB_PERIOD}_")), None
    )
    if lower_col is None or upper_col is None:
        return 0

    curr_close = float(df["close"].iloc[-1])
    prev_close = float(df["close"].iloc[-2])
    lower = float(bb_df[lower_col].iloc[-1])
    upper = float(bb_df[upper_col].iloc[-1])

    if any(v != v for v in (curr_close, prev_close, lower, upper)):
        return 0

    if curr_close < lower and curr_close > prev_close:   # bouncing up
        return 1

    if curr_close > upper and curr_close < prev_close:   # rolling over
        return -1

    return 0
