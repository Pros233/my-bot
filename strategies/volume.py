"""
strategies/volume.py — Directional volume spike strategy.

Group  : universal
Weight : 1.0

Signal logic:
  avg_vol = rolling mean of volume (period 20)
  +1 if volume > 2 * avg_vol AND close > open   (bullish volume spike)
  -1 if volume > 2 * avg_vol AND close < open   (bearish volume spike)
   0 otherwise
"""
from __future__ import annotations

import pandas as pd

import config

GROUP = "universal"
WEIGHT = 1.0
NAME = "Volume"


def get_signal(df: pd.DataFrame) -> int:
    """Return +1, 0, or -1 for the most recent completed candle."""
    min_len = config.VOLUME_MA_PERIOD + 1
    if len(df) < min_len:
        return 0

    avg_vol = df["volume"].rolling(config.VOLUME_MA_PERIOD).mean()

    curr_vol = float(df["volume"].iloc[-1])
    curr_avg = float(avg_vol.iloc[-1])
    curr_close = float(df["close"].iloc[-1])
    curr_open = float(df["open"].iloc[-1])

    if any(v != v for v in (curr_vol, curr_avg, curr_close, curr_open)):
        return 0

    if curr_avg == 0:
        return 0

    body_pct = abs(curr_close - curr_open) / curr_open

    if curr_vol > 2 * curr_avg and body_pct > 0.003:
        if curr_close > curr_open:   # green candle
            return 1
        if curr_close < curr_open:   # red candle
            return -1

    return 0
