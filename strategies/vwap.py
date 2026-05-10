"""
strategies/vwap.py — Session VWAP strategy.

Group  : universal
Weight : 1.0

VWAP resets each UTC calendar day (typical_price × volume cumsum / volume cumsum).

Signal logic:
  +1 if close > VWAP AND VWAP is rising  AND close > EMA200
  -1 if close < VWAP AND VWAP is falling AND close < EMA200
   0 otherwise
"""
from __future__ import annotations

import pandas as pd
import pandas_ta as ta  # noqa: F401

GROUP = "universal"
WEIGHT = 1.0
NAME = "VWAP"


def _compute_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Session VWAP that resets each UTC calendar day.

    Requires df to have a DatetimeIndex (UTC-aware or naive-UTC) and
    columns: high, low, close, volume.
    """
    work = df[["high", "low", "close", "volume"]].copy()

    # Normalise index to UTC date for grouping
    if work.index.tzinfo is not None:
        dates = work.index.normalize()
    else:
        dates = pd.to_datetime(work.index.date)

    typical = (work["high"] + work["low"] + work["close"]) / 3.0
    tp_vol = typical * work["volume"]

    # Cumulative sums within each calendar day
    work["_date"] = dates
    work["_tp_vol"] = tp_vol
    work["_cum_tp_vol"] = work.groupby("_date")["_tp_vol"].cumsum()
    work["_cum_vol"] = work.groupby("_date")["volume"].cumsum()

    vwap = work["_cum_tp_vol"] / work["_cum_vol"]
    vwap.name = "VWAP"
    return vwap


def get_signal(df: pd.DataFrame) -> int:
    """Return +1, 0, or -1 for the most recent completed candle."""
    if len(df) < 202:
        return 0

    vwap   = _compute_vwap(df)
    ema200 = df.ta.ema(length=200)

    if ema200 is None:
        return 0

    curr_vwap   = float(vwap.iloc[-1])
    prev_vwap   = float(vwap.iloc[-2])
    curr_close  = float(df["close"].iloc[-1])
    curr_ema200 = float(ema200.iloc[-1])

    if any(v != v for v in (curr_vwap, prev_vwap, curr_close, curr_ema200)):
        return 0

    vwap_rising  = curr_vwap > prev_vwap
    vwap_falling = curr_vwap < prev_vwap

    if curr_close > curr_vwap and vwap_rising and curr_close > curr_ema200:
        return 1

    if curr_close < curr_vwap and vwap_falling and curr_close < curr_ema200:
        return -1

    return 0
