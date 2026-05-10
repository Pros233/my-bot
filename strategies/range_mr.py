"""
strategies/range_mr.py — Range mean-reversion entry signal (live path).

Implements the promoted 2H research setup on the live candle feed:
  RMR_2H_NO_FAR_VWAP_LONG_BIASED_BLOCK_LONG_HIGHATR_HIGHVOL

Entry conditions (ALL must be true):
  1. Regime: NOT TRENDING  (ADX < ADX_TREND_THRESHOLD)
  2. Price extended below VWAP  (close < vwap OR low < vwap)
  3. Price at/below range midpoint OR below 24-bar range low
  4. VWAP distance NOT "far"  (|close - vwap| / stop_dist < RMR_VWAP_FAR_R)
  5. NOT (ATR-bucket=high AND volume-bucket=high) — catastrophic combination
  6. Reclaim pattern: bullish close, beats recent N-bar highs
  7. Rejection (hammer): lower wick >= candle body

LONG only. No short entries are emitted.

Caller usage:
    from strategies.range_mr import get_signal_2h

    sig = get_signal_2h(df_2h)   # df_2h must be 2H OHLCV with DatetimeIndex
    if sig.direction == "LONG":
        # sig.stop_price, sig.tp_price, sig.atr_value are ready to use
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd
import pandas_ta as ta  # noqa: F401 — registers .ta accessor

import config
from strategies.vwap import _compute_vwap


@dataclass
class RangeMRSignal:
    direction: str          # "LONG" or "NONE"
    entry_price: float      # estimated next-open fill (with slippage)
    stop_price: float       # hard stop below entry
    tp_price: float         # take-profit at 1.5R (MR_EXIT_0 partial-TP level)
    stop_distance: float    # ATR × ATR_STOP_MULTIPLIER
    atr_value: float
    vwap: float
    range_high: float       # prior RESEARCH_RANGE_LOOKBACK-bar high
    range_low: float        # prior RESEARCH_RANGE_LOOKBACK-bar low
    atr_bucket: str         # "low" | "medium" | "high"
    volume_bucket: str      # "low" | "medium" | "high"
    vwap_distance_r: float  # |close − vwap| / stop_distance
    reject_reason: str      # non-empty when direction == "NONE"
    signal_type: str = "MR" # "MR" = range mean-reversion | "TREND" = EMA crossover trend


# Minimum bars needed before the signal is reliable
_WARMUP = max(
    config.RESEARCH_RANGE_LOOKBACK + config.RESEARCH_RECLAIM_LOOKBACK + 10,
    config.ATR_PERIOD + 5,
    config.VOLUME_MA_PERIOD + 5,
    config.ADX_PERIOD + 5,
    202,  # EMA200 warmup (used by _compute_vwap dependency chain)
)


def _atr_bucket(atr_pct: float) -> str:
    if atr_pct < config.RMR_ATR_LOW_PCT:
        return "low"
    if atr_pct < config.RMR_ATR_HIGH_PCT:
        return "medium"
    return "high"


def _volume_bucket(volume_ratio: float) -> str:
    if volume_ratio < config.RMR_VOL_LOW:
        return "low"
    if volume_ratio < config.RMR_VOL_HIGH:
        return "medium"
    return "high"


def _trend_signal(
    df: pd.DataFrame,
    i: int,
    adx_val: float,
    atr_val: float,
    stop_dist: float,
    atr_bucket: str,
    vol_bucket: str,
) -> RangeMRSignal:
    """
    Trend-following entry when ADX is elevated but below RMR_TREND_ADX_MAX.

    Conditions (ALL required):
      • EMA9 crossed above EMA21 on this bar (or already above by ≥ 0)
      • RSI(14) > RMR_TREND_RSI_MIN
      • Close > VWAP
      • NOT (HIGH_ATR + HIGH_VOL) — same catastrophic filter as RMR

    Returns a LONG signal with signal_type="TREND" or a NONE reject.
    """
    def _no_t(reason: str) -> RangeMRSignal:
        return RangeMRSignal(
            direction="NONE", entry_price=0.0, stop_price=0.0, tp_price=0.0,
            stop_distance=0.0, atr_value=0.0, vwap=0.0,
            range_high=0.0, range_low=0.0,
            atr_bucket="", volume_bucket="", vwap_distance_r=0.0,
            reject_reason=reason, signal_type="TREND",
        )

    if atr_bucket == "high" and vol_bucket == "high":
        return _no_t("TREND blocked: HIGH_ATR + HIGH_VOL")

    # EMA crossover
    ema_fast = df["close"].ewm(span=config.EMA_FAST, adjust=False).mean()
    ema_slow = df["close"].ewm(span=config.EMA_SLOW, adjust=False).mean()
    ema_fast_now  = float(ema_fast.iloc[i])
    ema_fast_prev = float(ema_fast.iloc[i - 1])
    ema_slow_now  = float(ema_slow.iloc[i])
    ema_slow_prev = float(ema_slow.iloc[i - 1])

    if not (math.isfinite(ema_fast_now) and math.isfinite(ema_slow_now)):
        return _no_t("TREND: non-finite EMA")

    crossed_above = ema_fast_prev <= ema_slow_prev and ema_fast_now > ema_slow_now
    already_above = ema_fast_now > ema_slow_now
    if not (crossed_above or already_above):
        return _no_t("TREND: EMA9 not above EMA21")

    # RSI confirmation
    rsi_series = df.ta.rsi(length=config.RSI_PERIOD)
    rsi_val = float(rsi_series.iloc[i]) if rsi_series is not None else math.nan
    if not math.isfinite(rsi_val) or rsi_val < config.RMR_TREND_RSI_MIN:
        return _no_t(f"TREND: RSI {rsi_val:.1f} < {config.RMR_TREND_RSI_MIN}")

    # Price above VWAP
    vwap_s   = _compute_vwap(df)
    vwap_val = float(vwap_s.iloc[i])
    curr_close = float(df["close"].iloc[i])
    if not math.isfinite(vwap_val) or curr_close <= vwap_val:
        return _no_t("TREND: close not above VWAP")

    entry_price = curr_close * (1.0 + config.SLIPPAGE)
    stop_price  = entry_price - stop_dist
    tp_price    = entry_price + stop_dist * config.RMR_TP_RR_RATIO
    vwap_dist_r = abs(curr_close - vwap_val) / stop_dist if stop_dist > 0 else 0.0

    lb = config.RESEARCH_RANGE_LOOKBACK
    prior_high = float(df["high"].iloc[i - lb: i].max())
    prior_low  = float(df["low"].iloc[i - lb: i].min())

    return RangeMRSignal(
        direction="LONG",
        entry_price=round(entry_price, 2),
        stop_price=round(stop_price, 2),
        tp_price=round(tp_price, 2),
        stop_distance=round(stop_dist, 2),
        atr_value=round(atr_val, 2),
        vwap=round(vwap_val, 2),
        range_high=round(prior_high, 2),
        range_low=round(prior_low, 2),
        atr_bucket=atr_bucket,
        volume_bucket=vol_bucket,
        vwap_distance_r=round(vwap_dist_r, 3),
        reject_reason="",
        signal_type="TREND",
    )


def get_signal_2h(df: pd.DataFrame) -> RangeMRSignal:
    """
    Evaluate the range MR setup on the most recent completed 2H candle.

    *df* must be a 2H OHLCV DataFrame with a UTC-aware DatetimeIndex.
    The last row is treated as the current (just-closed) signal candle.

    Returns a RangeMRSignal; trade only when .direction == "LONG".
    """

    def _no(reason: str) -> RangeMRSignal:
        return RangeMRSignal(
            direction="NONE", entry_price=0.0, stop_price=0.0, tp_price=0.0,
            stop_distance=0.0, atr_value=0.0, vwap=0.0,
            range_high=0.0, range_low=0.0,
            atr_bucket="", volume_bucket="", vwap_distance_r=0.0,
            reject_reason=reason, signal_type="MR",
        )

    if len(df) < _WARMUP:
        return _no(f"insufficient bars ({len(df)} < {_WARMUP})")

    # ── Indicators ─────────────────────────────────────────────────────────────
    atr_series = df.ta.atr(length=config.ATR_PERIOD)
    adx_df     = df.ta.adx(length=config.ADX_PERIOD)
    vol_ma     = df["volume"].rolling(config.VOLUME_MA_PERIOD).mean()
    vwap_s     = _compute_vwap(df)

    i = len(df) - 1  # last completed candle

    atr_val = float(atr_series.iloc[i])
    if not math.isfinite(atr_val) or atr_val <= 0:
        return _no("invalid ATR")

    adx_col = f"ADX_{config.ADX_PERIOD}"
    adx_val = (
        float(adx_df[adx_col].iloc[i])
        if adx_df is not None and adx_col in adx_df.columns
        else math.nan
    )
    if not math.isfinite(adx_val):
        return _no("invalid ADX")

    curr_close = float(df["close"].iloc[i])
    curr_open  = float(df["open"].iloc[i])
    curr_high  = float(df["high"].iloc[i])
    curr_low   = float(df["low"].iloc[i])
    prev_close = float(df["close"].iloc[i - 1])
    vwap_val   = float(vwap_s.iloc[i])
    vol_ma_val = float(vol_ma.iloc[i])
    vol_ratio  = float(df["volume"].iloc[i]) / vol_ma_val if vol_ma_val > 0 else 0.0

    if not all(math.isfinite(v) for v in (curr_close, curr_open, curr_high, curr_low, prev_close, vwap_val)):
        return _no("non-finite OHLCV or VWAP")
    if curr_close <= 0 or vwap_val <= 0:
        return _no("non-positive price or VWAP")

    # ── Stop distance and bucket classification (needed by trend path too) ─────
    stop_dist  = atr_val * config.ATR_STOP_MULTIPLIER
    atr_pct    = (atr_val / curr_close) * 100.0
    atb        = _atr_bucket(atr_pct)
    volb       = _volume_bucket(vol_ratio)

    # ── 1. Regime gate ─────────────────────────────────────────────────────────
    if adx_val > config.RMR_ADX_THRESHOLD:
        # Trend-following path: EMA crossover + RSI confirmation (opt-in, not yet
        # walk-forward validated — set RMR_TREND_ENTRY=true to enable).
        if config.RMR_TREND_ENTRY and adx_val <= config.RMR_TREND_ADX_MAX:
            return _trend_signal(df, i, adx_val, atr_val, stop_dist, atb, volb)
        return _no(f"trending regime ADX={adx_val:.1f}")

    # ── Range boundaries ───────────────────────────────────────────────────────
    lb = config.RESEARCH_RANGE_LOOKBACK
    prior_high = float(df["high"].iloc[i - lb: i].max())
    prior_low  = float(df["low"].iloc[i - lb: i].min())
    range_width = prior_high - prior_low
    if range_width <= 0:
        return _no("degenerate range (width=0)")

    range_mid = (prior_high + prior_low) / 2.0

    # ── 2. LONG direction gate — price must be below VWAP ──────────────────────
    extended_below_vwap = curr_close < vwap_val or curr_low < vwap_val
    if not extended_below_vwap:
        return _no("price not extended below VWAP")

    # ── 3. Price at/below range mid or range low ────────────────────────────────
    broke_range_low = curr_low < prior_low
    if not (broke_range_low or curr_close <= range_mid):
        return _no("price not at/below range mid")

    # ── 4. VWAP distance filter — block "far" entries ──────────────────────────
    vwap_dist_r = abs(curr_close - vwap_val) / stop_dist
    if vwap_dist_r >= config.RMR_VWAP_FAR_R:
        return _no(f"VWAP distance too far ({vwap_dist_r:.2f}R)")

    # ── 5. Catastrophic filter: HIGH_ATR + HIGH_VOL LONG blocked ───────────────
    if atb == "high" and volb == "high":
        return _no("blocked: HIGH_ATR + HIGH_VOL (historical PF=0.07)")

    # ── 6. Reclaim pattern ─────────────────────────────────────────────────────
    rl = config.RESEARCH_RECLAIM_LOOKBACK
    recent_max     = float(df["close"].iloc[i - rl: i].max())
    failed_bkout   = broke_range_low and curr_close >= prior_low
    reclaim_up = (
        curr_close > curr_open
        and curr_close > recent_max
        and (failed_bkout or curr_close > prev_close)
    )
    if not reclaim_up:
        return _no("reclaim pattern absent")

    # ── 7. Rejection (hammer) ──────────────────────────────────────────────────
    body        = abs(curr_close - curr_open)
    lower_wick  = min(curr_open, curr_close) - curr_low
    rejection   = curr_close > curr_open and lower_wick >= body
    if not rejection:
        return _no("rejection (hammer) pattern absent")

    # ── Build trade prices ─────────────────────────────────────────────────────
    # Entry: next open with slippage
    entry_price = curr_close * (1.0 + config.SLIPPAGE)
    stop_price  = entry_price - stop_dist

    # TP: 1.5R above entry (matches MR_EXIT_0 partial-TP level = DEFAULT_PARTIAL_TP_R)
    tp_price = entry_price + stop_dist * config.RMR_TP_RR_RATIO

    return RangeMRSignal(
        direction="LONG",
        entry_price=round(entry_price, 2),
        stop_price=round(stop_price, 2),
        tp_price=round(tp_price, 2),
        stop_distance=round(stop_dist, 2),
        atr_value=round(atr_val, 2),
        vwap=round(vwap_val, 2),
        range_high=round(prior_high, 2),
        range_low=round(prior_low, 2),
        atr_bucket=atb,
        volume_bucket=volb,
        vwap_distance_r=round(vwap_dist_r, 3),
        reject_reason="",
    )


def resample_1h_to_2h(df_1h: pd.DataFrame) -> pd.DataFrame:
    """
    Resample a 1H OHLCV DataFrame to 2H, dropping any incomplete bar.

    *df_1h* must have a UTC-aware DatetimeIndex and columns open/high/low/close/volume.
    Only bars where exactly 2 source candles contributed are kept, so the last
    still-forming 2H bar is automatically excluded.
    """
    rule = "2h"
    counts = df_1h["close"].resample(rule, label="right", closed="right").count()
    resampled = pd.DataFrame({
        "open":   df_1h["open"].resample(rule, label="right", closed="right").first(),
        "high":   df_1h["high"].resample(rule, label="right", closed="right").max(),
        "low":    df_1h["low"].resample(rule, label="right", closed="right").min(),
        "close":  df_1h["close"].resample(rule, label="right", closed="right").last(),
        "volume": df_1h["volume"].resample(rule, label="right", closed="right").sum(),
    }).dropna()
    # Keep only complete 2H bars (exactly 2 contributing 1H candles)
    return resampled.loc[counts == 2]
