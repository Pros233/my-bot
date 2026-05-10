"""
regime_classifier.py — Extended market regime classifier (ROBUSTNESS #1).

Classifies each candle into one of:
  TRENDING_UP    — ADX > ADX_TREND_THRESHOLD AND close >= EMA200
  TRENDING_DOWN  — ADX > ADX_TREND_THRESHOLD AND close < EMA200
  RANGING        — ADX < ADX_RANGING_THRESHOLD
  HIGH_VOLATILITY — ATR(14) > ATR_HIGH_VOL_MULTIPLIER × rolling-median ATR
  UNKNOWN        — insufficient data (warmup period)

Classification priority: HIGH_VOLATILITY is checked first; if the market is
in HIGH_VOL the trending/ranging check is skipped.  Between the two ADX
thresholds (20–25) the classifier defaults to RANGING for MR safety.

Usage
-----
    from regime_classifier import RegimeClassifier

    clf = RegimeClassifier(log_to_csv=True)          # writes outputs/research/regime_log.csv
    regime_now = clf.classify(df)                    # string for current candle
    regime_series = clf.classify_series(df)          # pd.Series for all candles
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pandas_ta as ta  # noqa: F401 — registers .ta accessor

import config

REGIME_LOG_PATH = Path("outputs/research/regime_log.csv")
_LOG_HEADERS = ["timestamp", "adx", "atr", "atr_median", "natr", "close", "ema200", "regime"]

# Regime label constants
TRENDING_UP     = "TRENDING_UP"
TRENDING_DOWN   = "TRENDING_DOWN"
RANGING         = "RANGING"
HIGH_VOLATILITY = "HIGH_VOLATILITY"
UNKNOWN         = "UNKNOWN"

ALL_REGIMES = (TRENDING_UP, TRENDING_DOWN, RANGING, HIGH_VOLATILITY, UNKNOWN)


class RegimeClassifier:
    """
    Classify market regime using ADX and ATR.

    Parameters
    ----------
    adx_trending : float
        ADX threshold above which market is TRENDING (default from config).
    adx_ranging : float
        ADX threshold below which market is RANGING (default from config).
    atr_vol_multiplier : float
        ATR multiplier for HIGH_VOLATILITY detection.
    atr_vol_period : int
        Rolling window (bars) for computing ATR median.
    log_to_csv : bool
        Write each classify_series() result to regime_log.csv.
    """

    def __init__(
        self,
        adx_trending: float = config.ADX_TREND_THRESHOLD,
        adx_ranging: float = config.ADX_RANGING_THRESHOLD,
        atr_vol_multiplier: float = config.ATR_HIGH_VOL_MULTIPLIER,
        atr_vol_period: int = config.ATR_HIGH_VOL_PERIOD,
        log_to_csv: bool = False,
    ) -> None:
        self.adx_trending = adx_trending
        self.adx_ranging = adx_ranging
        self.atr_vol_mult = atr_vol_multiplier
        self.atr_vol_period = atr_vol_period
        self.log_to_csv = log_to_csv
        self._log_initialized = False

    # ── Public API ─────────────────────────────────────────────────────────────

    def classify(self, df: pd.DataFrame) -> str:
        """Classify regime for the *last* candle in df."""
        s = self.classify_series(df)
        return str(s.iloc[-1])

    def classify_series(self, df: pd.DataFrame) -> pd.Series:
        """
        Classify regime for every candle in df.

        Returns a pd.Series aligned to df.index with string labels.
        """
        adx_s, atr_s, atr_median_s, ema200_s = self._compute_indicators(df)
        regimes = self._vectorized_classify(df, adx_s, atr_s, atr_median_s, ema200_s)
        result = pd.Series(regimes, index=df.index, name="regime", dtype=str)

        if self.log_to_csv:
            self._write_log(df, result, adx_s, atr_s, atr_median_s, ema200_s)

        return result

    # ── Internals ──────────────────────────────────────────────────────────────

    def _compute_indicators(
        self, df: pd.DataFrame
    ) -> tuple[Optional[pd.Series], Optional[pd.Series], Optional[pd.Series], Optional[pd.Series]]:
        # ADX
        adx_df = df.ta.adx(length=config.ATR_PERIOD)
        adx_col = f"ADX_{config.ATR_PERIOD}"
        if adx_df is not None and adx_col in adx_df.columns:
            adx_s = adx_df[adx_col]
        elif adx_df is not None and not adx_df.empty:
            adx_s = adx_df[next(c for c in adx_df.columns if c.startswith("ADX_"))]
        else:
            adx_s = None

        # ATR
        atr_s = df.ta.atr(length=config.ATR_PERIOD)

        # ATR rolling median (for HIGH_VOL detection)
        atr_median_s = (
            atr_s.rolling(self.atr_vol_period, min_periods=max(1, self.atr_vol_period // 2)).median()
            if atr_s is not None
            else None
        )

        # EMA200
        ema200_s = df.ta.ema(length=200)

        return adx_s, atr_s, atr_median_s, ema200_s

    def _vectorized_classify(
        self,
        df: pd.DataFrame,
        adx_s: Optional[pd.Series],
        atr_s: Optional[pd.Series],
        atr_median_s: Optional[pd.Series],
        ema200_s: Optional[pd.Series],
    ) -> list[str]:
        n = len(df)
        result = [UNKNOWN] * n

        # Minimum lookback for valid indicators
        min_warmup = max(config.ATR_PERIOD, 200, self.atr_vol_period) + 5

        for i in range(n):
            if i < min_warmup:
                result[i] = UNKNOWN
                continue

            adx_val = float(adx_s.iloc[i]) if adx_s is not None else math.nan
            atr_val = float(atr_s.iloc[i]) if atr_s is not None else math.nan
            atr_med = float(atr_median_s.iloc[i]) if atr_median_s is not None else math.nan
            ema200_val = float(ema200_s.iloc[i]) if ema200_s is not None else math.nan
            close_val = float(df["close"].iloc[i])

            if not math.isfinite(adx_val) or not math.isfinite(atr_val):
                result[i] = UNKNOWN
                continue

            # HIGH_VOLATILITY has priority
            if math.isfinite(atr_med) and atr_med > 0:
                if atr_val > self.atr_vol_mult * atr_med:
                    result[i] = HIGH_VOLATILITY
                    continue

            # Trending vs ranging
            if adx_val > self.adx_trending:
                if math.isfinite(ema200_val) and close_val < ema200_val:
                    result[i] = TRENDING_DOWN
                else:
                    result[i] = TRENDING_UP
                continue

            if adx_val < self.adx_ranging:
                result[i] = RANGING
                continue

            # Between thresholds (adx_ranging .. adx_trending) — conservative: RANGING
            result[i] = RANGING

        return result

    def _write_log(
        self,
        df: pd.DataFrame,
        regimes: pd.Series,
        adx_s: Optional[pd.Series],
        atr_s: Optional[pd.Series],
        atr_median_s: Optional[pd.Series],
        ema200_s: Optional[pd.Series],
    ) -> None:
        REGIME_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        mode = "w" if not self._log_initialized else "a"
        with open(REGIME_LOG_PATH, mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_LOG_HEADERS)
            if not self._log_initialized:
                writer.writeheader()
                self._log_initialized = True
            for i in range(len(df)):
                ts = df.index[i]
                adx_v = float(adx_s.iloc[i]) if adx_s is not None else float("nan")
                atr_v = float(atr_s.iloc[i]) if atr_s is not None else float("nan")
                atr_m = float(atr_median_s.iloc[i]) if atr_median_s is not None else float("nan")
                ema_v = float(ema200_s.iloc[i]) if ema200_s is not None else float("nan")
                close_v = float(df["close"].iloc[i])
                natr = atr_v / close_v if close_v > 0 and math.isfinite(atr_v) else float("nan")
                writer.writerow({
                    "timestamp": str(ts),
                    "adx": _fmt(adx_v, 4),
                    "atr": _fmt(atr_v, 4),
                    "atr_median": _fmt(atr_m, 4),
                    "natr": _fmt(natr, 6),
                    "close": _fmt(close_v, 2),
                    "ema200": _fmt(ema_v, 2),
                    "regime": regimes.iloc[i],
                })


def _fmt(v: float, d: int) -> str:
    return f"{v:.{d}f}" if math.isfinite(v) else ""


def dominant_regime(regime_series: pd.Series) -> str:
    """Return the most frequent regime label in the series (excluding UNKNOWN)."""
    counts = regime_series[regime_series != UNKNOWN].value_counts()
    if counts.empty:
        return UNKNOWN
    return str(counts.index[0])
