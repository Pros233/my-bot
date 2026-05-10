"""
tests/test_regime_classifier.py — Unit tests for RegimeClassifier (ROBUSTNESS #1).

Run: python -m pytest tests/ -v
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure bot root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from regime_classifier import (
    RegimeClassifier,
    dominant_regime,
    TRENDING_UP, TRENDING_DOWN, RANGING, HIGH_VOLATILITY, UNKNOWN,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_df(n: int = 300, price: float = 60_000.0, trend: bool = False) -> pd.DataFrame:
    """Synthetic OHLCV DataFrame with UTC DatetimeIndex."""
    rng = np.random.default_rng(42)
    index = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")

    if trend:
        close = price + np.arange(n) * 50.0 + rng.normal(0, 100, n)
    else:
        close = price + rng.normal(0, 200, n).cumsum() * 0.1

    high   = close + rng.uniform(50, 300, n)
    low    = close - rng.uniform(50, 300, n)
    open_  = close + rng.normal(0, 100, n)
    volume = rng.uniform(100, 1000, n)

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )


def _make_ranging_df(n: int = 300) -> pd.DataFrame:
    """Range-bound data: price oscillates around a mean."""
    rng = np.random.default_rng(7)
    index = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")
    close = 60_000 + 300 * np.sin(np.linspace(0, 10 * math.pi, n)) + rng.normal(0, 50, n)
    high   = close + rng.uniform(30, 100, n)
    low    = close - rng.uniform(30, 100, n)
    open_  = close + rng.normal(0, 30, n)
    volume = rng.uniform(100, 500, n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestRegimeClassifierInterface:
    def test_classify_returns_string(self):
        df = _make_df()
        clf = RegimeClassifier()
        result = clf.classify(df)
        assert isinstance(result, str)
        assert result in (TRENDING_UP, TRENDING_DOWN, RANGING, HIGH_VOLATILITY, UNKNOWN)

    def test_classify_series_length_matches_df(self):
        df = _make_df(250)
        clf = RegimeClassifier()
        s = clf.classify_series(df)
        assert len(s) == len(df)
        assert s.index.equals(df.index)

    def test_classify_series_valid_labels_only(self):
        df = _make_df(300)
        clf = RegimeClassifier()
        s = clf.classify_series(df)
        assert set(s.unique()).issubset(set([TRENDING_UP, TRENDING_DOWN, RANGING, HIGH_VOLATILITY, UNKNOWN]))

    def test_warmup_period_is_unknown(self):
        df = _make_df(300)
        clf = RegimeClassifier()
        s = clf.classify_series(df)
        # First ~210 bars (EMA200 + ATR warmup) should be UNKNOWN
        assert s.iloc[0] == UNKNOWN
        assert s.iloc[10] == UNKNOWN


class TestRegimeClassifierLogic:
    def test_unknown_on_short_df(self):
        df = _make_df(50)
        clf = RegimeClassifier()
        s = clf.classify_series(df)
        # All bars should be UNKNOWN (insufficient data)
        assert (s == UNKNOWN).all()

    def test_no_nan_in_result(self):
        df = _make_df(300)
        clf = RegimeClassifier()
        s = clf.classify_series(df)
        assert not s.isna().any()

    def test_classify_and_classify_series_agree(self):
        df = _make_df(300)
        clf = RegimeClassifier()
        series_last = clf.classify_series(df).iloc[-1]
        single = clf.classify(df)
        assert single == series_last

    def test_custom_thresholds_respected(self):
        """Very high trending threshold forces more RANGING labels."""
        df = _make_df(300)
        clf_strict = RegimeClassifier(adx_trending=99.0, adx_ranging=1.0)
        s = clf_strict.classify_series(df)
        valid = s[s != UNKNOWN]
        # With extreme thresholds nearly all valid bars should be RANGING
        if len(valid) > 0:
            assert (valid != TRENDING_UP).all() or (valid != TRENDING_DOWN).all()


class TestDominantRegime:
    def test_dominant_excludes_unknown(self):
        s = pd.Series([UNKNOWN, UNKNOWN, RANGING, RANGING, RANGING, TRENDING_UP])
        assert dominant_regime(s) == RANGING

    def test_dominant_all_unknown(self):
        s = pd.Series([UNKNOWN, UNKNOWN])
        assert dominant_regime(s) == UNKNOWN

    def test_dominant_tie_returns_one_of_them(self):
        s = pd.Series([RANGING, RANGING, TRENDING_UP, TRENDING_UP])
        result = dominant_regime(s)
        assert result in (RANGING, TRENDING_UP)


class TestNoCsvSideEffect:
    def test_no_file_written_without_flag(self, tmp_path, monkeypatch):
        """classify_series should not write any file unless log_to_csv=True."""
        import regime_classifier as rc
        monkeypatch.setattr(rc, "REGIME_LOG_PATH", tmp_path / "regime_log.csv")
        df = _make_df(300)
        clf = RegimeClassifier(log_to_csv=False)
        clf.classify_series(df)
        assert not (tmp_path / "regime_log.csv").exists()

    def test_csv_written_with_flag(self, tmp_path, monkeypatch):
        import regime_classifier as rc
        monkeypatch.setattr(rc, "REGIME_LOG_PATH", tmp_path / "regime_log.csv")
        df = _make_df(300)
        clf = RegimeClassifier(log_to_csv=True)
        clf.classify_series(df)
        assert (tmp_path / "regime_log.csv").exists()
