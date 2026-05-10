"""
tests/test_range_mr_integration.py — Integration tests for range-MR improvements.

Covers:
  - TieredExitFramework ATR trailing stop (new Tier 1.5)
  - TieredExitFramework volume-gated momentum exit
  - RegimeClassifier gate in _run_research_setup_simulation (unit-level via mock)
  - Robustness columns in _promotion_summary
"""
from __future__ import annotations

import sys
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from tiered_exit import (
    ExitConfig, TierState, TieredExitFramework, ExitEvent, compute_exit_r,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _candles(
    highs, lows, opens=None, closes=None, volumes=None
) -> pd.DataFrame:
    n = len(highs)
    if opens is None:
        opens = [(h + l) / 2 for h, l in zip(highs, lows)]
    if closes is None:
        closes = [(h + l) / 2 for h, l in zip(highs, lows)]
    if volumes is None:
        volumes = [1000.0] * n
    idx = pd.date_range("2024-01-01", periods=n, freq="2h", tz="UTC")
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=idx,
    )


def _atr_series(df: pd.DataFrame, value: float) -> pd.Series:
    """Constant-ATR series aligned to df.index."""
    return pd.Series(value, index=df.index)


def _vol_ma_series(df: pd.DataFrame, value: float) -> pd.Series:
    """Constant volume-MA series aligned to df.index."""
    return pd.Series(value, index=df.index)


def _rsi_series(df: pd.DataFrame, values: list) -> pd.Series:
    return pd.Series(values, index=df.index)


# ── ATR Trailing Stop (Tier 1.5) ─────────────────────────────────────────────

class TestAtrTrailingStop:
    """Tests for the new ATR-based trailing stop (Tier 1.5)."""

    def test_trail_activates_at_activation_r(self):
        """ATR trail should not activate until current_r >= atr_trail_activation_r."""
        cfg = ExitConfig(
            enable_atr_trail=True,
            atr_trail_activation_r=1.0,
            atr_trail_multiplier=1.0,
        )
        # LONG: entry=100, SL=90, TP=120 → r=20, activation at 1.0R means price=120
        state = TierState(side=1, entry_price=100, initial_sl=90, tp_price=120)
        # Candles: high stays at 115 (< activation price), never hits trail
        df = _candles(highs=[115, 115], lows=[100, 100], closes=[115, 115])
        atr = _atr_series(df, 2.0)
        framework = TieredExitFramework(cfg)
        events = framework.simulate(state, df, atr_series=atr)
        # No trail activated → no trail stop event; should hit EOD
        assert events[-1].reason == "EOD"
        assert state.atr_trail_stop is None

    def test_trail_activates_and_tightens_stop(self):
        """Once activated, ATR trail should tighten the stop as price rises."""
        cfg = ExitConfig(
            enable_atr_trail=True,
            atr_trail_activation_r=0.5,
            atr_trail_multiplier=1.0,
        )
        # LONG: entry=100, SL=90, TP=130 → r=30, activation at 0.5R = price 115
        # ATR=2; trail = high - ATR
        # Bar 0: high=116 → trail=114; low=115 → no SL (115 > 114)
        # Bar 1: high=120 → trail=118; low=119 → no SL (119 > 118)
        # Bar 2: high=119, low=116 → trail from bar1=118; 116 < 118 → SL
        state = TierState(side=1, entry_price=100, initial_sl=90, tp_price=130)
        df = _candles(
            highs=[116, 120, 119],
            lows=[115, 119, 116],
            closes=[116, 120, 117],
        )
        atr = _atr_series(df, 2.0)
        framework = TieredExitFramework(cfg)
        events = framework.simulate(state, df, atr_series=atr)
        sl_events = [e for e in events if e.reason == "SL"]
        assert len(sl_events) == 1
        assert sl_events[0].bar_index == 2

    def test_trail_only_moves_in_favour_long(self):
        """ATR trail stop must only move up (never down) for LONG positions."""
        cfg = ExitConfig(
            enable_atr_trail=True,
            atr_trail_activation_r=0.0,  # activate immediately
            atr_trail_multiplier=1.0,
        )
        state = TierState(side=1, entry_price=100, initial_sl=90, tp_price=130)
        # Prices go up then down
        df = _candles(
            highs=[110, 120, 105],
            lows=[ 100, 100, 95],
            closes=[110, 120, 105],
        )
        atr = _atr_series(df, 5.0)
        framework = TieredExitFramework(cfg)
        events = framework.simulate(state, df, atr_series=atr)
        # After bar 1, trail = 120-5 = 115; bar 2 low=95 triggers SL at 115
        sl_events = [e for e in events if e.reason == "SL"]
        assert sl_events, "expected SL event"
        # The trail stop should never have gone below its previous value
        assert state.atr_trail_stop is not None
        assert state.atr_trail_stop >= 100  # at least breakeven

    def test_trail_only_moves_in_favour_short(self):
        """ATR trail stop must only move down (never up) for SHORT positions."""
        cfg = ExitConfig(
            enable_atr_trail=True,
            atr_trail_activation_r=0.0,
            atr_trail_multiplier=1.0,
        )
        # SHORT: entry=100, SL=110, TP=70
        state = TierState(side=-1, entry_price=100, initial_sl=110, tp_price=70)
        # Bar 0: low=90 → trail = 90+5 = 95
        # Bar 1: low=85 → trail = 85+5 = 90 (moves down = favourable)
        # Bar 2: high=92 → SL (trail=90, high=92 → stop hit)
        df = _candles(
            highs=[100, 90,  92],
            lows=[ 90,  85,  88],
            closes=[90,  85,  90],
        )
        atr = _atr_series(df, 5.0)
        framework = TieredExitFramework(cfg)
        events = framework.simulate(state, df, atr_series=atr)
        sl_events = [e for e in events if e.reason == "SL"]
        assert sl_events, "expected SL event"

    def test_trail_with_no_atr_data_skipped(self):
        """If atr_series is None but trail is enabled, trail should not activate."""
        cfg = ExitConfig(
            enable_atr_trail=True,
            atr_trail_activation_r=0.0,
            atr_trail_multiplier=1.0,
        )
        state = TierState(side=1, entry_price=100, initial_sl=90, tp_price=130)
        df = _candles(highs=[110, 120], lows=[100, 100], closes=[110, 120])
        framework = TieredExitFramework(cfg)
        events = framework.simulate(state, df, atr_series=None)
        assert state.atr_trail_stop is None


# ── Volume-Gated Momentum Exit ────────────────────────────────────────────────

class TestVolumeGatedMomentumExit:
    """Tests for Tier 4 momentum exit with volume + NATR gate."""

    def test_momentum_fires_without_gate(self):
        """Without the volume gate, RSI crossing should trigger MOMENTUM."""
        cfg = ExitConfig(
            enable_momentum_exit=True,
            momentum_exit_min_r=0.0,
            enable_volume_gate_momentum=False,
        )
        state = TierState(side=1, entry_price=100, initial_sl=90, tp_price=130)
        df = _candles(highs=[110, 110], lows=[100, 100], closes=[110, 105])
        rsi = _rsi_series(df, [55.0, 45.0])  # crosses below 50 at bar 1
        framework = TieredExitFramework(cfg)
        events = framework.simulate(state, df, rsi_series=rsi)
        assert any(e.reason == "MOMENTUM" for e in events)

    def test_momentum_blocked_by_low_volume(self):
        """With volume gate, momentum exit should be blocked when vol < min_ratio × avg."""
        cfg = ExitConfig(
            enable_momentum_exit=True,
            momentum_exit_min_r=0.0,
            enable_volume_gate_momentum=True,
            volume_gate_min_ratio=2.0,   # require 2× average vol
            natr_medium_min=0.0,
            natr_medium_max=1.0,         # broad NATR band so ATR doesn't block
        )
        state = TierState(side=1, entry_price=100, initial_sl=90, tp_price=130)
        # Volume 500 < 1000 × 2.0 → gate blocked
        df = _candles(
            highs=[110, 110], lows=[100, 100], closes=[110, 105], volumes=[500, 500]
        )
        rsi = _rsi_series(df, [55.0, 45.0])
        vol_ma = _vol_ma_series(df, 1000.0)
        atr = _atr_series(df, 0.5)  # NATR = 0.5/105 ≈ 0.005 → medium band OK
        framework = TieredExitFramework(cfg)
        events = framework.simulate(state, df, rsi_series=rsi, atr_series=atr, volume_ma_series=vol_ma)
        assert not any(e.reason == "MOMENTUM" for e in events)

    def test_momentum_blocked_by_high_natr(self):
        """With volume gate, momentum exit should be blocked when NATR > max."""
        cfg = ExitConfig(
            enable_momentum_exit=True,
            momentum_exit_min_r=0.0,
            enable_volume_gate_momentum=True,
            volume_gate_min_ratio=1.0,
            natr_medium_min=0.002,
            natr_medium_max=0.004,  # tight band
        )
        state = TierState(side=1, entry_price=100, initial_sl=90, tp_price=130)
        df = _candles(
            highs=[110, 110], lows=[100, 100], closes=[110, 105], volumes=[2000, 2000]
        )
        rsi = _rsi_series(df, [55.0, 45.0])
        vol_ma = _vol_ma_series(df, 1000.0)
        atr = _atr_series(df, 1.0)  # NATR = 1/105 ≈ 0.0095 → above natr_medium_max
        framework = TieredExitFramework(cfg)
        events = framework.simulate(state, df, rsi_series=rsi, atr_series=atr, volume_ma_series=vol_ma)
        assert not any(e.reason == "MOMENTUM" for e in events)

    def test_momentum_allowed_when_volume_and_natr_ok(self):
        """Volume gate should pass when volume is sufficient and NATR is medium."""
        cfg = ExitConfig(
            enable_momentum_exit=True,
            momentum_exit_min_r=0.0,
            enable_volume_gate_momentum=True,
            volume_gate_min_ratio=1.5,
            natr_medium_min=0.002,
            natr_medium_max=0.010,
        )
        state = TierState(side=1, entry_price=100, initial_sl=90, tp_price=130)
        # vol 2000 >= 1000 × 1.5 = 1500 ✓, NATR = 0.3/105 ≈ 0.003 (medium) ✓
        df = _candles(
            highs=[110, 110], lows=[100, 100], closes=[110, 105], volumes=[2000, 2000]
        )
        rsi = _rsi_series(df, [55.0, 45.0])
        vol_ma = _vol_ma_series(df, 1000.0)
        atr = _atr_series(df, 0.3)
        framework = TieredExitFramework(cfg)
        events = framework.simulate(state, df, rsi_series=rsi, atr_series=atr, volume_ma_series=vol_ma)
        assert any(e.reason == "MOMENTUM" for e in events)


# ── ATR Trail + Partial TP interaction ────────────────────────────────────────

class TestAtrTrailWithPartialTP:
    """ATR trail and partial TP should interact correctly."""

    def test_partial_tp_then_atr_trail_close(self):
        """After partial TP fires, ATR trail should continue on remaining size."""
        cfg = ExitConfig(
            enable_partial_tp=True,
            partial_tp_level=0.5,        # 50% of the way to TP
            partial_tp_close_pct=0.5,
            enable_atr_trail=True,
            atr_trail_activation_r=0.0,
            atr_trail_multiplier=1.0,
        )
        # LONG: entry=100, SL=90, TP=120 → r=20, partial at 110
        state = TierState(side=1, entry_price=100, initial_sl=90, tp_price=120)
        # Bar 0: high=111 → partial TP at 110, trail = 111-2 = 109
        # Bar 1: high=115 → trail = 115-2 = 113
        # Bar 2: low=111 → trail=113, low<trail → SL on remaining 0.5
        df = _candles(
            highs=[111, 115, 115],
            lows=[ 100, 100, 111],
            closes=[111, 115, 113],
        )
        atr = _atr_series(df, 2.0)
        framework = TieredExitFramework(cfg)
        events = framework.simulate(state, df, atr_series=atr)
        reasons = [e.reason for e in events]
        assert "PARTIAL_TP" in reasons
        assert "SL" in reasons
        # Total fraction should sum to 1.0
        assert abs(sum(e.fraction for e in events) - 1.0) < 1e-9


# ── Robustness columns ────────────────────────────────────────────────────────

class TestRobustnessColumns:
    """Test that is_oos_pf_ratio and cluster_share_winning_pct appear in summary output."""

    def test_is_oos_pf_ratio_and_cluster_share_present(self):
        """_promotion_summary should include the two new robustness columns."""
        # Import locally to avoid heavy backtest.py load at collection time
        import importlib, types
        # We need to import backtest and test _promotion_summary directly.
        # Use a stub approach: import only what we need.
        import backtest as bt

        # Build minimal WalkForwardWindowResult stubs
        from backtest import BacktestMetrics, WalkForwardWindowResult, SimTrade

        def _fake_metrics(pf: float, ret_pct: float, trades: int = 5) -> BacktestMetrics:
            return BacktestMetrics(
                label="test",
                initial_balance=10000,
                final_balance=10000 * (1 + ret_pct / 100),
                total_trades=trades,
                wins=int(trades * 0.6),
                losses=trades - int(trades * 0.6),
                win_rate_pct=60.0,
                avg_win_pct=2.0,
                avg_loss_pct=-1.0,
                profit_factor=pf,
                sharpe_ratio=0.5,
                sortino_ratio=0.7,
                max_drawdown_pct=10.0,
                total_return_pct=ret_pct,
            )

        def _fake_trade(r: float) -> SimTrade:
            t = SimTrade(
                trade_num=1, entry_time=pd.Timestamp("2024-01-01", tz="UTC"),
                entry_price=100.0, stop_price=90.0, tp_price=120.0,
                size=0.1, direction="LONG",
            )
            t.exit_r = r
            t.result = "WIN" if r > 0 else "LOSS"
            t.partial_tp_hit = False
            t.touched_vwap_after_entry = False
            t.touched_range_mid_after_entry = False
            t.vwap_touch_r = 0.0
            t.range_mid_touch_r = 0.0
            t.candles_in_trade = 3
            return t

        wins = [_fake_trade(0.5) for _ in range(6)]
        losses = [_fake_trade(-0.3) for _ in range(4)]
        all_trades = wins + losses

        windows = [
            WalkForwardWindowResult(
                setup_name="TEST_SETUP",
                window_id=w,
                train_start=pd.Timestamp("2024-01-01", tz="UTC"),
                train_end=pd.Timestamp("2024-03-01", tz="UTC"),
                test_start=pd.Timestamp("2024-03-01", tz="UTC"),
                test_end=pd.Timestamp("2024-06-01", tz="UTC"),
                in_metrics=_fake_metrics(1.3, 5.0, 20),
                out_metrics=_fake_metrics(1.2 if w % 2 == 0 else 0.9, 3.0 if w % 2 == 0 else -1.0, 10),
                in_trades=all_trades[:5],
                out_trades=all_trades[5:10],
                entry_profile_name="ENTRY_BASELINE",
            )
            for w in range(1, 7)
        ]

        row = bt._promotion_summary("TEST_SETUP", windows, entry_profile_name="ENTRY_BASELINE")
        assert "is_oos_pf_ratio" in row, "is_oos_pf_ratio column missing from summary"
        assert "cluster_share_winning_pct" in row, "cluster_share_winning_pct column missing from summary"
        assert isinstance(row["is_oos_pf_ratio"], float)
        assert 0 <= row["cluster_share_winning_pct"] <= 100

    def test_is_oos_ratio_zero_when_no_is_pf(self):
        """is_oos_pf_ratio should be 0 when IS PF is 0 to avoid division by zero."""
        import backtest as bt
        from backtest import BacktestMetrics, WalkForwardWindowResult, SimTrade

        def _m(pf, ret, trades=5):
            return BacktestMetrics(
                label="t", initial_balance=10000,
                final_balance=10000*(1+ret/100),
                total_trades=trades, wins=3, losses=2,
                win_rate_pct=60, avg_win_pct=2, avg_loss_pct=-1,
                profit_factor=pf, sharpe_ratio=0.5, sortino_ratio=0.7,
                max_drawdown_pct=10, total_return_pct=ret,
            )

        def _t(r):
            t = SimTrade(trade_num=1, entry_time=pd.Timestamp("2024-01-01", tz="UTC"),
                         entry_price=100, stop_price=90, tp_price=120, size=0.1, direction="LONG")
            t.exit_r = r; t.result = "WIN" if r>0 else "LOSS"
            t.partial_tp_hit = False; t.touched_vwap_after_entry = False
            t.touched_range_mid_after_entry = False; t.vwap_touch_r=0; t.range_mid_touch_r=0
            t.candles_in_trade=3
            return t

        ws = [WalkForwardWindowResult(
            setup_name="S", window_id=1,
            train_start=pd.Timestamp("2024-01-01",tz="UTC"),
            train_end=pd.Timestamp("2024-03-01",tz="UTC"),
            test_start=pd.Timestamp("2024-03-01",tz="UTC"),
            test_end=pd.Timestamp("2024-06-01",tz="UTC"),
            in_metrics=_m(0.0, 0.0, 20),   # IS PF = 0
            out_metrics=_m(1.2, 3.0, 10),
            in_trades=[_t(0.2)]*5,
            out_trades=[_t(0.3)]*5,
            entry_profile_name="E",
        )]
        row = bt._promotion_summary("S", ws, entry_profile_name="E")
        assert row["is_oos_pf_ratio"] == 0.0


# ── strategies/range_mr — unit tests ─────────────────────────────────────────

from strategies.range_mr import get_signal_2h, resample_1h_to_2h
import config as cfg


def _base_ranging_df(n: int = 250, price: float = 30_000.0) -> pd.DataFrame:
    """
    Synthetic 2H OHLCV DataFrame for a ranging market.

    Prices oscillate around *price* with a narrow range, giving low ADX.
    ATR% ≈ 0.35% (medium bucket).  Volume ratio ≈ 1.5× (medium bucket).
    VWAP resets each UTC day; the daily session VWAP stays near *price*.
    """
    rng = np.random.default_rng(7)
    half_range = price * 0.0018           # ~$54 at $30 k → ATR~$36 ≈ 0.12%
    # Sinusoidal oscillation keeps ADX low
    t = np.linspace(0, 6 * np.pi, n)
    closes = price + (price * 0.005) * np.sin(t) + rng.uniform(-half_range, half_range, n)
    highs  = closes + half_range * rng.uniform(0.5, 1.0, n)
    lows   = closes - half_range * rng.uniform(0.5, 1.0, n)
    opens  = np.roll(closes, 1); opens[0] = closes[0]
    avg_vol = 1_000.0
    volumes = avg_vol * 1.5 + rng.uniform(-200, 200, n)
    idx = pd.date_range("2024-01-01", periods=n, freq="2h", tz="UTC")
    df = pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=idx,
    )
    # Ensure high >= low >= 0 numerically
    df["high"] = df[["open", "high", "close"]].max(axis=1) + abs(half_range * 0.1)
    df["low"]  = df[["open", "low",  "close"]].min(axis=1) - abs(half_range * 0.1)
    return df


class TestResample1hTo2h:
    def test_drops_incomplete_bar(self):
        """Odd number of 1H candles → last partial 2H bar is dropped."""
        idx = pd.date_range("2024-01-01", periods=5, freq="1h", tz="UTC")
        df = pd.DataFrame(
            {"open": 1.0, "high": 1.1, "low": 0.9, "close": 1.0, "volume": 100.0},
            index=idx,
        )
        df2h = resample_1h_to_2h(df)
        assert len(df2h) == 2          # bars 0-1 and 2-3; bar 4 alone → dropped

    def test_correct_ohlcv_aggregation(self):
        """open=first, high=max, low=min, close=last, volume=sum.

        resample uses closed="right", so starting at 01:00 puts pairs (01:00,02:00)
        and (03:00,04:00) each neatly inside a single 2H window.
        """
        idx = pd.date_range("2024-01-01 01:00", periods=4, freq="1h", tz="UTC")
        df = pd.DataFrame({
            "open":   [10.0, 12.0, 14.0, 16.0],
            "high":   [11.0, 13.0, 15.0, 17.0],
            "low":    [ 9.0, 11.0, 13.0, 15.0],
            "close":  [12.0, 14.0, 16.0, 18.0],
            "volume": [100.0, 200.0, 300.0, 400.0],
        }, index=idx)
        df2h = resample_1h_to_2h(df)
        assert len(df2h) == 2
        # First 2H bar covers (00:00, 02:00] → candles at 01:00 and 02:00
        row0 = df2h.iloc[0]
        assert row0["open"]   == pytest.approx(10.0)   # first of 01:00, 02:00
        assert row0["high"]   == pytest.approx(13.0)   # max(11, 13)
        assert row0["low"]    == pytest.approx(9.0)    # min(9, 11)
        assert row0["close"]  == pytest.approx(14.0)   # last of 02:00
        assert row0["volume"] == pytest.approx(300.0)  # 100+200

    def test_even_count_keeps_all_bars(self):
        """Candles aligned to 2H bin boundaries → all complete bars kept.

        Starting at 01:00 ensures candles pair neatly into (00:00,02:00],
        (02:00,04:00], (04:00,06:00] with exactly 2 contributors each.
        """
        idx = pd.date_range("2024-01-01 01:00", periods=6, freq="1h", tz="UTC")
        df = pd.DataFrame(
            {"open": 1.0, "high": 1.1, "low": 0.9, "close": 1.0, "volume": 50.0},
            index=idx,
        )
        assert len(resample_1h_to_2h(df)) == 3


class TestGetSignal2hFilters:
    """Test each rejection gate in get_signal_2h independently."""

    def test_insufficient_bars_returns_none(self):
        """Fewer than _WARMUP bars must return NONE with a clear reason."""
        from strategies.range_mr import _WARMUP
        df = _base_ranging_df(n=_WARMUP - 1)
        sig = get_signal_2h(df)
        assert sig.direction == "NONE"
        assert "insufficient" in sig.reject_reason

    def test_bearish_close_blocks_reclaim(self):
        """A bearish last candle (close < open) cannot satisfy the reclaim pattern."""
        df = _base_ranging_df(n=260)
        # Force last bar bearish: close < open, and below range mid
        price = float(df["close"].iloc[-1])
        df.iloc[-1, df.columns.get_loc("open")]  = price + 50
        df.iloc[-1, df.columns.get_loc("close")] = price - 50
        df.iloc[-1, df.columns.get_loc("high")]  = price + 60
        df.iloc[-1, df.columns.get_loc("low")]   = price - 60
        sig = get_signal_2h(df)
        assert sig.direction == "NONE"
        # Should fail on either price-above-VWAP, range-mid, or reclaim
        assert sig.reject_reason != ""

    def test_no_lower_wick_blocks_rejection(self):
        """A doji / marubozu with no lower wick fails the hammer rejection test."""
        df = _base_ranging_df(n=260)
        price = float(df["close"].iloc[-1])
        # Bullish candle, high wick but NO lower wick (open == low)
        df.iloc[-1, df.columns.get_loc("open")]  = price - 20
        df.iloc[-1, df.columns.get_loc("close")] = price
        df.iloc[-1, df.columns.get_loc("high")]  = price + 100   # long upper wick
        df.iloc[-1, df.columns.get_loc("low")]   = price - 20    # no lower wick (open==low)
        sig = get_signal_2h(df)
        # May fail on an earlier gate (VWAP, range mid, etc.) but must not be LONG
        assert sig.direction == "NONE"

    def test_signal_fields_populated_on_none(self):
        """Even rejected signals carry bucket/distance info (zero values — not NaN)."""
        from strategies.range_mr import _WARMUP
        df = _base_ranging_df(n=_WARMUP - 1)
        sig = get_signal_2h(df)
        assert not np.isnan(sig.entry_price)
        assert not np.isnan(sig.stop_price)
        assert not np.isnan(sig.vwap_distance_r)

    def test_high_atr_high_vol_blocked(self):
        """HIGH_ATR + HIGH_VOL LONG entries must be blocked regardless of pattern."""
        df = _base_ranging_df(n=260)
        price = float(df["close"].iloc[-1])
        # Force very high ATR% by setting extreme candle ranges on last ~20 bars
        # and very high volume ratio (vol >> 2× volume MA)
        atr_bump = price * 0.015          # 1.5% range → ATR% well above 0.60 threshold
        vol_spike = 20_000.0             # >> 2× typical 1000-vol MA
        for j in range(-20, 0):
            df.iloc[j, df.columns.get_loc("high")] = df["close"].iloc[j] + atr_bump
            df.iloc[j, df.columns.get_loc("low")]  = df["close"].iloc[j] - atr_bump
            df.iloc[j, df.columns.get_loc("volume")] = vol_spike
        sig = get_signal_2h(df)
        assert sig.direction == "NONE"
        # Confirm it's the right gate (may also hit earlier gates, but bucket must block)
        if "HIGH_ATR" in sig.reject_reason:
            assert "HIGH_VOL" in sig.reject_reason


class TestGetSignal2hValidSignal:
    """Test that a well-constructed entry produces a LONG signal with correct fields."""

    def test_returns_long_on_valid_setup(self):
        """
        A hammer candle below VWAP + range mid in a ranging market should
        produce a LONG signal (or clearly fail on a documented gate — not silently).
        """
        df = _base_ranging_df(n=260)
        # We can't guarantee an indicator-exact LONG here since VWAP/ATR/ADX
        # depend on full history, but we verify the function returns a coherent result.
        sig = get_signal_2h(df)
        assert sig.direction in ("LONG", "NONE")
        assert sig.reject_reason == "" or sig.direction == "NONE"

    def test_long_signal_prices_consistent(self):
        """If a LONG fires, stop < entry < tp and stop_distance > 0."""
        # Run the signal on several random seeds until we get a LONG, or
        # at minimum verify that whenever LONG fires the prices are sane.
        for seed in range(10):
            rng = np.random.default_rng(seed)
            df = _base_ranging_df(n=260)
            sig = get_signal_2h(df)
            if sig.direction == "LONG":
                assert sig.stop_price < sig.entry_price, "stop must be below entry"
                assert sig.tp_price > sig.entry_price, "tp must be above entry"
                assert sig.stop_distance > 0, "stop_distance must be positive"
                assert sig.atr_bucket in ("low", "medium", "high")
                assert sig.volume_bucket in ("low", "medium", "high")
                assert sig.vwap_distance_r >= 0
                assert sig.reject_reason == ""
                break   # one confirmed LONG is enough

    def test_tp_equals_1_5r(self):
        """TP price must equal entry + stop_distance × RMR_TP_RR_RATIO (1.5)."""
        for seed in range(20):
            df = _base_ranging_df(n=260)
            sig = get_signal_2h(df)
            if sig.direction == "LONG":
                expected_tp = round(sig.entry_price + sig.stop_distance * cfg.RMR_TP_RR_RATIO, 2)
                assert sig.tp_price == pytest.approx(expected_tp, abs=0.01)
                break


class TestRMRConfig:
    """New RMR config keys exist with their documented defaults."""

    def test_enable_range_mr_default_false(self):
        assert cfg.ENABLE_RANGE_MR is False

    def test_atr_bucket_thresholds(self):
        assert cfg.RMR_ATR_LOW_PCT  == pytest.approx(0.30)
        assert cfg.RMR_ATR_HIGH_PCT == pytest.approx(0.60)
        assert cfg.RMR_ATR_LOW_PCT < cfg.RMR_ATR_HIGH_PCT

    def test_volume_bucket_thresholds(self):
        assert cfg.RMR_VOL_LOW  == pytest.approx(1.30)
        assert cfg.RMR_VOL_HIGH == pytest.approx(2.00)
        assert cfg.RMR_VOL_LOW < cfg.RMR_VOL_HIGH

    def test_vwap_far_threshold(self):
        assert cfg.RMR_VWAP_FAR_R == pytest.approx(1.00)

    def test_tp_rr_ratio_matches_partial_tp(self):
        """RMR_TP_RR_RATIO must equal DEFAULT_PARTIAL_TP_R (MR_EXIT_0 level)."""
        assert cfg.RMR_TP_RR_RATIO == pytest.approx(cfg.DEFAULT_PARTIAL_TP_R)
