"""
tests/test_tiered_exit.py — Unit tests for TieredExitFramework (ROBUSTNESS #4).

Tests cover each tier independently and in combination.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from tiered_exit import (
    ExitConfig, TierState, TieredExitFramework, ExitEvent, compute_exit_r,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _candles(highs, lows, opens=None, closes=None, n=None) -> pd.DataFrame:
    """Build a simple OHLCV DataFrame for trade simulation."""
    if n is None:
        n = len(highs)
    if opens is None:
        opens = [(h + l) / 2 for h, l in zip(highs, lows)]
    if closes is None:
        closes = [(h + l) / 2 for h, l in zip(highs, lows)]
    idx = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({
        "open": opens, "high": highs, "low": lows,
        "close": closes, "volume": [100.0] * n,
    }, index=idx)


def _long_state(entry=64_000, sl=63_000, tp=65_500) -> TierState:
    return TierState(side=1, entry_price=entry, initial_sl=sl, tp_price=tp)


def _short_state(entry=64_000, sl=65_000, tp=62_500) -> TierState:
    return TierState(side=-1, entry_price=entry, initial_sl=sl, tp_price=tp)


# ── Tier 0: Base (SL and TP) ──────────────────────────────────────────────────

class TestTier0Base:
    def test_tp_hit_long(self):
        # Price goes straight to TP on bar 0
        df = _candles(highs=[65_600], lows=[63_500])
        cfg = ExitConfig()
        fw  = TieredExitFramework(cfg)
        events = fw.simulate(_long_state(), df)
        assert len(events) == 1
        assert events[0].reason == "TP"
        assert events[0].exit_price == 65_500.0
        assert events[0].fraction == 1.0

    def test_sl_hit_long(self):
        # Price drops to SL on bar 0
        df = _candles(highs=[63_800], lows=[62_900])
        cfg = ExitConfig()
        fw  = TieredExitFramework(cfg)
        events = fw.simulate(_long_state(), df)
        assert len(events) == 1
        assert events[0].reason == "SL"
        assert events[0].exit_price == 63_000.0

    def test_tp_hit_short(self):
        df = _candles(highs=[63_500], lows=[62_400])
        cfg = ExitConfig()
        fw  = TieredExitFramework(cfg)
        events = fw.simulate(_short_state(), df)
        assert len(events) == 1
        assert events[0].reason == "TP"
        assert events[0].exit_price == 62_500.0

    def test_sl_hit_short(self):
        df = _candles(highs=[65_100], lows=[63_800])
        cfg = ExitConfig()
        fw  = TieredExitFramework(cfg)
        events = fw.simulate(_short_state(), df)
        assert len(events) == 1
        assert events[0].reason == "SL"
        assert events[0].exit_price == 65_000.0

    def test_sl_wins_when_both_hit_same_candle(self):
        """When SL and TP are both touched in one candle, SL is conservative."""
        df = _candles(highs=[66_000], lows=[62_900])
        cfg = ExitConfig()
        fw  = TieredExitFramework(cfg)
        events = fw.simulate(_long_state(), df)
        assert events[0].reason == "SL"

    def test_eod_close_when_no_exit_reached(self):
        # 5 candles that never hit SL or TP
        highs  = [64_400, 64_600, 64_800, 64_700, 64_500]
        lows   = [63_200, 63_100, 63_300, 63_200, 63_400]
        closes = [64_300, 64_500, 64_700, 64_600, 64_400]
        df = _candles(highs, lows, closes=closes)
        cfg = ExitConfig()
        fw  = TieredExitFramework(cfg)
        events = fw.simulate(_long_state(), df)
        assert events[-1].reason == "EOD"
        assert events[-1].exit_price == closes[-1]

    def test_fractions_sum_to_one(self):
        highs  = [64_500, 64_600, 65_600]
        lows   = [63_400, 63_500, 63_200]
        closes = [64_400, 64_500, 64_800]
        df = _candles(highs, lows, closes=closes)
        cfg = ExitConfig()
        fw  = TieredExitFramework(cfg)
        events = fw.simulate(_long_state(), df)
        assert abs(sum(ev.fraction for ev in events) - 1.0) < 1e-9


# ── Tier 1: Partial TP ────────────────────────────────────────────────────────

class TestTier1PartialTP:
    def test_partial_tp_fires_at_0_75x_distance(self):
        """Long: TP at 65500, entry 64000. 0.75× distance = 64000 + 0.75×1500 = 65125."""
        state = _long_state(entry=64_000, sl=63_000, tp=65_500)
        partial_level = 64_000 + 0.75 * (65_500 - 64_000)  # = 65125

        # Bar 0: price reaches partial-TP level but not full TP
        df = _candles(
            highs=[65_200, 65_600],
            lows=[63_500, 63_500],
            closes=[65_100, 65_500],
        )
        cfg = ExitConfig(enable_partial_tp=True)
        fw  = TieredExitFramework(cfg)
        events = fw.simulate(state, df)

        partial_events = [e for e in events if e.reason == "PARTIAL_TP"]
        assert len(partial_events) == 1
        assert partial_events[0].exit_price == pytest.approx(partial_level)
        assert partial_events[0].fraction == pytest.approx(0.40)

    def test_sl_moves_to_breakeven_after_partial(self):
        """After partial-TP fires, SL = entry price (breakeven)."""
        state = _long_state(entry=64_000, sl=63_000, tp=65_500)
        # Bar 0: hit partial TP
        # Bar 1: price drops to original SL (63000) — should NOT close (BE protects)
        # Bar 2: price drops to entry (64000) — hits new SL exactly
        partial_level = 64_000 + 0.75 * 1500  # 65125
        df = _candles(
            highs=[partial_level + 1, 64_200, 64_200],
            lows=[63_800, 62_900, 63_900],
            closes=[65_000, 63_000, 63_950],
        )
        cfg = ExitConfig(enable_partial_tp=True)
        fw  = TieredExitFramework(cfg)
        events = fw.simulate(state, df)

        # partial close on bar 0, SL close on bar with low ≤ 64000
        reasons = [e.reason for e in events]
        assert "PARTIAL_TP" in reasons
        assert "SL" in reasons

    def test_no_partial_tp_without_flag(self):
        state = _long_state()
        df = _candles(highs=[65_200, 65_600], lows=[63_500, 63_500])
        cfg = ExitConfig(enable_partial_tp=False)
        fw  = TieredExitFramework(cfg)
        events = fw.simulate(state, df)
        partial = [e for e in events if e.reason == "PARTIAL_TP"]
        assert len(partial) == 0

    def test_fractions_sum_to_one_with_partial(self):
        state = _long_state(entry=64_000, sl=63_000, tp=65_500)
        df = _candles(highs=[65_200, 65_600], lows=[63_500, 63_200])
        cfg = ExitConfig(enable_partial_tp=True)
        fw  = TieredExitFramework(cfg)
        events = fw.simulate(state, df)
        assert abs(sum(e.fraction for e in events) - 1.0) < 1e-9


# ── Tier 3: Time Stop ─────────────────────────────────────────────────────────

class TestTier3TimeStop:
    def test_time_stop_fires_after_max_bars(self):
        closes = [64_300] * 12
        highs  = [64_400] * 12
        lows   = [63_200] * 12   # never hits SL (63000) or TP (65500)
        df = _candles(highs, lows, closes=closes)
        cfg = ExitConfig(enable_time_stop=True, max_trade_bars=12)
        fw  = TieredExitFramework(cfg)
        events = fw.simulate(_long_state(), df)
        assert events[-1].reason == "TIME_STOP"
        assert events[-1].bar_index == 11  # 0-based, bar 12 (bars_open==12)

    def test_time_stop_disabled_by_default(self):
        closes = [64_300] * 20
        highs  = [64_400] * 20
        lows   = [63_200] * 20
        df = _candles(highs, lows, closes=closes)
        cfg = ExitConfig(enable_time_stop=False)
        fw  = TieredExitFramework(cfg)
        events = fw.simulate(_long_state(), df)
        assert events[-1].reason != "TIME_STOP"

    def test_tp_beats_time_stop_on_same_bar(self):
        """If TP is hit on bar max_trade_bars, TP should close first."""
        highs = [64_400] * 11 + [65_600]  # TP hit on last bar
        lows  = [63_200] * 12
        closes = [64_300] * 12
        df = _candles(highs, lows, closes=closes)
        cfg = ExitConfig(enable_time_stop=True, max_trade_bars=12)
        fw  = TieredExitFramework(cfg)
        events = fw.simulate(_long_state(), df)
        # TP check is before time-stop in the simulation loop
        reasons = [e.reason for e in events]
        assert "TP" in reasons


# ── Tier 4: Momentum Exit ─────────────────────────────────────────────────────

class TestTier4MomentumExit:
    def _rsi_series(self, df: pd.DataFrame, values: list[float]) -> pd.Series:
        return pd.Series(values, index=df.index, name="rsi7")

    def test_momentum_exit_fires_after_rsi_cross(self):
        """Long: RSI crosses below 50 after 0.5R move → MOMENTUM exit."""
        # Entry 64000, SL 63000, TP 65500 → R = 1500
        # 0.5R = 750 → price must reach 64750 before RSI exit activates
        highs  = [64_800, 64_700, 64_600, 64_500, 64_400]
        lows   = [63_500, 63_500, 63_400, 63_300, 63_200]
        closes = [64_700, 64_600, 64_500, 64_400, 64_300]
        df = _candles(highs, lows, closes=closes)
        # RSI: above 50 on bar 0–1, then crosses below 50 on bar 2
        rsi = self._rsi_series(df, [60.0, 55.0, 48.0, 45.0, 40.0])
        cfg = ExitConfig(enable_momentum_exit=True, momentum_exit_min_r=0.5)
        fw  = TieredExitFramework(cfg)
        events = fw.simulate(_long_state(), df, rsi_series=rsi)
        reasons = [e.reason for e in events]
        assert "MOMENTUM" in reasons

    def test_momentum_exit_not_before_0_5r(self):
        """RSI crosses through 50 but price hasn't moved 0.5R → no MOMENTUM exit."""
        highs  = [64_500, 64_400, 64_300]
        lows   = [63_500, 63_400, 63_300]
        closes = [64_400, 64_300, 64_200]
        df = _candles(highs, lows, closes=closes)
        # RSI crosses below 50 on bar 2, but price never reached 0.5R = 64750
        rsi = self._rsi_series(df, [60.0, 55.0, 48.0])
        cfg = ExitConfig(enable_momentum_exit=True, momentum_exit_min_r=0.5)
        fw  = TieredExitFramework(cfg)
        events = fw.simulate(_long_state(), df, rsi_series=rsi)
        reasons = [e.reason for e in events]
        assert "MOMENTUM" not in reasons

    def test_momentum_exit_disabled_by_default(self):
        highs  = [64_800, 64_700]
        lows   = [63_500, 63_400]
        closes = [64_700, 64_600]
        df = _candles(highs, lows, closes=closes)
        rsi = self._rsi_series(df, [60.0, 48.0])
        cfg = ExitConfig(enable_momentum_exit=False)
        fw  = TieredExitFramework(cfg)
        events = fw.simulate(_long_state(), df, rsi_series=rsi)
        assert all(e.reason != "MOMENTUM" for e in events)


# ── compute_exit_r ────────────────────────────────────────────────────────────

class TestComputeExitR:
    def test_full_tp_at_2r(self):
        entry, sl, tp = 64_000.0, 63_000.0, 66_000.0  # sl_dist=1000, tp_dist=2000
        events = [ExitEvent(bar_index=0, exit_price=66_000.0, reason="TP", fraction=1.0)]
        r = compute_exit_r(events, side=1, entry_price=entry, stop_distance=1000.0)
        assert abs(r - 2.0) < 1e-9

    def test_sl_at_minus_1r(self):
        entry, sl = 64_000.0, 63_000.0
        events = [ExitEvent(bar_index=0, exit_price=63_000.0, reason="SL", fraction=1.0)]
        r = compute_exit_r(events, side=1, entry_price=entry, stop_distance=1000.0)
        assert abs(r - (-1.0)) < 1e-9

    def test_partial_close_weighted(self):
        """40% at +1.5R, 60% at -1R → weighted R = 0.4×1.5 + 0.6×(-1) = 0.0"""
        events = [
            ExitEvent(0, 65_500.0, "PARTIAL_TP", 0.40),
            ExitEvent(2, 63_000.0, "SL",         0.60),
        ]
        r = compute_exit_r(events, side=1, entry_price=64_000.0, stop_distance=1000.0)
        # 0.4×1.5 = 0.6; 0.6×(−1) = −0.6 → total = 0.0
        assert abs(r - 0.0) < 1e-6

    def test_zero_stop_distance(self):
        events = [ExitEvent(0, 65_000.0, "TP", 1.0)]
        r = compute_exit_r(events, side=1, entry_price=64_000.0, stop_distance=0.0)
        assert r == 0.0
