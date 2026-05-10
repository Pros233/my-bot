"""
tests/test_sample_guard.py — Unit tests for SampleGuard (ROBUSTNESS #5).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from sample_guard import SampleGuard, WindowResult


def _wr(trade_count: int, regime: str = "RANGING", pf: float = 1.5) -> WindowResult:
    """Helper to build a minimal WindowResult."""
    return WindowResult(
        window_start="2023-01-01",
        window_end="2023-04-01",
        dominant_regime=regime,
        trade_count=trade_count,
        pf=pf,
        win_rate=0.55,
        median_r=0.3,
        worst_window_pf=pf,
        tp_hit_rate=0.5,
    )


class TestSampleGuardValidate:
    def test_sufficient_samples_valid(self):
        guard = SampleGuard(min_window_trades=20)
        windows = [_wr(25), _wr(30), _wr(22)]
        guard.validate(windows)
        assert all(w.sample_valid for w in windows)

    def test_insufficient_sample_flagged(self):
        guard = SampleGuard(min_window_trades=20)
        windows = [_wr(5), _wr(30)]
        guard.validate(windows)
        assert windows[0].sample_valid is False
        assert windows[0].exclusion_reason == "INSUFFICIENT_SAMPLE"
        assert windows[1].sample_valid is True

    def test_exactly_at_threshold_valid(self):
        guard = SampleGuard(min_window_trades=20)
        windows = [_wr(20)]
        guard.validate(windows)
        assert windows[0].sample_valid is True

    def test_one_below_threshold_invalid(self):
        guard = SampleGuard(min_window_trades=20)
        windows = [_wr(19)]
        guard.validate(windows)
        assert windows[0].sample_valid is False

    def test_validate_returns_same_list(self):
        guard = SampleGuard()
        windows = [_wr(25), _wr(30)]
        result = guard.validate(windows)
        assert result is windows


class TestSampleGuardMetrics:
    def test_valid_window_pct_all_valid(self):
        guard = SampleGuard(min_window_trades=10)
        windows = [_wr(15), _wr(20), _wr(25)]
        guard.validate(windows)
        assert guard.valid_window_pct(windows) == 1.0

    def test_valid_window_pct_partial(self):
        guard = SampleGuard(min_window_trades=20)
        windows = [_wr(5), _wr(5), _wr(25)]    # 2 invalid, 1 valid
        guard.validate(windows)
        assert abs(guard.valid_window_pct(windows) - 1 / 3) < 1e-9

    def test_valid_window_pct_empty(self):
        guard = SampleGuard()
        assert guard.valid_window_pct([]) == 0.0

    def test_passes_gate_true(self):
        guard = SampleGuard(min_window_trades=10, valid_window_pct_threshold=0.70)
        windows = [_wr(15)] * 8 + [_wr(3)] * 2   # 80% valid
        guard.validate(windows)
        assert guard.passes_gate(windows) is True

    def test_passes_gate_false(self):
        guard = SampleGuard(min_window_trades=10, valid_window_pct_threshold=0.70)
        windows = [_wr(15)] * 5 + [_wr(3)] * 5   # 50% valid
        guard.validate(windows)
        assert guard.passes_gate(windows) is False


class TestSampleGuardAggregate:
    def test_aggregate_valid_excludes_invalid(self):
        guard = SampleGuard(min_window_trades=20)
        # pf=2.0 for valid, pf=0.1 for invalid
        good = _wr(30, pf=2.0)
        bad  = _wr(5,  pf=0.1)
        windows = [good, bad]
        guard.validate(windows)
        agg = guard.aggregate_valid(windows)
        assert agg["valid_count"] == 1
        assert abs(agg["avg_pf"] - 2.0) < 1e-9

    def test_aggregate_empty_valid(self):
        guard = SampleGuard(min_window_trades=20)
        windows = [_wr(5)]
        guard.validate(windows)
        agg = guard.aggregate_valid(windows)
        assert agg["valid_count"] == 0
        assert agg["avg_pf"] == 0.0

    def test_aggregate_keys_present(self):
        guard = SampleGuard(min_window_trades=5)
        windows = [_wr(20), _wr(25)]
        guard.validate(windows)
        agg = guard.aggregate_valid(windows)
        expected_keys = {
            "valid_count", "total_count", "valid_window_pct",
            "avg_pf", "avg_win_rate", "median_r", "worst_pf", "avg_tp_hit_rate",
        }
        assert expected_keys.issubset(agg.keys())


class TestSampleGuardCSV:
    def test_excluded_written_to_csv(self, tmp_path, monkeypatch):
        import sample_guard as sg
        monkeypatch.setattr(sg, "EXCLUDED_WINDOWS_LOG", tmp_path / "excluded.csv")
        guard = SampleGuard(min_window_trades=20, log_to_csv=True)
        windows = [_wr(5), _wr(30)]
        guard.validate(windows)
        assert (tmp_path / "excluded.csv").exists()

    def test_no_csv_when_no_exclusions(self, tmp_path, monkeypatch):
        import sample_guard as sg
        monkeypatch.setattr(sg, "EXCLUDED_WINDOWS_LOG", tmp_path / "excluded.csv")
        guard = SampleGuard(min_window_trades=5, log_to_csv=True)
        windows = [_wr(30)]
        guard.validate(windows)
        # All valid — no excluded rows written
        assert not (tmp_path / "excluded.csv").exists()
