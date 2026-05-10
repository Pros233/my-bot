"""
tests/test_entry_buffer.py — Unit tests for EntryConfirmationBuffer (ROBUSTNESS #6).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from entry_buffer import EntryConfirmationBuffer


class TestEntryConfirmationBuffer:
    def test_no_drift_is_valid(self):
        buf = EntryConfirmationBuffer(buffer_pct=0.0015)
        assert buf.check(64_000.0, 64_000.0) is True

    def test_within_buffer_is_valid(self):
        buf = EntryConfirmationBuffer(buffer_pct=0.0015)
        # 0.1% drift — within 0.15% buffer
        assert buf.check(64_000.0, 64_064.0) is True

    def test_exactly_at_buffer_is_valid(self):
        buf = EntryConfirmationBuffer(buffer_pct=0.0015)
        entry = 64_000.0 * (1 + 0.0015)
        assert buf.check(64_000.0, entry) is True

    def test_beyond_buffer_is_invalid(self):
        buf = EntryConfirmationBuffer(buffer_pct=0.0015)
        # 0.5% drift — beyond 0.15%
        entry = 64_000.0 * 1.005
        assert buf.check(64_000.0, entry) is False

    def test_negative_drift_still_checked(self):
        buf = EntryConfirmationBuffer(buffer_pct=0.0015)
        # Price dropped 0.5% — also a chase (bot missed the dip)
        entry = 64_000.0 * 0.995
        assert buf.check(64_000.0, entry) is False

    def test_chase_skip_rate_zero_initially(self):
        buf = EntryConfirmationBuffer()
        assert buf.chase_skip_rate == 0.0

    def test_chase_skip_rate_accumulates(self):
        buf = EntryConfirmationBuffer(buffer_pct=0.0015)
        buf.check(64_000.0, 64_000.0)   # OK
        buf.check(64_000.0, 64_500.0)   # CHASED
        buf.check(64_000.0, 64_600.0)   # CHASED
        assert buf.total_checks == 3
        assert buf.chased_count == 2
        assert abs(buf.chase_skip_rate - 2 / 3) < 1e-9

    def test_reset_counters(self):
        buf = EntryConfirmationBuffer(buffer_pct=0.0015)
        buf.check(64_000.0, 65_000.0)   # CHASED
        buf.reset_counters()
        assert buf.total_checks == 0
        assert buf.chased_count == 0
        assert buf.chase_skip_rate == 0.0

    def test_invalid_signal_price(self):
        buf = EntryConfirmationBuffer()
        assert buf.check(0.0, 64_000.0) is False
        assert buf.check(float("nan"), 64_000.0) is False
        assert buf.check(64_000.0, float("inf")) is False

    def test_custom_buffer_pct(self):
        buf = EntryConfirmationBuffer(buffer_pct=0.005)  # 0.5%
        assert buf.check(64_000.0, 64_300.0) is True    # 0.47% drift → OK
        assert buf.check(64_000.0, 64_400.0) is False   # 0.63% drift → CHASED

    def test_csv_written(self, tmp_path, monkeypatch):
        import entry_buffer as eb
        monkeypatch.setattr(eb, "ENTRY_QUALITY_LOG", tmp_path / "entry_quality.csv")
        buf = EntryConfirmationBuffer(log_to_csv=True)
        buf.check(64_000.0, 64_500.0, timestamp="2023-01-01T00:00:00")
        assert (tmp_path / "entry_quality.csv").exists()

    def test_no_csv_without_flag(self, tmp_path, monkeypatch):
        import entry_buffer as eb
        monkeypatch.setattr(eb, "ENTRY_QUALITY_LOG", tmp_path / "entry_quality.csv")
        buf = EntryConfirmationBuffer(log_to_csv=False)
        buf.check(64_000.0, 64_500.0)
        assert not (tmp_path / "entry_quality.csv").exists()
