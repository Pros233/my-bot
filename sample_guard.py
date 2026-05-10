"""
sample_guard.py — Minimum Sample Enforcement (ROBUSTNESS #5).

Any walk-forward window with trade_count < MIN_WINDOW_TRADES is flagged as
INSUFFICIENT_SAMPLE and excluded from PF / win-rate aggregation.  The
strategy only passes if valid_window_pct >= VALID_WINDOW_PCT.

Usage
-----
    from sample_guard import SampleGuard, WindowResult

    guard = SampleGuard(log_to_csv=True)
    windows = guard.validate(window_list)     # marks sample_valid on each
    pct = guard.valid_window_pct(windows)     # e.g. 0.75
    passed = guard.passes_gate(windows)       # True/False
    agg = guard.aggregate_valid(windows)      # dict of aggregated metrics
"""
from __future__ import annotations

import csv
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import config

EXCLUDED_WINDOWS_LOG = Path("outputs/research/excluded_windows.csv")
_HEADERS = [
    "window_start", "window_end", "dominant_regime",
    "trade_count", "min_required", "reason",
]


@dataclass
class WindowResult:
    """Metrics for one walk-forward window."""
    window_start: str
    window_end: str
    dominant_regime: str
    trade_count: int
    pf: float           # profit factor (gross wins / gross losses)
    win_rate: float     # fraction of trades that were winners
    median_r: float     # median R-multiple
    worst_window_pf: float   # kept for compatibility; equals pf for a single window
    tp_hit_rate: float  # fraction of trades that hit TP
    avg_r: float = 0.0         # mean R-multiple across all trades
    sharpe_ratio: float = 0.0  # mean(R) / std(R); 0 when n<2 or std=0
    sortino_ratio: float = 0.0 # mean(R) / downside_std(R); 0 when no losses
    # Set by SampleGuard.validate()
    sample_valid: bool = True
    exclusion_reason: Optional[str] = None


class SampleGuard:
    """
    Validates walk-forward windows against minimum sample size rules.

    Parameters
    ----------
    min_window_trades : int
        Windows with fewer trades are marked INSUFFICIENT_SAMPLE.
    valid_window_pct_threshold : float
        Required fraction of valid windows to pass the robustness gate.
    log_to_csv : bool
        Write excluded windows to excluded_windows.csv.
    """

    def __init__(
        self,
        min_window_trades: int = config.MIN_WINDOW_TRADES,
        valid_window_pct_threshold: float = config.VALID_WINDOW_PCT,
        log_to_csv: bool = False,
    ) -> None:
        self.min_window_trades = min_window_trades
        self.valid_window_pct_threshold = valid_window_pct_threshold
        self.log_to_csv = log_to_csv
        self._initialized = False

    # ── Public ─────────────────────────────────────────────────────────────────

    def validate(self, windows: list[WindowResult]) -> list[WindowResult]:
        """
        Mark each window as sample_valid or INSUFFICIENT_SAMPLE in-place.

        Returns the same list with sample_valid and exclusion_reason set.
        """
        excluded = []
        for w in windows:
            if w.trade_count < self.min_window_trades:
                w.sample_valid = False
                w.exclusion_reason = "INSUFFICIENT_SAMPLE"
                excluded.append(w)
            else:
                w.sample_valid = True
                w.exclusion_reason = None

        if self.log_to_csv and excluded:
            self._log_excluded(excluded)

        return windows

    def valid_window_pct(self, windows: list[WindowResult]) -> float:
        """Fraction of windows where sample_valid=True."""
        if not windows:
            return 0.0
        return sum(1 for w in windows if w.sample_valid) / len(windows)

    def passes_gate(self, windows: list[WindowResult]) -> bool:
        """Return True if valid_window_pct meets the threshold."""
        return self.valid_window_pct(windows) >= self.valid_window_pct_threshold

    def aggregate_valid(self, windows: list[WindowResult]) -> dict:
        """Return aggregated metrics across sample-valid windows only."""
        valid = [w for w in windows if w.sample_valid]
        total = len(windows)

        if not valid:
            return {
                "valid_count": 0,
                "total_count": total,
                "valid_window_pct": 0.0,
                "avg_pf": 0.0,
                "avg_win_rate": 0.0,
                "median_r": 0.0,
                "worst_pf": 0.0,
                "avg_tp_hit_rate": 0.0,
                "avg_sharpe": 0.0,
                "avg_sortino": 0.0,
            }

        pfs = [w.pf for w in valid if math.isfinite(w.pf)]
        win_rates = [w.win_rate for w in valid]
        median_rs = [w.median_r for w in valid if math.isfinite(w.median_r)]
        tp_rates = [w.tp_hit_rate for w in valid]
        sharpes  = [w.sharpe_ratio  for w in valid if math.isfinite(w.sharpe_ratio)  and w.trade_count > 0]
        sortinos = [w.sortino_ratio for w in valid if math.isfinite(w.sortino_ratio) and w.trade_count > 0]

        return {
            "valid_count": len(valid),
            "total_count": total,
            "valid_window_pct": len(valid) / total,
            "avg_pf": statistics.mean(pfs) if pfs else 0.0,
            "avg_win_rate": statistics.mean(win_rates) if win_rates else 0.0,
            "median_r": statistics.median(median_rs) if median_rs else 0.0,
            "worst_pf": min(pfs) if pfs else 0.0,
            "avg_tp_hit_rate": statistics.mean(tp_rates) if tp_rates else 0.0,
            "avg_sharpe": statistics.mean(sharpes) if sharpes else 0.0,
            "avg_sortino": statistics.mean(sortinos) if sortinos else 0.0,
        }

    # ── Internals ──────────────────────────────────────────────────────────────

    def _log_excluded(self, excluded: list[WindowResult]) -> None:
        EXCLUDED_WINDOWS_LOG.parent.mkdir(parents=True, exist_ok=True)
        mode = "w" if not self._initialized else "a"
        with open(EXCLUDED_WINDOWS_LOG, mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_HEADERS)
            if not self._initialized:
                writer.writeheader()
                self._initialized = True
            for w in excluded:
                writer.writerow({
                    "window_start": w.window_start,
                    "window_end": w.window_end,
                    "dominant_regime": w.dominant_regime,
                    "trade_count": w.trade_count,
                    "min_required": self.min_window_trades,
                    "reason": w.exclusion_reason or "",
                })
