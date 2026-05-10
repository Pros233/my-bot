"""
entry_buffer.py — Entry Confirmation Buffer (ROBUSTNESS #6).

Prevents entering trades where price drifted too far from the signal close
before the entry executes on the next bar.

Signal fires on candle N (close price = signal_price).
Entry is attempted at candle N+1 (open price = entry_price).
If |entry_price - signal_price| / signal_price > ENTRY_BUFFER_PCT the
trade is marked CHASED_ENTRY and skipped.

Usage
-----
    from entry_buffer import EntryConfirmationBuffer

    buf = EntryConfirmationBuffer(log_to_csv=True)
    if buf.check(signal_price=64_000.0, entry_price=64_200.0):
        # execute trade
        ...
    print(f"Chase-skip rate: {buf.chase_skip_rate:.1%}")
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import config

ENTRY_QUALITY_LOG = Path("outputs/research/entry_quality.csv")
_HEADERS = ["timestamp", "signal_price", "next_open", "drift_pct", "buffer_pct", "status"]


class EntryConfirmationBuffer:
    """
    Validates that the execution price on candle N+1 has not strayed too far
    from the signal close on candle N.

    Parameters
    ----------
    buffer_pct : float
        Maximum allowed drift as a fraction of signal_price.
        Default: config.ENTRY_BUFFER_PCT (0.0015 = 0.15%).
    log_to_csv : bool
        Append each check result to outputs/research/entry_quality.csv.
    """

    def __init__(
        self,
        buffer_pct: float = config.ENTRY_BUFFER_PCT,
        log_to_csv: bool = False,
    ) -> None:
        self.buffer_pct = buffer_pct
        self.log_to_csv = log_to_csv
        self._initialized = False
        self.total_checks = 0
        self.chased_count = 0

    # ── Public ─────────────────────────────────────────────────────────────────

    def check(
        self,
        signal_price: float,
        entry_price: float,
        timestamp: str = "",
    ) -> bool:
        """
        Return True if the entry is valid, False if it is a CHASED_ENTRY.

        Parameters
        ----------
        signal_price : Close price at signal candle N.
        entry_price  : Open (or first tick) price at execution candle N+1.
        timestamp    : Optional string for CSV logging.
        """
        if signal_price <= 0 or not math.isfinite(signal_price) or not math.isfinite(entry_price):
            return False

        drift = abs(entry_price - signal_price) / signal_price
        valid = drift <= self.buffer_pct
        status = "OK" if valid else "CHASED_ENTRY"

        self.total_checks += 1
        if not valid:
            self.chased_count += 1

        if self.log_to_csv:
            self._log(timestamp, signal_price, entry_price, drift, status)

        return valid

    @property
    def chase_skip_rate(self) -> float:
        """Fraction of signals skipped due to price chase."""
        if self.total_checks == 0:
            return 0.0
        return self.chased_count / self.total_checks

    def reset_counters(self) -> None:
        self.total_checks = 0
        self.chased_count = 0

    # ── Internals ──────────────────────────────────────────────────────────────

    def _log(
        self,
        timestamp: str,
        signal_price: float,
        entry_price: float,
        drift: float,
        status: str,
    ) -> None:
        ENTRY_QUALITY_LOG.parent.mkdir(parents=True, exist_ok=True)
        mode = "w" if not self._initialized else "a"
        with open(ENTRY_QUALITY_LOG, mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_HEADERS)
            if not self._initialized:
                writer.writeheader()
                self._initialized = True
            writer.writerow({
                "timestamp": timestamp,
                "signal_price": f"{signal_price:.2f}",
                "next_open": f"{entry_price:.2f}",
                "drift_pct": f"{drift * 100:.4f}",
                "buffer_pct": f"{self.buffer_pct * 100:.4f}",
                "status": status,
            })
