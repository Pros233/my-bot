"""
monte_carlo.py — Bootstrap confidence intervals for walk-forward OOS trades.

Usage (standalone):
    from monte_carlo import bootstrap, MCResult
    result = bootstrap(r_multiples=[0.5, -1.0, 1.2, ...])
    print(result.pf_p50, result.prob_pf_above_1)

Used by backtest._promotion_summary to annotate each setup's OOS trade
distribution with percentile bands, giving a sense of how much the headline
PF/WR numbers are noise vs signal.
"""
from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass


_PF_CAP = 10.0  # cap to avoid inf in PF calculations


@dataclass
class MCResult:
    n_trades: int
    n_resamples: int
    # Profit-factor percentiles
    pf_p05: float
    pf_p50: float
    pf_p95: float
    # Win-rate percentiles
    win_rate_p05: float
    win_rate_p50: float
    win_rate_p95: float
    # Average-R percentiles
    avg_r_p05: float
    avg_r_p50: float
    avg_r_p95: float
    # Probability estimates
    prob_pf_above_1: float
    prob_avg_r_positive: float


def _profit_factor(r_multiples: list[float]) -> float:
    gross_profit = sum(r for r in r_multiples if r > 0)
    gross_loss = abs(sum(r for r in r_multiples if r < 0))
    if gross_loss == 0:
        return _PF_CAP
    raw = gross_profit / gross_loss
    return min(raw, _PF_CAP)


def _win_rate(r_multiples: list[float]) -> float:
    if not r_multiples:
        return 0.0
    return sum(1 for r in r_multiples if r > 0) / len(r_multiples)


def bootstrap(
    r_multiples: list[float],
    n_resamples: int = 1000,
    seed: int = 42,
) -> MCResult:
    """
    Bootstrap `r_multiples` with replacement to produce confidence intervals.

    Parameters
    ----------
    r_multiples : list of float
        Per-trade R-multiples (positive = win, negative = loss).
    n_resamples : int
        Number of bootstrap draws.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    MCResult
        Percentile bands and probability estimates for PF, WR, avg-R.
    """
    n = len(r_multiples)
    if n == 0:
        return MCResult(
            n_trades=0, n_resamples=n_resamples,
            pf_p05=0.0, pf_p50=0.0, pf_p95=0.0,
            win_rate_p05=0.0, win_rate_p50=0.0, win_rate_p95=0.0,
            avg_r_p05=0.0, avg_r_p50=0.0, avg_r_p95=0.0,
            prob_pf_above_1=0.0, prob_avg_r_positive=0.0,
        )

    rng = random.Random(seed)
    pfs: list[float] = []
    wrs: list[float] = []
    avg_rs: list[float] = []

    for _ in range(n_resamples):
        sample = [rng.choice(r_multiples) for _ in range(n)]
        pfs.append(_profit_factor(sample))
        wrs.append(_win_rate(sample))
        avg_rs.append(statistics.mean(sample))

    pfs.sort()
    wrs.sort()
    avg_rs.sort()

    def _pct(values: list[float], p: float) -> float:
        idx = max(0, min(len(values) - 1, int(p * len(values))))
        return values[idx]

    return MCResult(
        n_trades=n,
        n_resamples=n_resamples,
        pf_p05=round(_pct(pfs, 0.05), 4),
        pf_p50=round(_pct(pfs, 0.50), 4),
        pf_p95=round(_pct(pfs, 0.95), 4),
        win_rate_p05=round(_pct(wrs, 0.05), 4),
        win_rate_p50=round(_pct(wrs, 0.50), 4),
        win_rate_p95=round(_pct(wrs, 0.95), 4),
        avg_r_p05=round(_pct(avg_rs, 0.05), 4),
        avg_r_p50=round(_pct(avg_rs, 0.50), 4),
        avg_r_p95=round(_pct(avg_rs, 0.95), 4),
        prob_pf_above_1=round(sum(1 for p in pfs if p > 1.0) / n_resamples, 4),
        prob_avg_r_positive=round(sum(1 for r in avg_rs if r > 0) / n_resamples, 4),
    )
