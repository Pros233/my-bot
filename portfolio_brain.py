"""
portfolio_brain.py — Sector exposure tracking and portfolio health scoring.

Tracks live positions by sector (coin cluster), enforces MAX_SECTOR_EXPOSURE,
and computes a unified portfolio health score 0-100.

Sectors:
  BTC        → BTCUSDT
  ETH_LAYER1 → ETHUSDT, SOLUSDT, AVAXUSDT, SUIUSDT, TONUSDT
  ALTS       → LINKUSDT, ADAUSDT
  MEMES      → DOGEUSDT
  XRP        → XRPUSDT

Health score components (weights sum to 1.0):
  expectancy    0.25 — from engine_performance
  drawdown      0.20 — current max drawdown
  stability     0.15 — rolling win-rate consistency
  open_exposure 0.20 — how crowded current positions are
  sector_conc   0.20 — sector concentration penalty

Public API
----------
    get_sector(symbol)                → str
    check_sector_exposure(symbol, open_positions, balance) → (bool, str)
    compute_health_score(open_positions, balance) → float  (0-100)
    get_portfolio_summary(open_positions, balance) → dict

Never raises.
"""
from __future__ import annotations

import math
from typing import Optional

import logger

# ── Config ────────────────────────────────────────────────────────────────────

MAX_SECTOR_EXPOSURE = 0.40   # max fraction of open positions in one sector

_SECTORS: dict[str, list[str]] = {
    "BTC":        ["BTCUSDT"],
    "ETH_LAYER1": ["ETHUSDT", "SOLUSDT", "AVAXUSDT", "SUIUSDT", "TONUSDT"],
    "ALTS":       ["LINKUSDT", "ADAUSDT"],
    "MEMES":      ["DOGEUSDT"],
    "XRP":        ["XRPUSDT"],
}

_SYMBOL_TO_SECTOR: dict[str, str] = {
    sym: sector
    for sector, syms in _SECTORS.items()
    for sym in syms
}


# ── Sector helpers ────────────────────────────────────────────────────────────

def get_sector(symbol: str) -> str:
    """Return sector name for a symbol, or 'OTHER'."""
    return _SYMBOL_TO_SECTOR.get(symbol, "OTHER")


def check_sector_exposure(
    symbol: str,
    open_positions: list[str],   # list of currently open symbols
    balance: float = 0.0,
) -> tuple[bool, str]:
    """
    Return (allowed, reason).
    Blocks new position if adding it would push one sector over MAX_SECTOR_EXPOSURE.
    Fail-open: returns (True, "") on any error.
    """
    try:
        if not open_positions:
            return True, ""

        sector = get_sector(symbol)
        total  = len(open_positions) + 1   # +1 for prospective new trade

        # Count existing positions in the same sector
        same_sector = sum(
            1 for s in open_positions if get_sector(s) == sector
        ) + 1  # +1 for prospective

        exposure = same_sector / total
        if exposure > MAX_SECTOR_EXPOSURE:
            return (
                False,
                f"sector {sector} exposure {exposure*100:.0f}% > "
                f"{MAX_SECTOR_EXPOSURE*100:.0f}% limit "
                f"({same_sector}/{total} positions)",
            )
        return True, ""
    except Exception as exc:
        logger.log_warning(f"portfolio_brain.check_sector_exposure error: {exc}")
        return True, ""


# ── Health scoring ────────────────────────────────────────────────────────────

def _expectancy_component() -> float:
    """0-100 from engine expectancy.  Uses engine_performance if available."""
    try:
        import engine_performance as ep
        stats_list = [ep.get_engine_stats(e, days=30) for e in ep.ENGINE_NAMES]
        valid = [s for s in stats_list if s.get("trades", 0) >= 5]
        if not valid:
            return 50.0
        avg_exp = sum(s["expectancy"] for s in valid) / len(valid)
        # Map expectancy: +0.02 → 100, 0 → 50, -0.01 → 0
        score = 50.0 + avg_exp * 2500.0
        return max(0.0, min(100.0, score))
    except Exception:
        return 50.0


def _drawdown_component(balance: float) -> float:
    """0-100 from current equity protection state."""
    try:
        import equity_protection as ep
        summary = ep.get_summary(balance)
        state = summary.get("state", "normal")
        dd    = summary.get("max_drawdown_pct", 0.0)
        if state == "normal":
            return max(60.0, 100.0 - dd * 5)
        if state == "selective":
            return max(30.0, 60.0 - dd * 5)
        return max(0.0, 30.0 - dd * 5)
    except Exception:
        return 50.0


def _open_exposure_component(open_positions: list[str]) -> float:
    """0-100: fewer open positions = healthier (max 10 assumed)."""
    try:
        n = len(open_positions)
        if n == 0:
            return 100.0
        # 1 open = 90, 5 open = 50, 10+ = 0
        return max(0.0, 100.0 - n * 10.0)
    except Exception:
        return 50.0


def _sector_concentration_component(open_positions: list[str]) -> float:
    """0-100: penalise high concentration in one sector."""
    try:
        if not open_positions:
            return 100.0
        counts: dict[str, int] = {}
        for sym in open_positions:
            sec = get_sector(sym)
            counts[sec] = counts.get(sec, 0) + 1
        max_concentration = max(counts.values()) / len(open_positions)
        # 0.25 concentration → 100, 1.0 concentration → 0
        score = (1.0 - max_concentration) / 0.75 * 100.0
        return max(0.0, min(100.0, score))
    except Exception:
        return 50.0


def _stability_component() -> float:
    """0-100 from recent win-rate consistency across engines."""
    try:
        import engine_performance as ep
        stats_list = [ep.get_engine_stats(e, days=14) for e in ep.ENGINE_NAMES]
        valid = [s for s in stats_list if s.get("trades", 0) >= 5]
        if not valid:
            return 50.0
        win_rates = [s["win_rate"] for s in valid]
        avg_wr = sum(win_rates) / len(win_rates)
        # Variance penalty
        variance = sum((w - avg_wr) ** 2 for w in win_rates) / len(win_rates)
        score = avg_wr * 100.0 - math.sqrt(variance) * 50.0
        return max(0.0, min(100.0, score))
    except Exception:
        return 50.0


def compute_health_score(
    open_positions: list[str],
    balance: float = 0.0,
) -> float:
    """Return portfolio health 0-100.  Higher = healthier."""
    try:
        exp_c    = _expectancy_component()
        dd_c     = _drawdown_component(balance)
        stab_c   = _stability_component()
        exp_c2   = _open_exposure_component(open_positions)
        conc_c   = _sector_concentration_component(open_positions)

        score = (
            0.25 * exp_c  +
            0.20 * dd_c   +
            0.15 * stab_c +
            0.20 * exp_c2 +
            0.20 * conc_c
        )
        return round(max(0.0, min(100.0, score)), 1)
    except Exception as exc:
        logger.log_warning(f"portfolio_brain.compute_health_score error: {exc}")
        return 50.0


def get_portfolio_summary(
    open_positions: list[str],
    balance: float = 0.0,
) -> dict:
    """Return full portfolio summary for dashboard / Telegram."""
    try:
        health = compute_health_score(open_positions, balance)

        # Sector breakdown
        sector_counts: dict[str, int] = {}
        for sym in open_positions:
            sec = get_sector(sym)
            sector_counts[sec] = sector_counts.get(sec, 0) + 1

        total = len(open_positions)
        sector_exposure = {
            sec: {
                "count":    cnt,
                "exposure": round(cnt / total, 3) if total > 0 else 0.0,
                "over_limit": (cnt / total > MAX_SECTOR_EXPOSURE) if total > 0 else False,
            }
            for sec, cnt in sector_counts.items()
        }

        def _health_label(h: float) -> str:
            if h >= 75: return "STRONG"
            if h >= 50: return "MODERATE"
            if h >= 30: return "WEAK"
            return "CRITICAL"

        return {
            "health_score":    health,
            "health_label":    _health_label(health),
            "open_positions":  total,
            "sector_exposure": sector_exposure,
            "max_sector_limit": MAX_SECTOR_EXPOSURE,
            "components": {
                "expectancy":    round(_expectancy_component(), 1),
                "drawdown":      round(_drawdown_component(balance), 1),
                "stability":     round(_stability_component(), 1),
                "open_exposure": round(_open_exposure_component(open_positions), 1),
                "sector_conc":   round(_sector_concentration_component(open_positions), 1),
            },
        }
    except Exception as exc:
        logger.log_warning(f"portfolio_brain.get_portfolio_summary error: {exc}")
        return {"health_score": 50.0, "error": str(exc)}
