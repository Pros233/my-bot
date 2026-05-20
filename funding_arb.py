"""
funding_arb.py — Binance Futures funding rate arbitrage scanner (watch-only).

Uses the Binance FAPI public endpoint (no API key required for market data)
to identify symbols with extreme funding rates, signalling potential
long/short arbitrage opportunities.

Funding rate mechanics:
  - Positive rate (> +0.01%): longs pay shorts every 8h → shorts favoured
  - Negative rate (< -0.01%): shorts pay longs every 8h → longs favoured
  - Extreme: > ±0.05% per 8h = 2.19% APR annualised per side

This module is WATCH-ONLY. ARB_AUTO_TRADE is hard-locked to False.
Signal data feeds: Telegram alerts, dashboard panel.

Public API
----------
    get_funding_rates()         → dict[symbol → rate_data]
    get_arb_signals()           → list[dict]  (extreme rate opportunities)
    get_funding_summary()       → dict         (dashboard view)
    get_rate(symbol)            → float        (current funding rate %)

Never raises.
"""
from __future__ import annotations

import json
import threading
import time
import urllib.request
from datetime import datetime, timezone
from typing import Optional

import logger

# ── Config ────────────────────────────────────────────────────────────────────

_CACHE_TTL_S         = 600    # 10 minutes
_REQUEST_TIMEOUT     = 10
_EXTREME_RATE_PCT    = 0.05   # |rate| > 0.05% = extreme
_NOTABLE_RATE_PCT    = 0.02   # |rate| > 0.02% = notable

# Symbols to monitor (Binance Futures perpetuals)
_WATCH_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT",
    "ADAUSDT", "LINKUSDT", "AVAXUSDT", "BNBUSDT", "LTCUSDT",
    "DOTUSDT", "MATICUSDT", "SHIBUSDT", "SUIUSDT",
]

# ── Cache ─────────────────────────────────────────────────────────────────────

_lock          = threading.Lock()
_rates_cache:  dict[str, dict] = {}
_last_fetch:   float = 0.0

# Hard lock — never change this
ARB_AUTO_TRADE: bool = False


# ── Fetcher ───────────────────────────────────────────────────────────────────

def _fetch_funding_rates() -> bool:
    """
    Fetch current funding rates from Binance Futures.
    Uses the public /fapi/v1/premiumIndex endpoint (no auth required).
    """
    try:
        url = "https://fapi.binance.com/fapi/v1/premiumIndex"
        req = urllib.request.Request(url, headers={"User-Agent": "btcbot/1.0"})
        with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT) as resp:
            data = json.loads(resp.read().decode())

        if not isinstance(data, list):
            return False

        now_iso = datetime.now(timezone.utc).isoformat()
        new_cache: dict[str, dict] = {}

        for item in data:
            sym = item.get("symbol", "")
            if sym not in _WATCH_SYMBOLS:
                continue
            rate_str  = item.get("lastFundingRate", "0") or "0"
            mark_str  = item.get("markPrice", "0") or "0"
            index_str = item.get("indexPrice", "0") or "0"
            next_time = int(item.get("nextFundingTime", 0))

            rate_pct  = float(rate_str) * 100.0   # convert to %
            mark      = float(mark_str)
            index     = float(index_str)
            basis_pct = ((mark - index) / index * 100.0) if index > 0 else 0.0

            next_dt   = (
                datetime.fromtimestamp(next_time / 1000, tz=timezone.utc).strftime("%H:%M UTC")
                if next_time else "unknown"
            )

            new_cache[sym] = {
                "symbol":       sym,
                "rate_pct":     round(rate_pct, 5),
                "mark_price":   round(mark, 4),
                "index_price":  round(index, 4),
                "basis_pct":    round(basis_pct, 4),
                "next_funding": next_dt,
                "annualized_pct": round(rate_pct * 3 * 365, 2),  # 3 payouts/day × 365
                "fetched_at":   now_iso,
            }

        with _lock:
            _rates_cache.clear()
            _rates_cache.update(new_cache)
        return True

    except Exception as exc:
        logger.log_warning(f"funding_arb._fetch_funding_rates error: {exc}")
        return False


def _maybe_refresh() -> None:
    global _last_fetch
    with _lock:
        age = time.monotonic() - _last_fetch
    if age >= _CACHE_TTL_S:
        ok = _fetch_funding_rates()
        if ok:
            with _lock:
                _last_fetch = time.monotonic()


# ── Public API ────────────────────────────────────────────────────────────────

def get_funding_rates() -> dict[str, dict]:
    """Return all cached funding rate data. Fail-open → {}."""
    try:
        _maybe_refresh()
        with _lock:
            return dict(_rates_cache)
    except Exception:
        return {}


def get_rate(symbol: str) -> float:
    """Return current funding rate % for symbol. Fail-open → 0.0."""
    try:
        rates = get_funding_rates()
        return rates.get(symbol, {}).get("rate_pct", 0.0)
    except Exception:
        return 0.0


def get_arb_signals(min_rate_pct: float = _NOTABLE_RATE_PCT) -> list[dict]:
    """
    Return list of arb opportunities where |funding_rate| >= min_rate_pct.

    Each signal includes:
      - symbol, rate_pct, direction (LONG_FAVOURED / SHORT_FAVOURED)
      - strength (EXTREME / NOTABLE)
      - annualized_pct, basis_pct, next_funding

    Fail-open → [].
    """
    try:
        rates = get_funding_rates()
        signals = []
        for sym, data in rates.items():
            rate = data.get("rate_pct", 0.0)
            if abs(rate) < min_rate_pct:
                continue

            direction = "LONG_FAVOURED" if rate < 0 else "SHORT_FAVOURED"
            strength  = "EXTREME" if abs(rate) >= _EXTREME_RATE_PCT else "NOTABLE"

            signals.append({
                **data,
                "direction":    direction,
                "strength":     strength,
                "signal_note":  (
                    f"{'Shorts' if rate > 0 else 'Longs'} earn "
                    f"{abs(rate):.4f}%/8h ({abs(data['annualized_pct']):.1f}% ann.)"
                ),
            })

        # Sort by absolute rate, highest first
        signals.sort(key=lambda x: abs(x["rate_pct"]), reverse=True)
        return signals
    except Exception as exc:
        logger.log_warning(f"funding_arb.get_arb_signals error: {exc}")
        return []


def get_funding_summary() -> dict:
    """
    Return dashboard-friendly summary of current funding rates.
    Includes top 5 highest/lowest rates and overall market sentiment.
    Fail-open → empty structure.
    """
    try:
        rates   = get_funding_rates()
        signals = get_arb_signals()

        if not rates:
            return {"rates": {}, "arb_signals": [], "market_funding_bias": "neutral"}

        # Market funding bias
        avg_rate = sum(d["rate_pct"] for d in rates.values()) / len(rates) if rates else 0.0
        if avg_rate > 0.02:    bias = "heavily_long"
        elif avg_rate > 0.005: bias = "slightly_long"
        elif avg_rate < -0.02: bias = "heavily_short"
        elif avg_rate < -0.005: bias = "slightly_short"
        else:                  bias = "neutral"

        # Sort by rate for display
        sorted_rates = sorted(rates.values(), key=lambda x: x["rate_pct"], reverse=True)

        with _lock:
            cache_age = round(time.monotonic() - _last_fetch, 0)

        return {
            "rates":                {r["symbol"]: r for r in sorted_rates},
            "arb_signals":          signals,
            "market_funding_bias":  bias,
            "avg_rate_pct":         round(avg_rate, 5),
            "extreme_count":        sum(1 for s in signals if s["strength"] == "EXTREME"),
            "cache_age_s":          cache_age,
            "auto_trade_locked":    True,   # always shown as locked
        }
    except Exception as exc:
        logger.log_warning(f"funding_arb.get_funding_summary error: {exc}")
        return {"rates": {}, "arb_signals": [], "market_funding_bias": "neutral"}
