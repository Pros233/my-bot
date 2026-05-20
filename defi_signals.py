"""
defi_signals.py — DeFi ecosystem signals from DeFiLlama (free public API).

Data sources:
  - https://api.llama.fi/v2/chains    → per-chain TVL (momentum signal)
  - https://yields.llama.fi/pools     → top yield pools (yield context)

Signals derived:
  1. Chain TVL momentum: chains with >5% TVL growth are bullish for their L1 tokens
  2. Top yield pools: very high yields signal risk appetite OR protocol distress
  3. Combined: defi_signal_modifier(symbol) adds -8 to +8 to rank_score

Use cases:
  - ETH bullish if Ethereum chain TVL growing strongly
  - SOL/AVAX bullish if their chains show TVL inflow
  - Caution if top yield pools showing >200% APY (unsustainable, risk-on)

Public API
----------
    get_chain_tvl()               → dict[chain → tvl_data]
    get_top_pools(n)              → list[dict]
    defi_signal_modifier(symbol)  → float  (-8 to +8 rank_score delta)
    get_defi_summary()            → dict   (dashboard view)

1-hour cache. Fail-open on every call.
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

_CACHE_TTL_S      = 3600    # 1 hour
_REQUEST_TIMEOUT  = 12
_TVL_GROWTH_BULL  = 5.0     # TVL +5% change → bullish
_TVL_GROWTH_BEAR  = -5.0    # TVL -5% change → bearish
_YIELD_RISK_PCT   = 100.0   # APY > 100% = risk-on caution

# Chain → symbol mapping (which token benefits from TVL growth)
_CHAIN_TO_SYMBOL: dict[str, str] = {
    "Ethereum":    "ETHUSDT",
    "Solana":      "SOLUSDT",
    "Avalanche":   "AVAXUSDT",
    "BSC":         "BNBUSDT",
    "Polygon":     "MATICUSDT",
    "Arbitrum":    "ETHUSDT",   # Arbitrum TVL → ETH beneficiary
    "Optimism":    "ETHUSDT",
    "Base":        "ETHUSDT",
}

# ── Cache ─────────────────────────────────────────────────────────────────────

_lock           = threading.Lock()
_chain_cache:   dict[str, dict] = {}
_pools_cache:   list[dict] = []
_last_fetch:    float = 0.0


# ── HTTP helper ───────────────────────────────────────────────────────────────

def _get_json(url: str) -> Optional[dict | list]:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "btcbot/1.0"})
        with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT) as resp:
            return json.loads(resp.read().decode())
    except Exception as exc:
        logger.log_warning(f"defi_signals._get_json error ({url[:60]}): {exc}")
        return None


# ── Fetchers ──────────────────────────────────────────────────────────────────

def _fetch_chain_tvl() -> bool:
    data = _get_json("https://api.llama.fi/v2/chains")
    if not data or not isinstance(data, list):
        return False
    try:
        new_cache: dict[str, dict] = {}
        for chain in data:
            name    = chain.get("name", "")
            tvl     = float(chain.get("tvl") or 0.0)
            change1d = float(chain.get("change_1d") or 0.0)
            change7d = float(chain.get("change_7d") or 0.0)
            if name not in _CHAIN_TO_SYMBOL:
                continue
            new_cache[name] = {
                "chain":      name,
                "tvl_usd":    tvl,
                "change_1d":  change1d,
                "change_7d":  change7d,
                "symbol":     _CHAIN_TO_SYMBOL[name],
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            }
        with _lock:
            _chain_cache.clear()
            _chain_cache.update(new_cache)
        return True
    except Exception as exc:
        logger.log_warning(f"defi_signals._fetch_chain_tvl parse error: {exc}")
        return False


def _fetch_top_pools(limit: int = 20) -> bool:
    data = _get_json("https://yields.llama.fi/pools")
    if not data or not isinstance(data, dict):
        return False
    try:
        pools = data.get("data", [])
        # Sort by TVL, take top N
        sorted_pools = sorted(pools, key=lambda x: float(x.get("tvlUsd") or 0), reverse=True)
        top = []
        for pool in sorted_pools[:limit]:
            apy     = float(pool.get("apy") or 0.0)
            tvl     = float(pool.get("tvlUsd") or 0.0)
            project = pool.get("project", "")
            chain   = pool.get("chain", "")
            symbol  = pool.get("symbol", "")
            if tvl < 1_000_000:   # skip tiny pools
                continue
            top.append({
                "project":  project,
                "chain":    chain,
                "symbol":   symbol,
                "apy":      round(apy, 2),
                "tvl_usd":  tvl,
            })

        with _lock:
            _pools_cache.clear()
            _pools_cache.extend(top[:limit])
        return True
    except Exception as exc:
        logger.log_warning(f"defi_signals._fetch_top_pools parse error: {exc}")
        return False


def _maybe_refresh() -> None:
    global _last_fetch
    with _lock:
        age = time.monotonic() - _last_fetch
    if age >= _CACHE_TTL_S:
        ok_chain = _fetch_chain_tvl()
        ok_pools = _fetch_top_pools()
        if ok_chain or ok_pools:
            with _lock:
                _last_fetch = time.monotonic()


# ── Public API ────────────────────────────────────────────────────────────────

def get_chain_tvl() -> dict[str, dict]:
    """Return per-chain TVL data. Fail-open → {}."""
    try:
        _maybe_refresh()
        with _lock:
            return dict(_chain_cache)
    except Exception:
        return {}


def get_top_pools(n: int = 10) -> list[dict]:
    """Return top N yield pools by TVL. Fail-open → []."""
    try:
        _maybe_refresh()
        with _lock:
            return list(_pools_cache[:n])
    except Exception:
        return []


def defi_signal_modifier(symbol: str) -> float:
    """
    Return rank_score delta based on DeFi signals for the given symbol.
    Range: -8 to +8.

    Logic:
      - If symbol's chain shows strong TVL growth (>5%/day): +4 to +8
      - If symbol's chain shows TVL decline (< -5%/day): -4 to -8
      - If top yield pools averaging very high APY (>150%): -2 (risk-on caution)
      - Otherwise: 0

    Fail-open → 0.0.
    """
    try:
        _maybe_refresh()

        # Find chains associated with this symbol
        chain_data = {}
        with _lock:
            for chain, data in _chain_cache.items():
                if data.get("symbol") == symbol:
                    chain_data[chain] = data

        if not chain_data:
            return 0.0

        # Aggregate TVL change across relevant chains
        changes = [d["change_1d"] for d in chain_data.values()]
        avg_change = sum(changes) / len(changes) if changes else 0.0

        if avg_change >= 10.0:    modifier = +8.0
        elif avg_change >= 5.0:   modifier = +4.0
        elif avg_change >= 2.0:   modifier = +2.0
        elif avg_change <= -10.0: modifier = -8.0
        elif avg_change <= -5.0:  modifier = -4.0
        elif avg_change <= -2.0:  modifier = -2.0
        else:                     modifier = 0.0

        # Yield caution: if avg top-pool APY very high, reduce slightly
        with _lock:
            pools = list(_pools_cache[:10])
        if pools:
            avg_apy = sum(p["apy"] for p in pools) / len(pools)
            if avg_apy > 200.0:
                modifier = max(-8.0, modifier - 2.0)  # unsustainable yields = caution

        return round(max(-8.0, min(8.0, modifier)), 1)

    except Exception as exc:
        logger.log_warning(f"defi_signals.defi_signal_modifier({symbol}) error: {exc}")
        return 0.0


def get_defi_summary() -> dict:
    """
    Return dashboard-friendly DeFi summary.
    Fail-open → empty structure.
    """
    try:
        chains = get_chain_tvl()
        pools  = get_top_pools(10)

        with _lock:
            cache_age = round(time.monotonic() - _last_fetch, 0)

        # Summarise chain health
        chain_summary = {}
        for name, data in chains.items():
            chg = data.get("change_1d", 0.0)
            if chg >= 5:    health = "growing"
            elif chg <= -5: health = "shrinking"
            else:           health = "stable"
            chain_summary[name] = {
                **data,
                "health": health,
            }

        # Top yields
        top_yields = sorted(pools, key=lambda x: x["apy"], reverse=True)[:5]

        avg_apy = (
            sum(p["apy"] for p in pools) / len(pools) if pools else 0.0
        )
        yield_environment = (
            "risk_on" if avg_apy > 150
            else "moderate" if avg_apy > 50
            else "conservative"
        )

        return {
            "chains":             chain_summary,
            "top_yield_pools":    top_yields,
            "avg_top10_apy":      round(avg_apy, 1),
            "yield_environment":  yield_environment,
            "cache_age_s":        cache_age,
        }
    except Exception as exc:
        logger.log_warning(f"defi_signals.get_defi_summary error: {exc}")
        return {"chains": {}, "top_yield_pools": [], "avg_top10_apy": 0.0}
