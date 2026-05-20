"""
correlation_guard.py — Correlated position exposure guard.

Prevents the bot from holding multiple highly correlated positions
simultaneously, avoiding hidden cluster risk.

Correlation clusters:
  BTC_CLUSTER    BTCUSDT
  ALTS_MAJOR     ETH, SOL, AVAX, LINK, ADA
  MEMES          DOGE, SHIB, PEPE
  XRP_CLUSTER    XRP
  NEW_L1         TON, SUI

Config:
  MAX_CORRELATED_POSITIONS=2  (default: 2)
  ENABLE_CORRELATION_GUARD=true (default: true — conservative)

Public API
----------
    check_new_position(symbol, open_symbols) → (allowed: bool, reason: str)
    cluster_for(symbol)                      → str cluster name
    get_exposure_summary(open_symbols)       → dict of cluster → [symbols]

Never raises — fails open (returns allowed=True) on any error so
the correlation guard never blocks trading due to a bug.
"""
from __future__ import annotations

import config
import logger

# ── Correlation clusters ──────────────────────────────────────────────────────

_CLUSTERS: dict[str, list[str]] = {
    "BTC":       ["BTCUSDT"],
    "ALTS_MAJOR": ["ETHUSDT", "SOLUSDT", "AVAXUSDT", "LINKUSDT", "ADAUSDT"],
    "MEMES":     ["DOGEUSDT", "SHIBUSDT", "PEPEUSDT"],
    "XRP":       ["XRPUSDT"],
    "NEW_L1":    ["TONUSDT", "SUIUSDT"],
}

# Reverse map: symbol → cluster name
_SYMBOL_TO_CLUSTER: dict[str, str] = {
    sym: cluster
    for cluster, symbols in _CLUSTERS.items()
    for sym in symbols
}


def cluster_for(symbol: str) -> str:
    """Return the cluster name for a symbol, or 'OTHER' if unknown."""
    return _SYMBOL_TO_CLUSTER.get(symbol.upper(), "OTHER")


def get_exposure_summary(open_symbols: list[str]) -> dict[str, list[str]]:
    """
    Return {cluster_name: [symbols in that cluster currently open]}.
    Only includes clusters with ≥ 1 open position.
    """
    exposure: dict[str, list[str]] = {}
    for sym in open_symbols:
        cluster = cluster_for(sym)
        exposure.setdefault(cluster, []).append(sym)
    return exposure


def check_new_position(symbol: str, open_symbols: list[str]) -> tuple[bool, str]:
    """
    Check whether opening a new position in `symbol` is allowed given
    currently open positions.

    Returns
    -------
    (True, "")           — allowed, no cluster risk
    (False, reason_str)  — blocked, reason explains which cluster is full
    """
    if not getattr(config, "ENABLE_CORRELATION_GUARD", True):
        return True, ""

    try:
        max_corr = getattr(config, "MAX_CORRELATED_POSITIONS", 2)
        new_cluster = cluster_for(symbol)

        # BTC cluster is always solo — only 1 BTC position allowed
        if new_cluster == "BTC":
            btc_open = [s for s in open_symbols if cluster_for(s) == "BTC"]
            if len(btc_open) >= 1:
                return False, f"BTC cluster already open: {', '.join(btc_open)}"
            return True, ""

        # Count how many positions are already in the same cluster
        same_cluster_open = [s for s in open_symbols if cluster_for(s) == new_cluster]
        if len(same_cluster_open) >= max_corr:
            return False, (
                f"correlation limit: {new_cluster} already has "
                f"{len(same_cluster_open)} open ({', '.join(same_cluster_open)}) "
                f"— MAX_CORRELATED_POSITIONS={max_corr}"
            )

        return True, ""

    except Exception as exc:
        logger.log_warning(
            f"correlation_guard.check_new_position error "
            f"(fail-open, allowing trade): {exc}"
        )
        return True, ""
