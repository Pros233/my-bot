"""
sentiment_engine.py — CoinGecko-based sentiment filter.

Uses CoinGecko public /coins/markets endpoint (no API key required) to
derive a lightweight sentiment signal from 24h price change velocity and
volume ratios.

Rules:
  - Filter-only: sentiment can REDUCE rank_score, never drives entries alone
  - Sentiment score: -1.0 (extreme fear) to +1.0 (extreme greed)
  - sentiment_modifier(symbol) → float added to rank_score (-15 to +15)
  - 15-minute cache; fail-open (returns 0.0 on any error)

Public API
----------
    sentiment_modifier(symbol)     → float  (rank_score delta, -15 to +15)
    get_sentiment(symbol)          → dict   (raw sentiment data)
    get_all_sentiments()           → dict   (all tracked symbols)
    refresh_sentiment()            → None   (force cache refresh)
    get_sentiment_summary()        → dict   (for dashboard / Telegram)

Never raises.
"""
from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Optional

import logger

# ── Config ────────────────────────────────────────────────────────────────────

_CACHE_TTL_S   = 900      # 15 minutes
_MAX_MODIFIER  = 15.0     # max rank_score boost/penalty from sentiment
_REQUEST_TIMEOUT = 8      # seconds

# CoinGecko symbol → coin id mapping
_SYMBOL_TO_ID: dict[str, str] = {
    "BTCUSDT":  "bitcoin",
    "ETHUSDT":  "ethereum",
    "SOLUSDT":  "solana",
    "XRPUSDT":  "ripple",
    "DOGEUSDT": "dogecoin",
    "ADAUSDT":  "cardano",
    "LINKUSDT": "chainlink",
    "AVAXUSDT": "avalanche-2",
    "SUIUSDT":  "sui",
    "TONUSDT":  "the-open-network",
}

_ALL_IDS = list(set(_SYMBOL_TO_ID.values()))

# ── Cache ─────────────────────────────────────────────────────────────────────

_lock = threading.Lock()
_cache: dict[str, dict] = {}   # coin_id → data dict
_last_fetch: float = 0.0


def _fetch_coingecko() -> bool:
    """Fetch market data from CoinGecko. Returns True on success."""
    try:
        import urllib.request
        import json

        ids_param = ",".join(_ALL_IDS)
        url = (
            "https://api.coingecko.com/api/v3/coins/markets"
            f"?vs_currency=usd&ids={ids_param}"
            "&order=market_cap_desc&per_page=50&page=1"
            "&price_change_percentage=1h,24h,7d"
        )
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "btcbot/1.0"},
        )
        with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT) as resp:
            data = json.loads(resp.read().decode())

        new_cache: dict[str, dict] = {}
        for coin in data:
            cid = coin.get("id", "")
            if not cid:
                continue
            new_cache[cid] = {
                "id":           cid,
                "price":        coin.get("current_price", 0.0),
                "change_1h":    coin.get("price_change_percentage_1h_in_currency", 0.0) or 0.0,
                "change_24h":   coin.get("price_change_percentage_24h", 0.0) or 0.0,
                "change_7d":    coin.get("price_change_percentage_7d_in_currency", 0.0) or 0.0,
                "volume_24h":   coin.get("total_volume", 0.0) or 0.0,
                "market_cap":   coin.get("market_cap", 0.0) or 0.0,
                "fetched_at":   datetime.now(timezone.utc).isoformat(),
            }

        with _lock:
            _cache.clear()
            _cache.update(new_cache)

        return True
    except Exception as exc:
        logger.log_warning(f"sentiment_engine._fetch_coingecko error: {exc}")
        return False


def _maybe_refresh() -> None:
    """Refresh cache if stale."""
    global _last_fetch
    with _lock:
        age = time.monotonic() - _last_fetch
    if age >= _CACHE_TTL_S:
        ok = _fetch_coingecko()
        if ok:
            with _lock:
                _last_fetch = time.monotonic()


# ── Sentiment scoring ─────────────────────────────────────────────────────────

def _score_from_data(data: dict) -> float:
    """
    Derive a sentiment score -1.0 to +1.0 from CoinGecko market data.

    Formula:
      - 24h change is the primary signal (weight 0.5)
      - 1h momentum adds short-term bias  (weight 0.3)
      - 7d trend for context              (weight 0.2)

    Normalise each component against expected ranges:
      24h: ±5% = full signal  → clamp to [-5, +5] / 5
      1h:  ±2% = full signal  → clamp to [-2, +2] / 2
      7d: ±20% = full signal  → clamp to [-20, +20] / 20
    """
    def clamp(v: float, hi: float) -> float:
        return max(-1.0, min(1.0, v / hi))

    c24 = clamp(data.get("change_24h", 0.0), 5.0)
    c1h = clamp(data.get("change_1h",  0.0), 2.0)
    c7d = clamp(data.get("change_7d",  0.0), 20.0)

    score = 0.5 * c24 + 0.3 * c1h + 0.2 * c7d
    return round(max(-1.0, min(1.0, score)), 3)


# ── Public API ────────────────────────────────────────────────────────────────

def get_sentiment(symbol: str) -> dict:
    """Return raw sentiment data for a symbol. Fail-open → empty dict."""
    try:
        _maybe_refresh()
        coin_id = _SYMBOL_TO_ID.get(symbol)
        if not coin_id:
            return {}
        with _lock:
            data = dict(_cache.get(coin_id, {}))
        if not data:
            return {}
        score = _score_from_data(data)
        return {**data, "sentiment_score": score}
    except Exception:
        return {}


def sentiment_modifier(symbol: str) -> float:
    """
    Return a rank_score delta based on sentiment.
    Range: -_MAX_MODIFIER to +_MAX_MODIFIER.
    Returns 0.0 on any error (fail-open).
    """
    try:
        data = get_sentiment(symbol)
        if not data:
            return 0.0
        score = data.get("sentiment_score", 0.0)
        return round(score * _MAX_MODIFIER, 2)
    except Exception:
        return 0.0


def get_all_sentiments() -> dict[str, dict]:
    """Return sentiment data for all tracked symbols."""
    try:
        _maybe_refresh()
        result = {}
        for sym, cid in _SYMBOL_TO_ID.items():
            with _lock:
                data = dict(_cache.get(cid, {}))
            if data:
                score = _score_from_data(data)
                result[sym] = {**data, "sentiment_score": score}
        return result
    except Exception:
        return {}


def refresh_sentiment() -> None:
    """Force a cache refresh."""
    global _last_fetch
    with _lock:
        _last_fetch = 0.0
    _maybe_refresh()


def get_sentiment_summary() -> dict:
    """
    Return dashboard-friendly summary: top bullish, top bearish,
    avg sentiment, per-symbol scores.
    """
    try:
        all_data = get_all_sentiments()
        if not all_data:
            return {"symbols": {}, "avg_score": 0.0, "top_bullish": [], "top_bearish": []}

        scores = {
            sym: d.get("sentiment_score", 0.0)
            for sym, d in all_data.items()
        }

        avg = round(sum(scores.values()) / len(scores), 3) if scores else 0.0

        sorted_syms = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_bullish = [s for s, sc in sorted_syms if sc > 0][:3]
        top_bearish = [s for s, sc in sorted_syms if sc < 0][:3]

        # Categorise
        def _label(sc: float) -> str:
            if sc >= 0.5:  return "greed"
            if sc >= 0.15: return "mild_bullish"
            if sc <= -0.5: return "fear"
            if sc <= -0.15:return "mild_bearish"
            return "neutral"

        symbols_out = {}
        for sym, d in all_data.items():
            sc = d.get("sentiment_score", 0.0)
            symbols_out[sym] = {
                "score":      sc,
                "label":      _label(sc),
                "modifier":   round(sc * _MAX_MODIFIER, 1),
                "change_24h": round(d.get("change_24h", 0.0), 2),
                "change_1h":  round(d.get("change_1h",  0.0), 2),
            }

        with _lock:
            fetch_age_s = time.monotonic() - _last_fetch

        return {
            "symbols":     symbols_out,
            "avg_score":   avg,
            "avg_label":   _label(avg),
            "top_bullish": top_bullish,
            "top_bearish": top_bearish,
            "cache_age_s": round(fetch_age_s, 0),
            "cache_ttl_s": _CACHE_TTL_S,
        }
    except Exception as exc:
        logger.log_warning(f"sentiment_engine.get_sentiment_summary error: {exc}")
        return {"symbols": {}, "avg_score": 0.0, "error": str(exc)}
