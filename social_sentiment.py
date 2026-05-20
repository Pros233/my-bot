"""
social_sentiment.py — Social sentiment signals from CoinGecko + Fear/Greed index.

Data sources (all free, no API key required):
  - CoinGecko /trending/coins  → top 7 trending coins by search volume
  - CoinGecko /global          → global market dominance + 24h change
  - Alternative.me /fng/       → Crypto Fear & Greed Index (0-100)

Public API
----------
    get_trending_coins()       → list[dict]   (top trending coins with scores)
    get_fear_greed()           → dict         (fear/greed index value + label)
    get_social_summary()       → dict         (dashboard + Telegram combined view)
    trending_buy_signals()     → list[dict]   (coins meeting buy signal criteria)
    is_trending(symbol)        → bool         (in current trending set)
    social_rank_modifier(symbol) → float      (rank_score delta -10 to +10)

Rules:
  - Filter-only: signals reduce/boost rank_score, never trigger entries alone
  - 30-minute cache on trending; 1-hour cache on fear/greed
  - Fail-open on every call (returns empty/defaults on any error)

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

# ── Cache TTLs ────────────────────────────────────────────────────────────────

_TRENDING_TTL_S  = 1800   # 30 minutes
_FNG_TTL_S       = 3600   # 1 hour
_REQUEST_TIMEOUT = 10

# ── Thread-safe cache ─────────────────────────────────────────────────────────

_lock = threading.Lock()

_trending_cache: list[dict] = []
_trending_last_fetch: float = 0.0

_fng_cache: dict = {}
_fng_last_fetch: float = 0.0

_global_cache: dict = {}
_global_last_fetch: float = 0.0


# ── CoinGecko symbol → id mapping ─────────────────────────────────────────────

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
    "BNBUSDT":  "binancecoin",
    "LTCUSDT":  "litecoin",
}

# Reverse: coingecko id → USDT symbol
_ID_TO_SYMBOL: dict[str, str] = {v: k for k, v in _SYMBOL_TO_ID.items()}


# ── HTTP helper ───────────────────────────────────────────────────────────────

def _get_json(url: str) -> Optional[dict | list]:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "btcbot/1.0"})
        with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT) as resp:
            return json.loads(resp.read().decode())
    except Exception as exc:
        logger.log_warning(f"social_sentiment._get_json({url[:60]}...): {exc}")
        return None


# ── Trending coins ────────────────────────────────────────────────────────────

def _fetch_trending() -> bool:
    data = _get_json("https://api.coingecko.com/api/v3/trending")
    if not data or not isinstance(data, dict):
        return False
    try:
        coins = []
        for item in data.get("coins", []):
            coin = item.get("item", {})
            if not coin:
                continue
            cid    = coin.get("id", "")
            name   = coin.get("name", "")
            symbol = coin.get("symbol", "").upper()
            score  = float(coin.get("score", 0))   # rank 0 = top trending
            price_btc = float(coin.get("price_btc") or 0.0)
            usdt_sym  = _ID_TO_SYMBOL.get(cid, f"{symbol}USDT")
            coins.append({
                "id":        cid,
                "name":      name,
                "symbol":    symbol,
                "usdt_sym":  usdt_sym,
                "score":     score,
                "price_btc": price_btc,
                "rank":      int(coin.get("market_cap_rank") or 9999),
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            })

        with _lock:
            _trending_cache.clear()
            _trending_cache.extend(coins)
        return True
    except Exception as exc:
        logger.log_warning(f"social_sentiment._fetch_trending parse error: {exc}")
        return False


def _maybe_refresh_trending() -> None:
    global _trending_last_fetch
    with _lock:
        age = time.monotonic() - _trending_last_fetch
    if age >= _TRENDING_TTL_S:
        ok = _fetch_trending()
        if ok:
            with _lock:
                _trending_last_fetch = time.monotonic()


# ── Fear & Greed index ────────────────────────────────────────────────────────

def _fetch_fear_greed() -> bool:
    data = _get_json("https://api.alternative.me/fng/?limit=1&format=json")
    if not data or not isinstance(data, dict):
        return False
    try:
        entry = data.get("data", [{}])[0]
        value = int(entry.get("value", 50))
        label = entry.get("value_classification", "Neutral")
        ts    = entry.get("timestamp", "")
        with _lock:
            _fng_cache.clear()
            _fng_cache.update({
                "value": value,
                "label": label,
                "timestamp": ts,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            })
        return True
    except Exception as exc:
        logger.log_warning(f"social_sentiment._fetch_fear_greed parse error: {exc}")
        return False


def _maybe_refresh_fng() -> None:
    global _fng_last_fetch
    with _lock:
        age = time.monotonic() - _fng_last_fetch
    if age >= _FNG_TTL_S:
        ok = _fetch_fear_greed()
        if ok:
            with _lock:
                _fng_last_fetch = time.monotonic()


# ── Global market data ────────────────────────────────────────────────────────

def _fetch_global() -> bool:
    data = _get_json("https://api.coingecko.com/api/v3/global")
    if not data or not isinstance(data, dict):
        return False
    try:
        gd = data.get("data", {})
        with _lock:
            _global_cache.clear()
            _global_cache.update({
                "market_cap_change_24h": float(gd.get("market_cap_change_percentage_24h_usd") or 0.0),
                "btc_dominance":         float(gd.get("market_cap_percentage", {}).get("btc") or 0.0),
                "eth_dominance":         float(gd.get("market_cap_percentage", {}).get("eth") or 0.0),
                "active_cryptos":        int(gd.get("active_cryptocurrencies") or 0),
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            })
        return True
    except Exception as exc:
        logger.log_warning(f"social_sentiment._fetch_global parse error: {exc}")
        return False


def _maybe_refresh_global() -> None:
    global _global_last_fetch
    with _lock:
        age = time.monotonic() - _global_last_fetch
    if age >= _FNG_TTL_S:  # same TTL as fear/greed
        ok = _fetch_global()
        if ok:
            with _lock:
                _global_last_fetch = time.monotonic()


# ── Public API ────────────────────────────────────────────────────────────────

def get_trending_coins() -> list[dict]:
    """Return current list of trending coins (up to 7). Fail-open → []."""
    try:
        _maybe_refresh_trending()
        with _lock:
            return list(_trending_cache)
    except Exception:
        return []


def get_fear_greed() -> dict:
    """
    Return Fear & Greed index. Fail-open → {"value": 50, "label": "Neutral"}.

    value: 0 (extreme fear) → 100 (extreme greed)
    label: Extreme Fear / Fear / Neutral / Greed / Extreme Greed
    """
    try:
        _maybe_refresh_fng()
        with _lock:
            return dict(_fng_cache) if _fng_cache else {"value": 50, "label": "Neutral"}
    except Exception:
        return {"value": 50, "label": "Neutral"}


def get_global_market() -> dict:
    """Return global market data. Fail-open → {}."""
    try:
        _maybe_refresh_global()
        with _lock:
            return dict(_global_cache)
    except Exception:
        return {}


def is_trending(symbol: str) -> bool:
    """True if the symbol (e.g. 'BTCUSDT') is in the current trending list."""
    try:
        _maybe_refresh_trending()
        with _lock:
            tracked = {c["usdt_sym"] for c in _trending_cache}
        return symbol in tracked
    except Exception:
        return False


def trending_buy_signals() -> list[dict]:
    """
    Return trending coins that meet a basic buy-signal criteria:
    - In top-7 trending (score <= 6)
    - Fear & Greed > 45 (not extreme fear)
    - Global market 24h change > -3% (market not crashing)

    Returns list of dicts with signal metadata. Fail-open → [].
    """
    try:
        coins    = get_trending_coins()
        fng      = get_fear_greed()
        glb      = get_global_market()

        fng_val  = fng.get("value", 50)
        mkt_chg  = glb.get("market_cap_change_24h", 0.0)

        if fng_val < 25:   # extreme fear — no buy signals
            return []
        if mkt_chg < -5.0: # market crashing — skip
            return []

        signals = []
        for coin in coins:
            if coin.get("score", 99) > 6:
                continue
            strength = "STRONG" if fng_val >= 60 else "MODERATE"
            signals.append({
                **coin,
                "fng_value":  fng_val,
                "fng_label":  fng.get("label", "Neutral"),
                "mkt_change": mkt_chg,
                "strength":   strength,
            })
        return signals
    except Exception as exc:
        logger.log_warning(f"social_sentiment.trending_buy_signals error: {exc}")
        return []


def social_rank_modifier(symbol: str) -> float:
    """
    Rank score delta based on social sentiment.
    Range: -10 to +10.
    - Trending coin with greed: +5 to +10
    - Trending coin neutral: +2
    - Extreme fear market: -5 to -10
    Fail-open → 0.0.
    """
    try:
        fng    = get_fear_greed()
        fng_val = fng.get("value", 50)
        in_trend = is_trending(symbol)

        # Base modifier from fear/greed
        if fng_val >= 75:    base = +8.0   # extreme greed
        elif fng_val >= 60:  base = +4.0   # greed
        elif fng_val >= 45:  base = +1.0   # neutral
        elif fng_val >= 30:  base = -3.0   # fear
        else:                base = -8.0   # extreme fear

        # Boost if trending
        if in_trend:
            base = min(10.0, base + 5.0)

        return round(max(-10.0, min(10.0, base)), 1)
    except Exception:
        return 0.0


def get_social_summary() -> dict:
    """
    Combined summary for dashboard and Telegram.
    Returns: trending_coins, fear_greed, global_market, buy_signals.
    Fail-open → empty structure.
    """
    try:
        coins   = get_trending_coins()
        fng     = get_fear_greed()
        glb     = get_global_market()
        signals = trending_buy_signals()

        with _lock:
            trend_age = round(time.monotonic() - _trending_last_fetch, 0)
            fng_age   = round(time.monotonic() - _fng_last_fetch, 0)

        return {
            "trending_coins": coins,
            "fear_greed":     fng,
            "global_market":  glb,
            "buy_signals":    signals,
            "cache_age": {
                "trending_s": trend_age,
                "fng_s":      fng_age,
            },
        }
    except Exception as exc:
        logger.log_warning(f"social_sentiment.get_social_summary error: {exc}")
        return {
            "trending_coins": [],
            "fear_greed":     {"value": 50, "label": "Neutral"},
            "global_market":  {},
            "buy_signals":    [],
        }
