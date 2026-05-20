"""
websocket_feed.py — Binance live price and candle cache via WebSocket.

Runs a background daemon thread subscribing to Binance combined streams:
  - <symbol>@miniTicker  → latest price, 24h stats (updates every ~1s)
  - <symbol>@kline_1h    → current forming 1H candle (open/high/low/close/vol)

The cache is read-only from the main trading loop's perspective — it never
influences trade decisions (which still use closed candles from hourly polling).

Use cases:
  - Dashboard live price display (more accurate than last hourly close)
  - Funding arb: current price for basis calculation
  - Grid engine: intra-hour price tracking
  - Logging: show live price in cycle logs

Safety guarantees:
  - Daemon thread — dies with main process, no cleanup needed
  - All exceptions caught and logged; thread auto-reconnects
  - Cache never written by main thread (read-only for trading logic)
  - WebSocket failure never blocks or crashes main loop

Public API
----------
    start(symbols)              → None  (launch background thread)
    get_live_price(symbol)      → float | None
    get_live_candle(symbol)     → dict  | None
    get_live_ticker(symbol)     → dict  | None
    get_feed_status()           → dict  (dashboard view)
    is_connected()              → bool

Never raises.
"""
from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from typing import Optional

import logger

# ── Config ────────────────────────────────────────────────────────────────────

_RECONNECT_DELAY_S = 5      # seconds before reconnect attempt
_MAX_RECONNECTS    = 999    # effectively infinite reconnects
_PING_INTERVAL_S   = 20     # keepalive ping interval
_STREAM_BASE_URL   = "wss://stream.binance.com:9443/stream"

# ── Thread-safe cache ─────────────────────────────────────────────────────────

_lock         = threading.Lock()
_price_cache: dict[str, float] = {}       # symbol → latest price
_ticker_cache: dict[str, dict] = {}       # symbol → miniTicker data
_candle_cache: dict[str, dict] = {}       # symbol → current forming candle
_connected:    bool = False
_last_msg_ts:  float = 0.0
_reconnect_count: int = 0

_thread: Optional[threading.Thread] = None
_started: bool = False
_watched_symbols: list[str] = []


# ── WebSocket message handler ──────────────────────────────────────────────────

def _handle_message(msg: dict) -> None:
    global _last_msg_ts
    _last_msg_ts = time.monotonic()

    stream = msg.get("stream", "")
    data   = msg.get("data", {})

    if not data or not stream:
        return

    # miniTicker: e.g. "BTCUSDT@miniTicker"
    if "@miniTicker" in stream:
        sym = data.get("s", "")
        if not sym:
            return
        price = float(data.get("c", 0) or 0)
        with _lock:
            _price_cache[sym] = price
            _ticker_cache[sym] = {
                "symbol":      sym,
                "price":       price,
                "open_24h":    float(data.get("o", 0) or 0),
                "high_24h":    float(data.get("h", 0) or 0),
                "low_24h":     float(data.get("l", 0) or 0),
                "volume_24h":  float(data.get("v", 0) or 0),
                "change_24h":  round((price / float(data.get("o", price) or price) - 1) * 100, 3) if price else 0.0,
                "updated_at":  datetime.now(timezone.utc).isoformat(),
            }

    # kline: e.g. "BTCUSDT@kline_1h"
    elif "@kline_" in stream:
        k   = data.get("k", {})
        sym = k.get("s", "")
        if not sym:
            return
        with _lock:
            _candle_cache[sym] = {
                "symbol":     sym,
                "open":       float(k.get("o", 0) or 0),
                "high":       float(k.get("h", 0) or 0),
                "low":        float(k.get("l", 0) or 0),
                "close":      float(k.get("c", 0) or 0),
                "volume":     float(k.get("v", 0) or 0),
                "is_closed":  bool(k.get("x", False)),
                "open_time":  int(k.get("t", 0)),
                "interval":   k.get("i", "1h"),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            # Also update price cache from kline close
            if float(k.get("c", 0) or 0) > 0:
                _price_cache[sym] = float(k["c"])


# ── Background feed thread ────────────────────────────────────────────────────

def _run_feed(symbols: list[str]) -> None:
    """Background thread: connect → subscribe → receive → reconnect loop."""
    global _connected, _reconnect_count

    try:
        import websocket
    except ImportError:
        logger.log_warning("websocket_feed: websocket-client not installed — feed disabled")
        return

    # Build combined stream URL
    streams = []
    for sym in symbols:
        s = sym.lower()
        streams.append(f"{s}@miniTicker")
        streams.append(f"{s}@kline_1h")

    url = f"{_STREAM_BASE_URL}?streams=" + "/".join(streams)

    reconnects = 0
    while reconnects < _MAX_RECONNECTS:
        try:
            logger.log_info(f"websocket_feed: connecting ({len(symbols)} symbols, {len(streams)} streams)")

            ws = websocket.WebSocketApp(
                url,
                on_message=lambda ws, msg: _on_message(msg),
                on_error=lambda ws, err: _on_error(err),
                on_close=lambda ws, code, msg: _on_close(),
                on_open=lambda ws: _on_open(),
            )
            ws.run_forever(ping_interval=_PING_INTERVAL_S, ping_timeout=10)

        except Exception as exc:
            logger.log_warning(f"websocket_feed: connection error: {exc}")

        with _lock:
            _connected = False
        reconnects += 1
        _reconnect_count = reconnects
        logger.log_info(f"websocket_feed: reconnecting in {_RECONNECT_DELAY_S}s (attempt {reconnects})")
        time.sleep(_RECONNECT_DELAY_S)

    logger.log_warning("websocket_feed: max reconnects reached — feed stopped")


def _on_open() -> None:
    global _connected
    with _lock:
        _connected = True
    logger.log_info("websocket_feed: connected")


def _on_close() -> None:
    global _connected
    with _lock:
        _connected = False


def _on_error(err) -> None:
    logger.log_warning(f"websocket_feed: error: {err}")


def _on_message(raw: str) -> None:
    try:
        msg = json.loads(raw)
        _handle_message(msg)
    except Exception as exc:
        logger.log_warning(f"websocket_feed._on_message parse error: {exc}")


# ── Public API ────────────────────────────────────────────────────────────────

def start(symbols: list[str]) -> None:
    """
    Launch background WebSocket daemon thread.
    Safe to call multiple times — only starts once.
    """
    global _thread, _started, _watched_symbols
    if _started:
        return
    try:
        import websocket  # noqa — check available before starting
    except ImportError:
        logger.log_warning("websocket_feed: websocket-client not installed — skipping feed")
        return

    _watched_symbols = list(symbols)
    _started = True
    t = threading.Thread(
        target=_run_feed,
        args=(symbols,),
        name="websocket_feed",
        daemon=True,
    )
    t.start()
    _thread = t
    logger.log_info(f"websocket_feed: started for {symbols}")


def get_live_price(symbol: str) -> Optional[float]:
    """Return latest live price for symbol. None if not yet received."""
    try:
        with _lock:
            return _price_cache.get(symbol)
    except Exception:
        return None


def get_live_ticker(symbol: str) -> Optional[dict]:
    """Return full miniTicker data dict for symbol. None if not received."""
    try:
        with _lock:
            data = _ticker_cache.get(symbol)
            return dict(data) if data else None
    except Exception:
        return None


def get_live_candle(symbol: str) -> Optional[dict]:
    """Return current forming 1H candle data. None if not received."""
    try:
        with _lock:
            data = _candle_cache.get(symbol)
            return dict(data) if data else None
    except Exception:
        return None


def is_connected() -> bool:
    """True if WebSocket is currently connected."""
    try:
        with _lock:
            return _connected
    except Exception:
        return False


def get_feed_status() -> dict:
    """
    Return feed health summary for dashboard.
    Fail-open → empty structure.
    """
    try:
        with _lock:
            conn        = _connected
            n_prices    = len(_price_cache)
            n_candles   = len(_candle_cache)
            last_msg    = _last_msg_ts
            reconnects  = _reconnect_count
            prices_snap = {sym: round(p, 2) for sym, p in _price_cache.items()}
            tickers_snap = {
                sym: {
                    "change_24h": d.get("change_24h", 0.0),
                    "volume_24h": d.get("volume_24h", 0.0),
                }
                for sym, d in _ticker_cache.items()
            }

        age_s = round(time.monotonic() - last_msg, 1) if last_msg > 0 else None

        return {
            "connected":       conn,
            "symbols_tracked": n_prices,
            "candles_cached":  n_candles,
            "last_message_s":  age_s,
            "reconnect_count": reconnects,
            "live_prices":     prices_snap,
            "tickers_24h":     tickers_snap,
            "started":         _started,
        }
    except Exception as exc:
        return {"connected": False, "error": str(exc)}
