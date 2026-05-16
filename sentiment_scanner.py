"""
sentiment_scanner.py — Placeholder sentiment module.

Currently returns "neutral" for all symbols.
Structured to accept a future Twitter/X API integration.

Usage:
    from sentiment_scanner import get_sentiment
    sentiment = get_sentiment("BTCUSDT")   # "bullish" | "bearish" | "neutral"
"""
from __future__ import annotations


def get_sentiment(symbol: str) -> str:
    """
    Return sentiment for *symbol*.

    Current implementation: always "neutral" (no data source connected).

    Future: query Twitter/X API, parse recent tweets mentioning the coin,
    run keyword or model-based classification, return "bullish" / "bearish" /
    "neutral".

    Args:
        symbol: Binance ticker, e.g. "BTCUSDT".

    Returns:
        One of "bullish", "bearish", or "neutral".
    """
    # ── Future integration point ───────────────────────────────────────────
    # base = symbol.replace("USDT", "").replace("BTC", "").upper()
    # tweets = _fetch_tweets(f"#{base} OR ${base}", limit=100)
    # return _classify(tweets)
    return "neutral"
