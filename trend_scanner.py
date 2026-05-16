"""
trend_scanner.py — Trending coin scanner with advanced quality filters
and multi-timeframe (MTF) confirmation.

Scoring (100 pts base + up to 15 MTF bonus = 115 max):
  Volume spike vs pool median  0–30 pts
  1h momentum                  0–25 pts
  4h momentum                  0–20 pts
  Quote volume / liquidity     0–15 pts
  Bid/ask spread               0–10 pts
  15m bullish                  +5 pts bonus
  1h bullish                   +5 pts bonus
  4h bullish                   +5 pts bonus
  Upper wick penalty           up to −10 pts
  Vol/price divergence penalty up to −10 pts

Quality filters (hard reject before scoring):
  |24h move| > TREND_MAX_24H_MOVE_PCT   — overextended, likely fading
  wick_ratio > TREND_MAX_WICK_RATIO     — manipulation candle
  volume spike ≥ 3× with |1h move| < 0.3%  — wash trading proxy
  spread > TREND_MAX_SPREAD_PCT         — illiquid

Grade system:
  A+  score ≥ 85, all 3 TFs bullish, vol_spike ≥ 2.5, spread ≤ 0.05%
  A   score ≥ 70, ≥ 2 TFs bullish
  B   score ≥ 55, ≥ 1 TF bullish
  C   score ≥ 40, passed quality filters
  Reject — failed quality filter OR (MTF required AND 0 bullish TFs)
            OR ≥ 2 bearish TFs

Alerts: only A+ and A grades trigger Telegram messages.
This module NEVER places trades. ENABLE_TREND_AUTO_TRADE is safety-locked.
"""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from binance.client import Client

import config
import logger
from sentiment_scanner import get_sentiment
from trend_alerts import send_trend_alert

DB_PATH = Path("trend_watchlist.db")

# ── DB schema ─────────────────────────────────────────────────────────────────

_CREATE_SIGNALS = """
CREATE TABLE IF NOT EXISTS trend_signals (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    detected_at_utc    TEXT NOT NULL,
    symbol             TEXT NOT NULL,
    price              REAL NOT NULL,
    price_change_1h    REAL NOT NULL,
    price_change_4h    REAL NOT NULL,
    volume_spike       REAL NOT NULL,
    spread_pct         REAL NOT NULL,
    volatility_pct     REAL NOT NULL,
    quote_volume       REAL NOT NULL,
    sentiment          TEXT NOT NULL,
    score              REAL NOT NULL,
    grade              TEXT NOT NULL DEFAULT 'C',
    mtf_15m            TEXT NOT NULL DEFAULT 'neutral',
    mtf_1h             TEXT NOT NULL DEFAULT 'neutral',
    mtf_4h             TEXT NOT NULL DEFAULT 'neutral',
    wick_ratio         REAL NOT NULL DEFAULT 0.0,
    move_24h_pct       REAL NOT NULL DEFAULT 0.0,
    max_move_pct       REAL,
    min_move_pct       REAL,
    continuation_score REAL,
    reversal_score     REAL,
    metrics_computed   INTEGER NOT NULL DEFAULT 0,
    alert_sent         INTEGER NOT NULL DEFAULT 0,
    created_at         TEXT NOT NULL
)
"""

_CREATE_OUTCOMES = """
CREATE TABLE IF NOT EXISTS trend_outcomes (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id      INTEGER NOT NULL,
    checked_at_utc TEXT NOT NULL,
    horizon        TEXT NOT NULL,
    original_price REAL NOT NULL,
    current_price  REAL NOT NULL,
    return_pct     REAL NOT NULL,
    created_at     TEXT NOT NULL,
    FOREIGN KEY (signal_id) REFERENCES trend_signals(id)
)
"""

# Columns added in this version — applied as migrations on existing DBs.
_SIGNAL_MIGRATIONS: dict[str, str] = {
    "grade":               "TEXT NOT NULL DEFAULT 'C'",
    "mtf_15m":             "TEXT NOT NULL DEFAULT 'neutral'",
    "mtf_1h":              "TEXT NOT NULL DEFAULT 'neutral'",
    "mtf_4h":              "TEXT NOT NULL DEFAULT 'neutral'",
    "wick_ratio":          "REAL NOT NULL DEFAULT 0.0",
    "move_24h_pct":        "REAL NOT NULL DEFAULT 0.0",
    "max_move_pct":        "REAL",
    "min_move_pct":        "REAL",
    "continuation_score":  "REAL",
    "reversal_score":      "REAL",
    "metrics_computed":    "INTEGER NOT NULL DEFAULT 0",
}


def _ensure_db() -> None:
    """Create tables and migrate any missing columns on existing DBs."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(_CREATE_SIGNALS)
        conn.execute(_CREATE_OUTCOMES)
        existing = {row[1] for row in conn.execute("PRAGMA table_info(trend_signals)")}
        for col, defn in _SIGNAL_MIGRATIONS.items():
            if col not in existing:
                conn.execute(f"ALTER TABLE trend_signals ADD COLUMN {col} {defn}")
        conn.commit()


# ── Data class ────────────────────────────────────────────────────────────────

@dataclass
class CoinScore:
    symbol: str
    price: float
    price_change_1h: float
    price_change_4h: float
    volume_spike: float
    spread_pct: float
    volatility_pct: float
    quote_volume: float
    sentiment: str
    score: float
    grade: str
    mtf_15m: str
    mtf_1h: str
    mtf_4h: str
    wick_ratio: float
    move_24h_pct: float


# ── Base scoring functions ─────────────────────────────────────────────────────

def _score_volume_spike(spike: float) -> float:
    if spike >= 5.0: return 30.0
    if spike >= 3.0: return 22.0
    if spike >= 2.0: return 15.0
    if spike >= 1.5: return 8.0
    return 0.0


def _score_momentum(change_pct: float, max_pts: float) -> float:
    if change_pct >= 5.0: return max_pts
    if change_pct >= 3.0: return max_pts * 0.75
    if change_pct >= 1.5: return max_pts * 0.50
    if change_pct >= 0.5: return max_pts * 0.25
    return 0.0


def _score_liquidity(quote_vol: float) -> float:
    if quote_vol >= 500_000_000: return 15.0
    if quote_vol >= 100_000_000: return 12.0
    if quote_vol >= 50_000_000:  return 8.0
    if quote_vol >= 10_000_000:  return 4.0
    return 0.0


def _score_spread(spread_pct: float) -> float:
    if spread_pct <= 0.02: return 10.0
    if spread_pct <= 0.05: return 7.0
    if spread_pct <= 0.10: return 4.0
    if spread_pct <= 0.20: return 1.0
    return 0.0


# ── Kline analysis ─────────────────────────────────────────────────────────────

def _analyze_klines(
    client: Client,
    symbol: str,
    interval: str,
    min_change_pct: float,
) -> tuple[float, str, float]:
    """
    Fetch 20 candles and return (price_change_pct, structure, wick_ratio).

    structure: 'bullish' | 'bearish' | 'neutral'
      Bullish requires ≥ 3 of 5 signals: above recent average, higher highs,
      higher lows, last candle green, period change ≥ min_change_pct.

    wick_ratio: upper_wick / body of the most recent candle.
      High ratio indicates potential manipulation or exhaustion.
    """
    try:
        klines = client.get_klines(symbol=symbol, interval=interval, limit=20)
        if len(klines) < 10:
            return 0.0, "neutral", 0.0

        opens  = [float(k[1]) for k in klines]
        highs  = [float(k[2]) for k in klines]
        lows   = [float(k[3]) for k in klines]
        closes = [float(k[4]) for k in klines]

        # Overall period change
        period_change = (
            (closes[-1] - closes[0]) / closes[0] * 100.0
            if closes[0] > 0 else 0.0
        )

        # Upper wick ratio on most recent candle
        body       = abs(closes[-1] - opens[-1])
        upper_wick = highs[-1] - max(closes[-1], opens[-1])
        wick_ratio = upper_wick / body if body > 1e-8 else 999.0

        # Structure signals
        avg_close    = sum(closes[-10:-1]) / 9
        above_avg    = closes[-1] > avg_close
        higher_highs = max(highs[-3:]) > max(highs[-6:-3])
        higher_lows  = min(lows[-3:])  > min(lows[-6:-3])
        last_bullish = closes[-1] > opens[-1]

        bullish_count = sum([
            above_avg,
            higher_highs,
            higher_lows,
            last_bullish,
            period_change >= min_change_pct,
        ])

        if bullish_count >= 3 and period_change >= min_change_pct:
            structure = "bullish"
        elif bullish_count <= 1 or period_change <= -min_change_pct:
            structure = "bearish"
        else:
            structure = "neutral"

        return period_change, structure, wick_ratio

    except Exception:
        return 0.0, "neutral", 0.0


def _get_orderbook_spread(client: Client, symbol: str) -> float:
    """Return bid/ask spread as % of mid price. Returns 999.0 on failure."""
    try:
        ob = client.get_order_book(symbol=symbol, limit=5)
        best_bid = float(ob["bids"][0][0])
        best_ask = float(ob["asks"][0][0])
        mid = (best_bid + best_ask) / 2.0
        if mid == 0:
            return 999.0
        return (best_ask - best_bid) / mid * 100.0
    except Exception:
        return 999.0


# ── Quality filters ────────────────────────────────────────────────────────────

def _quality_check(
    move_24h_pct: float,
    volume_spike: float,
    change_1h: float,
    wick_ratio: float,
    spread_pct: float,
) -> tuple[bool, list[str]]:
    """
    Hard quality gates. ALL must pass to proceed to scoring.
    Returns (passed, list_of_rejection_reasons).
    """
    reasons: list[str] = []

    if abs(move_24h_pct) > config.TREND_MAX_24H_MOVE_PCT:
        reasons.append(
            f"overextended_24h={move_24h_pct:.1f}% "
            f"(max {config.TREND_MAX_24H_MOVE_PCT}%)"
        )

    if wick_ratio > config.TREND_MAX_WICK_RATIO:
        reasons.append(
            f"wick_ratio={wick_ratio:.1f} "
            f"(max {config.TREND_MAX_WICK_RATIO})"
        )

    if config.TREND_REQUIRE_VOLUME_CONFIRMATION:
        if volume_spike >= 3.0 and abs(change_1h) < 0.3:
            reasons.append(
                f"vol_no_price_confirm: spike={volume_spike:.1f}× "
                f"but 1h_move={change_1h:.2f}%"
            )

    if spread_pct > config.TREND_MAX_SPREAD_PCT:
        reasons.append(
            f"spread={spread_pct:.3f}% "
            f"(max {config.TREND_MAX_SPREAD_PCT}%)"
        )

    return len(reasons) == 0, reasons


# ── Grading ────────────────────────────────────────────────────────────────────

def _compute_grade(
    score: float,
    mtf_15m: str,
    mtf_1h: str,
    mtf_4h: str,
    quality_passed: bool,
    volume_spike: float,
    spread_pct: float,
) -> str:
    """
    A+ — all 3 TFs bullish, score ≥ 85, vol ≥ 2.5×, tight spread
    A  — ≥ 2 TFs bullish, score ≥ 70
    B  — ≥ 1 TF bullish, score ≥ 55
    C  — passed quality filters, score ≥ 40
    Reject — failed quality filter, 0 bullish TFs (if MTF required),
              or ≥ 2 bearish TFs
    """
    if not quality_passed:
        return "Reject"

    tf_bullish = sum(1 for tf in (mtf_15m, mtf_1h, mtf_4h) if tf == "bullish")
    tf_bearish = sum(1 for tf in (mtf_15m, mtf_1h, mtf_4h) if tf == "bearish")

    if config.TREND_REQUIRE_MULTI_TIMEFRAME and tf_bullish == 0:
        return "Reject"
    if tf_bearish >= 2:
        return "Reject"

    if score >= 85 and tf_bullish == 3 and volume_spike >= 2.5 and spread_pct <= 0.05:
        return "A+"
    if score >= 70 and tf_bullish >= 2:
        return "A"
    if score >= 55 and tf_bullish >= 1:
        return "B"
    if score >= 40:
        return "C"
    return "Reject"


# ── Ticker filter ──────────────────────────────────────────────────────────────

_EXCLUDE_SUFFIXES = ("UPUSDT", "DOWNUSDT", "BULLUSDT", "BEARUSDT")
_STABLE_BASES = {"USDC", "BUSD", "TUSD", "USDP", "DAI", "FDUSD", "USDT"}


def _filter_tickers(tickers: list[dict]) -> list[dict]:
    """Keep liquid USDT spot pairs; exclude stablecoins and leveraged tokens."""
    filtered = []
    for t in tickers:
        sym = t["symbol"]
        if not sym.endswith("USDT"):
            continue
        if any(sym.endswith(s) for s in _EXCLUDE_SUFFIXES):
            continue
        base = sym[:-4]
        if base in _STABLE_BASES:
            continue
        try:
            qv = float(t.get("quoteVolume", 0))
        except (ValueError, TypeError):
            continue
        if qv < config.TREND_MIN_QUOTE_VOLUME:
            continue
        filtered.append(t)
    return filtered


# ── Main scan ──────────────────────────────────────────────────────────────────

def run_scan(client: Client) -> list[CoinScore]:
    """
    Full scan: fetch tickers → filter → MTF analysis → quality check →
    score → grade → save DB → send A+/A alerts.

    Returns the saved CoinScore list (excludes Reject grades).
    All failures are non-fatal — the trading bot continues regardless.

    Safety lock: ENABLE_TREND_AUTO_TRADE is ignored; scanner is WATCH ONLY.
    """
    if config.ENABLE_TREND_AUTO_TRADE:
        logger.log_warning(
            "TREND AUTO TRADE DISABLED BY SAFETY LOCK — "
            "ENABLE_TREND_AUTO_TRADE has no effect; scanner is watch-only."
        )

    _ensure_db()

    # ── 1. Fetch all 24h tickers (weight=40) ──────────────────────────────
    try:
        tickers = client.get_ticker()
    except Exception as exc:
        logger.log_warning(f"trend_scanner: get_ticker failed: {exc}")
        return []

    candidates = _filter_tickers(tickers)
    pre_sorted = sorted(
        candidates,
        key=lambda t: float(t.get("quoteVolume", 0)),
        reverse=True,
    )
    top_pool = pre_sorted[: config.TREND_SCANNER_TOP_N * 5]

    pool_volumes = [float(t.get("quoteVolume", 0)) for t in top_pool]
    pool_median  = sorted(pool_volumes)[len(pool_volumes) // 2] if pool_volumes else 1.0

    # ── 2. Analyse each candidate ─────────────────────────────────────────
    scored: list[CoinScore] = []

    for t in top_pool:
        sym = t["symbol"]
        try:
            price = float(t.get("lastPrice", 0))
            if price == 0:
                continue

            qv           = float(t.get("quoteVolume", 0))
            volume_spike = qv / pool_median if pool_median > 0 else 1.0

            # Early cheap reject
            if volume_spike < config.TREND_MIN_VOLUME_SPIKE:
                continue

            try:
                move_24h_pct = float(t.get("priceChangePercent", 0))
            except (ValueError, TypeError):
                move_24h_pct = 0.0

            # 24h overextension early reject (no API call needed)
            if abs(move_24h_pct) > config.TREND_MAX_24H_MOVE_PCT:
                continue

            # ── Kline fetches ──────────────────────────────────────────────
            time.sleep(0.05)
            change_1h, struct_1h, wick_ratio = _analyze_klines(
                client, sym,
                Client.KLINE_INTERVAL_1HOUR,
                config.TREND_MIN_1H_CONFIRMATION,
            )

            time.sleep(0.05)
            change_4h, struct_4h, _ = _analyze_klines(
                client, sym,
                Client.KLINE_INTERVAL_4HOUR,
                config.TREND_MIN_4H_CONFIRMATION,
            )

            struct_15m = "neutral"
            if config.TREND_REQUIRE_MULTI_TIMEFRAME:
                time.sleep(0.05)
                _, struct_15m, _ = _analyze_klines(
                    client, sym,
                    Client.KLINE_INTERVAL_15MINUTE,
                    config.TREND_MIN_15M_CONFIRMATION,
                )

            # ── Spread ────────────────────────────────────────────────────
            time.sleep(0.05)
            spread_pct = _get_orderbook_spread(client, sym)
            if spread_pct > config.TREND_MAX_SPREAD_PCT:
                continue

            volatility_pct = abs(move_24h_pct)

            # ── Quality check ─────────────────────────────────────────────
            quality_passed, reject_reasons = _quality_check(
                move_24h_pct, volume_spike, change_1h, wick_ratio, spread_pct,
            )

            # ── Base score ────────────────────────────────────────────────
            base_score = (
                _score_volume_spike(volume_spike)
                + _score_momentum(change_1h, 25.0)
                + _score_momentum(change_4h, 20.0)
                + _score_liquidity(qv)
                + _score_spread(spread_pct)
            )

            # ── MTF modifiers ─────────────────────────────────────────────
            for tf_struct in (struct_15m, struct_1h, struct_4h):
                if tf_struct == "bullish":
                    base_score += 5.0
                elif tf_struct == "bearish":
                    base_score -= 5.0

            # Wick penalty — large upper wick suggests exhaustion/manipulation
            if wick_ratio > 2.0:
                base_score -= min(10.0, (wick_ratio - 2.0) * 4.0)

            # Volume/price divergence penalty — spike not confirmed by move
            if volume_spike >= 3.0 and abs(change_1h) < 0.5:
                base_score -= 10.0

            score = max(0.0, min(115.0, base_score))

            # ── Grade ─────────────────────────────────────────────────────
            grade = _compute_grade(
                score, struct_15m, struct_1h, struct_4h,
                quality_passed, volume_spike, spread_pct,
            )

            if reject_reasons:
                logger.log_info(
                    f"trend_scanner: {sym} grade={grade} "
                    f"quality=[{'; '.join(reject_reasons)}]"
                )

            if grade == "Reject":
                continue

            scored.append(CoinScore(
                symbol=sym,
                price=price,
                price_change_1h=change_1h,
                price_change_4h=change_4h,
                volume_spike=volume_spike,
                spread_pct=spread_pct,
                volatility_pct=volatility_pct,
                quote_volume=qv,
                sentiment=get_sentiment(sym),
                score=score,
                grade=grade,
                mtf_15m=struct_15m,
                mtf_1h=struct_1h,
                mtf_4h=struct_4h,
                wick_ratio=wick_ratio,
                move_24h_pct=move_24h_pct,
            ))

        except Exception as exc:
            logger.log_warning(f"trend_scanner: scoring {sym} failed: {exc}")
            continue

    # ── 3. Rank and keep top N ────────────────────────────────────────────
    scored.sort(key=lambda c: c.score, reverse=True)
    top = scored[: config.TREND_SCANNER_TOP_N]

    if not top:
        logger.log_info("trend_scanner: no coins passed quality filters this cycle")
        return []

    # ── 4. Save to DB and send alerts ─────────────────────────────────────
    now_utc = datetime.now(timezone.utc).isoformat()
    saved: list[CoinScore] = []

    with sqlite3.connect(DB_PATH) as conn:
        for coin in top:
            alert_sent = 0
            # Only A+ and A grades get Telegram alerts
            if coin.grade in ("A+", "A"):
                ok = send_trend_alert(
                    symbol=coin.symbol,
                    price=coin.price,
                    price_change_1h=coin.price_change_1h,
                    price_change_4h=coin.price_change_4h,
                    volume_spike=coin.volume_spike,
                    spread_pct=coin.spread_pct,
                    volatility_pct=coin.volatility_pct,
                    sentiment=coin.sentiment,
                    score=coin.score,
                    grade=coin.grade,
                    mtf_15m=coin.mtf_15m,
                    mtf_1h=coin.mtf_1h,
                    mtf_4h=coin.mtf_4h,
                )
                if ok:
                    alert_sent = 1

            conn.execute(
                """
                INSERT INTO trend_signals
                    (detected_at_utc, symbol, price, price_change_1h,
                     price_change_4h, volume_spike, spread_pct, volatility_pct,
                     quote_volume, sentiment, score, grade,
                     mtf_15m, mtf_1h, mtf_4h,
                     wick_ratio, move_24h_pct, alert_sent, created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    now_utc, coin.symbol, coin.price, coin.price_change_1h,
                    coin.price_change_4h, coin.volume_spike, coin.spread_pct,
                    coin.volatility_pct, coin.quote_volume, coin.sentiment,
                    coin.score, coin.grade,
                    coin.mtf_15m, coin.mtf_1h, coin.mtf_4h,
                    coin.wick_ratio, coin.move_24h_pct, alert_sent, now_utc,
                ),
            )
            saved.append(coin)

        conn.commit()

    summary = ", ".join(f"{c.symbol}({c.grade},{c.score:.0f})" for c in top)
    logger.log_info(f"trend_scanner: top {len(top)} — {summary}")
    return saved
