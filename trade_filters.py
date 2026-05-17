"""
trade_filters.py — Modular trade-quality filter engine.

Runs before every execute_buy call. Returns a list of FilterResult objects
that feed into trade_grader.py for final A+/A/B/C/Reject grading.

All filters are fail-safe: any exception → pass=True (never block a trade
due to a filter crash). All filters are non-raising.

Filters:
  A. btc_alignment      — Altcoin longs blocked if BTC strongly bearish
  B. volume_quality     — Reject low-volume / fake-spike setups
  C. candle_extension   — Reject entries after parabolic candles
  D. spread             — Reject if bid/ask spread exceeds ATR-adjusted limit
  E. bb_compression     — Prefer expansion after volatility squeeze
  F. session            — Score by trading session (Asia/London/NY)
  G. news_risk          — Block entries near major macro events
  H. symbol_cooldown    — Cooldown after stop-loss

Safety: if any filter function raises, FilterResult(passed=True) is returned
so the bot continues trading without filter protection.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import config
import logger


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class FilterResult:
    name: str
    passed: bool
    reason: str = ""
    hard_fail: bool = False    # True → always REJECT, ignore grade
    grade_penalty: int = 0     # subtracted from grade score if not passed


# ── Module-level cooldown state ───────────────────────────────────────────────

_cooldowns: dict[str, float] = {}   # symbol → monotonic time of last SL close


def record_stop_loss(symbol: str) -> None:
    """Call from main.py after a stop-loss closes to start cooldown."""
    _cooldowns[symbol] = time.monotonic()
    logger.log_info(f"FILTER | cooldown started for {symbol} ({config.SYMBOL_COOLDOWN_CANDLES} candles)")


# ── Session helper ────────────────────────────────────────────────────────────

def hour_to_session(hour: int) -> str:
    """Map UTC hour (0-23) to named trading session."""
    if 0 <= hour < 8:
        return "Asia"
    elif 8 <= hour < 13:
        return "London"
    elif 13 <= hour < 16:
        return "NY/London"
    elif 16 <= hour < 21:
        return "New York"
    else:
        return "Off-hours"


# ── Individual filters ────────────────────────────────────────────────────────

def filter_btc_alignment(symbol: str, scan_results: dict) -> FilterResult:
    """
    A. BTC Trend Alignment
    Reject altcoin longs if BTC is strongly bearish (EMA fast < EMA slow
    and price meaningfully below EMA fast).
    """
    name = "btc_alignment"
    if not getattr(config, "BTC_ALIGNMENT_FILTER", True):
        return FilterResult(name, True, "disabled")
    try:
        if symbol == "BTCUSDT":
            return FilterResult(name, True, "BTC itself — no alignment check")

        btc = scan_results.get("BTCUSDT")
        if btc is None or btc.df.empty or len(btc.df) < 22:
            return FilterResult(name, True, "BTCUSDT data unavailable — neutral")

        df = btc.df
        ema_fast = df["close"].ewm(span=config.EMA_FAST, adjust=False).mean().iloc[-1]
        ema_slow = df["close"].ewm(span=config.EMA_SLOW, adjust=False).mean().iloc[-1]
        price    = float(df["close"].iloc[-1])

        bearish_gap = (ema_fast - ema_slow) / ema_slow * 100   # negative = bearish

        if bearish_gap < -0.5 and price < ema_fast * 0.998:
            return FilterResult(
                name, False,
                f"BTC strongly bearish: EMA{config.EMA_FAST}={ema_fast:.0f} < EMA{config.EMA_SLOW}={ema_slow:.0f} ({bearish_gap:.2f}%)",
                hard_fail=True,
                grade_penalty=5,
            )
        if bearish_gap < -0.2:
            return FilterResult(
                name, False,
                f"BTC moderately bearish: gap={bearish_gap:.2f}%",
                hard_fail=False,
                grade_penalty=2,
            )

        return FilterResult(name, True, f"BTC alignment OK: gap={bearish_gap:+.2f}%")

    except Exception as exc:
        logger.log_warning(f"FILTER | {name} error (non-critical): {exc}")
        return FilterResult(name, True, f"error — pass: {exc}")


def filter_volume_quality(df) -> FilterResult:
    """
    B. Volume Quality
    Reject setups where current volume is below rolling median threshold,
    or where a volume spike is not accompanied by price movement (fake spike).
    """
    name = "volume_quality"
    if not getattr(config, "VOLUME_QUALITY_FILTER", True):
        return FilterResult(name, True, "disabled")
    try:
        if df.empty or len(df) < 21:
            return FilterResult(name, True, "insufficient data")

        min_ratio = getattr(config, "VOLUME_QUALITY_MIN_RATIO", 0.5)
        rolling_median = float(df["volume"].rolling(20).median().iloc[-1])
        current_vol    = float(df["volume"].iloc[-1])

        if rolling_median <= 0:
            return FilterResult(name, True, "median=0 — pass")

        ratio = current_vol / rolling_median

        if ratio < min_ratio:
            return FilterResult(
                name, False,
                f"low volume: {ratio:.2f}x median (min={min_ratio:.2f}x)",
                hard_fail=False,
                grade_penalty=3,
            )

        # Fake spike: volume spike but price barely moved
        if ratio > 5.0:
            price_change_pct = abs(
                float(df["close"].iloc[-1]) / float(df["close"].iloc[-2]) - 1
            ) * 100
            if price_change_pct < 0.15:
                return FilterResult(
                    name, False,
                    f"fake spike: vol={ratio:.1f}x median but price_chg={price_change_pct:.3f}%",
                    hard_fail=False,
                    grade_penalty=2,
                )

        return FilterResult(name, True, f"volume OK: {ratio:.2f}x median")

    except Exception as exc:
        logger.log_warning(f"FILTER | {name} error (non-critical): {exc}")
        return FilterResult(name, True, f"error — pass: {exc}")


def filter_candle_extension(df) -> FilterResult:
    """
    C. Candle Extension
    Reject entries after oversized candles that suggest chasing a parabolic move.
    """
    name = "candle_extension"
    if not getattr(config, "CANDLE_EXTENSION_FILTER", True):
        return FilterResult(name, True, "disabled")
    try:
        if df.empty or len(df) < 15:
            return FilterResult(name, True, "insufficient data")

        import pandas_ta as ta  # noqa
        atr_series = df.ta.atr(length=14)
        if atr_series is None or atr_series.empty:
            return FilterResult(name, True, "ATR unavailable")

        atr = float(atr_series.iloc[-1])
        if atr <= 0:
            return FilterResult(name, True, "ATR=0 — pass")

        last = df.iloc[-1]
        body = abs(float(last["close"]) - float(last["open"]))
        mult = getattr(config, "CANDLE_EXTENSION_ATR_MULTIPLE", 2.5)

        if body > mult * 3 * atr:
            return FilterResult(
                name, False,
                f"extreme candle body: {body:.2f} = {body/atr:.1f}x ATR (hard reject at >{mult*3:.1f}x)",
                hard_fail=True,
                grade_penalty=6,
            )
        if body > mult * atr:
            return FilterResult(
                name, False,
                f"extended candle: {body:.2f} = {body/atr:.1f}x ATR (limit={mult:.1f}x)",
                hard_fail=False,
                grade_penalty=3,
            )

        return FilterResult(name, True, f"candle body OK: {body/atr:.2f}x ATR")

    except Exception as exc:
        logger.log_warning(f"FILTER | {name} error (non-critical): {exc}")
        return FilterResult(name, True, f"error — pass: {exc}")


def filter_spread(symbol: str, client) -> FilterResult:
    """
    D. Spread Filter
    Reject entries if bid/ask spread exceeds the ATR-adjusted threshold.
    """
    name = "spread"
    if not getattr(config, "SPREAD_FILTER", True):
        return FilterResult(name, True, "disabled")
    try:
        if client is None:
            return FilterResult(name, True, "no client")

        ticker = client.get_orderbook_ticker(symbol=symbol)
        bid = float(ticker["bidPrice"])
        ask = float(ticker["askPrice"])
        if bid <= 0:
            return FilterResult(name, True, "bid=0 — pass")

        spread_bps = (ask - bid) / bid * 10_000   # basis points
        max_bps    = getattr(config, "MAX_SPREAD_BPS", 10)

        if spread_bps > max_bps * 3:
            return FilterResult(
                name, False,
                f"very wide spread: {spread_bps:.1f} bps (max={max_bps})",
                hard_fail=True,
                grade_penalty=4,
            )
        if spread_bps > max_bps:
            return FilterResult(
                name, False,
                f"wide spread: {spread_bps:.1f} bps (max={max_bps})",
                hard_fail=False,
                grade_penalty=2,
            )

        return FilterResult(name, True, f"spread OK: {spread_bps:.1f} bps")

    except Exception as exc:
        logger.log_warning(f"FILTER | {name} error (non-critical): {exc}")
        return FilterResult(name, True, f"error — pass: {exc}")


def filter_bb_compression(df) -> FilterResult:
    """
    E. Volatility Compression / Breakout Filter
    Prefer entries where BB width is expanding after a recent squeeze.
    Penalise entries into ongoing compression.
    """
    name = "bb_compression"
    if not getattr(config, "VOLATILITY_COMPRESSION_FILTER", True):
        return FilterResult(name, True, "disabled")
    try:
        if df.empty or len(df) < 25:
            return FilterResult(name, True, "insufficient data")

        period = getattr(config, "BB_PERIOD", 20)
        std    = getattr(config, "BB_STD", 2.0)

        bb_mid   = df["close"].rolling(period).mean()
        bb_std_s = df["close"].rolling(period).std()
        bb_upper = bb_mid + std * bb_std_s
        bb_lower = bb_mid - std * bb_std_s
        bb_width = (bb_upper - bb_lower) / bb_mid   # normalized

        if bb_width.isnull().all():
            return FilterResult(name, True, "BB unavailable")

        w_now  = float(bb_width.iloc[-1])
        w_5    = float(bb_width.iloc[-5])   # 5 bars ago
        w_10   = float(bb_width.iloc[-10])  # 10 bars ago

        # 25th percentile of recent width = squeeze threshold
        recent_widths = bb_width.dropna().iloc[-50:]
        squeeze_thresh = float(recent_widths.quantile(0.25))

        expanding_after_squeeze = w_5 < squeeze_thresh and w_now > w_5 * 1.1
        compressing = w_now < w_5 * 0.9 and w_now < squeeze_thresh

        if compressing:
            return FilterResult(
                name, False,
                f"volatility compressing: width={w_now:.4f} < threshold={squeeze_thresh:.4f}",
                hard_fail=False,
                grade_penalty=1,
            )
        if expanding_after_squeeze:
            # Bonus — handled in grader
            return FilterResult(name, True, f"squeeze breakout: width expanding {w_5:.4f}→{w_now:.4f}")

        return FilterResult(name, True, f"BB width normal: {w_now:.4f}")

    except Exception as exc:
        logger.log_warning(f"FILTER | {name} error (non-critical): {exc}")
        return FilterResult(name, True, f"error — pass: {exc}")


def filter_session(now_utc: datetime) -> FilterResult:
    """
    F. Session Filter
    Classify the current trading session and apply a grade penalty for
    low-quality sessions. By default does not block (no hard_fail).
    """
    name = "session"
    try:
        session = hour_to_session(now_utc.hour)
        weights = {
            "NY/London": 3,    # highest quality
            "London":    2,
            "New York":  2,
            "Asia":      0,
            "Off-hours": -1,
        }
        weight = weights.get(session, 0)

        if weight < 0:
            return FilterResult(
                name, False,
                f"session={session} (low quality)",
                hard_fail=False,
                grade_penalty=abs(weight),
            )
        return FilterResult(name, True, f"session={session}")

    except Exception as exc:
        logger.log_warning(f"FILTER | {name} error (non-critical): {exc}")
        return FilterResult(name, True, "error — pass")


def filter_news_risk(now_utc: datetime) -> FilterResult:
    """
    G. News Risk Filter
    Block entries within a configurable window around scheduled macro events.
    Reads from news_schedule.json.
    """
    name = "news_risk"
    if not getattr(config, "NEWS_RISK_FILTER", True):
        return FilterResult(name, True, "disabled")
    try:
        schedule_paths = [
            Path("/opt/btcbot/news_schedule.json"),
            Path("news_schedule.json"),
        ]
        schedule = []
        for p in schedule_paths:
            if p.exists():
                schedule = json.loads(p.read_text())
                break

        if not schedule:
            return FilterResult(name, True, "no schedule file")

        now_ts = now_utc.timestamp()
        for event in schedule:
            try:
                ev_dt  = datetime.fromisoformat(event["datetime_utc"]).replace(tzinfo=timezone.utc)
                window = event.get("window_minutes", 60) * 60
                delta  = abs(now_ts - ev_dt.timestamp())
                if delta < window:
                    return FilterResult(
                        name, False,
                        f"news window: {event['event']} in {int(delta//60)}m",
                        hard_fail=True,
                        grade_penalty=8,
                    )
            except Exception:
                continue

        return FilterResult(name, True, "no upcoming news events")

    except Exception as exc:
        logger.log_warning(f"FILTER | {name} error (non-critical): {exc}")
        return FilterResult(name, True, f"error — pass: {exc}")


def filter_symbol_cooldown(symbol: str, now_utc: datetime) -> FilterResult:
    """
    H. Symbol Cooldown Filter
    After a stop-loss, block the symbol for SYMBOL_COOLDOWN_CANDLES × interval.
    """
    name = "symbol_cooldown"
    if not getattr(config, "SYMBOL_COOLDOWN_CANDLES", 3):
        return FilterResult(name, True, "disabled (cooldown=0)")
    try:
        last_sl = _cooldowns.get(symbol)
        if last_sl is None:
            return FilterResult(name, True, "no recent SL")

        cooldown_secs = getattr(config, "SYMBOL_COOLDOWN_CANDLES", 3) * 3600
        elapsed       = time.monotonic() - last_sl
        remaining     = cooldown_secs - elapsed

        if remaining > 0:
            return FilterResult(
                name, False,
                f"cooldown active: {int(remaining//3600)}h {int((remaining%3600)//60)}m remaining",
                hard_fail=False,
                grade_penalty=4,
            )

        del _cooldowns[symbol]
        return FilterResult(name, True, "cooldown expired")

    except Exception as exc:
        logger.log_warning(f"FILTER | {name} error (non-critical): {exc}")
        return FilterResult(name, True, f"error — pass: {exc}")


# ── Public API ────────────────────────────────────────────────────────────────

def run_all(
    symbol: str,
    df,
    now_utc: datetime,
    scan_results: dict,
    client,
) -> list[FilterResult]:
    """
    Run all 8 filters. Never raises. Returns list of FilterResult.
    If ENABLE_TRADE_FILTERS=false, returns a single pass-all result.
    """
    if not getattr(config, "ENABLE_TRADE_FILTERS", True):
        return [FilterResult("all_filters", True, "ENABLE_TRADE_FILTERS=false")]

    results = [
        filter_btc_alignment(symbol, scan_results),
        filter_volume_quality(df),
        filter_candle_extension(df),
        filter_spread(symbol, client),
        filter_bb_compression(df),
        filter_session(now_utc),
        filter_news_risk(now_utc),
        filter_symbol_cooldown(symbol, now_utc),
    ]

    passed  = [f.name for f in results if f.passed]
    failed  = [(f.name, f.reason) for f in results if not f.passed]
    hard    = [f.name for f in results if f.hard_fail]

    logger.log_info(
        f"FILTERS | {symbol} | pass={len(passed)} fail={len(failed)} hard={len(hard)} "
        + (f"| blocked={hard}" if hard else "")
    )
    for f in results:
        if not f.passed:
            logger.log_info(f"  FILTER FAIL | {f.name}: {f.reason}")

    return results


def has_hard_fail(results: list[FilterResult]) -> bool:
    """True if any filter has hard_fail=True (trade must be rejected)."""
    return any(r.hard_fail for r in results)


def total_grade_penalty(results: list[FilterResult]) -> int:
    """Sum of grade_penalty from all failed filters."""
    return sum(r.grade_penalty for r in results if not r.passed)


def session_from_results(results: list[FilterResult]) -> str:
    """Extract session name from session filter result."""
    for r in results:
        if r.name == "session":
            for part in r.reason.split():
                if "=" in part:
                    return part.split("=", 1)[1]
    return "Unknown"
