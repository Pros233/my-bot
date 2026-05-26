"""
candidate_scanner_15m.py — 15-minute intraday candidate generator.

Scans 15m OHLCV data for early trade setups. Does NOT execute trades.
Acts as a supplementary candidate source when the 1H loop finds nothing.

Default: ENABLE_15M_CANDIDATE_SCAN=false  (off unless explicitly enabled)

Public API:
    scan_15m_candidates(client, data_client, symbols, now_utc) -> list[Candidate15m]
    get_stats()                                                 -> dict
    record_scan(candidates, confirmed, rejected, ...)          -> None
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

import config
import logger

_STATS_PATH = Path(__file__).parent / "candidate_scanner_15m_stats.json"
_CANDLES    = 100   # 15m candles per symbol (~25 h of history)


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class Candidate15m:
    symbol:          str
    side:            str    = "BUY"
    engine:          str    = ""
    setup_name:      str    = ""
    interval:        str    = "15m"
    entry_reference: float  = 0.0
    stop_reference:  float  = 0.0
    tp_reference:    float  = 0.0
    rank_score:      float  = 0.0
    grade_estimate:  str    = "B"
    volume_ratio:    float  = 1.0
    atr_pct:         float  = 0.0
    rsi:             float  = 50.0
    adx:             float  = 0.0
    reason:          str    = ""
    timestamp:       str    = ""


# ── Indicator helpers ─────────────────────────────────────────────────────────

def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, min_periods=period).mean()
    rs   = gain / loss.replace(0, 1e-10)
    return (100 - (100 / (1 + rs))).fillna(50.0)


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, prev_c = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([high - low,
                    (high - prev_c).abs(),
                    (low  - prev_c).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period).mean()


def _vol_ratio(volume: pd.Series, period: int = 20) -> pd.Series:
    ma = volume.rolling(period).mean()
    return (volume / ma.replace(0, 1)).fillna(1.0)


def _adx_simple(df: pd.DataFrame, period: int = 14) -> float:
    """Scalar ADX from last row (simplified, no external dependency)."""
    try:
        high, low = df["high"], df["low"]
        plus_dm  = high.diff().clip(lower=0)
        minus_dm = (-low.diff()).clip(lower=0)
        mask_p = plus_dm  < minus_dm; plus_dm[mask_p]  = 0.0
        mask_m = minus_dm < plus_dm;  minus_dm[mask_m] = 0.0
        atr_s = _atr(df, period)
        pdi   = 100 * plus_dm.ewm(alpha=1/period).mean()  / atr_s.replace(0, 1)
        mdi   = 100 * minus_dm.ewm(alpha=1/period).mean() / atr_s.replace(0, 1)
        dx    = (100 * (pdi - mdi).abs() / (pdi + mdi + 1e-10))
        return float(dx.ewm(alpha=1/period).mean().iloc[-1])
    except Exception:
        return 20.0


# ── Setup detectors ───────────────────────────────────────────────────────────

def _cfg(key: str, default):
    return getattr(config, key, default)


def _base_checks(df: pd.DataFrame, atr_s: pd.Series, vol_s: pd.Series,
                 extra_vol: float = 0.0) -> tuple[bool, float, float, float, float]:
    """
    Validate ATR% and volume thresholds.
    Returns (ok, curr_close, curr_atr, curr_atr_pct, curr_vol).
    """
    curr_close  = float(df["close"].iloc[-1])
    curr_atr    = float(atr_s.iloc[-1])
    curr_atr_pct = curr_atr / curr_close * 100 if curr_close > 0 else 0.0
    curr_vol    = float(vol_s.iloc[-1])

    if curr_atr_pct < _cfg("CANDIDATE_15M_MIN_ATR_PCT", 0.2):
        return False, curr_close, curr_atr, curr_atr_pct, curr_vol
    if curr_atr_pct > _cfg("CANDIDATE_15M_MAX_ATR_PCT", 3.5):
        return False, curr_close, curr_atr, curr_atr_pct, curr_vol
    min_vol = max(_cfg("CANDIDATE_15M_MIN_VOLUME_RATIO", 1.1), extra_vol)
    if curr_vol < min_vol:
        return False, curr_close, curr_atr, curr_atr_pct, curr_vol
    return True, curr_close, curr_atr, curr_atr_pct, curr_vol


def _detect_pullback_reclaim(df: pd.DataFrame, symbol: str, now_utc) -> Optional[Candidate15m]:
    """15M_PULLBACK_RECLAIM: price dipped below EMA21 then reclaimed it last bar."""
    try:
        if len(df) < 30:
            return None
        close = df["close"]
        ema21 = _ema(close, 21)
        ema9  = _ema(close, 9)
        atr_s = _atr(df)
        vol_s = _vol_ratio(df["volume"])
        rsi_s = _rsi(close)

        ok, curr_c, curr_atr, curr_atr_pct, curr_vol = _base_checks(df, atr_s, vol_s)
        if not ok:
            return None

        prev_c    = float(close.iloc[-2])
        prev_e21  = float(ema21.iloc[-2])
        curr_e21  = float(ema21.iloc[-1])
        curr_rsi  = float(rsi_s.iloc[-1])
        ema9_above = float(ema9.iloc[-1]) > curr_e21

        # Core pattern: price crossed above EMA21
        if not (prev_c < prev_e21 and curr_c > curr_e21):
            return None
        if not (40 <= curr_rsi <= 72):
            return None

        stop = curr_c - 1.5 * curr_atr
        tp   = curr_c + 2.5 * curr_atr
        rank = 40.0 + (10.0 if ema9_above else 0.0) + min(curr_vol * 5, 20.0)

        return Candidate15m(
            symbol=symbol, side="BUY", engine="PULLBACK",
            setup_name="15M_PULLBACK_RECLAIM",
            entry_reference=curr_c,
            stop_reference=round(stop, 8),
            tp_reference=round(tp, 8),
            rank_score=round(rank, 1),
            grade_estimate="B",
            volume_ratio=round(curr_vol, 2),
            atr_pct=round(curr_atr_pct, 3),
            rsi=round(curr_rsi, 1),
            adx=round(_adx_simple(df), 1),
            reason=f"EMA21 reclaim | RSI={curr_rsi:.0f} | vol={curr_vol:.1f}x",
            timestamp=now_utc.strftime("%Y-%m-%d %H:%M UTC"),
        )
    except Exception:
        return None


def _detect_micro_breakout(df: pd.DataFrame, symbol: str, now_utc) -> Optional[Candidate15m]:
    """15M_MICRO_BREAKOUT: close broke above 20-bar high with volume surge."""
    try:
        if len(df) < 25:
            return None
        close = df["close"]
        high  = df["high"]
        atr_s = _atr(df)
        vol_s = _vol_ratio(df["volume"])
        rsi_s = _rsi(close)

        ok, curr_c, curr_atr, curr_atr_pct, curr_vol = _base_checks(
            df, atr_s, vol_s, extra_vol=1.3)
        if not ok:
            return None

        prev_high_20 = float(high.iloc[-21:-2].max())
        curr_rsi     = float(rsi_s.iloc[-1])

        if not (curr_c > prev_high_20):
            return None
        if not (50 <= curr_rsi <= 78):
            return None

        stop = prev_high_20 - curr_atr * 0.5
        tp   = curr_c + 2.0 * curr_atr
        rank = 45.0 + min(curr_vol * 6, 25.0)

        return Candidate15m(
            symbol=symbol, side="BUY", engine="BREAKOUT",
            setup_name="15M_MICRO_BREAKOUT",
            entry_reference=curr_c,
            stop_reference=round(stop, 8),
            tp_reference=round(tp, 8),
            rank_score=round(rank, 1),
            grade_estimate="B",
            volume_ratio=round(curr_vol, 2),
            atr_pct=round(curr_atr_pct, 3),
            rsi=round(curr_rsi, 1),
            adx=round(_adx_simple(df), 1),
            reason=f"Break {prev_high_20:.6g} high | vol={curr_vol:.1f}x | RSI={curr_rsi:.0f}",
            timestamp=now_utc.strftime("%Y-%m-%d %H:%M UTC"),
        )
    except Exception:
        return None


def _detect_momentum_continuation(df: pd.DataFrame, symbol: str, now_utc) -> Optional[Candidate15m]:
    """15M_MOMENTUM_CONTINUATION: EMA9>EMA21, RSI 50-70, price above both EMAs."""
    try:
        if len(df) < 30:
            return None
        close = df["close"]
        ema9  = _ema(close, 9)
        ema21 = _ema(close, 21)
        atr_s = _atr(df)
        vol_s = _vol_ratio(df["volume"])
        rsi_s = _rsi(close)

        ok, curr_c, curr_atr, curr_atr_pct, curr_vol = _base_checks(df, atr_s, vol_s)
        if not ok:
            return None

        curr_e9  = float(ema9.iloc[-1])
        curr_e21 = float(ema21.iloc[-1])
        curr_rsi = float(rsi_s.iloc[-1])

        if not (curr_e9 > curr_e21):
            return None
        if not (50 <= curr_rsi <= 70):
            return None
        if curr_c < curr_e9:
            return None

        stop = curr_e21 - 0.5 * curr_atr
        tp   = curr_c + 2.0 * curr_atr
        rank = 35.0 + (curr_rsi - 50) * 0.5 + min(curr_vol * 4, 15.0)

        return Candidate15m(
            symbol=symbol, side="BUY", engine="MOMENTUM",
            setup_name="15M_MOMENTUM_CONTINUATION",
            entry_reference=curr_c,
            stop_reference=round(stop, 8),
            tp_reference=round(tp, 8),
            rank_score=round(rank, 1),
            grade_estimate="B",
            volume_ratio=round(curr_vol, 2),
            atr_pct=round(curr_atr_pct, 3),
            rsi=round(curr_rsi, 1),
            adx=round(_adx_simple(df), 1),
            reason=f"EMA9>EMA21 | RSI={curr_rsi:.0f} | vol={curr_vol:.1f}x",
            timestamp=now_utc.strftime("%Y-%m-%d %H:%M UTC"),
        )
    except Exception:
        return None


def _detect_range_bounce(df: pd.DataFrame, symbol: str, now_utc) -> Optional[Candidate15m]:
    """15M_RANGE_BOUNCE: near 20-bar range low with RSI recovering from oversold."""
    try:
        if len(df) < 30:
            return None
        close = df["close"]
        low   = df["low"]
        atr_s = _atr(df)
        vol_s = _vol_ratio(df["volume"])
        rsi_s = _rsi(close)

        ok, curr_c, curr_atr, curr_atr_pct, curr_vol = _base_checks(df, atr_s, vol_s)
        if not ok:
            return None

        range_low = float(low.iloc[-21:-1].min())
        curr_rsi  = float(rsi_s.iloc[-1])
        prev_rsi  = float(rsi_s.iloc[-2])

        if curr_c > range_low + curr_atr:      # too far from support
            return None
        if not (prev_rsi <= 42 and curr_rsi > prev_rsi):
            return None

        stop = range_low - 0.5 * curr_atr
        tp   = curr_c + 1.5 * curr_atr
        rank = 32.0 + (42 - min(prev_rsi, 42)) * 0.5 + min(curr_vol * 3, 12.0)

        return Candidate15m(
            symbol=symbol, side="BUY", engine="RANGE_MR",
            setup_name="15M_RANGE_BOUNCE",
            entry_reference=curr_c,
            stop_reference=round(stop, 8),
            tp_reference=round(tp, 8),
            rank_score=round(rank, 1),
            grade_estimate="C",
            volume_ratio=round(curr_vol, 2),
            atr_pct=round(curr_atr_pct, 3),
            rsi=round(curr_rsi, 1),
            adx=round(_adx_simple(df), 1),
            reason=f"Range bounce near {range_low:.6g} | RSI {prev_rsi:.0f}→{curr_rsi:.0f}",
            timestamp=now_utc.strftime("%Y-%m-%d %H:%M UTC"),
        )
    except Exception:
        return None


def _detect_vol_expansion(df: pd.DataFrame, symbol: str, now_utc) -> Optional[Candidate15m]:
    """15M_VOL_EXPANSION: ATR expanding >1.3× recent, bullish price move."""
    try:
        if len(df) < 25:
            return None
        close = df["close"]
        atr_s = _atr(df)
        vol_s = _vol_ratio(df["volume"])
        rsi_s = _rsi(close)

        ok, curr_c, curr_atr, curr_atr_pct, curr_vol = _base_checks(
            df, atr_s, vol_s, extra_vol=1.4)
        if not ok:
            return None

        prev_c   = float(close.iloc[-4])
        prev_atr = float(atr_s.iloc[-5])
        curr_rsi = float(rsi_s.iloc[-1])

        if prev_atr <= 0 or curr_atr < prev_atr * 1.3:
            return None
        if curr_c <= prev_c:
            return None
        if not (52 <= curr_rsi <= 75):
            return None

        stop = curr_c - 2.0 * curr_atr
        tp   = curr_c + 2.0 * curr_atr
        rank = (38.0
                + min(curr_vol * 5, 20.0)
                + min((curr_atr / prev_atr - 1.0) * 20, 15.0))

        return Candidate15m(
            symbol=symbol, side="BUY", engine="VOL_EXPANSION",
            setup_name="15M_VOL_EXPANSION",
            entry_reference=curr_c,
            stop_reference=round(stop, 8),
            tp_reference=round(tp, 8),
            rank_score=round(rank, 1),
            grade_estimate="B",
            volume_ratio=round(curr_vol, 2),
            atr_pct=round(curr_atr_pct, 3),
            rsi=round(curr_rsi, 1),
            adx=round(_adx_simple(df), 1),
            reason=(f"ATR expand {prev_atr:.6g}→{curr_atr:.6g} | "
                    f"RSI={curr_rsi:.0f} | vol={curr_vol:.1f}x"),
            timestamp=now_utc.strftime("%Y-%m-%d %H:%M UTC"),
        )
    except Exception:
        return None


_DETECTORS = [
    _detect_pullback_reclaim,
    _detect_micro_breakout,
    _detect_momentum_continuation,
    _detect_range_bounce,
    _detect_vol_expansion,
]


# ── Kline → DataFrame (no main.py import) ────────────────────────────────────

def _klines_to_df(klines: list) -> pd.DataFrame:
    df = pd.DataFrame(klines, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)
    return df[["open", "high", "low", "close", "volume"]].iloc[:-1]  # drop forming candle


# ── Stats ─────────────────────────────────────────────────────────────────────

def _load_stats() -> dict:
    try:
        if _STATS_PATH.exists():
            return json.loads(_STATS_PATH.read_text())
    except Exception:
        pass
    return {
        "last_scan_time": None, "last_scan_symbol_count": 0,
        "candidates_found_today": 0, "confirmed_today": 0,
        "rejected_today": 0, "rejection_reasons": {},
        "last_selected": None, "scan_date": None,
    }


def _save_stats(stats: dict) -> None:
    try:
        _STATS_PATH.write_text(json.dumps(stats, indent=2))
    except Exception:
        pass


def get_stats() -> dict:
    return _load_stats()


def record_scan(
    candidates: list,
    confirmed:  int = 0,
    rejected:   int = 0,
    rejection_reasons: list | None = None,
    last_selected: dict | None = None,
    symbol_count: int = 0,
) -> None:
    stats = _load_stats()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if stats.get("scan_date") != today:
        stats.update({
            "candidates_found_today": 0, "confirmed_today": 0,
            "rejected_today": 0, "rejection_reasons": {}, "scan_date": today,
        })
    stats["last_scan_time"]         = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    stats["last_scan_symbol_count"] = symbol_count
    stats["candidates_found_today"] += len(candidates)
    stats["confirmed_today"]        += confirmed
    stats["rejected_today"]         += rejected
    for r in (rejection_reasons or []):
        stats["rejection_reasons"][r] = stats["rejection_reasons"].get(r, 0) + 1
    if last_selected:
        stats["last_selected"] = last_selected
    _save_stats(stats)


# ── Public scan API ───────────────────────────────────────────────────────────

def scan_15m_candidates(
    client,
    data_client,
    symbols: list,
    now_utc,
) -> list:
    """
    Scan 15m candles for all symbols and return candidates sorted by rank.
    Never raises — returns empty list on any error.
    Only runs when ENABLE_15M_CANDIDATE_SCAN=true.
    """
    if not getattr(config, "ENABLE_15M_CANDIDATE_SCAN", False):
        return []

    max_cands  = getattr(config, "CANDIDATE_15M_MAX_CANDIDATES_PER_CYCLE", 5)
    min_rank   = getattr(config, "CANDIDATE_15M_MIN_RANK_SCORE", 20)
    syms_cfg   = getattr(config, "CANDIDATE_SCAN_SYMBOLS", "")
    if syms_cfg.strip():
        allowed = [s.strip() for s in syms_cfg.split(",") if s.strip()]
        symbols = [s for s in symbols if s in allowed]

    logger.log_info(f"15M_SCAN_START | symbols={len(symbols)}")
    all_candidates: list = []

    for sym in symbols:
        try:
            klines = (data_client or client).get_klines(
                symbol=sym, interval="15m", limit=_CANDLES + 1
            )
            if not klines or len(klines) < 32:
                continue
            df = _klines_to_df(klines)
            if len(df) < 30:
                continue
            for detect in _DETECTORS:
                c = detect(df, sym, now_utc)
                if c and c.rank_score >= min_rank:
                    logger.log_info(
                        f"15M_CANDIDATE | symbol={c.symbol} | setup={c.setup_name}"
                        f" | grade={c.grade_estimate} | rank={c.rank_score:.0f}"
                        f" | {c.reason}"
                    )
                    all_candidates.append(c)
        except Exception as exc:
            logger.log_warning(f"15m scan error for {sym}: {exc}")

    all_candidates.sort(key=lambda c: c.rank_score, reverse=True)
    return all_candidates[:max_cands]
