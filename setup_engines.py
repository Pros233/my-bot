"""
setup_engines.py — Additional live setup engines (all opt-in via config flags).

Each engine evaluates a 1H OHLCV DataFrame and returns an EngineSignal.
Engines are independent of the primary RMR strategy and run after it in the
scan pipeline.  All are disabled by default — enable via .env:

    ENABLE_PULLBACK_SETUP=true
    ENABLE_BREAKOUT_SETUP=true
    ENABLE_NY_MOMENTUM_SETUP=true
    ENABLE_MEAN_REVERSION_SETUP=true

Safety guarantees
-----------------
* No engine ever raises — all exceptions return EngineSignal(direction="NONE").
* No engine modifies shared state.
* Engines use ONLY the df passed to them (no Binance API calls).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

import config


# ── Signal dataclass ──────────────────────────────────────────────────────────

@dataclass
class EngineSignal:
    direction: str          # "LONG" | "NONE"
    engine: str             # engine name: "PULLBACK" | "BREAKOUT" | "NY_MOMENTUM" | "MICRO_MR"
    entry_price: float = 0.0
    stop_price:  float = 0.0
    tp_price:    float = 0.0
    stop_distance: float = 0.0
    atr_value:   float = 0.0
    rank_boost:  float = 0.0    # added to rank_score in scan_symbol
    reason:      str   = ""
    reject_reason: str = ""


def _no_signal(engine: str, reason: str) -> EngineSignal:
    return EngineSignal(direction="NONE", engine=engine, reject_reason=reason)


# ── Shared helpers ────────────────────────────────────────────────────────────

def _compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    try:
        import pandas_ta as ta  # noqa: F401
        return df.ta.atr(length=period)
    except Exception:
        high_low = df["high"] - df["low"]
        return high_low.rolling(period).mean()


def _compute_ema(df: pd.DataFrame, length: int) -> pd.Series:
    return df["close"].ewm(span=length, adjust=False).mean()


def _compute_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    try:
        import pandas_ta as ta  # noqa: F401
        return df.ta.rsi(length=period)
    except Exception:
        delta = df["close"].diff()
        gain  = delta.clip(lower=0).rolling(period).mean()
        loss  = (-delta.clip(upper=0)).rolling(period).mean()
        rs    = gain / loss.replace(0, float("nan"))
        return 100 - 100 / (1 + rs)


def _hammer_pattern(df: pd.DataFrame) -> bool:
    """True if the last closed candle is a hammer (lower wick >= body)."""
    row   = df.iloc[-1]
    body  = abs(row["close"] - row["open"])
    lower = row["open"] - row["low"] if row["close"] >= row["open"] else row["close"] - row["low"]
    if body == 0:
        return False
    return lower >= body


# ── Engine A: Pullback continuation ──────────────────────────────────────────

def get_pullback_signal(df: pd.DataFrame) -> EngineSignal:
    """
    Trend continuation entry on EMA21 pullback reclaim.

    Conditions (ALL required):
    1. EMA9 > EMA21 (uptrend)
    2. Close pulled back to within 0.5% of EMA21 at some point in last 3 bars
    3. Last close > EMA21 (reclaim)
    4. RSI 40-65 (not overbought, not oversold)
    5. Volume on reclaim bar >= 0.8× 20-bar average

    Stop: below EMA21 - 0.5×ATR
    TP:   entry + 2×stop_distance
    """
    engine = "PULLBACK"
    try:
        if len(df) < 30:
            return _no_signal(engine, "insufficient data")

        ema9  = _compute_ema(df, 9)
        ema21 = _compute_ema(df, 21)
        atr   = _compute_atr(df, 14)
        rsi   = _compute_rsi(df, 14)
        vol_ma = df["volume"].rolling(20).mean()

        c = df["close"].iloc[-1]
        e9 = float(ema9.iloc[-1])
        e21 = float(ema21.iloc[-1])
        atr_val = float(atr.iloc[-1])
        rsi_val = float(rsi.iloc[-1])
        vol_now = float(df["volume"].iloc[-1])
        vol_avg = float(vol_ma.iloc[-1])

        if e9 <= e21:
            return _no_signal(engine, f"EMA9={e9:.2f} not above EMA21={e21:.2f}")

        # Check pullback within last 3 bars (low touched EMA21 ± 0.5%)
        touched = any(
            abs(float(df["low"].iloc[-i]) - float(ema21.iloc[-i])) / float(ema21.iloc[-i]) <= 0.005
            for i in range(1, 4)
        )
        if not touched:
            return _no_signal(engine, "no EMA21 touch in last 3 bars")

        if c <= e21:
            return _no_signal(engine, f"close={c:.2f} not above EMA21={e21:.2f}")

        if not (40 <= rsi_val <= 65):
            return _no_signal(engine, f"RSI={rsi_val:.1f} outside 40-65")

        if vol_now < 0.8 * vol_avg:
            return _no_signal(engine, f"volume={vol_now:.0f} < 0.8×avg={vol_avg:.0f}")

        entry = c * (1 + config.SLIPPAGE)
        stop  = e21 - 0.5 * atr_val
        stop_dist = entry - stop
        if stop_dist <= 0:
            return _no_signal(engine, "stop_distance <= 0")
        tp = entry + 2.0 * stop_dist

        return EngineSignal(
            direction="LONG", engine=engine,
            entry_price=round(entry, 4), stop_price=round(stop, 4),
            tp_price=round(tp, 4), stop_distance=round(stop_dist, 4),
            atr_value=round(atr_val, 4), rank_boost=50.0,
            reason=f"EMA21 pullback reclaim | RSI={rsi_val:.0f} | vol_ratio={vol_now/vol_avg:.1f}x",
        )

    except Exception as exc:
        return _no_signal(engine, f"error: {exc}")


# ── Engine B: Volatility breakout ─────────────────────────────────────────────

def get_breakout_signal(df: pd.DataFrame) -> EngineSignal:
    """
    Bollinger Band squeeze → expansion breakout with volume confirmation.

    Conditions (ALL required):
    1. BB width in bottom 20% of last 20-bar range (squeeze)
    2. Last close breaks above upper BB
    3. Volume on breakout bar >= 1.5× 20-bar average
    4. RSI 50-75 (momentum aligned, not extreme)

    Stop: prior BB midline (SMA20)
    TP:   entry + 2×stop_distance
    """
    engine = "BREAKOUT"
    try:
        if len(df) < 30:
            return _no_signal(engine, "insufficient data")

        period = 20
        sma   = df["close"].rolling(period).mean()
        std   = df["close"].rolling(period).std()
        upper = sma + 2 * std
        lower = sma - 2 * std
        bb_width = (upper - lower) / sma
        atr = _compute_atr(df, 14)
        rsi = _compute_rsi(df, 14)
        vol_ma = df["volume"].rolling(period).mean()

        c         = float(df["close"].iloc[-1])
        upper_val = float(upper.iloc[-1])
        sma_val   = float(sma.iloc[-1])
        atr_val   = float(atr.iloc[-1])
        rsi_val   = float(rsi.iloc[-1])
        vol_now   = float(df["volume"].iloc[-1])
        vol_avg   = float(vol_ma.iloc[-1])

        # Squeeze: current BB width in bottom 20% of rolling 20-bar window
        width_now  = float(bb_width.iloc[-1])
        width_min  = float(bb_width.iloc[-20:].min())
        width_max  = float(bb_width.iloc[-20:].max())
        width_pctile = (width_now - width_min) / (width_max - width_min + 1e-9)
        if width_pctile > 0.20:
            return _no_signal(engine, f"BB not squeezed (pctile={width_pctile:.2f})")

        if c <= upper_val:
            return _no_signal(engine, f"close={c:.2f} not above upper BB={upper_val:.2f}")

        if vol_now < 1.5 * vol_avg:
            return _no_signal(engine, f"volume={vol_now:.0f} < 1.5×avg={vol_avg:.0f}")

        if not (50 <= rsi_val <= 75):
            return _no_signal(engine, f"RSI={rsi_val:.1f} outside 50-75")

        entry     = c * (1 + config.SLIPPAGE)
        stop      = sma_val
        stop_dist = entry - stop
        if stop_dist <= 0:
            return _no_signal(engine, "stop_distance <= 0")
        tp = entry + 2.0 * stop_dist

        return EngineSignal(
            direction="LONG", engine=engine,
            entry_price=round(entry, 4), stop_price=round(stop, 4),
            tp_price=round(tp, 4), stop_distance=round(stop_dist, 4),
            atr_value=round(atr_val, 4), rank_boost=60.0,
            reason=f"BB squeeze breakout | RSI={rsi_val:.0f} | vol={vol_now/vol_avg:.1f}x | width_pctile={width_pctile:.2f}",
        )

    except Exception as exc:
        return _no_signal(engine, f"error: {exc}")


# ── Engine C: NY open momentum ─────────────────────────────────────────────────

def get_ny_momentum_signal(
    df: pd.DataFrame,
    now_utc: Optional[datetime] = None,
) -> EngineSignal:
    """
    New York open momentum burst (13:00-16:00 UTC only).

    Conditions (ALL required):
    1. Current UTC hour in [13, 14, 15] (NY open session)
    2. Last 2 bars are consecutive up-bars (close > open)
    3. Volume on both bars >= 1.2× 20-bar average
    4. RSI 50-70

    Stop: low of the 2-bar move
    TP:   entry + 2×stop_distance
    """
    engine = "NY_MOMENTUM"
    try:
        if now_utc is None:
            now_utc = datetime.now(timezone.utc)

        if now_utc.hour not in (13, 14, 15):
            return _no_signal(engine, f"outside NY open window (hour={now_utc.hour})")

        if len(df) < 25:
            return _no_signal(engine, "insufficient data")

        atr    = _compute_atr(df, 14)
        rsi    = _compute_rsi(df, 14)
        vol_ma = df["volume"].rolling(20).mean()

        atr_val = float(atr.iloc[-1])
        rsi_val = float(rsi.iloc[-1])
        vol_avg = float(vol_ma.iloc[-1])

        # Last 2 bars must be up-bars with sufficient volume
        for i in (1, 2):
            row = df.iloc[-i]
            if float(row["close"]) <= float(row["open"]):
                return _no_signal(engine, f"bar-{i} is not an up-bar")
            if float(df["volume"].iloc[-i]) < 1.2 * vol_avg:
                return _no_signal(engine,
                    f"bar-{i} volume={df['volume'].iloc[-i]:.0f} < 1.2×avg={vol_avg:.0f}")

        if not (50 <= rsi_val <= 70):
            return _no_signal(engine, f"RSI={rsi_val:.1f} outside 50-70")

        c     = float(df["close"].iloc[-1])
        entry = c * (1 + config.SLIPPAGE)
        # Stop below the low of the 2-bar move
        stop  = float(min(df["low"].iloc[-1], df["low"].iloc[-2])) - 0.25 * atr_val
        stop_dist = entry - stop
        if stop_dist <= 0:
            return _no_signal(engine, "stop_distance <= 0")
        tp = entry + 2.0 * stop_dist

        return EngineSignal(
            direction="LONG", engine=engine,
            entry_price=round(entry, 4), stop_price=round(stop, 4),
            tp_price=round(tp, 4), stop_distance=round(stop_dist, 4),
            atr_value=round(atr_val, 4), rank_boost=70.0,
            reason=f"NY open momentum | RSI={rsi_val:.0f} | 2 consecutive up-bars",
        )

    except Exception as exc:
        return _no_signal(engine, f"error: {exc}")


# ── Engine D: Mean reversion micro ────────────────────────────────────────────

def get_mean_reversion_signal(df: pd.DataFrame) -> EngineSignal:
    """
    Mean reversion micro setup (complement to RMR — 1H timeframe).

    Conditions (ALL required):
    1. ADX < 22 (ranging market)
    2. RSI < 38 OR close below VWAP (price extended below fair value)
    3. Hammer pattern on last closed bar
    4. Volume >= 0.7× 20-bar average (some participation)

    Stop: last bar low - 0.25×ATR
    TP:   entry + 1.5×stop_distance
    """
    engine = "MICRO_MR"
    try:
        if len(df) < 30:
            return _no_signal(engine, "insufficient data")

        try:
            import pandas_ta as ta  # noqa: F401
            adx_df  = df.ta.adx(length=config.ADX_PERIOD)
            adx_val = float(adx_df[f"ADX_{config.ADX_PERIOD}"].iloc[-1])
        except Exception:
            adx_val = 15.0  # assume ranging if ADX fails

        atr    = _compute_atr(df, 14)
        rsi    = _compute_rsi(df, 14)
        vol_ma = df["volume"].rolling(20).mean()

        atr_val = float(atr.iloc[-1])
        rsi_val = float(rsi.iloc[-1])
        vol_now = float(df["volume"].iloc[-1])
        vol_avg = float(vol_ma.iloc[-1])
        c       = float(df["close"].iloc[-1])

        if adx_val >= 22:
            return _no_signal(engine, f"ADX={adx_val:.1f} >= 22 (trending)")

        # VWAP: cumulative within-day approximation
        try:
            from strategies.vwap import _compute_vwap
            vwap_series = _compute_vwap(df)
            vwap_val    = float(vwap_series.iloc[-1])
            below_vwap  = c < vwap_val
        except Exception:
            below_vwap = False
            vwap_val   = c

        oversold = rsi_val < 38
        if not (oversold or below_vwap):
            return _no_signal(engine,
                f"RSI={rsi_val:.1f} not < 38 and close not below VWAP={vwap_val:.2f}")

        if not _hammer_pattern(df):
            return _no_signal(engine, "no hammer pattern on last bar")

        if vol_now < 0.7 * vol_avg:
            return _no_signal(engine, f"volume={vol_now:.0f} < 0.7×avg={vol_avg:.0f}")

        entry     = c * (1 + config.SLIPPAGE)
        stop      = float(df["low"].iloc[-1]) - 0.25 * atr_val
        stop_dist = entry - stop
        if stop_dist <= 0:
            return _no_signal(engine, "stop_distance <= 0")
        tp = entry + 1.5 * stop_dist

        cond = f"RSI={rsi_val:.0f}" if oversold else f"below VWAP={vwap_val:.2f}"
        return EngineSignal(
            direction="LONG", engine=engine,
            entry_price=round(entry, 4), stop_price=round(stop, 4),
            tp_price=round(tp, 4), stop_distance=round(stop_dist, 4),
            atr_value=round(atr_val, 4), rank_boost=40.0,
            reason=f"1H mean reversion | ADX={adx_val:.1f} | {cond} | hammer",
        )

    except Exception as exc:
        return _no_signal(engine, f"error: {exc}")


# ── Engine E: Intraday scalp momentum ─────────────────────────────────────────

def get_intraday_scalp_signal(
    df: pd.DataFrame,
    now_utc: Optional[datetime] = None,
) -> EngineSignal:
    """
    Intraday momentum scalp — active 07:00-20:00 UTC only.

    Targets short-duration momentum bursts with tight TP (1.2R) and fast stop.
    Designed to capture intraday continuation moves in high-activity sessions.

    Conditions (ALL required):
    1. UTC hour 07-19 (active intraday window)
    2. EMA9 > EMA21 (uptrend confirmed)
    3. RSI 45-62 (momentum zone, not extended or overbought)
    4. Last 3 consecutive closes each higher than the previous (acceleration)
    5. Volume >= 1.3x 20-bar average (participation present)
    6. ADX < 35 (not blow-off, momentum still healthy)
    7. ATR not in top 10% of 50-bar range (avoid chaotic bars)

    Stop: EMA9 - 0.3xATR  (tight scalp stop)
    TP:   entry + 1.2xstop_distance (quick profit target)
    """
    engine = "INTRADAY_SCALP"
    try:
        if now_utc is None:
            now_utc = datetime.now(timezone.utc)

        if now_utc.hour not in range(7, 20):
            return _no_signal(engine, f"outside scalp window (hour={now_utc.hour} UTC)")

        if len(df) < 55:
            return _no_signal(engine, "insufficient data (<55 bars)")

        ema9    = _compute_ema(df, 9)
        ema21   = _compute_ema(df, 21)
        atr     = _compute_atr(df, 14)
        rsi     = _compute_rsi(df, 14)
        vol_ma  = df["volume"].rolling(20).mean()

        e9      = float(ema9.iloc[-1])
        e21     = float(ema21.iloc[-1])
        atr_val = float(atr.iloc[-1])
        rsi_val = float(rsi.iloc[-1])
        vol_now = float(df["volume"].iloc[-1])
        vol_avg = float(vol_ma.iloc[-1])
        c       = float(df["close"].iloc[-1])

        if e9 <= e21:
            return _no_signal(engine, f"EMA9={e9:.2f} <= EMA21={e21:.2f}")

        if not (45 <= rsi_val <= 62):
            return _no_signal(engine, f"RSI={rsi_val:.1f} outside 45-62")

        # Three accelerating closes (index -1 is most recent)
        closes = [float(df["close"].iloc[-i]) for i in range(1, 5)]
        if not (closes[0] > closes[1] > closes[2] > closes[3]):
            return _no_signal(engine, "no 3-bar close acceleration")

        if vol_now < 1.3 * vol_avg:
            return _no_signal(engine, f"vol={vol_now:.0f} < 1.3x avg={vol_avg:.0f}")

        try:
            import pandas_ta as _ta  # noqa: F401
            adx_df  = df.ta.adx(length=config.ADX_PERIOD)
            adx_val = float(adx_df[f"ADX_{config.ADX_PERIOD}"].iloc[-1])
        except Exception:
            adx_val = 20.0
        if adx_val >= 35:
            return _no_signal(engine, f"ADX={adx_val:.1f} >= 35 (blow-off risk)")

        atr_p90 = float(atr.iloc[-50:].quantile(0.90))
        if atr_val >= atr_p90:
            return _no_signal(engine, f"ATR in top 10% of 50-bar range — chaotic")

        entry     = c * (1 + config.SLIPPAGE)
        stop      = e9 - 0.3 * atr_val
        stop_dist = entry - stop
        if stop_dist <= 0:
            return _no_signal(engine, "stop_distance <= 0")
        tp = entry + 1.2 * stop_dist

        return EngineSignal(
            direction="LONG", engine=engine,
            entry_price=round(entry, 4), stop_price=round(stop, 4),
            tp_price=round(tp, 4), stop_distance=round(stop_dist, 4),
            atr_value=round(atr_val, 4), rank_boost=45.0,
            reason=(
                f"Intraday scalp | RSI={rsi_val:.0f} | 3-bar accel"
                f" | ADX={adx_val:.1f} | vol={vol_now/vol_avg:.1f}x"
            ),
        )

    except Exception as exc:
        return _no_signal(engine, f"error: {exc}")
