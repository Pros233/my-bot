"""
anomaly_detector.py — Market and system anomaly detection.

Monitors per-cycle metrics and triggers alerts when thresholds are exceeded.

Anomaly types:
  ADX_SPIKE        — ADX jumped >2× its recent average (trending surge)
  VOL_SPIKE        — ATR jumped >3× recent average (volatility explosion)
  SPREAD_ANOMALY   — estimated spread proxy (wick ratio) >80% of range
  PRICE_MOVE       — bar close moved >3× ATR in one candle (flash move)
  EXEC_DELAY       — execution pipeline took >30s (network/exchange lag)
  CORR_BREAKDOWN   — BTC dropped >3% while alt is being entered (divergence)

Auto-responses:
  WARNING  → reduce_aggressiveness (risk_scale=0.75, tighten 1 grade level)
  CRITICAL → pause_new_entries (no new trades until anomaly clears)

Deduplication: same anomaly type not re-alerted within _DEDUP_WINDOW_S.

Public API
----------
    check_market_anomalies(df, symbol, adx, atr_pct) → list[AnomalyEvent]
    check_system_anomalies(exec_delay_s)              → list[AnomalyEvent]
    record_anomaly(event)                             → None
    get_active_anomalies()                            → list[AnomalyEvent]
    clear_expired_anomalies()                         → int  (count cleared)
    should_reduce_aggressiveness()                    → bool
    should_pause_entries()                            → bool
    get_anomaly_summary()                             → dict

Never raises.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import logger

# ── Config ────────────────────────────────────────────────────────────────────

_DEDUP_WINDOW_S  = 4 * 3600   # don't re-alert same type within 4 hours
_EXPIRY_S        = 8 * 3600   # anomalies auto-clear after 8 hours
_EXEC_DELAY_WARN = 30.0       # seconds
_ADX_SPIKE_MULT  = 2.0        # ADX > N× recent avg
_VOL_SPIKE_MULT  = 3.0        # ATR > N× recent avg
_PRICE_MOVE_MULT = 3.0        # bar close > N× ATR
_WICK_RATIO_CRIT = 0.80       # wick/range > this → SPREAD_ANOMALY
_BTC_DROP_CRIT   = -3.0       # BTC 1-bar % change for CORR_BREAKDOWN

SEVERITY_WARNING  = "WARNING"
SEVERITY_CRITICAL = "CRITICAL"

# ── Persistence ───────────────────────────────────────────────────────────────

_SAVE_PATHS = [
    Path("/opt/btcbot/anomaly_detector.json"),
    Path("anomaly_detector.json"),
]
_lock = threading.Lock()
# {anomaly_type+symbol: {severity, ts_first, ts_last, value, count}}
_active: dict = {}


def _save_path() -> Path:
    for p in _SAVE_PATHS:
        if p.parent.exists():
            return p
    return _SAVE_PATHS[-1]


def _load() -> None:
    p = _save_path()
    if not p.exists():
        return
    try:
        data = json.loads(p.read_text())
        with _lock:
            _active.update(data)
    except Exception:
        pass


def _save() -> None:
    p = _save_path()
    try:
        with _lock:
            snap = json.dumps(_active, indent=2)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(snap)
        tmp.replace(p)
    except Exception:
        pass


_load()


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class AnomalyEvent:
    anomaly_type: str
    symbol:       str
    severity:     str
    value:        float
    threshold:    float
    message:      str
    ts:           str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ── Detection helpers ─────────────────────────────────────────────────────────

def _check_adx_spike(df, adx: float, symbol: str) -> Optional[AnomalyEvent]:
    try:
        if len(df) < 22:
            return None
        adx_col = [c for c in df.columns if c.startswith("ADX_")]
        if adx_col:
            adx_series = df[adx_col[0]].dropna()
        else:
            return None
        if len(adx_series) < 20:
            return None
        avg_adx = float(adx_series.iloc[-21:-1].mean())
        if avg_adx <= 0:
            return None
        ratio = adx / avg_adx
        if ratio >= _ADX_SPIKE_MULT:
            sev = SEVERITY_CRITICAL if ratio >= 3.0 else SEVERITY_WARNING
            return AnomalyEvent(
                anomaly_type="ADX_SPIKE", symbol=symbol, severity=sev,
                value=round(adx, 1), threshold=round(avg_adx * _ADX_SPIKE_MULT, 1),
                message=f"ADX {adx:.1f} = {ratio:.1f}× recent avg {avg_adx:.1f}",
            )
    except Exception:
        pass
    return None


def _check_vol_spike(df, atr_pct: float, symbol: str) -> Optional[AnomalyEvent]:
    try:
        if len(df) < 22:
            return None
        atr_col = [c for c in df.columns if c.startswith("ATRr_") or c.startswith("ATR_") or c == "atr"]
        if atr_col:
            atr_series = df[atr_col[0]].dropna()
        else:
            # Estimate from bar range
            atr_series = (df["high"] - df["low"]).astype(float)
        avg_atr = float(atr_series.iloc[-21:-1].mean())
        cur_atr = float(atr_series.iloc[-1])
        if avg_atr <= 0:
            return None
        ratio = cur_atr / avg_atr
        if ratio >= _VOL_SPIKE_MULT:
            sev = SEVERITY_CRITICAL if ratio >= 5.0 else SEVERITY_WARNING
            return AnomalyEvent(
                anomaly_type="VOL_SPIKE", symbol=symbol, severity=sev,
                value=round(ratio, 2), threshold=_VOL_SPIKE_MULT,
                message=f"ATR {ratio:.1f}× recent avg — volatility explosion",
            )
    except Exception:
        pass
    return None


def _check_price_move(df, symbol: str) -> Optional[AnomalyEvent]:
    try:
        if len(df) < 22 or "close" not in df.columns:
            return None
        closes = df["close"].astype(float)
        bar_move = abs(float(closes.iloc[-1]) - float(closes.iloc[-2]))
        atr_series = (df["high"] - df["low"]).astype(float)
        avg_atr = float(atr_series.iloc[-21:-1].mean())
        if avg_atr <= 0:
            return None
        ratio = bar_move / avg_atr
        if ratio >= _PRICE_MOVE_MULT:
            sev = SEVERITY_CRITICAL if ratio >= 5.0 else SEVERITY_WARNING
            return AnomalyEvent(
                anomaly_type="PRICE_MOVE", symbol=symbol, severity=sev,
                value=round(ratio, 2), threshold=_PRICE_MOVE_MULT,
                message=f"1-bar move {ratio:.1f}× ATR — possible flash move",
            )
    except Exception:
        pass
    return None


def _check_spread_anomaly(df, symbol: str) -> Optional[AnomalyEvent]:
    try:
        if len(df) < 4 or not all(c in df.columns for c in ("open","high","low","close")):
            return None
        opens  = df["open"].astype(float).iloc[-4:]
        highs  = df["high"].astype(float).iloc[-4:]
        lows   = df["low"].astype(float).iloc[-4:]
        closes = df["close"].astype(float).iloc[-4:]
        ratios = []
        for i in range(len(opens)):
            rng = float(highs.iloc[i] - lows.iloc[i])
            if rng <= 0:
                continue
            body = abs(float(closes.iloc[i] - opens.iloc[i]))
            ratios.append((rng - body) / rng)
        if not ratios:
            return None
        avg_wick = sum(ratios) / len(ratios)
        if avg_wick >= _WICK_RATIO_CRIT:
            return AnomalyEvent(
                anomaly_type="SPREAD_ANOMALY", symbol=symbol, severity=SEVERITY_WARNING,
                value=round(avg_wick, 3), threshold=_WICK_RATIO_CRIT,
                message=f"avg wick ratio {avg_wick*100:.0f}% — spread explosion",
            )
    except Exception:
        pass
    return None


# ── Public API ────────────────────────────────────────────────────────────────

def check_market_anomalies(
    df,
    symbol:  str,
    adx:     float = 0.0,
    atr_pct: float = 0.0,
) -> list[AnomalyEvent]:
    """Run all market anomaly checks for one symbol. Returns list of new events."""
    events: list[AnomalyEvent] = []
    try:
        for check in [
            _check_adx_spike(df, adx, symbol),
            _check_vol_spike(df, atr_pct, symbol),
            _check_price_move(df, symbol),
            _check_spread_anomaly(df, symbol),
        ]:
            if check is not None:
                events.append(check)
    except Exception as exc:
        logger.log_warning(f"anomaly_detector.check_market_anomalies error: {exc}")
    return events


def check_system_anomalies(exec_delay_s: float) -> list[AnomalyEvent]:
    """Check system-level anomalies (execution delay)."""
    events: list[AnomalyEvent] = []
    try:
        if exec_delay_s >= _EXEC_DELAY_WARN:
            sev = SEVERITY_CRITICAL if exec_delay_s >= 60 else SEVERITY_WARNING
            events.append(AnomalyEvent(
                anomaly_type="EXEC_DELAY", symbol="SYSTEM", severity=sev,
                value=round(exec_delay_s, 1), threshold=_EXEC_DELAY_WARN,
                message=f"execution cycle took {exec_delay_s:.0f}s — network/exchange lag",
            ))
    except Exception:
        pass
    return events


def record_anomaly(event: AnomalyEvent) -> bool:
    """
    Record an anomaly. Returns True if this is a new/novel event (not dedup'd).
    Dedup: same anomaly_type+symbol is suppressed within _DEDUP_WINDOW_S.
    """
    try:
        key = f"{event.anomaly_type}:{event.symbol}"
        ts_now = time.monotonic()
        with _lock:
            existing = _active.get(key)
            if existing:
                last_ts = existing.get("ts_mono", 0)
                if ts_now - last_ts < _DEDUP_WINDOW_S:
                    # Update count but don't re-alert
                    _active[key]["count"] = existing.get("count", 1) + 1
                    _active[key]["ts_last"] = event.ts
                    return False

            _active[key] = {
                "anomaly_type": event.anomaly_type,
                "symbol":       event.symbol,
                "severity":     event.severity,
                "value":        event.value,
                "threshold":    event.threshold,
                "message":      event.message,
                "ts_first":     event.ts,
                "ts_last":      event.ts,
                "ts_mono":      ts_now,
                "count":        1,
            }
        _save()
        logger.log_info(
            f"ANOMALY | {event.severity} | {event.anomaly_type} | "
            f"{event.symbol} | {event.message}"
        )
        return True
    except Exception as exc:
        logger.log_warning(f"anomaly_detector.record_anomaly error: {exc}")
        return False


def clear_expired_anomalies() -> int:
    """Remove anomalies older than _EXPIRY_S. Returns count cleared."""
    cleared = 0
    try:
        ts_now = time.monotonic()
        with _lock:
            expired = [
                k for k, v in _active.items()
                if ts_now - v.get("ts_mono", 0) > _EXPIRY_S
            ]
            for k in expired:
                del _active[k]
                cleared += 1
        if cleared:
            _save()
    except Exception:
        pass
    return cleared


def get_active_anomalies() -> list[dict]:
    """Return list of currently active anomaly records."""
    try:
        clear_expired_anomalies()
        with _lock:
            return [dict(v) for v in _active.values()]
    except Exception:
        return []


def should_reduce_aggressiveness() -> bool:
    """True if any WARNING-level anomaly is active."""
    try:
        anomalies = get_active_anomalies()
        return any(a.get("severity") in (SEVERITY_WARNING, SEVERITY_CRITICAL) for a in anomalies)
    except Exception:
        return False


def should_pause_entries() -> bool:
    """True if any CRITICAL-level anomaly is active."""
    try:
        anomalies = get_active_anomalies()
        return any(a.get("severity") == SEVERITY_CRITICAL for a in anomalies)
    except Exception:
        return False


def get_anomaly_summary() -> dict:
    """Return dashboard/Telegram-friendly summary."""
    try:
        anomalies = get_active_anomalies()
        warnings  = [a for a in anomalies if a.get("severity") == SEVERITY_WARNING]
        criticals = [a for a in anomalies if a.get("severity") == SEVERITY_CRITICAL]
        return {
            "total":          len(anomalies),
            "warnings":       len(warnings),
            "criticals":      len(criticals),
            "reduce_aggressiveness": should_reduce_aggressiveness(),
            "pause_entries":  should_pause_entries(),
            "active":         anomalies,
        }
    except Exception as exc:
        logger.log_warning(f"anomaly_detector.get_anomaly_summary error: {exc}")
        return {"total": 0, "warnings": 0, "criticals": 0, "active": []}
