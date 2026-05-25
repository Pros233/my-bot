"""
exchange_filters.py — Binance exchange filter cache and order validation.

Fetches and caches LOT_SIZE and MIN_NOTIONAL/NOTIONAL filters from
exchangeInfo. Validates and adjusts order quantity before any API call.

Public API
----------
    get_filters(client, symbol)              -> SymbolFilters
    validate_order(client, symbol, qty, price) -> ValidationResult
    send_skip_alert(symbol, reason)          -> None  (rate-limited, 1/hr/symbol)
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import logger

# ── Module-level cache ────────────────────────────────────────────────────────
_filters_cache: dict = {}   # symbol -> SymbolFilters
_alert_last_ts: dict = {}   # symbol -> float epoch (rate-limit Telegram alerts)
_ALERT_COOLDOWN_S = 3600    # one alert per symbol per hour max


@dataclass
class SymbolFilters:
    symbol:       str
    step_size:    float
    min_qty:      float
    min_notional: float


@dataclass
class ValidationResult:
    valid:        bool
    adjusted_qty: float
    reason:       str
    min_notional: float
    notional:     float
    step_size:    float


def get_filters(client, symbol: str) -> SymbolFilters:
    """Return cached SymbolFilters for symbol, fetching from Binance if needed."""
    if symbol in _filters_cache:
        return _filters_cache[symbol]

    step_size    = 0.00001
    min_qty      = 0.00001
    min_notional = 5.0

    try:
        info = None
        for attempt in range(3):
            try:
                info = client.get_symbol_info(symbol)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)

        if info:
            for flt in info.get("filters", []):
                ft = flt.get("filterType", "")
                if ft == "LOT_SIZE":
                    step_size = float(flt["stepSize"])
                    min_qty   = float(flt["minQty"])
                elif ft in ("MIN_NOTIONAL", "NOTIONAL"):
                    min_notional = float(flt.get("minNotional", min_notional))
    except Exception as exc:
        logger.log_warning(
            f"exchange_filters: could not fetch filters for {symbol}: {exc} — using safe defaults"
        )

    filters = SymbolFilters(
        symbol=symbol,
        step_size=step_size,
        min_qty=min_qty,
        min_notional=min_notional,
    )
    _filters_cache[symbol] = filters
    return filters


def validate_order(client, symbol: str, qty: float, price: float) -> ValidationResult:
    """
    Validate and round qty for a Binance order.

    Returns ValidationResult:
      valid         — True if order can proceed
      adjusted_qty  — qty rounded DOWN to stepSize
      reason        — 'ok' or human-readable failure description
      min_notional  — symbol minimum notional value
      notional      — adjusted_qty * price
      step_size     — LOT_SIZE stepSize used for rounding
    """
    f = get_filters(client, symbol)

    # Round DOWN to step size
    adj = qty
    if f.step_size > 0:
        adj = math.floor(qty / f.step_size) * f.step_size
        if f.step_size >= 1.0:
            adj = float(int(adj))
        else:
            precision = max(0, round(-math.log10(f.step_size)))
            adj = round(adj, precision)

    notional = adj * price

    if adj < f.min_qty:
        return ValidationResult(
            valid=False,
            adjusted_qty=adj,
            reason=f"lot_size: qty {adj} < min_qty {f.min_qty}",
            min_notional=f.min_notional,
            notional=notional,
            step_size=f.step_size,
        )

    if notional < f.min_notional:
        return ValidationResult(
            valid=False,
            adjusted_qty=adj,
            reason=(
                f"min_notional: ${notional:.2f} < ${f.min_notional:.2f} "
                f"(qty={adj} price={price:.4f})"
            ),
            min_notional=f.min_notional,
            notional=notional,
            step_size=f.step_size,
        )

    return ValidationResult(
        valid=True,
        adjusted_qty=adj,
        reason="ok",
        min_notional=f.min_notional,
        notional=notional,
        step_size=f.step_size,
    )


def send_skip_alert(symbol: str, reason: str) -> None:
    """Send Telegram alert for a skipped order. Rate-limited to once per hour per symbol."""
    now = time.time()
    if now - _alert_last_ts.get(symbol, 0) < _ALERT_COOLDOWN_S:
        return
    _alert_last_ts[symbol] = now
    try:
        import alerts
        alerts.alert_order_failed(symbol, "ORDER SKIP", reason)
    except Exception:
        pass
