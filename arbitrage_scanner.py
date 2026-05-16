"""
arbitrage_scanner.py — Watch-only arbitrage opportunity scanner.

Scans Binance spot for:
  A. Triangular arbitrage (USDT -> coin A -> coin B -> USDT)
  B. Cross-symbol spot imbalance (implied vs actual cross price)

SAFETY LOCK: _ARB_EXECUTION_HARD_LOCK = True. No orders are ever placed.
             If ARB_AUTO_TRADE=true appears in .env it is ignored and logged.
             All exceptions are caught — scanner failure never crashes the bot.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone

import alerts
import config
import logger

# ── Safety lock ───────────────────────────────────────────────────────────────
_ARB_EXECUTION_HARD_LOCK: bool = True   # Never set to False. Ever.

_DB_PATH: str = "arbitrage_watchlist.db"
_START_USDT: float = 1_000.0            # Simulation notional for profit calc

# ── Triangular routes ─────────────────────────────────────────────────────────
# Each leg: (symbol, action)
#   "buy"  → buy base with quote  → consume ask price
#   "sell" → sell base for quote  → consume bid price
_TRIANGULAR_ROUTES: list[tuple[str, list[tuple[str, str]]]] = [
    ("USDT->BTC->ETH->USDT",  [("BTCUSDT","buy"), ("ETHBTC","buy"),  ("ETHUSDT","sell")]),
    ("USDT->ETH->BTC->USDT",  [("ETHUSDT","buy"), ("ETHBTC","sell"), ("BTCUSDT","sell")]),
    ("USDT->BTC->BNB->USDT",  [("BTCUSDT","buy"), ("BNBBTC","buy"),  ("BNBUSDT","sell")]),
    ("USDT->BNB->BTC->USDT",  [("BNBUSDT","buy"), ("BNBBTC","sell"), ("BTCUSDT","sell")]),
    ("USDT->ETH->BNB->USDT",  [("ETHUSDT","buy"), ("BNBETH","buy"),  ("BNBUSDT","sell")]),
    ("USDT->BNB->ETH->USDT",  [("BNBUSDT","buy"), ("BNBETH","sell"), ("ETHUSDT","sell")]),
    ("USDT->BTC->LTC->USDT",  [("BTCUSDT","buy"), ("LTCBTC","buy"),  ("LTCUSDT","sell")]),
    ("USDT->LTC->BTC->USDT",  [("LTCUSDT","buy"), ("LTCBTC","sell"), ("BTCUSDT","sell")]),
    ("USDT->BTC->XRP->USDT",  [("BTCUSDT","buy"), ("XRPBTC","buy"),  ("XRPUSDT","sell")]),
    ("USDT->XRP->BTC->USDT",  [("XRPUSDT","buy"), ("XRPBTC","sell"), ("BTCUSDT","sell")]),
    ("USDT->BTC->ADA->USDT",  [("BTCUSDT","buy"), ("ADABTC","buy"),  ("ADAUSDT","sell")]),
    ("USDT->ADA->BTC->USDT",  [("ADAUSDT","buy"), ("ADABTC","sell"), ("BTCUSDT","sell")]),
    ("USDT->BTC->DOT->USDT",  [("BTCUSDT","buy"), ("DOTBTC","buy"),  ("DOTUSDT","sell")]),
    ("USDT->DOT->BTC->USDT",  [("DOTUSDT","buy"), ("DOTBTC","sell"), ("BTCUSDT","sell")]),
    ("USDT->BTC->LINK->USDT", [("BTCUSDT","buy"), ("LINKBTC","buy"), ("LINKUSDT","sell")]),
    ("USDT->LINK->BTC->USDT", [("LINKUSDT","buy"),("LINKBTC","sell"),("BTCUSDT","sell")]),
]

# ── Cross-symbol imbalance pairs ──────────────────────────────────────────────
# (cross_symbol, base_usdt_sym, quote_usdt_sym)
# Checks whether the implied cross price (via USDT) differs meaningfully from
# the actual cross-pair price.
_CROSS_PAIRS: list[tuple[str, str, str]] = [
    ("ETHBTC",  "ETHUSDT",  "BTCUSDT"),
    ("BNBBTC",  "BNBUSDT",  "BTCUSDT"),
    ("BNBETH",  "BNBUSDT",  "ETHUSDT"),
    ("LTCBTC",  "LTCUSDT",  "BTCUSDT"),
    ("XRPBTC",  "XRPUSDT",  "BTCUSDT"),
    ("ADABTC",  "ADAUSDT",  "BTCUSDT"),
]


# ── Signal dataclass ──────────────────────────────────────────────────────────

@dataclass
class ArbSignal:
    arb_type: str           # "TRIANGULAR" | "CROSS_SYMBOL"
    route: str
    symbols: str            # comma-separated
    start_amount_usdt: float
    expected_end_usdt: float
    gross_profit_pct: float
    net_profit_pct: float
    estimated_fees: float
    estimated_slippage: float
    liquidity_score: str    # "OK" | "LOW" | "POOR"
    spread_pct: float


# ── DB ────────────────────────────────────────────────────────────────────────

def _ensure_db() -> None:
    with sqlite3.connect(_DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS arbitrage_signals (
                id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                detected_at_utc    TEXT    NOT NULL,
                arb_type           TEXT    NOT NULL,
                route              TEXT    NOT NULL,
                symbols            TEXT    NOT NULL,
                start_amount_usdt  REAL    NOT NULL,
                expected_end_usdt  REAL    NOT NULL,
                gross_profit_pct   REAL    NOT NULL,
                net_profit_pct     REAL    NOT NULL,
                estimated_fees     REAL    NOT NULL,
                estimated_slippage REAL    NOT NULL,
                liquidity_score    TEXT    NOT NULL,
                spread_pct         REAL    NOT NULL,
                alert_sent         INTEGER NOT NULL DEFAULT 0,
                created_at         TEXT    NOT NULL
            )
        """)
        conn.commit()


def _save_signal(sig: ArbSignal, alert_sent: bool) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(_DB_PATH) as conn:
        conn.execute("""
            INSERT INTO arbitrage_signals
              (detected_at_utc, arb_type, route, symbols,
               start_amount_usdt, expected_end_usdt,
               gross_profit_pct, net_profit_pct,
               estimated_fees, estimated_slippage,
               liquidity_score, spread_pct, alert_sent, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            now, sig.arb_type, sig.route, sig.symbols,
            sig.start_amount_usdt, sig.expected_end_usdt,
            sig.gross_profit_pct, sig.net_profit_pct,
            sig.estimated_fees, sig.estimated_slippage,
            sig.liquidity_score, sig.spread_pct,
            1 if alert_sent else 0, now,
        ))
        conn.commit()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _liquidity_score(spread_pct: float) -> str:
    if spread_pct < 0.05:
        return "OK"
    if spread_pct < 0.15:
        return "LOW"
    return "POOR"


def _fetch_books(client, needed: set) -> dict:
    """
    Fetch best bid/ask for all needed symbols in a single API call
    via GET /api/v3/ticker/bookTicker (weight: 2).
    Returns {symbol: {bid, ask, spread_pct}}.
    """
    try:
        tickers = client.get_orderbook_tickers()
    except Exception as exc:
        logger.log_warning(f"ARB | get_orderbook_tickers failed: {exc}")
        return {}

    books: dict = {}
    for t in tickers:
        sym = t.get("symbol", "")
        if sym not in needed:
            continue
        try:
            bid = float(t["bidPrice"])
            ask = float(t["askPrice"])
        except (KeyError, ValueError):
            continue
        if bid <= 0 or ask <= 0 or ask < bid:
            continue
        mid = (bid + ask) / 2
        spread_pct = (ask - bid) / mid * 100 if mid > 0 else 999.0
        books[sym] = {"bid": bid, "ask": ask, "spread_pct": spread_pct}
    return books


# ── Scanners ──────────────────────────────────────────────────────────────────

def _scan_triangular(books: dict) -> list[ArbSignal]:
    """
    Evaluate all predefined triangular routes.

    Leg mechanics:
      "buy"  action on symbol XY (X=base, Y=quote):
             spend Y, receive X → divide by ask, apply fee
      "sell" action on symbol XY:
             spend X, receive Y → multiply by bid, apply fee

    Gross profit = profit after fees only.
    Net profit   = gross after additional slippage buffer.
    """
    fee_rate = config.ARB_FEE_PCT / 100
    slip_rate = config.ARB_SLIPPAGE_BUFFER_PCT / 100
    max_spread = config.ARB_MAX_SPREAD_PCT

    signals: list[ArbSignal] = []

    for route_label, legs in _TRIANGULAR_ROUTES:
        # All symbols must be in the fetched book
        if not all(sym in books for sym, _ in legs):
            continue

        # Reject routes where any leg's spread exceeds the max
        if any(books[sym]["spread_pct"] > max_spread for sym, _ in legs):
            continue

        amount = _START_USDT
        ok = True
        spreads = []

        for sym, action in legs:
            b = books[sym]
            spreads.append(b["spread_pct"])
            if action == "buy":
                if b["ask"] <= 0:
                    ok = False
                    break
                amount = (amount / b["ask"]) * (1.0 - fee_rate)
            else:  # sell
                if b["bid"] <= 0:
                    ok = False
                    break
                amount = amount * b["bid"] * (1.0 - fee_rate)

        if not ok:
            continue

        gross_end = amount
        gross_profit_pct = (gross_end - _START_USDT) / _START_USDT * 100

        # Only save signals with any gross profit (filters obvious losers)
        if gross_profit_pct <= 0:
            continue

        net_end = gross_end * (1.0 - slip_rate)
        net_profit_pct = (net_end - _START_USDT) / _START_USDT * 100
        estimated_fees = _START_USDT * len(legs) * fee_rate
        estimated_slippage = gross_end * slip_rate
        avg_spread = sum(spreads) / len(spreads)

        signals.append(ArbSignal(
            arb_type="TRIANGULAR",
            route=route_label,
            symbols=", ".join(sym for sym, _ in legs),
            start_amount_usdt=_START_USDT,
            expected_end_usdt=round(net_end, 4),
            gross_profit_pct=round(gross_profit_pct, 4),
            net_profit_pct=round(net_profit_pct, 4),
            estimated_fees=round(estimated_fees, 4),
            estimated_slippage=round(estimated_slippage, 4),
            liquidity_score=_liquidity_score(avg_spread),
            spread_pct=round(avg_spread, 4),
        ))

    return signals


def _scan_cross_symbol(books: dict) -> list[ArbSignal]:
    """
    Detect implied vs actual cross-pair price discrepancies.

    For cross pair BASE/QUOTE (e.g. ETHBTC):
      Implied ask = ask_BASE_USDT / bid_QUOTE_USDT
                    (cost to buy BASE using QUOTE, routed through USDT)
      Actual  ask = ask_CROSS

    If actual_ask < implied_ask the cross pair is cheaper → potential arb.
    Gross profit % = (implied_ask - actual_ask) / implied_ask * 100
    """
    fee_rate = config.ARB_FEE_PCT / 100
    slip_rate = config.ARB_SLIPPAGE_BUFFER_PCT / 100
    max_spread = config.ARB_MAX_SPREAD_PCT

    signals: list[ArbSignal] = []

    for cross_sym, base_usdt, quote_usdt in _CROSS_PAIRS:
        if not all(s in books for s in (cross_sym, base_usdt, quote_usdt)):
            continue

        b_cross = books[cross_sym]
        b_base  = books[base_usdt]
        b_quote = books[quote_usdt]

        if b_cross["spread_pct"] > max_spread:
            continue

        # Implied cross ask and bid via USDT routing
        if b_quote["bid"] <= 0 or b_quote["ask"] <= 0:
            continue
        implied_ask = b_base["ask"] / b_quote["bid"]
        implied_bid = b_base["bid"] / b_quote["ask"]
        implied_mid = (implied_ask + implied_bid) / 2

        actual_ask = b_cross["ask"]
        actual_bid = b_cross["bid"]
        actual_mid = (actual_ask + actual_bid) / 2

        if implied_mid <= 0 or actual_mid <= 0:
            continue

        # Spread between implied and actual prices
        spread_pct = abs(implied_mid - actual_mid) / implied_mid * 100

        # Gross profit: whichever direction has the better edge
        buy_edge  = (implied_ask - actual_ask) / implied_ask * 100  # buy cross directly
        sell_edge = (actual_bid - implied_bid) / actual_bid  * 100  # sell cross directly
        gross_profit_pct = max(buy_edge, sell_edge)

        if gross_profit_pct <= 0:
            continue

        total_fee_pct = 2 * fee_rate * 100   # 2 legs
        net_profit_pct = gross_profit_pct - total_fee_pct - slip_rate * 100
        estimated_fees = _START_USDT * 2 * fee_rate
        estimated_slippage = _START_USDT * slip_rate
        expected_end = _START_USDT * (1.0 + net_profit_pct / 100)

        signals.append(ArbSignal(
            arb_type="CROSS_SYMBOL",
            route=f"USDT->{cross_sym}->USDT (implied vs actual)",
            symbols=f"{cross_sym}, {base_usdt}, {quote_usdt}",
            start_amount_usdt=_START_USDT,
            expected_end_usdt=round(expected_end, 4),
            gross_profit_pct=round(gross_profit_pct, 4),
            net_profit_pct=round(net_profit_pct, 4),
            estimated_fees=round(estimated_fees, 4),
            estimated_slippage=round(estimated_slippage, 4),
            liquidity_score=_liquidity_score(b_cross["spread_pct"]),
            spread_pct=round(spread_pct, 4),
        ))

    return signals


# ── Main entry point ──────────────────────────────────────────────────────────

def run_scan(client) -> None:
    """
    Called from main.py once per candle cycle (hourly).
    Scans for arbitrage opportunities, saves to DB, sends Telegram alerts.
    Never raises — all exceptions are caught internally.
    """
    # Safety lock check — warn if someone tries to enable auto-trade via env
    if _ARB_EXECUTION_HARD_LOCK and config.ARB_AUTO_TRADE:
        logger.log_warning("ARB AUTO TRADE DISABLED BY SAFETY LOCK")

    if not config.ENABLE_ARBITRAGE_SCANNER:
        return

    try:
        _ensure_db()
    except Exception as exc:
        logger.log_warning(f"ARB | DB init failed (non-critical): {exc}")
        return

    # Collect all symbols needed across all routes
    needed: set = set()
    for _, legs in _TRIANGULAR_ROUTES:
        for sym, _ in legs:
            needed.add(sym)
    for cross_sym, base_usdt, quote_usdt in _CROSS_PAIRS:
        needed.update([cross_sym, base_usdt, quote_usdt])

    books = _fetch_books(client, needed)
    if not books:
        logger.log_warning("ARB | No order books available — skipping scan")
        return

    try:
        tri_signals   = _scan_triangular(books)
        cross_signals = _scan_cross_symbol(books)
    except Exception as exc:
        logger.log_warning(f"ARB | Scan computation failed (non-critical): {exc}")
        return

    all_signals = tri_signals + cross_signals
    min_net = config.ARB_MIN_NET_PROFIT_PCT

    qualifying = [s for s in all_signals if s.net_profit_pct >= min_net]
    qualifying.sort(key=lambda s: s.net_profit_pct, reverse=True)
    top = qualifying[: config.ARB_TOP_N]

    logger.log_info(
        f"ARB | {len(tri_signals)} triangular + {len(cross_signals)} cross-symbol "
        f"signals with gross profit > 0 | "
        f"{len(qualifying)} above {min_net:.2f}% net threshold | "
        f"{len(top)} alert(s) queued"
    )

    # Save and alert top qualifying signals
    for sig in top:
        try:
            _save_signal(sig, alert_sent=True)
        except Exception as exc:
            logger.log_warning(f"ARB | DB save failed: {exc}")
        try:
            alerts.alert_arb_opportunity(sig)
        except Exception as exc:
            logger.log_warning(f"ARB | Telegram alert failed: {exc}")

    # Save below-threshold signals for research (no alert)
    below = [s for s in all_signals if s.net_profit_pct < min_net]
    for sig in below:
        try:
            _save_signal(sig, alert_sent=False)
        except Exception as exc:
            logger.log_warning(f"ARB | DB save (below threshold) failed: {exc}")
