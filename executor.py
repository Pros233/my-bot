"""
executor.py — Live order execution engine with safeguards.

Responsibilities:
  - Place MARKET BUY + LIMIT TP + STOP_LOSS_LIMIT orders
  - Verify fills with up to 3 retries (exponential backoff)
  - Clock drift detection
  - Ghost/duplicate order prevention on reconnect
  - Emergency shutdown (cancel all orders, log, exit)
"""
from __future__ import annotations

import math
import signal
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import pandas_ta as ta  # noqa: F401 — registers .ta accessor
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceOrderException

import alerts
import config
import logger
from risk import TradeParams

# ── Position state ────────────────────────────────────────────────────────────

@dataclass
class OpenPosition:
    side: str                  # "BUY" (long only for this bot)
    entry_price: float
    fill_price: float          # actual fill from order response
    stop_price: float
    tp_price: float
    size: float
    market_order_id: int
    tp_order_id: Optional[int] = None
    stop_order_id: Optional[int] = None
    entry_time: float = field(default_factory=time.time)
    # ── Partial TP state (populated when ENABLE_PARTIAL_TP=True) ──────────────
    partial_tp_order_id: Optional[int] = None  # LIMIT sell for partial_tp_size
    partial_tp_size: float = 0.0               # qty closed at partial TP
    remaining_size: float = 0.0               # qty still open after partial TP
    partial_tp_hit: bool = False               # True after partial TP fills
    be_stop_order_id: Optional[int] = None     # BE stop for remaining position
    # ── MR_EXIT_14 state (ENABLE_MOMENTUM_EXIT) ───────────────────────────────
    prev_rsi7: Optional[float] = None          # RSI(7) from previous candle check
    # ── Trade journal context (set in execute_buy) ────────────────────────────
    strategy: str = "UNKNOWN"
    regime: str = ""
    adx: float = 0.0
    atr_pct: float = 0.0
    score_pct: float = 0.0
    entry_balance: float = 0.0
    fees_estimated: float = 0.0
    opened_at_utc: str = ""
    session: str = ""        # trading session at entry (set by main.py after execute_buy)
    trade_grade: str = ""    # A+/A/B/C (set by main.py after execute_buy)


# ── Retry helper ──────────────────────────────────────────────────────────────

def _with_retry(fn, max_retries: int = 3, base_delay: float = 2.0):
    """Call *fn* up to *max_retries* times with exponential backoff."""
    last_exc = None
    for attempt in range(max_retries):
        try:
            return fn()
        except (BinanceAPIException, BinanceOrderException) as exc:
            last_exc = exc
            delay = base_delay * (2 ** attempt)
            logger.log_warning(
                f"API error (attempt {attempt + 1}/{max_retries}): {exc}. "
                f"Retrying in {delay:.0f}s."
            )
            time.sleep(delay)
        except Exception as exc:
            last_exc = exc
            logger.log_error("Unexpected error in API call", exc)
            break
    raise RuntimeError(f"API call failed after {max_retries} retries") from last_exc


# ── Execution engine ──────────────────────────────────────────────────────────

class ExecutionEngine:
    def __init__(self, client: Client, symbol: str = config.SYMBOL) -> None:
        self.client = client
        self.symbol = symbol
        self.position: Optional[OpenPosition] = None
        self._shutdown_requested = False

        # Register clean-shutdown handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    # ── Public interface ──────────────────────────────────────────────────────

    def has_open_position(self) -> bool:
        return self.position is not None

    def check_clock_sync(self) -> None:
        """Pause and alert if system clock drifts >1 000 ms from Binance."""
        try:
            server_time = _with_retry(
                lambda: self.client.get_server_time()
            )["serverTime"]
        except Exception as exc:
            logger.log_warning(f"Could not fetch server time: {exc}")
            return

        local_time = int(time.time() * 1000)
        drift_ms = abs(local_time - server_time)

        if drift_ms > 1000:
            msg = (
                f"CLOCK DRIFT ALERT: {drift_ms} ms. "
                "Sync your system clock (e.g. ntpdate). Pausing 60 s."
            )
            logger.log_warning(msg)
            print(f"\n[!] {msg}\n")
            time.sleep(60)

    def sync_open_orders(self) -> None:
        """
        On (re)connect: reconcile local position state with Binance.
        Prevents ghost positions from a prior crashed session.
        """
        try:
            open_orders = _with_retry(
                lambda: self.client.get_open_orders(symbol=self.symbol)
            )
        except Exception as exc:
            logger.log_error("Could not fetch open orders on sync", exc)
            return

        if self.position is None and open_orders:
            logger.log_warning(
                f"Found {len(open_orders)} open order(s) on Binance but no "
                "local position. Cancelling all to start clean."
            )
            self._cancel_all_orders()

    def execute_buy(
        self,
        params: TradeParams,
        strategy: str = "CONSENSUS",
        regime: str = "",
        adx: float = 0.0,
        atr_pct: float = 0.0,
        score_pct: float = 0.0,
        balance: float = 0.0,
    ) -> bool:
        """
        Full BUY flow:
          1. Market buy
          2. Verify fill (3 retries)
          3. Record fill_price
          4. Place TP limit sell
          5. Place stop-loss limit sell
          6. Log everything

        Returns True if all orders placed successfully, False otherwise.
        """
        if self.position is not None:
            logger.log_warning("execute_buy called with an existing open position. Skipping.")
            return False

        qty = params.position_size

        # 1. Place market buy
        try:
            buy_order = _with_retry(lambda: self.client.order_market_buy(
                symbol=self.symbol,
                quantity=qty,
            ))
        except Exception as exc:
            logger.log_error("Market BUY order failed", exc)
            alerts.alert_order_failed(self.symbol, "MARKET BUY", str(exc))
            return False

        buy_id = buy_order["orderId"]

        # 2. Verify fill (up to 3× with 2 s backoff)
        fill_price = self._await_fill(buy_id, max_retries=3)
        if fill_price is None:
            logger.log_error(f"Could not confirm fill for order {buy_id}. Aborting.")
            alerts.alert_order_failed(self.symbol, "MARKET BUY fill unconfirmed", f"order_id={buy_id}")
            self._cancel_all_orders()
            return False

        if config.ENABLE_PARTIAL_TP:
            # Partial TP flow (MR_EXIT_0 live approximation):
            #   • Close PARTIAL_TP_CLOSE_PCT (40%) at TP1 = params.tp_price
            #   • Keep full-size SL active; once partial TP fills, move SL to BE
            partial_qty   = round(qty * config.PARTIAL_TP_CLOSE_PCT, 5)
            remaining_qty = round(qty - partial_qty, 5)

            # 3a. Partial LIMIT sell at TP
            partial_tp_id = self._place_limit_sell(partial_qty, params.tp_price)
            # 3b. Full-size SL (protects entire position until partial TP fires)
            stop_id = self._place_stop_limit_sell(qty, params.stop_price)

            self.position = OpenPosition(
                side="BUY",
                entry_price=params.entry_price,
                fill_price=fill_price,
                stop_price=params.stop_price,
                tp_price=params.tp_price,
                size=qty,
                market_order_id=buy_id,
                stop_order_id=stop_id,
                partial_tp_order_id=partial_tp_id,
                partial_tp_size=partial_qty,
                remaining_size=remaining_qty,
            )
            tp_id = partial_tp_id   # for logging below
        else:
            # 3. Place full TP limit sell
            tp_id = self._place_limit_sell(qty, params.tp_price)

            # 4. Place stop-loss limit sell
            stop_id = self._place_stop_limit_sell(qty, params.stop_price)

            # 5. Record position
            self.position = OpenPosition(
                side="BUY",
                entry_price=params.entry_price,
                fill_price=fill_price,
                stop_price=params.stop_price,
                tp_price=params.tp_price,
                size=qty,
                market_order_id=buy_id,
                tp_order_id=tp_id,
                stop_order_id=stop_id,
            )

        # Populate journal context on the position
        self.position.strategy = strategy
        self.position.regime = regime
        self.position.adx = adx
        self.position.atr_pct = atr_pct
        self.position.score_pct = score_pct
        self.position.entry_balance = balance
        self.position.fees_estimated = params.fee_estimate
        self.position.opened_at_utc = datetime.now(timezone.utc).isoformat()

        logger.log_trade_open(
            timestamp=datetime.now(timezone.utc),
            side="BUY",
            entry=fill_price,
            stop=params.stop_price,
            tp=params.tp_price,
            size=qty,
            fee_est=params.fee_estimate,
            order_ids={"buy": buy_id, "tp": tp_id, "stop": stop_id},
        )
        alerts.alert_trade_open(
            symbol=self.symbol,
            side="BUY",
            entry=fill_price,
            stop=params.stop_price,
            tp=params.tp_price,
            size=qty,
            strategy=strategy,
        )
        return True

    def check_position(self, df: Optional[pd.DataFrame] = None) -> Optional[float]:
        """
        Poll TP and stop orders; return exit_price if position fully closed,
        None if still open.

        Two-phase state machine when ENABLE_PARTIAL_TP=True:
          Phase 1 (partial_tp_hit=False):
            • Partial TP fills → cancel full SL, place BE stop for remaining qty.
            • SL fills before partial TP → cancel partial TP, close position.
          Phase 2 (partial_tp_hit=True):
            • BE stop fills → close remaining position.

        MR_EXIT_14 (ENABLE_MOMENTUM_EXIT=True):
          When profit ≥ MOMENTUM_EXIT_MIN_R and RSI(7) crosses back below 50,
          cancel all remaining orders and close at market.
        """
        if self.position is None:
            return None

        # ── MR_EXIT_14: RSI(7) cross-50 momentum exit ─────────────────────────
        if config.ENABLE_MOMENTUM_EXIT and df is not None and len(df) >= 10:
            pos = self.position
            rsi7_series = df.ta.rsi(length=7)
            if rsi7_series is not None and len(rsi7_series) >= 2:
                curr_rsi = float(rsi7_series.iloc[-1])
                prev_rsi = pos.prev_rsi7
                pos.prev_rsi7 = curr_rsi if math.isfinite(curr_rsi) else prev_rsi

                if math.isfinite(curr_rsi) and prev_rsi is not None:
                    curr_close = float(df["close"].iloc[-1])
                    stop_dist = abs(pos.fill_price - pos.stop_price)
                    profit_r = (curr_close - pos.fill_price) / stop_dist if stop_dist > 0 else 0.0
                    rsi_cross_below_50 = prev_rsi >= 50.0 and curr_rsi < 50.0

                    if profit_r >= config.MOMENTUM_EXIT_MIN_R and rsi_cross_below_50:
                        close_qty = pos.remaining_size if pos.remaining_size > 0 else pos.size
                        exit_price = self._close_at_market(close_qty)
                        if exit_price is not None:
                            logger.log_info(
                                f"MR_EXIT_14: RSI(7) crossed below 50 at {curr_rsi:.1f} "
                                f"(prev={prev_rsi:.1f}) with profit={profit_r:.2f}R — "
                                f"closing {close_qty} at {exit_price:.2f}"
                            )
                            alerts.alert_trade_close(
                                self.symbol, exit_price, pos.fill_price, close_qty, "MARKET CLOSE"
                            )
                            self._clear_position(exit_price, "MARKET CLOSE")
                            return exit_price

        if config.ENABLE_PARTIAL_TP:
            pos = self.position

            if not pos.partial_tp_hit:
                # ── Phase 1: waiting for partial TP or SL ─────────────────────
                partial_filled = self._order_is_filled(pos.partial_tp_order_id)
                stop_filled    = self._order_is_filled(pos.stop_order_id)

                if partial_filled:
                    # 1. Cancel old full-size stop
                    cancel_note = "ok"
                    try:
                        self._cancel_order_safe(pos.stop_order_id)
                    except Exception as exc:
                        cancel_note = f"failed ({exc})"
                    logger.log_info(
                        f"Partial TP filled at {partial_filled:.2f} "
                        f"({pos.partial_tp_size} BTC closed, "
                        f"{pos.remaining_size} BTC remaining) | "
                        f"old stop {pos.stop_order_id} cancel: {cancel_note}"
                    )
                    alerts.alert_partial_tp(
                        self.symbol, partial_filled, pos.partial_tp_size, pos.remaining_size
                    )

                    # 2. Fetch current price
                    try:
                        current_price = float(_with_retry(
                            lambda: self.client.get_symbol_ticker(symbol=self.symbol)
                        )["price"])
                    except Exception as exc:
                        logger.log_warning(f"Could not fetch current price for BE calc: {exc}")
                        current_price = partial_filled

                    # 3. Calculate and attempt BE stop placement
                    stop_dist = pos.fill_price - pos.stop_price
                    be_price  = round(pos.fill_price + config.BE_OFFSET_R * stop_dist, 2)
                    logger.log_info(
                        f"Current price: {current_price:.2f} | "
                        f"Attempting BE stop at {be_price:.2f} for {pos.remaining_size} BTC"
                    )
                    be_id = self._place_stop_limit_sell(pos.remaining_size, be_price)

                    if be_id is not None:
                        # 4a. BE stop placed successfully
                        pos.be_stop_order_id = be_id
                        pos.partial_tp_hit   = True
                        logger.log_info(f"BE stop placed — order ID {be_id}")
                        return None   # position still partially open

                    # 4b. BE stop failed — close remaining at market
                    logger.log_warning(
                        "BE stop placement failed — closing remaining position at market "
                        "to avoid unprotected exposure"
                    )
                    exit_price = self._close_at_market(pos.remaining_size)
                    if exit_price is not None:
                        logger.log_info(
                            f"Emergency market close executed at {exit_price:.2f} "
                            f"for {pos.remaining_size} BTC"
                        )
                        self._clear_position(exit_price, "EMERGENCY_MARKET_CLOSE")
                        return exit_price
                    # Market close also failed — avoid infinite loop, flag for manual review
                    logger.log_error(
                        "Emergency market close failed — position may be orphaned. "
                        "Manual intervention required."
                    )
                    pos.partial_tp_hit   = True
                    pos.be_stop_order_id = None
                    return None

                if stop_filled:
                    self._cancel_order_safe(pos.partial_tp_order_id)
                    alerts.alert_trade_close(
                        self.symbol, stop_filled, pos.fill_price, pos.size, "STOP HIT"
                    )
                    self._clear_position(stop_filled, "STOP HIT")
                    return stop_filled

            else:
                # ── Phase 2: waiting for BE stop (remaining position) ──────────
                be_filled = self._order_is_filled(pos.be_stop_order_id)
                if be_filled:
                    alerts.alert_trade_close(
                        self.symbol, be_filled, pos.fill_price, pos.remaining_size, "BE STOP"
                    )
                    self._clear_position(be_filled, "BE STOP")
                    return be_filled

            return None

        # ── Standard (no partial TP) ───────────────────────────────────────────
        tp_filled   = self._order_is_filled(self.position.tp_order_id)
        stop_filled = self._order_is_filled(self.position.stop_order_id)

        if tp_filled:
            self._cancel_order_safe(self.position.stop_order_id)
            alerts.alert_trade_close(
                self.symbol, tp_filled, self.position.fill_price, self.position.size, "TP HIT"
            )
            self._clear_position(tp_filled, "TP HIT")
            return tp_filled

        if stop_filled:
            self._cancel_order_safe(self.position.tp_order_id)
            alerts.alert_trade_close(
                self.symbol, stop_filled, self.position.fill_price, self.position.size, "STOP HIT"
            )
            self._clear_position(stop_filled, "STOP HIT")
            return stop_filled

        return None

    def emergency_shutdown(self, reason: str = "Unhandled exception") -> None:
        """Cancel all open orders, log state, and exit the process."""
        logger.log_error(f"EMERGENCY SHUTDOWN: {reason}")
        alerts.alert_emergency_shutdown(reason)
        self._cancel_all_orders()
        sys.exit(1)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _await_fill(self, order_id: int, max_retries: int = 3) -> Optional[float]:
        """Poll order status until FILLED; return avg fill price or None."""
        for attempt in range(max_retries):
            try:
                order = _with_retry(lambda: self.client.get_order(
                    symbol=self.symbol, orderId=order_id
                ))
            except Exception:
                time.sleep(2 ** attempt)
                continue

            if order.get("status") == "FILLED":
                executed_qty = float(order.get("executedQty", 0))
                cumulative_quote = float(order.get("cummulativeQuoteQty", 0))
                if executed_qty > 0:
                    return cumulative_quote / executed_qty
            time.sleep(2 ** attempt)

        return None

    def _place_limit_sell(self, qty: float, price: float) -> Optional[int]:
        try:
            order = _with_retry(lambda: self.client.order_limit_sell(
                symbol=self.symbol,
                quantity=qty,
                price=f"{price:.2f}",
                timeInForce="GTC",
            ))
            return order["orderId"]
        except Exception as exc:
            logger.log_error("Failed to place TP limit sell", exc)
            return None

    def _place_stop_limit_sell(self, qty: float, stop_price: float) -> Optional[int]:
        # Limit price slightly below stop to ensure fill in fast-moving markets
        limit_price = round(stop_price * 0.999, 2)
        try:
            order = _with_retry(lambda: self.client.create_order(
                symbol=self.symbol,
                side="SELL",
                type="STOP_LOSS_LIMIT",
                quantity=qty,
                price=f"{limit_price:.2f}",
                stopPrice=f"{stop_price:.2f}",
                timeInForce="GTC",
            ))
            return order["orderId"]
        except Exception as exc:
            logger.log_error("Failed to place stop-loss limit order", exc)
            return None

    def _order_is_filled(self, order_id: Optional[int]) -> Optional[float]:
        """Return avg fill price if FILLED, else None."""
        if order_id is None:
            return None
        try:
            order = _with_retry(lambda: self.client.get_order(
                symbol=self.symbol, orderId=order_id
            ))
            if order.get("status") == "FILLED":
                executed_qty = float(order.get("executedQty", 0))
                cumulative_quote = float(order.get("cummulativeQuoteQty", 0))
                if executed_qty > 0:
                    return cumulative_quote / executed_qty
        except Exception as exc:
            logger.log_error(f"Error checking order {order_id}", exc)
        return None

    def _cancel_order_safe(self, order_id: Optional[int]) -> None:
        if order_id is None:
            return
        try:
            _with_retry(lambda: self.client.cancel_order(
                symbol=self.symbol, orderId=order_id
            ))
        except Exception as exc:
            logger.log_warning(f"Could not cancel order {order_id}: {exc}")

    def _cancel_all_orders(self) -> None:
        try:
            self.client.cancel_all_open_orders(symbol=self.symbol)
            logger.log_info("All open orders cancelled.")
        except BinanceAPIException as exc:
            if exc.code == -2011:
                logger.log_info("No open orders to cancel.")
            else:
                logger.log_error("Failed to cancel all open orders", exc)
        except Exception as exc:
            logger.log_error("Failed to cancel all open orders", exc)

    def _close_at_market(self, qty: float) -> Optional[float]:
        """Cancel all open orders then place a market sell for qty. Returns fill price."""
        self._cancel_all_orders()
        try:
            order = _with_retry(lambda: self.client.order_market_sell(
                symbol=self.symbol,
                quantity=qty,
            ))
            sell_id = order["orderId"]
            fill = self._await_fill(sell_id, max_retries=3)
            return fill
        except Exception as exc:
            logger.log_error("Market SELL (momentum exit) failed", exc)
            return None

    def _clear_position(self, exit_price: float, close_type: str = "UNKNOWN") -> None:
        """Wipe local position state, report PnL to pause_manager, write trade journal."""
        if self.position is not None:
            realized_pnl = (exit_price - self.position.fill_price) * self.position.size
            try:
                import pause_manager
                pause_manager.record_close(realized_pnl, close_type)
            except Exception as exc:
                logger.log_warning(f"pause_manager.record_close failed (non-critical): {exc}")
            try:
                import trade_journal
                trade_journal.record_trade(
                    symbol=self.symbol,
                    position=self.position,
                    exit_price=exit_price,
                    close_reason=close_type,
                    session=self.position.session,
                    trade_grade=self.position.trade_grade,
                )
            except Exception as exc:
                logger.log_warning(f"trade_journal.record_trade failed (non-critical): {exc}")
        self.position = None

    def _signal_handler(self, signum, frame) -> None:  # noqa: ANN001
        self.emergency_shutdown(f"Signal {signum} received")
