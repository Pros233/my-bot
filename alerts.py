"""
alerts.py — Telegram alert dispatcher.

Sends alerts to a Telegram chat for critical trading events.
All functions fail silently — Telegram errors are logged as warnings,
never raised to the caller. The bot must never stop trading due to
an alert failure.
"""
from __future__ import annotations

from datetime import datetime, timezone

import requests

import config
import logger


def _mode() -> str:
    return "TESTNET" if config.TESTNET else "LIVE"


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _send(text: str) -> None:
    """Send a Telegram message. Never raises."""
    if not config.ENABLE_TELEGRAM_ALERTS:
        return
    if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
        logger.log_warning(
            "Telegram alerts enabled but TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set."
        )
        return
    url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        resp = requests.post(
            url,
            json={
                "chat_id": config.TELEGRAM_CHAT_ID,
                "text": text,
                "parse_mode": "Markdown",
            },
            timeout=10,
        )
        if not resp.ok:
            logger.log_warning(
                f"Telegram API error: {resp.status_code} — {resp.text[:200]}"
            )
    except Exception as exc:
        logger.log_warning(f"Telegram send failed (non-critical): {exc}")


# ── Event alerts ───────────────────────────────────────────────────────────────

def alert_startup(symbols: list[str]) -> None:
    _send(
        f"*Bot started* | {_mode()}\n"
        f"Symbols: {', '.join(symbols)}\n"
        f"Strategy: {'RMR' if config.ENABLE_RANGE_MR else 'Consensus'}\n"
        f"Risk/trade: {config.RISK_PER_TRADE * 100:.2f}% | "
        f"Max open: {config.MAX_OPEN_TRADES}\n"
        f"Time: {_ts()}"
    )


def alert_trade_open(
    symbol: str,
    side: str,
    entry: float,
    stop: float,
    tp: float,
    size: float,
    strategy: str = "CONSENSUS",
) -> None:
    sl_pct = abs(entry - stop) / entry * 100
    tp_pct = abs(tp - entry) / entry * 100
    _send(
        f"*TRADE OPEN* | {_mode()}\n"
        f"Symbol: `{symbol}` | Strategy: {strategy}\n"
        f"Side: {side}\n"
        f"Entry:  `${entry:,.2f}`\n"
        f"Stop:   `${stop:,.2f}` (-{sl_pct:.2f}%)\n"
        f"TP:     `${tp:,.2f}` (+{tp_pct:.2f}%)\n"
        f"Size:   `{size:.5f}`\n"
        f"Time: {_ts()}"
    )


def alert_trade_close(
    symbol: str,
    exit_price: float,
    entry_price: float,
    size: float,
    close_type: str,  # "TP HIT" | "STOP HIT" | "MARKET CLOSE" | "BE STOP"
) -> None:
    pnl = round((exit_price - entry_price) * size, 4)
    pnl_str = f"+${pnl:.4f}" if pnl >= 0 else f"-${abs(pnl):.4f}"
    _send(
        f"*{close_type}* | {_mode()}\n"
        f"Symbol: `{symbol}`\n"
        f"Entry: `${entry_price:,.2f}` -> Exit: `${exit_price:,.2f}`\n"
        f"Size:  `{size:.5f}`\n"
        f"PnL:   `{pnl_str}` (gross)\n"
        f"Time: {_ts()}"
    )


def alert_partial_tp(
    symbol: str,
    price: float,
    qty_closed: float,
    qty_remaining: float,
) -> None:
    _send(
        f"*PARTIAL TP* | {_mode()}\n"
        f"Symbol: `{symbol}`\n"
        f"Price:     `${price:,.2f}`\n"
        f"Closed:    `{qty_closed:.5f}`\n"
        f"Remaining: `{qty_remaining:.5f}`\n"
        f"Time: {_ts()}"
    )


def alert_emergency_shutdown(reason: str) -> None:
    _send(
        f"*EMERGENCY SHUTDOWN* | {_mode()}\n"
        f"Reason: {reason}\n"
        f"Time: {_ts()}"
    )


def alert_order_failed(symbol: str, order_type: str, reason: str) -> None:
    _send(
        f"*ORDER FAILED* | {_mode()}\n"
        f"Symbol: `{symbol}`\n"
        f"Type: {order_type}\n"
        f"Reason: {reason[:300]}\n"
        f"Time: {_ts()}"
    )


def alert_exception(exc: Exception, context: str = "") -> None:
    _send(
        f"*UNHANDLED EXCEPTION* | {_mode()}\n"
        f"Context: {context}\n"
        f"Error: `{type(exc).__name__}: {str(exc)[:300]}`\n"
        f"Time: {_ts()}"
    )


def alert_paused(
    reason: str,
    daily_pnl_pct: float,
    weekly_pnl_pct: float,
    consecutive_losses: int,
) -> None:
    reason_labels = {
        "daily_loss_limit": "Daily loss limit hit",
        "weekly_loss_limit": "Weekly loss limit hit",
        "consecutive_losses": f"{consecutive_losses} consecutive losses",
    }
    label = reason_labels.get(reason, reason)
    _send(
        f"*TRADING PAUSED* | {_mode()}\n"
        f"Reason: {label}\n"
        f"Daily PnL:  `{daily_pnl_pct*100:+.2f}%`\n"
        f"Weekly PnL: `{weekly_pnl_pct*100:+.2f}%`\n"
        f"Consecutive losses: `{consecutive_losses}`\n"
        f"No new entries until unpaused.\n"
        f"Time: {_ts()}"
    )


def alert_resumed(resume_type: str) -> None:
    labels = {
        "auto_daily": "Auto-resumed — new UTC day",
        "auto_consecutive": "Auto-resumed — new UTC day (consecutive reset)",
        "manual": "Manually unpaused",
    }
    label = labels.get(resume_type, resume_type)
    _send(
        f"*TRADING RESUMED* | {_mode()}\n"
        f"Type: {label}\n"
        f"Time: {_ts()}"
    )


def alert_arb_opportunity(sig: object) -> None:
    """Send a watch-only arbitrage opportunity alert."""
    fees_slip_pct = sig.gross_profit_pct - sig.net_profit_pct
    _send(
        f"*ARB WATCH | {sig.arb_type.replace('_', ' ').title()}*\n"
        f"Route: `{sig.route}`\n"
        f"Gross:         `+{sig.gross_profit_pct:.2f}%`\n"
        f"Fees/slippage: `-{fees_slip_pct:.2f}%`\n"
        f"Net:           `+{sig.net_profit_pct:.2f}%`\n"
        f"Liquidity: {sig.liquidity_score} | Spread: {sig.spread_pct:.3f}%\n"
        f"Action: WATCH ONLY\n"
        f"No trade placed.\n"
        f"Time: {_ts()}"
    )


def alert_daily_report(
    total_pnl: float,
    daily_pnl: float,
    weekly_pnl: float,
    total_trades: int,
    win_rate: float,
    max_dd_pct: float,
    best_sym: str,
    worst_sym: str,
    open_positions: int,
    is_paused: bool,
    symbols: list[str],
    report_type: str = "Daily",
) -> None:
    status = "PAUSED" if is_paused else "ACTIVE"
    pf_sign = "+" if total_pnl >= 0 else ""
    _send(
        f"*{report_type} Report* | {_mode()} | {status}\n"
        f"Open: {open_positions}/{len(symbols)} positions\n"
        f"────────────────\n"
        f"Total PnL:   `{pf_sign}{total_pnl:.4f}` USDT\n"
        f"Daily PnL:   `{daily_pnl:+.4f}` USDT\n"
        f"Weekly PnL:  `{weekly_pnl:+.4f}` USDT\n"
        f"────────────────\n"
        f"Trades:      {total_trades}\n"
        f"Win rate:    {win_rate:.1%}\n"
        f"Max DD:      {max_dd_pct:.2f}%\n"
        f"Best sym:    {best_sym or '—'}\n"
        f"Worst sym:   {worst_sym or '—'}\n"
        f"Time: {_ts()}"
    )
