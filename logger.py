"""
logger.py — Structured trade logging to trades.log and equity.csv.
"""
from __future__ import annotations

import csv
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import config

LOG_DIR = Path(__file__).parent
TRADE_LOG = LOG_DIR / "trades.log"
EQUITY_CSV = LOG_DIR / "equity.csv"

_EQUITY_HEADERS = [
    "timestamp", "trade_num", "entry", "exit",
    "pnl_net", "balance", "drawdown_pct",
]


def _setup_logger() -> logging.Logger:
    level = getattr(logging, config.LOG_LEVEL, logging.INFO)
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    logger = logging.getLogger("tradingbot")
    logger.setLevel(level)

    if not logger.handlers:
        # File handler
        fh = logging.FileHandler(TRADE_LOG, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter(fmt, datefmt))
        logger.addHandler(fh)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(logging.Formatter(fmt, datefmt))
        logger.addHandler(ch)

    return logger


log = _setup_logger()


def _ensure_csv() -> None:
    if not EQUITY_CSV.exists():
        with open(EQUITY_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_EQUITY_HEADERS)
            writer.writeheader()


def log_cycle(
    timestamp: datetime,
    trend: str,
    vol: str,
    adx: float,
    atr_pct: float,
    consensus,       # ConsensusResult
    decision: str,
) -> None:
    """Log one candle cycle summary."""
    ts = timestamp.strftime("%Y-%m-%d %H:%M UTC")
    log.info(
        "CYCLE | %s | Regime: %s+%s | ADX: %.1f | ATR%%: %.2f%% | "
        "Decision: %s | Score: %.2f/%.2f (%.1f%%)",
        ts, trend, vol, adx, atr_pct, decision,
        consensus.score, consensus.max_possible,
        consensus.ratio * 100,
    )
    for s in consensus.breakdown:
        log.info(
            "  %-12s | signal: %+d | weight: %.1f | contrib: %+.2f",
            s.name, s.signal, s.weight, s.contribution,
        )


def log_skip(timestamp: datetime, reason: str) -> None:
    ts = timestamp.strftime("%Y-%m-%d %H:%M UTC")
    log.warning("SKIP | %s | %s", ts, reason)


def log_trade_open(
    timestamp: datetime,
    side: str,
    entry: float,
    stop: float,
    tp: float,
    size: float,
    fee_est: float,
    order_ids: dict,
) -> None:
    ts = timestamp.strftime("%Y-%m-%d %H:%M UTC")
    log.info(
        "TRADE OPEN | %s | %s | Entry: %.2f | Stop: %.2f | TP: %.2f | "
        "Size: %.5f BTC | Est. Fee: $%.4f | Orders: %s",
        ts, side, entry, stop, tp, size, fee_est, order_ids,
    )


def log_trade_close(
    timestamp: datetime,
    exit_price: float,
    entry_price: float,
    size: float,
    pnl_net: float,
    balance: float,
    trade_num: int,
    peak_balance: float,
) -> None:
    ts = timestamp.strftime("%Y-%m-%d %H:%M UTC")
    win_loss = "WIN" if pnl_net > 0 else "LOSS"
    drawdown = ((peak_balance - balance) / peak_balance * 100) if peak_balance > 0 else 0.0
    log.info(
        "TRADE CLOSE | %s | %s | Exit: %.2f | Entry: %.2f | "
        "PnL Net: $%.4f | Balance: $%.2f | Drawdown: %.2f%%",
        ts, win_loss, exit_price, entry_price, pnl_net, balance, drawdown,
    )
    _write_equity_row(
        timestamp=timestamp,
        trade_num=trade_num,
        entry=entry_price,
        exit_price=exit_price,
        pnl_net=pnl_net,
        balance=balance,
        drawdown_pct=drawdown,
    )


def log_error(msg: str, exc: Optional[Exception] = None) -> None:
    if exc:
        log.error("%s: %s", msg, exc, exc_info=True)
    else:
        log.error(msg)


def log_info(msg: str) -> None:
    log.info(msg)


def log_warning(msg: str) -> None:
    log.warning(msg)


def _write_equity_row(
    timestamp: datetime,
    trade_num: int,
    entry: float,
    exit_price: float,
    pnl_net: float,
    balance: float,
    drawdown_pct: float,
) -> None:
    _ensure_csv()
    row = {
        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "trade_num": trade_num,
        "entry": round(entry, 2),
        "exit": round(exit_price, 2),
        "pnl_net": round(pnl_net, 4),
        "balance": round(balance, 2),
        "drawdown_pct": round(drawdown_pct, 4),
    }
    with open(EQUITY_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_EQUITY_HEADERS)
        writer.writerow(row)
