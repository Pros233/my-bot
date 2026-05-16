"""
trade_journal.py — SQLite trade journal.

Writes every fully closed trade to trades.db.
All writes are fail-safe: a DB error never crashes the bot.

trades.db auto-creates on first write. The bot can run without any
pre-existing database file.
"""
from __future__ import annotations

import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import config
import logger

if TYPE_CHECKING:
    from executor import OpenPosition

DB_PATH = Path("trades.db")

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS trades (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    opened_at_utc     TEXT    NOT NULL,
    closed_at_utc     TEXT    NOT NULL,
    symbol            TEXT    NOT NULL,
    strategy          TEXT    NOT NULL DEFAULT 'UNKNOWN',
    side              TEXT    NOT NULL DEFAULT 'BUY',
    entry_price       REAL    NOT NULL,
    exit_price        REAL    NOT NULL,
    stop_price        REAL    NOT NULL,
    tp_price          REAL    NOT NULL,
    quantity          REAL    NOT NULL,
    pnl_usdt          REAL    NOT NULL,
    pnl_pct           REAL    NOT NULL,
    fees_estimated    REAL    NOT NULL DEFAULT 0.0,
    duration_minutes  REAL    NOT NULL DEFAULT 0.0,
    close_reason      TEXT    NOT NULL,
    regime            TEXT    NOT NULL DEFAULT '',
    adx               REAL    NOT NULL DEFAULT 0.0,
    atr_pct           REAL    NOT NULL DEFAULT 0.0,
    score_pct         REAL    NOT NULL DEFAULT 0.0,
    testnet           INTEGER NOT NULL DEFAULT 1,
    created_at        TEXT    NOT NULL
)
"""

_INSERT = """
INSERT INTO trades (
    opened_at_utc, closed_at_utc, symbol, strategy, side,
    entry_price, exit_price, stop_price, tp_price, quantity,
    pnl_usdt, pnl_pct, fees_estimated, duration_minutes, close_reason,
    regime, adx, atr_pct, score_pct, testnet, created_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def _ensure_db() -> None:
    """Create trades table if it doesn't exist. Safe to call on every startup."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(_CREATE_TABLE)
            conn.commit()
    except Exception as exc:
        logger.log_warning(f"trade_journal: could not initialise DB: {exc}")


def record_trade(
    symbol: str,
    position: "OpenPosition",
    exit_price: float,
    close_reason: str,
) -> None:
    """
    Write a completed trade to trades.db.
    Never raises — all errors logged as warnings so trading is never blocked.

    Note: for partial-TP positions (ENABLE_PARTIAL_TP=True), pnl_usdt is
    approximated as (exit_price - fill_price) * full_size. The partial TP
    profit portion is not separately captured. Partial TP is disabled by default.
    """
    try:
        _ensure_db()
        now = datetime.now(timezone.utc).isoformat()
        duration_min = (time.time() - position.entry_time) / 60.0
        pnl_usdt = (exit_price - position.fill_price) * position.size
        pnl_pct = (
            (pnl_usdt / position.entry_balance * 100)
            if position.entry_balance > 0
            else 0.0
        )

        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(_INSERT, (
                position.opened_at_utc or now,   # opened_at_utc
                now,                              # closed_at_utc
                symbol,
                position.strategy,
                position.side,
                position.fill_price,             # entry_price (actual fill)
                exit_price,
                position.stop_price,
                position.tp_price,
                position.size,
                round(pnl_usdt, 6),
                round(pnl_pct, 6),
                position.fees_estimated,
                round(duration_min, 2),
                close_reason,
                position.regime,
                position.adx,
                position.atr_pct,
                position.score_pct,
                1 if config.TESTNET else 0,
                now,
            ))
            conn.commit()

        logger.log_info(
            f"JOURNAL | {symbol} | {close_reason} | "
            f"pnl={pnl_usdt:+.4f} USDT ({pnl_pct:+.3f}%) | "
            f"entry={position.fill_price:.2f} exit={exit_price:.2f} qty={position.size:.5f}"
        )

    except Exception as exc:
        logger.log_warning(f"trade_journal: write failed (non-critical): {exc}")
