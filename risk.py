"""
risk.py — Position sizing and trade parameter calculation.

All functions are pure (no side effects).
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pandas_ta as ta  # noqa: F401

import config


@dataclass
class TradeParams:
    entry_price: float         # raw close price at signal
    effective_entry: float     # entry_price * (1 + slippage)
    stop_price: float          # ATR-based stop loss
    tp_price: float            # 2:1 reward-to-risk TP
    stop_distance: float       # ATR * multiplier (in $ terms)
    position_size: float       # BTC units (rounded to 5 dp)
    risk_amount: float         # $ at risk on this trade
    fee_estimate: float        # estimated round-trip fee ($ )
    halved: bool               # True when regime forced half-size


def calculate(
    df: pd.DataFrame,
    entry_price: float,
    account_balance: float,
    halve: bool = False,
) -> TradeParams:
    """
    Calculate all trade parameters for a LONG position.

    Parameters
    ----------
    df              : OHLCV DataFrame used to compute ATR.
    entry_price     : Close price at signal candle.
    account_balance : Current USDT free balance.
    halve           : True when TRENDING + HIGH_VOL regime.
    """
    # ATR-based stop distance
    atr_series = df.ta.atr(length=config.ATR_PERIOD)
    atr_val = float(atr_series.iloc[-1])

    stop_distance = atr_val * config.ATR_STOP_MULTIPLIER
    stop_price = entry_price - stop_distance
    tp_price = entry_price + (stop_distance * config.TP_RR_RATIO)

    # Slippage-adjusted effective entry
    effective_entry = entry_price * (1.0 + config.SLIPPAGE)

    # Position sizing: risk a fixed fraction of balance
    risk_amount = account_balance * config.RISK_PER_TRADE
    if halve:
        risk_amount /= 2.0

    position_size = risk_amount / stop_distance

    # Hard cap: position value <= MAX_POSITION_PCT of account
    max_value = account_balance * config.MAX_POSITION_PCT
    max_size = max_value / entry_price
    position_size = min(position_size, max_size)

    # Round to 5 decimal places (BTC precision on Binance)
    position_size = round(position_size, 5)

    # Fee estimate: 0.1% entry + 0.1% exit on effective_entry + tp price
    fee_estimate = (
        position_size * effective_entry * config.MAKER_FEE
        + position_size * tp_price * config.MAKER_FEE
    )

    return TradeParams(
        entry_price=entry_price,
        effective_entry=effective_entry,
        stop_price=round(stop_price, 2),
        tp_price=round(tp_price, 2),
        stop_distance=round(stop_distance, 2),
        position_size=position_size,
        risk_amount=round(risk_amount, 4),
        fee_estimate=round(fee_estimate, 4),
        halved=halve,
    )


def net_pnl(
    position_size: float,
    entry_price: float,
    exit_price: float,
) -> float:
    """
    Net PnL after deducting maker fees on both legs.

    Returns positive value for a winner, negative for a loser.
    """
    gross = position_size * (exit_price - entry_price)
    entry_fee = position_size * entry_price * config.MAKER_FEE
    exit_fee = position_size * exit_price * config.MAKER_FEE
    return round(gross - entry_fee - exit_fee, 6)
