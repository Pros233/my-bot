"""
dashboard.py — Terminal status block printed after each candle close.

Output example:
  ┌─ BTC/USDT 1H ──────────────────────────────────────┐
  │ Regime: TRENDING | ADX: 31.2 | ATR%: 1.8%          │
  │ Active strategies: EMA, MACD, BB, Volume, VWAP      │
  │ EMA Cross  [+1] x2 = +2.0                          │
  │ MACD       [ 0] x2 =  0.0                          │
  │ ...                                                  │
  │ Score: +5.0 / 7.0 = 71.4% → BUY                  │
  │ Position: 0.00412 BTC | Stop: $61,230 | TP: $64,100│
  │ Balance: $1,042.30 | Open trades: 1                 │
  └────────────────────────────────────────────────────┘
"""
from __future__ import annotations

from typing import Optional

import config

_WIDTH = 56  # inner width (between │ chars)


def _line(content: str = "") -> str:
    padded = f" {content}"
    return f"│{padded:<{_WIDTH}}│"


def _fmt_signal(sig: int) -> str:
    if sig == 1:
        return "+1"
    if sig == -1:
        return "-1"
    return " 0"


def print_status(
    symbol: str,
    interval: str,
    trend: str,
    vol: str,
    adx: float,
    atr_pct: float,
    consensus,                   # ConsensusResult
    decision: str,
    balance: float,
    open_trades: int,
    position_size: Optional[float] = None,
    stop_price: Optional[float] = None,
    tp_price: Optional[float] = None,
) -> None:
    """Print the status dashboard block to stdout."""
    header_label = f"─ {symbol} {interval} "
    header = f"┌{header_label}{'─' * (_WIDTH - len(header_label) + 1)}┐"

    lines = [header]

    # Regime row
    lines.append(_line(
        f"Regime: {trend} | ADX: {adx:.1f} | ATR%: {atr_pct:.2f}%"
    ))

    # Active strategies
    names = ", ".join(s.name for s in consensus.breakdown)
    lines.append(_line(f"Active strategies: {names}"))

    lines.append(_line("─" * (_WIDTH - 1)))

    # Per-strategy breakdown
    for s in consensus.breakdown:
        sig_str = _fmt_signal(s.signal)
        contrib_sign = "+" if s.contribution >= 0 else ""
        row = (
            f" {s.name:<12} [{sig_str}] x{s.weight:.1f} = "
            f"{contrib_sign}{s.contribution:.1f}"
        )
        lines.append(_line(row))

    lines.append(_line("─" * (_WIDTH - 1)))

    # Consensus score
    score_sign = "+" if consensus.score >= 0 else ""
    ratio_pct = consensus.ratio * 100
    lines.append(_line(
        f" Score: {score_sign}{consensus.score:.1f} / {consensus.max_possible:.1f}"
        f" = {ratio_pct:.1f}%  →  {decision}"
    ))

    # Position info
    if position_size and stop_price and tp_price:
        lines.append(_line(
            f" Position: {position_size:.5f} BTC"
            f" | Stop: ${stop_price:,.0f}"
            f" | TP: ${tp_price:,.0f}"
        ))
    else:
        lines.append(_line(" Position: none"))

    # Account
    trades_str = f"{open_trades} open"
    lines.append(_line(
        f" Balance: ${balance:,.2f} | Trades: {trades_str}"
    ))

    footer = f"└{'─' * (_WIDTH + 1)}┘"
    lines.append(footer)

    print("\n".join(lines))
    print()  # blank line after block
