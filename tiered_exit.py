"""
tiered_exit.py — Tiered Exit Framework (ROBUSTNESS #4).

Six-tier system for research simulations (all higher tiers opt-in):

  Tier 0 (Base):        Always-active: hard stop + full TP.
  Tier 1 (Partial TP):  At partial_tp_level × TP distance → close
                         partial_tp_close_pct, move SL to breakeven.
  Tier 1.5 (ATR Trail): ATR-based trailing stop, activates at
                         atr_trail_activation_r R.  End-of-bar update.
  Tier 1.9 (Mom.Fade):  2-bar consecutive-close momentum fade exit after
                         momentum_fade_min_r R (pure price, no RSI).
  Tier 2 (Full TP):     Close remaining at full TP target.
  Tier 2.5 (Partial 1R):Close partial_1r_close_pct at exactly 1.0×R,
                         move SL to entry, remainder trails ATR.
  Tier 3 (Time Stop):   Close at market after max_trade_bars bars.
  Tier 4 (RSI Mom.):    After momentum_exit_min_r R, exit if RSI(7)
                         crosses back through 50.

All tiers above Tier 0 are disabled by default; no existing profile
behaviour is changed.

Usage
-----
    from tiered_exit import ExitConfig, TierState, TieredExitFramework

    cfg = ExitConfig(enable_partial_tp=True, enable_time_stop=True)
    state = TierState(side=1, entry_price=64_000, initial_sl=63_000, tp_price=65_500)
    framework = TieredExitFramework(cfg)
    events = framework.simulate(state, df_after_entry, rsi7_series)
    for ev in events:
        print(ev.reason, ev.exit_price, ev.fraction)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import pandas_ta as ta  # noqa: F401

import config


# ── Configuration ──────────────────────────────────────────────────────────────

@dataclass
class ExitConfig:
    """
    Per-simulation exit configuration.

    Passed to TieredExitFramework; mirrors the per-profile opt-in flags from
    the spec without requiring a full profiles module.
    """
    enable_partial_tp: bool = config.ENABLE_PARTIAL_TP
    enable_time_stop: bool = config.ENABLE_TIME_STOP
    enable_momentum_exit: bool = config.ENABLE_MOMENTUM_EXIT
    max_trade_bars: int = config.MAX_TRADE_BARS
    partial_tp_level: float = config.PARTIAL_TP_LEVEL          # 0.75 × TP distance
    partial_tp_close_pct: float = config.PARTIAL_TP_CLOSE_PCT  # 0.40 = 40%
    momentum_exit_min_r: float = config.MOMENTUM_EXIT_MIN_R    # 0.5R min before RSI exit
    # Tier 1.5: ATR-based trailing stop (opt-in)
    enable_atr_trail: bool = config.ENABLE_ATR_TRAIL
    atr_trail_activation_r: float = config.ATR_TRAIL_ACTIVATION_R   # R at which ATR trail activates
    atr_trail_multiplier: float = config.ATR_TRAIL_MULTIPLIER        # trail = high/low ± ATR×mult
    # Tier 1.9: 2-bar price momentum-fade exit (opt-in)
    enable_momentum_fade: bool = config.ENABLE_MOMENTUM_FADE_EXIT
    momentum_fade_min_r: float = config.MOMENTUM_FADE_MIN_R
    # Tier 2.5: Partial exit at exactly 1.0R, breakeven SL, remainder trails ATR (opt-in)
    enable_partial_1r: bool = config.ENABLE_PARTIAL_1R
    partial_1r_close_pct: float = config.PARTIAL_1R_CLOSE_PCT
    # Tier 4 volume gate: momentum exit requires medium NATR + above-avg volume
    enable_volume_gate_momentum: bool = config.ENABLE_VOLUME_GATE_MOMENTUM
    volume_gate_min_ratio: float = config.VOLUME_GATE_MIN_RATIO
    natr_medium_min: float = config.NATR_MEDIUM_MIN
    natr_medium_max: float = config.NATR_MEDIUM_MAX


# ── Trade state ────────────────────────────────────────────────────────────────

@dataclass
class TierState:
    """Mutable state for one open simulated trade."""
    side: int           # +1 LONG, -1 SHORT
    entry_price: float
    initial_sl: float
    tp_price: float
    size: float = 1.0   # normalised position; 1.0 = full

    # Tier 1 state
    partial_closed: bool = False
    breakeven_sl: Optional[float] = None   # None until Tier 1 fires
    remaining_size: float = 1.0

    # Tier 1.5 state (ATR trailing stop)
    atr_trail_stop: Optional[float] = None   # None until activated

    # Tier 1.9 / Tier 4 state
    max_favorable_r: float = 0.0
    prev_close: Optional[float] = None       # one bar ago (for 2-bar fade check)
    prev2_close: Optional[float] = None      # two bars ago

    # Tier 2.5 state (partial at 1.0R)
    partial_1r_done: bool = False

    bars_open: int = 0

    @property
    def current_sl(self) -> float:
        """Active stop level: ATR trail > breakeven > initial, whichever is most favourable."""
        sl = self.breakeven_sl if self.breakeven_sl is not None else self.initial_sl
        if self.atr_trail_stop is not None:
            # ATR trail must always be on the favourable side of the base stop
            if self.side == 1:
                sl = max(sl, self.atr_trail_stop)
            else:
                sl = min(sl, self.atr_trail_stop)
        return sl

    @property
    def r_distance(self) -> float:
        return abs(self.tp_price - self.entry_price)

    @property
    def partial_tp_trigger(self) -> float:
        """Price level that triggers Tier 1 partial close."""
        dist = self.tp_price - self.entry_price  # signed
        return self.entry_price + config.PARTIAL_TP_LEVEL * dist


# ── Exit event ─────────────────────────────────────────────────────────────────

@dataclass
class ExitEvent:
    bar_index: int    # 0-based index within the post-entry slice
    exit_price: float
    reason: str       # SL | PARTIAL_TP | MOMENTUM_FADE | TP | PARTIAL_1R | TIME_STOP | MOMENTUM | EOD
    fraction: float   # fraction of original position closed (0–1)


# ── Framework ──────────────────────────────────────────────────────────────────

class TieredExitFramework:
    """
    Simulate trade exits against a sequence of OHLCV candles.

    Parameters
    ----------
    config : ExitConfig
        Which tiers to activate and their parameters.

    Usage
    -----
        framework = TieredExitFramework(ExitConfig(enable_partial_tp=True))
        state = TierState(side=+1, entry_price=..., initial_sl=..., tp_price=...)
        events = framework.simulate(state, df_after_entry, rsi7_series)
    """

    def __init__(self, exit_config: ExitConfig) -> None:
        self.cfg = exit_config

    def simulate(
        self,
        state: TierState,
        df: pd.DataFrame,
        rsi_series: Optional[pd.Series] = None,
        atr_series: Optional[pd.Series] = None,
        volume_ma_series: Optional[pd.Series] = None,
    ) -> list[ExitEvent]:
        """
        Walk through df bar-by-bar emitting ExitEvents.

        df               : OHLCV DataFrame sliced from the bar *after* entry.
        rsi_series       : Pre-computed RSI(7) aligned to df.index (for Tier 4).
        atr_series       : Pre-computed ATR aligned to df.index (for Tier 1.5 trail).
        volume_ma_series : Pre-computed volume moving average aligned to df.index
                           (for Tier 4 volume gate).

        Returns a list of ExitEvents.  Fractions sum to 1.0 across all events.
        The final event always closes the remaining position.
        """
        events: list[ExitEvent] = []
        r = state.r_distance

        for i, (_, candle) in enumerate(df.iterrows()):
            state.bars_open += 1
            high  = float(candle["high"])
            low   = float(candle["low"])
            close = float(candle["close"])

            # Track max favourable excursion (in R-units)
            if r > 0:
                fav = (high - state.entry_price) / r if state.side == 1 else (state.entry_price - low) / r
                state.max_favorable_r = max(state.max_favorable_r, fav)

            # ── Tier 1.9: 2-bar momentum-fade exit ────────────────────────────
            # Fires when: (a) max_favorable_r >= momentum_fade_min_r AND
            #             (b) two consecutive closes fade against our direction.
            # Uses close history tracked at end-of-bar (prev2_close, prev_close).
            if (
                self.cfg.enable_momentum_fade
                and state.max_favorable_r >= self.cfg.momentum_fade_min_r
                and state.prev_close is not None
                and state.prev2_close is not None
            ):
                if state.side == 1:
                    two_bar_fade = close < state.prev_close < state.prev2_close
                else:
                    two_bar_fade = close > state.prev_close > state.prev2_close
                if two_bar_fade:
                    events.append(ExitEvent(i, close, "MOMENTUM_FADE", state.remaining_size))
                    return events

            # ── Tier 1: Partial TP ─────────────────────────────────────────────
            if self.cfg.enable_partial_tp and not state.partial_closed:
                ptp = state.partial_tp_trigger
                triggered = (state.side == 1 and high >= ptp) or (state.side == -1 and low <= ptp)
                if triggered:
                    frac = self.cfg.partial_tp_close_pct
                    state.partial_closed = True
                    state.remaining_size = round(1.0 - frac, 10)
                    state.breakeven_sl = state.entry_price
                    events.append(ExitEvent(i, ptp, "PARTIAL_TP", frac))

            # ── Tier 0 / Tier 2: Hard stop and full TP ────────────────────────
            current_sl = state.current_sl

            sl_hit = (state.side == 1 and low <= current_sl) or (state.side == -1 and high >= current_sl)
            tp_hit = (state.side == 1 and high >= state.tp_price) or (state.side == -1 and low <= state.tp_price)

            # Conservative: if both touch same candle, SL wins
            if sl_hit and tp_hit:
                tp_hit = False

            if sl_hit:
                events.append(ExitEvent(i, current_sl, "SL", state.remaining_size))
                return events

            if tp_hit:
                events.append(ExitEvent(i, state.tp_price, "TP", state.remaining_size))
                return events

            # ── Tier 2.5: Partial exit at 1.0R ────────────────────────────────
            # Fires the first time price reaches entry ± 1.0×stop_distance.
            # Closes partial_1r_close_pct of remaining size, moves SL to entry.
            # ATR trail (if enabled) then keeps the remainder tighter.
            if self.cfg.enable_partial_1r and not state.partial_1r_done and r > 0:
                one_r_target = (
                    state.entry_price + r if state.side == 1
                    else state.entry_price - r
                )
                hit_1r = (state.side == 1 and high >= one_r_target) or (
                    state.side == -1 and low <= one_r_target
                )
                if hit_1r:
                    frac = self.cfg.partial_1r_close_pct
                    state.partial_1r_done = True
                    close_qty = round(state.remaining_size * frac, 10)
                    state.remaining_size = round(state.remaining_size - close_qty, 10)
                    # Move stop to breakeven (keep best of current SL and entry)
                    if state.side == 1:
                        state.breakeven_sl = max(
                            state.breakeven_sl if state.breakeven_sl is not None else state.initial_sl,
                            state.entry_price,
                        )
                    else:
                        state.breakeven_sl = min(
                            state.breakeven_sl if state.breakeven_sl is not None else state.initial_sl,
                            state.entry_price,
                        )
                    events.append(ExitEvent(i, one_r_target, "PARTIAL_1R", close_qty))

            # ── Tier 3: Time stop ──────────────────────────────────────────────
            if self.cfg.enable_time_stop and state.bars_open >= self.cfg.max_trade_bars:
                events.append(ExitEvent(i, close, "TIME_STOP", state.remaining_size))
                return events

            # ── Tier 4: Momentum exit ─────────────────────────────────────────
            if (
                self.cfg.enable_momentum_exit
                and state.max_favorable_r >= self.cfg.momentum_exit_min_r
                and rsi_series is not None
                and i > 0
            ):
                # Volume gate: if enabled, only allow momentum exit when NATR is
                # in the medium band AND current volume exceeds the moving average.
                volume_gate_ok = True
                if self.cfg.enable_volume_gate_momentum:
                    atr_val = _series_at(atr_series, df, i)
                    vol_ma_val = _series_at(volume_ma_series, df, i)
                    curr_vol = float(candle.get("volume", 0.0))
                    if atr_val is not None and close > 0:
                        natr = atr_val / close
                        in_medium_band = self.cfg.natr_medium_min <= natr <= self.cfg.natr_medium_max
                    else:
                        in_medium_band = True  # no ATR data → don't gate
                    above_avg_vol = (
                        vol_ma_val is not None
                        and curr_vol >= vol_ma_val * self.cfg.volume_gate_min_ratio
                    )
                    volume_gate_ok = in_medium_band and above_avg_vol

                if volume_gate_ok:
                    curr_rsi = _rsi_at(rsi_series, df, i)
                    prev_rsi = _rsi_at(rsi_series, df, i - 1)
                    if curr_rsi is not None and prev_rsi is not None:
                        if state.side == 1 and prev_rsi >= 50 and curr_rsi < 50:
                            events.append(ExitEvent(i, close, "MOMENTUM", state.remaining_size))
                            return events
                        if state.side == -1 and prev_rsi <= 50 and curr_rsi > 50:
                            events.append(ExitEvent(i, close, "MOMENTUM", state.remaining_size))
                            return events

            # ── Tier 1.5: ATR trailing stop update (end-of-bar, applies next bar) ──
            # Updated after all exit checks so the new trail level only takes
            # effect starting with the NEXT bar — consistent with how _advance_stop
            # works in backtest.py (stop advances at end of position management).
            if self.cfg.enable_atr_trail and r > 0:
                current_r = (
                    (high - state.entry_price) / r if state.side == 1
                    else (state.entry_price - low) / r
                )
                if current_r >= self.cfg.atr_trail_activation_r:
                    atr_val = _series_at(atr_series, df, i)
                    if atr_val is not None and atr_val > 0:
                        trail_offset = atr_val * self.cfg.atr_trail_multiplier
                        if state.side == 1:
                            raw_trail = high - trail_offset
                            if state.atr_trail_stop is None or raw_trail > state.atr_trail_stop:
                                state.atr_trail_stop = round(raw_trail, 2)
                        else:
                            raw_trail = low + trail_offset
                            if state.atr_trail_stop is None or raw_trail < state.atr_trail_stop:
                                state.atr_trail_stop = round(raw_trail, 2)

            # ── End-of-bar: update close history for Tier 1.9 ────────────────
            state.prev2_close = state.prev_close
            state.prev_close = close

        # End of data — close at last available close
        if state.remaining_size > 0:
            last_close = float(df["close"].iloc[-1]) if not df.empty else state.entry_price
            events.append(ExitEvent(max(0, len(df) - 1), last_close, "EOD", state.remaining_size))

        return events


def _series_at(series: Optional[pd.Series], df: pd.DataFrame, i: int) -> Optional[float]:
    """Get value from any aligned Series at position i. Returns None if unavailable."""
    if series is None:
        return None
    try:
        idx = df.index[i]
        val = float(series.loc[idx]) if idx in series.index else float(series.iloc[i])
        return None if not math.isfinite(val) else val
    except (IndexError, KeyError, TypeError):
        return None


def _rsi_at(rsi_series: pd.Series, df: pd.DataFrame, i: int) -> Optional[float]:
    """Get RSI value at position i, aligned to df.index."""
    try:
        idx = df.index[i]
        val = float(rsi_series.loc[idx]) if idx in rsi_series.index else float(rsi_series.iloc[i])
        return None if not math.isfinite(val) else val
    except (IndexError, KeyError, TypeError):
        return None


def compute_exit_r(
    events: list[ExitEvent],
    side: int,
    entry_price: float,
    stop_distance: float,
) -> float:
    """
    Compute the weighted-average exit R-multiple for a list of exit events.

    For partial closes, each event contributes its fraction × R.
    """
    if not events or stop_distance <= 0:
        return 0.0

    total_r = 0.0
    for ev in events:
        if side == 1:
            r = (ev.exit_price - entry_price) / stop_distance
        else:
            r = (entry_price - ev.exit_price) / stop_distance
        total_r += r * ev.fraction

    return total_r
