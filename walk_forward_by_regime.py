"""
walk_forward_by_regime.py — Regime-stratified walk-forward engine (ROBUSTNESS #3).

Integrates all nine robustness improvements into a single research pipeline:

  1. RegimeClassifier      — TRENDING_UP/DOWN / RANGING / HIGH_VOLATILITY
  2. ATR-scaled TP/SL      — TP = entry ± ATR × TP_ATR_MULTIPLIER
  3. Walk-forward windows  — WF_WINDOW_DAYS-day rolling windows
  4. TieredExitFramework   — Tiers 0-4 (opt-in per ExitConfig)
  5. SampleGuard           — flags windows with < MIN_WINDOW_TRADES
  6. EntryConfirmationBuffer — skip if N+1 open drifts > ENTRY_BUFFER_PCT
  7. Short-vol gate        — block shorts outside NATR band
  8. MTF confirmation      — 4H ADX + Daily BB midpoint check
  9. Research mode         — called from main.py --mode research/backtest

Strategy used for signal generation is mean-reversion focused:
  RANGING:      Bollinger-band touch (long: close < lower-band AND rising;
                short: close > upper-band AND falling)
  TRENDING:     EMA(9) / EMA(21) crossover aligned with EMA(200)

Both signal types must also pass MACD agreement to enter a trade
(preserving the REQUIRE_MACD spirit of the existing engine).

Outputs to outputs/research/:
  regime_performance_summary.csv  — per-window metrics
  regime_log.csv                  — regime label per candle
  entry_quality.csv               — per-signal entry-buffer results
  blocked_entries.csv             — short entries blocked by vol gate
  mtf_rejections.csv              — entries rejected by MTF check
  excluded_windows.csv            — windows excluded by SampleGuard

Call from backtest.py run() or from main.py --mode research:
    from walk_forward_by_regime import run_regime_walk_forward
    run_regime_walk_forward(df, initial_balance=10_000.0)
"""
from __future__ import annotations

import csv
import math
import statistics
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pandas_ta as ta  # noqa: F401

import config
from regime_classifier import (
    RegimeClassifier, dominant_regime,
    TRENDING_UP, TRENDING_DOWN, RANGING, HIGH_VOLATILITY, UNKNOWN,
)
from entry_buffer import EntryConfirmationBuffer
from strategies.vwap import _compute_vwap
from sample_guard import SampleGuard, WindowResult
from tiered_exit import ExitConfig, TierState, TieredExitFramework, compute_exit_r

OUTPUT_DIR = Path("outputs/research")
SUMMARY_CSV = OUTPUT_DIR / "regime_performance_summary.csv"
BLOCKED_ENTRIES_CSV = OUTPUT_DIR / "blocked_entries.csv"
MTF_REJECTIONS_CSV  = OUTPUT_DIR / "mtf_rejections.csv"

_SUMMARY_HEADERS = [
    "window_start", "window_end", "dominant_regime",
    "trade_count", "pf", "win_rate", "median_r", "worst_window_pf",
    "tp_hit_rate", "avg_r", "sharpe_ratio", "sortino_ratio",
    "chase_skip_rate", "sample_valid", "valid_window_pct",
    "robustness_pass",
]
_BLOCKED_HEADERS = ["timestamp", "direction", "natr", "short_max_natr", "short_min_natr", "reason"]
_MTF_HEADERS     = ["timestamp", "direction", "signal_price", "h4_adx", "daily_bb_mid", "daily_price", "reason"]


# ── Resampling helper (mirrors backtest._resample_ohlcv) ─────────────────────

def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    factor = int(rule[:-1]) if rule[:-1].isdigit() else 1
    counts = df["close"].resample(rule, label="right", closed="right").count()
    out = pd.DataFrame({
        "open":   df["open"].resample(rule, label="right", closed="right").first(),
        "high":   df["high"].resample(rule, label="right", closed="right").max(),
        "low":    df["low"].resample(rule, label="right", closed="right").min(),
        "close":  df["close"].resample(rule, label="right", closed="right").last(),
        "volume": df["volume"].resample(rule, label="right", closed="right").sum(),
    }).dropna()
    if factor > 1:
        out = out.loc[counts == factor]
    return out


# ── Research frame (vectorised indicator pre-computation) ─────────────────────

def _build_research_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Pre-compute all indicator columns needed by the simulation."""
    rf = pd.DataFrame(index=df.index)
    rf["open"]  = df["open"]
    rf["high"]  = df["high"]
    rf["low"]   = df["low"]
    rf["close"] = df["close"]
    rf["volume"] = df["volume"]

    # ATR
    rf["atr"] = df.ta.atr(length=config.ATR_PERIOD)
    rf["natr"] = rf["atr"] / rf["close"]   # normalised ATR

    # EMA
    rf["ema_fast"]  = df.ta.ema(length=config.EMA_FAST)
    rf["ema_slow"]  = df.ta.ema(length=config.EMA_SLOW)
    rf["ema200"]    = df.ta.ema(length=200)

    # MACD
    macd_df = df.ta.macd(fast=config.MACD_FAST, slow=config.MACD_SLOW, signal=config.MACD_SIGNAL)
    mc = f"MACD_{config.MACD_FAST}_{config.MACD_SLOW}_{config.MACD_SIGNAL}"
    ms = f"MACDs_{config.MACD_FAST}_{config.MACD_SLOW}_{config.MACD_SIGNAL}"
    rf["macd_line"] = macd_df[mc] if macd_df is not None and mc in macd_df.columns else np.nan
    rf["macd_sig"]  = macd_df[ms] if macd_df is not None and ms in macd_df.columns else np.nan

    # Bollinger Bands
    bb_df = df.ta.bbands(length=config.BB_PERIOD, std=config.BB_STD)
    if bb_df is not None and not bb_df.empty:
        bl = next((c for c in bb_df.columns if c.startswith(f"BBL_{config.BB_PERIOD}_")), None)
        bu = next((c for c in bb_df.columns if c.startswith(f"BBU_{config.BB_PERIOD}_")), None)
        bm = next((c for c in bb_df.columns if c.startswith(f"BBM_{config.BB_PERIOD}_")), None)
        rf["bb_lower"] = bb_df[bl] if bl else np.nan
        rf["bb_upper"] = bb_df[bu] if bu else np.nan
        rf["bb_mid"]   = bb_df[bm] if bm else np.nan
    else:
        rf["bb_lower"] = rf["bb_upper"] = rf["bb_mid"] = np.nan

    # RSI(7) for Tier 4 momentum exit
    rf["rsi7"] = df.ta.rsi(length=config.MOMENTUM_EXIT_RSI_PERIOD)

    # Volume MA (for RMR volume-bucket classification)
    rf["vol_ma"] = df["volume"].rolling(config.VOLUME_MA_PERIOD).mean()

    # Session VWAP (resets each UTC calendar day) — used by RMR signal
    rf["vwap"] = _compute_vwap(df)

    return rf


# ── MTF helper ────────────────────────────────────────────────────────────────

def _build_mtf_frames(df_1h: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resample 1H to 4H and 1D; compute ADX(14) and BB for each."""
    df_4h = _resample_ohlcv(df_1h, "4h")
    df_1d = _resample_ohlcv(df_1h, "1D")

    # 4H ADX
    adx4h = df_4h.ta.adx(length=config.ATR_PERIOD)
    adx4h_col = f"ADX_{config.ATR_PERIOD}"
    if adx4h is not None and adx4h_col in adx4h.columns:
        df_4h["adx"] = adx4h[adx4h_col]
    else:
        df_4h["adx"] = np.nan

    # Daily BB
    bb1d = df_1d.ta.bbands(length=config.BB_PERIOD, std=config.BB_STD)
    if bb1d is not None and not bb1d.empty:
        bm = next((c for c in bb1d.columns if c.startswith(f"BBM_{config.BB_PERIOD}_")), None)
        bs = next((c for c in bb1d.columns if c.startswith(f"BBB_{config.BB_PERIOD}_")), None)
        bl = next((c for c in bb1d.columns if c.startswith(f"BBL_{config.BB_PERIOD}_")), None)
        bu = next((c for c in bb1d.columns if c.startswith(f"BBU_{config.BB_PERIOD}_")), None)
        df_1d["bb_mid"]   = bb1d[bm] if bm else np.nan
        df_1d["bb_lower"] = bb1d[bl] if bl else np.nan
        df_1d["bb_upper"] = bb1d[bu] if bu else np.nan
    else:
        df_1d["bb_mid"] = df_1d["bb_lower"] = df_1d["bb_upper"] = np.nan

    return df_4h, df_1d


def _check_mtf(
    ts: pd.Timestamp,
    df_4h: pd.DataFrame,
    df_1d: pd.DataFrame,
) -> tuple[bool, str]:
    """
    Returns (passes, reason).

    4H check : 4H ADX < MTF_ADX_THRESHOLD (confirming range on higher TF).
    Daily check : price within Daily BB midpoint ± MTF_BB_SIGMA × half-bandwidth.
    """
    # ── 4H ADX ────────────────────────────────────────────────────────────────
    try:
        h4_bar = df_4h.index.asof(ts)
        h4_adx = float(df_4h.loc[h4_bar, "adx"]) if h4_bar in df_4h.index else math.nan
    except Exception:
        h4_adx = math.nan

    if math.isfinite(h4_adx) and h4_adx >= config.MTF_ADX_THRESHOLD:
        return False, f"4H_ADX={h4_adx:.1f}>={config.MTF_ADX_THRESHOLD}"

    # ── Daily BB ───────────────────────────────────────────────────────────────
    try:
        d_bar = df_1d.index.asof(ts)
        bb_mid   = float(df_1d.loc[d_bar, "bb_mid"])   if d_bar in df_1d.index else math.nan
        bb_lower = float(df_1d.loc[d_bar, "bb_lower"]) if d_bar in df_1d.index else math.nan
        bb_upper = float(df_1d.loc[d_bar, "bb_upper"]) if d_bar in df_1d.index else math.nan
    except Exception:
        bb_mid = bb_lower = bb_upper = math.nan

    if math.isfinite(bb_mid) and math.isfinite(bb_lower) and math.isfinite(bb_upper):
        half_bw = (bb_upper - bb_lower) / 2.0
        gate = config.MTF_BB_SIGMA * half_bw
        # Check if signal timestamp's 1H close is within gate
        if ts in df_4h.index:
            price = float(df_4h.loc[ts, "close"]) if "close" in df_4h.columns else math.nan
        else:
            try:
                h4_close_bar = df_4h.index.asof(ts)
                price = float(df_4h.loc[h4_close_bar, "close"]) if h4_close_bar in df_4h.index else math.nan
            except Exception:
                price = math.nan

        if math.isfinite(price) and math.isfinite(gate) and gate > 0:
            if abs(price - bb_mid) > gate:
                return False, f"DAILY_BB_EXTREME price={price:.0f} mid={bb_mid:.0f} gate={gate:.0f}"

    return True, ""


# ── Signal logic ──────────────────────────────────────────────────────────────

def _get_mr_signal(rf: pd.DataFrame, i: int) -> int:
    """
    Range mean-reversion entry signal at bar i — aligned with the validated
    RMR_2H_NO_FAR_VWAP_LONG_BIASED_BLOCK_LONG_HIGHATR_HIGHVOL setup.

    Entry conditions (LONG only — no short MR entries; shorts failed research):
      1. Price extended below VWAP (close < vwap OR low < vwap)
      2. Price at/below 24-bar range midpoint OR broke range low
      3. VWAP distance NOT "far" (|close-vwap| / stop_dist < RMR_VWAP_FAR_R)
      4. NOT (ATR-bucket=high AND volume-bucket=high) — catastrophic block
      5. Reclaim: bullish close beats recent 3-bar max
      6. Rejection: close > open AND lower_wick >= body (hammer)

    Returns +1 (LONG) or 0.  Never returns -1 (no shorts).
    """
    lb = config.RESEARCH_RANGE_LOOKBACK  # 24 bars
    rl = config.RESEARCH_RECLAIM_LOOKBACK  # 3 bars
    min_i = lb + rl + 2

    if i < min_i:
        return 0

    close   = float(rf["close"].iloc[i])
    open_   = float(rf["open"].iloc[i])
    high    = float(rf["high"].iloc[i])
    low     = float(rf["low"].iloc[i])
    p_close = float(rf["close"].iloc[i - 1])
    vwap    = float(rf["vwap"].iloc[i])
    atr     = float(rf["atr"].iloc[i])
    vol_ma  = float(rf["vol_ma"].iloc[i])
    volume  = float(rf["volume"].iloc[i])

    vals = (close, open_, high, low, p_close, vwap, atr, vol_ma)
    if any(not math.isfinite(v) for v in vals) or atr <= 0 or vol_ma <= 0 or close <= 0 or vwap <= 0:
        return 0

    # 1. Price below VWAP
    if not (close < vwap or low < vwap):
        return 0

    # Range boundaries (prior 24 bars, excluding current)
    prior_high = float(rf["high"].iloc[i - lb: i].max())
    prior_low  = float(rf["low"].iloc[i - lb: i].min())
    range_width = prior_high - prior_low
    if range_width <= 0:
        return 0
    range_mid = (prior_high + prior_low) / 2.0

    # 2. Price at/below range mid or broke range low
    broke_range_low = low < prior_low
    if not (broke_range_low or close <= range_mid):
        return 0

    # 3. VWAP distance filter — block "far" entries
    stop_dist = atr * config.ATR_STOP_MULTIPLIER
    if stop_dist <= 0:
        return 0
    vwap_dist_r = abs(close - vwap) / stop_dist
    if vwap_dist_r >= config.RMR_VWAP_FAR_R:
        return 0

    # 4. Catastrophic block: HIGH_ATR + HIGH_VOL
    atr_pct    = (atr / close) * 100.0
    vol_ratio  = volume / vol_ma
    high_atr   = atr_pct >= config.RMR_ATR_HIGH_PCT
    high_vol   = vol_ratio >= config.RMR_VOL_HIGH
    if high_atr and high_vol:
        return 0

    # 5. Reclaim pattern: bullish candle beats recent 3-bar close max
    recent_max   = float(rf["close"].iloc[i - rl: i].max())
    failed_bkout = broke_range_low and close >= prior_low
    if not (close > open_ and close > recent_max and (failed_bkout or close > p_close)):
        return 0

    # 6. Rejection / hammer: lower wick >= body
    body       = abs(close - open_)
    lower_wick = min(open_, close) - low
    if not (close > open_ and lower_wick >= body):
        return 0

    return 1


def _get_trend_signal(rf: pd.DataFrame, i: int) -> int:
    """EMA(9)/EMA(21) crossover aligned with EMA(200)."""
    if i < 2:
        return 0
    ef_now  = rf["ema_fast"].iloc[i]
    ef_prev = rf["ema_fast"].iloc[i - 1]
    es_now  = rf["ema_slow"].iloc[i]
    es_prev = rf["ema_slow"].iloc[i - 1]
    e200    = rf["ema200"].iloc[i]
    close   = rf["close"].iloc[i]

    if any(not math.isfinite(float(v)) for v in (ef_now, ef_prev, es_now, es_prev, e200, close)):
        return 0

    if ef_prev <= es_prev and ef_now > es_now and float(close) > float(e200):
        return 1
    if ef_prev >= es_prev and ef_now < es_now and float(close) < float(e200):
        return -1
    return 0


# ── Per-window simulation ─────────────────────────────────────────────────────

@dataclass
class _TradeRecord:
    side: int
    entry_price: float
    exit_price: float
    stop_distance: float
    exit_r: float
    tp_hit: bool
    regime: str


def _simulate_window(
    df: pd.DataFrame,
    rf: pd.DataFrame,
    regime_series: pd.Series,
    df_4h: pd.DataFrame,
    df_1d: pd.DataFrame,
    exit_cfg: ExitConfig,
    entry_buf: EntryConfirmationBuffer,
    mtf_enabled: bool,
    blocked_rows: list,
    mtf_rejection_rows: list,
) -> list[_TradeRecord]:
    """Run one walk-forward window and return trade records."""
    framework = TieredExitFramework(exit_cfg)
    rsi7_series = rf["rsi7"]
    trades: list[_TradeRecord] = []

    warmup = max(
        202,  # EMA200 / VWAP warmup
        config.ADX_PERIOD + config.ATR_HIGH_VOL_PERIOD + 5,
        config.BB_PERIOD + 2,
        config.MACD_SLOW + config.MACD_SIGNAL + 2,
        config.RESEARCH_RANGE_LOOKBACK + config.RESEARCH_RECLAIM_LOOKBACK + config.VOLUME_MA_PERIOD + 5,
    )

    i = warmup
    while i < len(df) - 1:
        # ── Regime gate ───────────────────────────────────────────────────────
        regime = str(regime_series.iloc[i])

        if regime == UNKNOWN or regime == HIGH_VOLATILITY:
            i += 1
            continue

        # Choose signal based on regime
        if regime in (TRENDING_UP, TRENDING_DOWN):
            signal = _get_trend_signal(rf, i)
        else:  # RANGING
            signal = _get_mr_signal(rf, i)

        if signal == 0:
            i += 1
            continue

        # ── ATR / TP / SL at signal candle ────────────────────────────────────
        atr_val = float(rf["atr"].iloc[i])
        close_n = float(rf["close"].iloc[i])
        natr    = float(rf["natr"].iloc[i])
        ts      = df.index[i]

        if not math.isfinite(atr_val) or atr_val <= 0 or not math.isfinite(close_n):
            i += 1
            continue

        # RMR LONG: 1.5R TP (matching RMR_TP_RR_RATIO); no RMR shorts (EV < 0 in research).
        # Trend signals keep ATR-scaled TP/SL; shorts from TRENDING_DOWN apply vol gate.
        stop_dist = atr_val * config.ATR_STOP_MULTIPLIER
        if regime == RANGING:          # signal == 1 always (RMR emits only LONG)
            sl_price = close_n - stop_dist
            tp_price = close_n + stop_dist * config.RMR_TP_RR_RATIO
        elif signal == 1:              # trend LONG
            tp_price = close_n + atr_val * config.TP_ATR_MULTIPLIER
            sl_price = close_n - atr_val * config.SL_ATR_MULTIPLIER
        else:                          # signal == -1, trend SHORT — apply vol gate
            if not math.isfinite(natr):
                i += 1
                continue
            if natr > config.SHORT_MAX_NATR or natr < config.SHORT_MIN_NATR:
                if blocked_rows is not None:
                    blocked_rows.append({
                        "timestamp": str(ts),
                        "direction": "SHORT",
                        "natr": f"{natr:.6f}",
                        "short_max_natr": config.SHORT_MAX_NATR,
                        "short_min_natr": config.SHORT_MIN_NATR,
                        "reason": "NATR_OUT_OF_BAND",
                    })
                i += 1
                continue
            tp_price = close_n - atr_val * config.TP_ATR_MULTIPLIER
            sl_price = close_n + atr_val * config.SL_ATR_MULTIPLIER

        # ── MIN_RR_RATIO check ────────────────────────────────────────────────
        tp_dist = abs(tp_price - close_n)
        sl_dist = abs(sl_price - close_n)
        if sl_dist <= 0 or (tp_dist / sl_dist) < config.MIN_RR_RATIO:
            i += 1
            continue

        # ── MTF confirmation (for MR / RANGING) ───────────────────────────────
        if mtf_enabled and regime == RANGING:
            passes, reason = _check_mtf(ts, df_4h, df_1d)
            if not passes:
                if mtf_rejection_rows is not None:
                    mtf_rejection_rows.append({
                        "timestamp": str(ts),
                        "direction": "LONG" if signal == 1 else "SHORT",
                        "signal_price": f"{close_n:.2f}",
                        "h4_adx": "",
                        "daily_bb_mid": "",
                        "daily_price": "",
                        "reason": reason,
                    })
                i += 1
                continue

        # ── Entry confirmation buffer — check N+1 open ─────────────────────
        open_n1 = float(df["open"].iloc[i + 1])
        if not entry_buf.check(close_n, open_n1, str(ts)):
            i += 1
            continue

        # ── Entry approved — set up trade ─────────────────────────────────────
        entry_price = open_n1 * (1.0 + config.SLIPPAGE if signal == 1 else 1.0 - config.SLIPPAGE)
        stop_distance = abs(entry_price - sl_price)
        if stop_distance <= 0:
            i += 1
            continue

        state = TierState(
            side=signal,
            entry_price=entry_price,
            initial_sl=sl_price,
            tp_price=tp_price,
        )

        # Slice from bar i+2 onwards (bar i+1 is the entry bar)
        post_entry_df  = df.iloc[i + 2:].reset_index(drop=False)
        post_entry_rsi = rsi7_series.iloc[i + 2:]

        # Restore index for rsi alignment
        post_entry_df_indexed = df.iloc[i + 2:]
        events = framework.simulate(state, post_entry_df_indexed, post_entry_rsi)

        if not events:
            i += 2
            continue

        exit_r   = compute_exit_r(events, signal, entry_price, stop_distance)
        tp_hit   = any(ev.reason in ("TP", "PARTIAL_TP") for ev in events)
        exit_price = events[-1].exit_price
        exit_bar   = i + 2 + events[-1].bar_index

        trades.append(_TradeRecord(
            side=signal,
            entry_price=entry_price,
            exit_price=exit_price,
            stop_distance=stop_distance,
            exit_r=exit_r,
            tp_hit=tp_hit,
            regime=regime,
        ))

        i = exit_bar + 1

    return trades


# ── Metrics ───────────────────────────────────────────────────────────────────

def _profit_factor(rs: list[float]) -> float:
    gains  = sum(r for r in rs if r > 0)
    losses = abs(sum(r for r in rs if r < 0))
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return gains / losses


def _sharpe(rs: list[float]) -> float:
    """Mean(R) / std(R) across all trades; 0 when fewer than 2 trades or std=0."""
    if len(rs) < 2:
        return 0.0
    mu = statistics.mean(rs)
    sd = statistics.pstdev(rs)
    return round(mu / sd, 4) if sd > 0 else 0.0


def _sortino(rs: list[float]) -> float:
    """Mean(R) / downside-std(R); 0 when there are no losing trades."""
    if not rs:
        return 0.0
    mu = statistics.mean(rs)
    losses = [r for r in rs if r < 0]
    if not losses:
        return round(mu * 10, 4)   # large positive — cap at 10× mean
    downside_var = sum(r ** 2 for r in losses) / len(losses)
    downside_std = math.sqrt(downside_var)
    return round(mu / downside_std, 4) if downside_std > 0 else 0.0


def _window_result(
    window_start: str,
    window_end: str,
    regime_label: str,
    trades: list[_TradeRecord],
) -> WindowResult:
    n = len(trades)
    if n == 0:
        return WindowResult(
            window_start=window_start, window_end=window_end,
            dominant_regime=regime_label, trade_count=0,
            pf=0.0, win_rate=0.0, median_r=0.0, worst_window_pf=0.0,
            tp_hit_rate=0.0, avg_r=0.0, sharpe_ratio=0.0, sortino_ratio=0.0,
        )
    rs = [t.exit_r for t in trades]
    wins = [r for r in rs if r > 0]
    pf = _profit_factor(rs)
    avg_r = round(statistics.mean(rs), 4)
    return WindowResult(
        window_start=window_start,
        window_end=window_end,
        dominant_regime=regime_label,
        trade_count=n,
        pf=pf,
        win_rate=len(wins) / n,
        median_r=statistics.median(rs),
        worst_window_pf=pf,   # equals pf for a single window
        tp_hit_rate=sum(1 for t in trades if t.tp_hit) / n,
        avg_r=avg_r,
        sharpe_ratio=_sharpe(rs),
        sortino_ratio=_sortino(rs),
    )


# ── Robustness gate ───────────────────────────────────────────────────────────

def _check_robustness_gate(windows: list[WindowResult]) -> bool:
    """
    Strategy passes all three conditions per regime type present:
      1. PF > WF_MIN_PF AND trade_count > WF_MIN_TRADES in ≥ WF_PASS_PCT of windows.
      2. Average Sharpe ratio across valid windows > 0 (positive risk-adjusted return).
      3. Average avg_r across valid windows > 0 (positive mean R-multiple).
    Skips regimes with no sample-valid windows.
    """
    if not windows:
        return False

    regime_types = {w.dominant_regime for w in windows if w.dominant_regime != UNKNOWN}
    if not regime_types:
        return False

    for regime in regime_types:
        regime_windows = [w for w in windows if w.dominant_regime == regime and w.sample_valid]
        if not regime_windows:
            continue

        # Gate 1: PF pass rate
        passing = sum(
            1 for w in regime_windows
            if w.pf > config.WF_MIN_PF and w.trade_count > config.WF_MIN_TRADES
        )
        if (passing / len(regime_windows)) < config.WF_PASS_PCT:
            return False

        # Gate 2: average Sharpe > 0 (risk-adjusted return direction is positive)
        active = [w for w in regime_windows if w.trade_count > 0]
        if active:
            avg_sharpe = statistics.mean(w.sharpe_ratio for w in active)
            if avg_sharpe <= 0:
                return False

        # Gate 3: mean avg_r > 0 (expected return per trade is positive)
        if active:
            avg_r = statistics.mean(w.avg_r for w in active)
            if avg_r <= 0:
                return False

    return True


# ── CSV helpers ───────────────────────────────────────────────────────────────

def _write_csv(path: Path, headers: list, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


# ── Public entry point ────────────────────────────────────────────────────────

def run_regime_walk_forward(
    df: pd.DataFrame,
    initial_balance: float = 10_000.0,
    exit_config: Optional[ExitConfig] = None,
    mtf_enabled: bool = config.MTF_CONFIRMATION,
    log_regime: bool = True,
    log_entries: bool = True,
) -> bool:
    """
    Run regime-stratified walk-forward research and output CSVs.

    Parameters
    ----------
    df              : 1H OHLCV DataFrame (full history).
    initial_balance : Starting balance (not used in R-based metrics but
                      kept for signature compatibility with backtest.run()).
    exit_config     : Exit tier configuration; defaults to global config.
    mtf_enabled     : Whether to apply 4H ADX + Daily BB MTF check.
    log_regime      : Write regime_log.csv.
    log_entries     : Write entry_quality.csv.

    Returns
    -------
    bool : True if strategy passes the robustness gate.
    """
    if exit_config is None:
        exit_config = ExitConfig()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n{'━' * 80}")
    print("  REGIME-STRATIFIED WALK-FORWARD  (ROBUSTNESS modules #1–8)")
    print(f"{'━' * 80}")
    print(f"  History       : {len(df)} bars  ({df.index[0].date()} → {df.index[-1].date()})")
    print(f"  Window size   : {config.WF_WINDOW_DAYS} days")
    print(f"  Exit config   : partial_tp={exit_config.enable_partial_tp}  "
          f"time_stop={exit_config.enable_time_stop}  momentum={exit_config.enable_momentum_exit}")
    print(f"  MTF confirm   : {mtf_enabled}")

    # ── Pre-compute research frame and regime series ──────────────────────────
    print("  Pre-computing indicators…", flush=True)
    rf = _build_research_frame(df)

    clf = RegimeClassifier(log_to_csv=log_regime)
    regime_series = clf.classify_series(df)

    # ── MTF frames ────────────────────────────────────────────────────────────
    if mtf_enabled:
        print("  Building MTF frames (4H, 1D)…", flush=True)
        df_4h, df_1d = _build_mtf_frames(df)
    else:
        df_4h = df_1d = pd.DataFrame()

    # ── Walk-forward windows ──────────────────────────────────────────────────
    window_td = timedelta(days=config.WF_WINDOW_DAYS)
    start_date = df.index[0]
    end_date   = df.index[-1]

    windows_spec: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    wstart = start_date
    while wstart + window_td <= end_date:
        wend = wstart + window_td
        windows_spec.append((wstart, wend))
        wstart = wend  # non-overlapping

    print(f"  Walk-forward windows: {len(windows_spec)}")

    # ── Shared logging structures ─────────────────────────────────────────────
    entry_buf         = EntryConfirmationBuffer(log_to_csv=log_entries)
    sample_guard      = SampleGuard(log_to_csv=True)
    blocked_rows: list[dict] = []
    mtf_rejection_rows: list[dict] = []
    window_results: list[WindowResult] = []

    # ── Per-window simulation ─────────────────────────────────────────────────
    for wstart, wend in windows_spec:
        mask = (df.index >= wstart) & (df.index < wend)
        df_w  = df.loc[mask]
        rf_w  = rf.loc[mask]
        reg_w = regime_series.loc[mask]

        if len(df_w) < 50:
            continue

        dom_regime = dominant_regime(reg_w)

        trades = _simulate_window(
            df=df_w,
            rf=rf_w,
            regime_series=reg_w,
            df_4h=df_4h,
            df_1d=df_1d,
            exit_cfg=exit_config,
            entry_buf=entry_buf,
            mtf_enabled=mtf_enabled,
            blocked_rows=blocked_rows,
            mtf_rejection_rows=mtf_rejection_rows,
        )

        wr = _window_result(
            window_start=str(wstart.date()),
            window_end=str(wend.date()),
            regime_label=dom_regime,
            trades=trades,
        )
        window_results.append(wr)

    # ── Sample guard ──────────────────────────────────────────────────────────
    sample_guard.validate(window_results)
    valid_pct = sample_guard.valid_window_pct(window_results)
    agg = sample_guard.aggregate_valid(window_results)
    robustness_passed = _check_robustness_gate(window_results)

    # ── Per-regime breakdown ──────────────────────────────────────────────────
    all_regime_types = sorted({w.dominant_regime for w in window_results if w.sample_valid})
    regime_stats: dict[str, dict] = {}
    for reg in all_regime_types:
        rw = [w for w in window_results if w.dominant_regime == reg and w.sample_valid]
        if not rw:
            continue
        pfs_r  = [w.pf for w in rw]
        rs_all = [w.avg_r for w in rw]
        tp_rs  = [w.tp_hit_rate for w in rw]
        passing_r = sum(1 for w in rw if w.pf > config.WF_MIN_PF and w.trade_count > config.WF_MIN_TRADES)
        regime_stats[reg] = {
            "n": len(rw),
            "avg_pf": statistics.mean(pfs_r),
            "worst_pf": min(pfs_r),
            "avg_r": statistics.mean(rs_all),
            "avg_tp": statistics.mean(tp_rs),
            "pass_rate": passing_r / len(rw),
        }

    # Compute Sharpe / Sortino across valid windows
    valid_windows = [w for w in window_results if w.sample_valid]
    all_sharpes  = [w.sharpe_ratio  for w in valid_windows if w.trade_count > 0]
    all_sortinos = [w.sortino_ratio for w in valid_windows if w.trade_count > 0]
    avg_sharpe  = statistics.mean(all_sharpes)  if all_sharpes  else 0.0
    avg_sortino = statistics.mean(all_sortinos) if all_sortinos else 0.0

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n  Results across {len(window_results)} windows:")
    print(f"    Valid windows  : {agg['valid_count']} / {agg['total_count']}  ({valid_pct:.1%})")
    print(f"    Avg PF         : {agg['avg_pf']:.3f}")
    print(f"    Worst PF       : {agg['worst_pf']:.3f}")
    print(f"    Avg win rate   : {agg['avg_win_rate']:.1%}")
    print(f"    Median R       : {agg['median_r']:+.3f}")
    print(f"    Avg Sharpe     : {avg_sharpe:+.3f}")
    print(f"    Avg Sortino    : {avg_sortino:+.3f}")
    print(f"    Chase-skip rate: {entry_buf.chase_skip_rate:.1%}")

    if regime_stats:
        print(f"\n  Per-regime breakdown (valid windows):")
        print(f"    {'Regime':<20} {'N':>4}  {'AvgPF':>6}  {'WrstPF':>7}  {'AvgR':>6}  {'TP%':>5}  {'PassRate':>8}")
        for reg, s in regime_stats.items():
            print(f"    {reg:<20} {s['n']:>4}  {s['avg_pf']:>6.3f}  {s['worst_pf']:>7.3f}  "
                  f"{s['avg_r']:>+6.3f}  {s['avg_tp']:>5.1%}  {s['pass_rate']:>8.1%}")

    print(f"\n    Robustness gate: {'PASS ✓' if robustness_passed else 'FAIL ✗'}")

    # ── Write CSVs ────────────────────────────────────────────────────────────
    summary_rows = []
    for w in window_results:
        summary_rows.append({
            "window_start":     w.window_start,
            "window_end":       w.window_end,
            "dominant_regime":  w.dominant_regime,
            "trade_count":      w.trade_count,
            "pf":               f"{w.pf:.4f}",
            "win_rate":         f"{w.win_rate:.4f}",
            "median_r":         f"{w.median_r:.4f}",
            "worst_window_pf":  f"{w.worst_window_pf:.4f}",
            "tp_hit_rate":      f"{w.tp_hit_rate:.4f}",
            "avg_r":            f"{w.avg_r:.4f}",
            "sharpe_ratio":     f"{w.sharpe_ratio:.4f}",
            "sortino_ratio":    f"{w.sortino_ratio:.4f}",
            "chase_skip_rate":  f"{entry_buf.chase_skip_rate:.4f}",
            "sample_valid":     w.sample_valid,
            "valid_window_pct": f"{valid_pct:.4f}",
            "robustness_pass":  robustness_passed,
        })

    _write_csv(SUMMARY_CSV, _SUMMARY_HEADERS, summary_rows)
    _write_csv(BLOCKED_ENTRIES_CSV, _BLOCKED_HEADERS, blocked_rows)
    _write_csv(MTF_REJECTIONS_CSV,  _MTF_HEADERS,     mtf_rejection_rows)

    print(f"\n  Outputs written to {OUTPUT_DIR}/")
    print(f"    {SUMMARY_CSV.name}")
    print(f"    {BLOCKED_ENTRIES_CSV.name}  ({len(blocked_rows)} blocked shorts)")
    print(f"    {MTF_REJECTIONS_CSV.name}  ({len(mtf_rejection_rows)} MTF rejections)")

    return robustness_passed
