"""
backtest.py — Historical simulation with full strategy stack.

Run via:  python main.py --backtest

Features:
  - Fetches historical klines from Binance (public endpoint)
  - Simulates slippage (0.1%), maker fees (0.1%), entry at next candle open
  - Walk-forward validation: 70 % in-sample / 30 % out-of-sample
  - Metrics: total trades, win rate, avg win/loss %, profit factor,
             Sharpe ratio (annualised), max drawdown
  - ASCII equity curve printed to terminal
  - equity.csv written to bot directory
"""
from __future__ import annotations

import csv
import math
import statistics
from dataclasses import dataclass, field  # noqa: F401 — field used in SimTrade
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
from binance.client import Client

import config
import consensus as con
import regime as reg
import risk
import logger
import strategies.vwap as strat_vwap
import monte_carlo as mc

EQUITY_CSV = Path(__file__).parent / "equity.csv"
SKIPPED_SETUPS_CSV = Path(__file__).parent / "skipped_setups.csv"
SETUP_TRADES_CSV = Path(__file__).parent / "setup_trades.csv"
WALK_FORWARD_WINDOWS_CSV = Path(__file__).parent / "walk_forward_windows.csv"
WALK_FORWARD_SUMMARY_CSV = Path(__file__).parent / "walk_forward_summary.csv"
RANGE_EXIT_WINDOWS_CSV = Path(__file__).parent / "range_mean_reversion_exit_windows.csv"
RANGE_EXIT_SUMMARY_CSV = Path(__file__).parent / "range_mean_reversion_exit_summary.csv"
ENTRY_PROFILE_SKIPS_CSV = Path(__file__).parent / "skipped_entry_profiles.csv"
ENTRY_PROFILE_TRADES_CSV = Path(__file__).parent / "entry_profile_trades.csv"
ENTRY_PROFILE_WINDOWS_CSV = Path(__file__).parent / "entry_profile_windows.csv"
ENTRY_PROFILE_SUMMARY_CSV = Path(__file__).parent / "entry_profile_summary.csv"
ENTRY_ALT_HORIZON_SKIPS_CSV = Path(__file__).parent / "entry_alt_horizon_skips.csv"
ENTRY_ALT_HORIZON_TRADES_CSV = Path(__file__).parent / "entry_alt_horizon_trades.csv"
ENTRY_ALT_HORIZON_WINDOWS_CSV = Path(__file__).parent / "entry_alt_horizon_windows.csv"
ENTRY_ALT_HORIZON_SUMMARY_CSV = Path(__file__).parent / "entry_alt_horizon_summary.csv"
ENTRY_TARGET_TRADES_CSV = Path(__file__).parent / "entry_target_diagnostics_trades.csv"
ENTRY_TARGET_WINDOWS_CSV = Path(__file__).parent / "entry_target_diagnostics_windows.csv"
ENTRY_TARGET_SUMMARY_CSV = Path(__file__).parent / "entry_target_diagnostics_summary.csv"
ENTRY_FAILURE_DIAGNOSTICS_CSV = Path(__file__).parent / "entry_failure_diagnostics.csv"
ENTRY_DIAGNOSIS_SUMMARY_CSV = Path(__file__).parent / "entry_diagnosis_summary.csv"
ENTRY_DIAGNOSIS_WINDOWS_CSV = Path(__file__).parent / "entry_diagnosis_windows.csv"
ENTRY_DIAGNOSIS_REPORT_MD = Path(__file__).parent / "entry_diagnosis_report.md"
RANGE_MR_RESEARCH_SKIPS_CSV = Path(__file__).parent / "range_mr_research_skips.csv"
RANGE_MR_RESEARCH_TRADES_CSV = Path(__file__).parent / "range_mr_research_trades.csv"
RANGE_MR_RESEARCH_WINDOWS_CSV = Path(__file__).parent / "range_mr_research_windows.csv"
RANGE_MR_RESEARCH_SUMMARY_CSV = Path(__file__).parent / "range_mr_research_summary.csv"
RANGE_MR_RESEARCH_REGIME_SUMMARY_CSV = Path(__file__).parent / "range_mr_research_regime_summary.csv"
RANGE_MR_RESEARCH_REPORT_MD = Path(__file__).parent / "range_mr_research_report.md"

_EQUITY_HEADERS = [
    "timestamp", "trade_num", "direction", "entry", "exit",
    "pnl_net", "balance", "drawdown_pct",
]


# ── Data fetching ─────────────────────────────────────────────────────────────

def _klines_to_df(klines: list) -> pd.DataFrame:
    df = pd.DataFrame(klines, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)
    return df[["open", "high", "low", "close", "volume"]]


def fetch_historical(client: Client, days: int) -> pd.DataFrame:
    """Fetch *days* of 1H klines from Binance (public endpoint)."""
    logger.log_info(f"Fetching {days} days of 1H klines for {config.SYMBOL}…")

    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=days)

    start_str = start_dt.strftime("%d %b %Y %H:%M:%S")
    end_str = end_dt.strftime("%d %b %Y %H:%M:%S")

    klines = client.get_historical_klines(
        symbol=config.SYMBOL,
        interval=Client.KLINE_INTERVAL_1HOUR,
        start_str=start_str,
        end_str=end_str,
    )

    df = _klines_to_df(klines)
    logger.log_info(f"Fetched {len(df)} candles ({df.index[0]} → {df.index[-1]})")
    return df


def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    Research-only horizon helper.

    Resample an OHLCV frame to a higher timeframe while dropping incomplete bars.
    This is used only by backtest research and never in live execution.
    """
    factor = int(rule[:-1]) if rule.endswith("h") else 1
    counts = df["close"].resample(rule, label="right", closed="right").count()
    resampled = pd.DataFrame({
        "open": df["open"].resample(rule, label="right", closed="right").first(),
        "high": df["high"].resample(rule, label="right", closed="right").max(),
        "low": df["low"].resample(rule, label="right", closed="right").min(),
        "close": df["close"].resample(rule, label="right", closed="right").last(),
        "volume": df["volume"].resample(rule, label="right", closed="right").sum(),
    }).dropna()
    if factor > 1:
        resampled = resampled.loc[counts == factor]
    return resampled


# ── Single-trade simulation ───────────────────────────────────────────────────

@dataclass
class SimTrade:
    trade_num: int
    entry_time: pd.Timestamp
    entry_price: float
    stop_price: float            # initial hard stop (reference only)
    tp_price: float              # kept for CSV/reference
    size: float
    direction: str = "LONG"
    signals_fired: list = field(default_factory=list)
    # R-unit tracking
    atr_at_entry: float = 0.0
    stop_distance: float = 0.0   # R unit = ATR × ATR_STOP_MULTIPLIER
    # Dynamic exit state
    remaining_size: float = 0.0
    current_stop: float = 0.0    # dynamic stop — only ever moves in favour
    tp_hit: bool = False         # True after partial TP fires
    highest_close: float = 0.0
    lowest_close: float = float("inf")
    partial_pnl: float = 0.0
    # Regime-modulated parameters
    stage_b_r: float = 0.8       # profit-R threshold to activate Stage-B trail
    stage_b_atr_mult: float = 1.2
    partial_tp_r: float = 1.0    # profit-R target for partial TP
    stage_c_floor_offset_r: float = 0.1
    # Trade management counters
    candles_in_trade: int = 0
    trail_delay: int = 0         # winner-extension delay (candles)
    # Results
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    pnl_net: Optional[float] = None
    result: Optional[str] = None
    exit_reason: str = ""        # STOP | STALL | TIME | TRAIL | FORCE_CLOSE
    exit_r: float = 0.0          # net PnL in R-multiples
    # Path tracking (updated each candle; used by forensics)
    mfe_price: float = 0.0       # max high (LONG) / min low (SHORT) seen
    mae_price: float = 0.0       # min low (LONG) / max high (SHORT) seen
    candle_at_mfe: int = 0       # candles_in_trade when MFE was set
    candle_at_mae: int = 0       # candles_in_trade when MAE was set
    reached_0_5r: bool = False
    reached_1_0r: bool = False
    reached_1_5r: bool = False
    reached_2_0r: bool = False
    candles_to_0_5r: int = -1    # -1 = never reached
    candles_to_1_0r: int = -1
    candles_to_1_5r: int = -1
    candle_first_neg_1r: int = -1  # first candle MAE exceeded 1R adverse
    # Research-only tags and diagnostics (backtest/reporting only)
    setup_name: str = ""
    entry_profile_name: str = ""
    signal_time: Optional[pd.Timestamp] = None
    signal_tags: list[str] = field(default_factory=list)
    entry_profile_tags: list[str] = field(default_factory=list)
    atr_bucket: str = ""
    volume_bucket: str = ""
    vwap_distance_bucket: str = ""
    pre_entry_move_3c_bucket: str = ""
    pre_entry_move_6c_bucket: str = ""
    entry_close_price: float = 0.0
    entry_vwap: float = 0.0
    entry_atr_pct: float = 0.0
    entry_volume_ratio: float = 0.0
    entry_macd_histogram: float = 0.0
    entry_macd_slope: float = 0.0
    entry_trend: str = ""
    entry_vol_regime: str = ""
    breakout_level: float = math.nan
    macd_agrees: bool = False
    price_vs_vwap: str = ""
    trend_context_reason: str = ""
    pullback_detected: bool = False
    reclaim_detected: bool = False
    pullback_depth_r: float = math.nan
    candles_since_impulse: int = -1
    range_high: float = math.nan
    range_low: float = math.nan
    range_mid: float = math.nan
    distance_from_range_boundary_r: float = math.nan
    rejection_detected: bool = False
    candles_outside_range: int = -1
    touched_vwap_after_entry: bool = False
    candles_to_vwap_touch: int = -1
    vwap_touch_r: float = math.nan
    touched_range_mid_after_entry: bool = False
    candles_to_range_mid_touch: int = -1
    range_mid_touch_r: float = math.nan
    max_continuation_away_from_vwap_before_reversion_r: float = 0.0
    research_exit_code: str = ""
    research_exit_progress: int = 0


@dataclass(frozen=True)
class ResearchBaseline:
    name: str
    status: str
    notes: str
    config_snapshot: dict[str, object]


RESEARCH_BASELINE = ResearchBaseline(
    name="MACD_VWAP_F2_BASELINE",
    status="FAILED_730_DAY_VALIDATION",
    notes=(
        "Research baseline only. Do not treat as live-ready; "
        "730-day validation failed."
    ),
    config_snapshot={
        "STALL_EXIT_ENABLED": False,
        "DEFAULT_PARTIAL_TP_R": 1.5,
        "STAGE_C_ATR_MULT": 1.5,
        "STOCHASTIC_WEIGHT": 0.0,
        "REQUIRE_MACD": True,
        "VWAP_VOLUME_STOCH_ONLY_BLOCKED": True,
    },
)


@dataclass(frozen=True)
class SetupDecision:
    allowed: bool
    setup_name: str
    rejection_reason: str = ""
    signal_tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResearchSetup:
    name: str
    candidate_family: str
    evaluator: Callable[["CandidateContext"], SetupDecision]


@dataclass(frozen=True)
class EntryDecision:
    allowed: bool
    entry_profile_name: str
    rejection_reason: str = ""
    entry_tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResearchEntryProfile:
    name: str
    description: str
    evaluator: Callable[["CandidateContext", ResearchSetup, SetupDecision], EntryDecision]


@dataclass(frozen=True)
class MeanReversionExitSpec:
    code: str
    description: str


@dataclass(frozen=True)
class EntryTargetSpec:
    code: str
    description: str


@dataclass(frozen=True)
class RegimeThresholds:
    atr_q1: float
    atr_q2: float
    volume_q1: float
    volume_q2: float
    vwap_q1: float
    vwap_q2: float
    move3_q1: float
    move3_q2: float
    move6_q1: float
    move6_q2: float

    def atr_bucket(self, value: float) -> str:
        return _bucket_tertile(value, self.atr_q1, self.atr_q2)

    def volume_bucket(self, value: float) -> str:
        return _bucket_tertile(value, self.volume_q1, self.volume_q2)

    def vwap_bucket(self, value_abs_r: float) -> str:
        if value_abs_r <= self.vwap_q1:
            return "close"
        if value_abs_r <= self.vwap_q2:
            return "medium"
        return "far"

    def move3_bucket(self, value: float) -> str:
        return _bucket_tertile(value, self.move3_q1, self.move3_q2)

    def move6_bucket(self, value: float) -> str:
        return _bucket_tertile(value, self.move6_q1, self.move6_q2)


@dataclass(frozen=True)
class CandidateContext:
    signal_pos: int
    signal_time: pd.Timestamp
    entry_time: pd.Timestamp
    candidate_family: str
    direction: str
    close_price: float
    entry_price: float
    vwap: float
    atr_value: float
    atr_pct: float
    volume_ratio: float
    distance_from_vwap_r: float
    price_move_3r: float
    price_move_6r: float
    macd_histogram: float
    macd_slope: float
    trend: str
    vol_regime: str
    halve_position: bool
    signals_fired: tuple[str, ...]
    signal_tags: tuple[str, ...]
    atr_bucket: str = ""
    volume_bucket: str = ""
    vwap_distance_bucket: str = ""
    pre_entry_move_3c_bucket: str = ""
    pre_entry_move_6c_bucket: str = ""
    breakout_level: float = math.nan
    prior_range_high: float = math.nan
    prior_range_low: float = math.nan
    range_width: float = math.nan
    breakout_distance_r: float = math.nan
    breakout_retest_confirmed: bool = False
    macd_agrees: bool = False
    price_vs_vwap: str = ""
    trend_context_reason: str = ""
    pullback_detected: bool = False
    reclaim_detected: bool = False
    pullback_depth_r: float = math.nan
    candles_since_impulse: int = -1
    range_high: float = math.nan
    range_low: float = math.nan
    range_mid: float = math.nan
    distance_from_range_boundary_r: float = math.nan
    rejection_detected: bool = False
    candles_outside_range: int = -1

    @property
    def signals_set(self) -> set[str]:
        return set(self.signals_fired)


def _warmup_bars() -> int:
    return max(
        202,
        config.ADX_PERIOD,
        config.ATR_PERIOD,
        config.MACD_SLOW + config.MACD_SIGNAL,
        config.BB_PERIOD,
        config.VOLUME_MA_PERIOD,
        config.EMA_SLOW,
        config.RSI_PERIOD,
        config.STOCH_K + config.STOCH_D,
    ) + 5


def _bucket_tertile(value: float, q1: float, q2: float) -> str:
    if value <= q1:
        return "low"
    if value <= q2:
        return "medium"
    return "high"


RESEARCH_TRAIN_DAYS = 180
RESEARCH_TEST_DAYS = 60
RESEARCH_STEP_DAYS = 30
ENTRY_ALT_HORIZONS = ("1h", "2h", "4h")
ENTRY_ALT_SETUPS = ("VOLUME_BREAKOUT_CONTINUATION", "PULLBACK_TO_TREND_CONTINUATION")
ENTRY_ALT_PROFILES = (
    "ENTRY_BASELINE",
    "ENTRY_ANTI_CHASE_LONG_ONLY",
    "ENTRY_ANTI_CHASE_LONG_ONLY_6C_GUARD",
    "ENTRY_ANTI_CHASE_LONG_ONLY_BALANCED",
    "ENTRY_ANTI_CHASE_LONG_ONLY_RELAXED_VWAP",
    "ENTRY_ANTI_CHASE_LONG_ONLY_RELAXED_EXTENSION",
    "ENTRY_ANTI_CHASE_LONG_ONLY_ADAPTIVE",
)
ENTRY_TARGET_SETUPS = ("VOLUME_BREAKOUT_CONTINUATION",)
ENTRY_TARGET_PROFILES = (
    "ENTRY_BASELINE",
    "ENTRY_ANTI_CHASE_LONG_ONLY",
    "ENTRY_ANTI_CHASE_LONG_ONLY_6C_GUARD",
    "ENTRY_ANTI_CHASE_LONG_ONLY_BALANCED",
    "ENTRY_ANTI_CHASE_LONG_ONLY_RELAXED_VWAP",
    "ENTRY_ANTI_CHASE_LONG_ONLY_RELAXED_EXTENSION",
    "ENTRY_ANTI_CHASE_LONG_ONLY_ADAPTIVE",
)
ENTRY_TARGET_SPECS = (
    EntryTargetSpec("CONT_TARGET_BASELINE", "Current F2 baseline exit."),
    EntryTargetSpec("CONT_TARGET_1_0R", "Full exit at 1.0R if reached, else baseline stop/time logic."),
    EntryTargetSpec("CONT_TARGET_0_75R", "Full exit at 0.75R if reached, else baseline stop/time logic."),
    EntryTargetSpec("CONT_TARGET_VWAP_TOUCH", "Full exit at first post-entry VWAP touch if reached, else baseline stop/time logic."),
    EntryTargetSpec("CONT_TARGET_RANGE_MID", "Full exit at prior-range midpoint touch if reached, else baseline stop/time logic."),
)


def _score_profit_factor(rs: list[float]) -> float:
    gains = sum(r for r in rs if r > 0)
    losses = abs(sum(r for r in rs if r < 0))
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return gains / losses


def _compute_research_frame(df: pd.DataFrame) -> pd.DataFrame:
    rf = pd.DataFrame(index=df.index)
    rf["open"] = df["open"]
    rf["high"] = df["high"]
    rf["low"] = df["low"]
    rf["close"] = df["close"]
    rf["volume"] = df["volume"]

    adx_df = df.ta.adx(length=config.ADX_PERIOD)
    adx_col = f"ADX_{config.ADX_PERIOD}"
    rf["adx"] = adx_df[adx_col] if adx_df is not None and adx_col in adx_df.columns else np.nan

    rf["atr"] = df.ta.atr(length=config.ATR_PERIOD)
    rf["atr_pct"] = rf["atr"] / rf["close"] * 100.0

    rf["ema9"] = df.ta.ema(length=config.EMA_FAST)
    rf["ema_fast"] = df.ta.ema(length=config.EMA_FAST)
    rf["ema_slow"] = df.ta.ema(length=config.EMA_SLOW)
    rf["ema200"] = df.ta.ema(length=200)

    macd_df = df.ta.macd(
        fast=config.MACD_FAST,
        slow=config.MACD_SLOW,
        signal=config.MACD_SIGNAL,
    )
    macd_col = f"MACD_{config.MACD_FAST}_{config.MACD_SLOW}_{config.MACD_SIGNAL}"
    macd_sig_col = f"MACDs_{config.MACD_FAST}_{config.MACD_SLOW}_{config.MACD_SIGNAL}"
    macd_hist_col = f"MACDh_{config.MACD_FAST}_{config.MACD_SLOW}_{config.MACD_SIGNAL}"
    rf["macd_line"] = macd_df[macd_col] if macd_df is not None and macd_col in macd_df.columns else np.nan
    rf["macd_signal"] = macd_df[macd_sig_col] if macd_df is not None and macd_sig_col in macd_df.columns else np.nan
    rf["macd_hist"] = macd_df[macd_hist_col] if macd_df is not None and macd_hist_col in macd_df.columns else np.nan
    rf["macd_hist_slope"] = rf["macd_hist"].diff()

    rf["rsi"] = df.ta.rsi(length=config.RSI_PERIOD)
    rf["rsi7"] = df.ta.rsi(length=7)

    stoch_df = df.ta.stoch(
        k=config.STOCH_K,
        d=config.STOCH_D,
        smooth_k=config.STOCH_SMOOTH_K,
    )
    stoch_k_col = f"STOCHk_{config.STOCH_K}_{config.STOCH_D}_{config.STOCH_SMOOTH_K}"
    stoch_d_col = f"STOCHd_{config.STOCH_K}_{config.STOCH_D}_{config.STOCH_SMOOTH_K}"
    rf["stoch_k"] = stoch_df[stoch_k_col] if stoch_df is not None and stoch_k_col in stoch_df.columns else np.nan
    rf["stoch_d"] = stoch_df[stoch_d_col] if stoch_df is not None and stoch_d_col in stoch_df.columns else np.nan

    bb_df = df.ta.bbands(length=config.BB_PERIOD, std=config.BB_STD)
    if bb_df is not None and not bb_df.empty:
        lower_col = next((c for c in bb_df.columns if c.startswith(f"BBL_{config.BB_PERIOD}_")), None)
        upper_col = next((c for c in bb_df.columns if c.startswith(f"BBU_{config.BB_PERIOD}_")), None)
        rf["bb_lower"] = bb_df[lower_col] if lower_col else np.nan
        rf["bb_upper"] = bb_df[upper_col] if upper_col else np.nan
    else:
        rf["bb_lower"] = np.nan
        rf["bb_upper"] = np.nan

    rf["vwap"] = strat_vwap._compute_vwap(df)
    rf["volume_ma"] = df["volume"].rolling(config.VOLUME_MA_PERIOD).mean()
    rf["volume_ratio"] = rf["volume"] / rf["volume_ma"]
    rf["body_pct"] = (rf["close"] - rf["open"]).abs() / rf["open"]

    rf["trend"] = np.where(rf["adx"] > config.ADX_TREND_THRESHOLD, reg.TRENDING, reg.RANGING)
    rf["vol_regime"] = np.where(
        rf["atr_pct"] > config.ATR_HIGH_VOL_THRESHOLD_PCT,
        reg.HIGH_VOLATILITY,
        reg.NORMAL,
    )
    rf["regime_allows"] = ~(
        (rf["trend"] == reg.RANGING) & (rf["vol_regime"] == reg.HIGH_VOLATILITY)
    )
    rf["halve_position"] = (
        (rf["trend"] == reg.TRENDING) & (rf["vol_regime"] == reg.HIGH_VOLATILITY)
    )

    rf["signal_ema"] = 0
    ema_long = (
        (rf["ema_fast"].shift(1) <= rf["ema_slow"].shift(1))
        & (rf["ema_fast"] > rf["ema_slow"])
        & (rf["close"] > rf["ema200"])
    )
    ema_short = (
        (rf["ema_fast"].shift(1) >= rf["ema_slow"].shift(1))
        & (rf["ema_fast"] < rf["ema_slow"])
        & (rf["close"] < rf["ema200"])
    )
    rf.loc[ema_long, "signal_ema"] = 1
    rf.loc[ema_short, "signal_ema"] = -1

    rf["signal_macd"] = 0
    macd_long = (
        (rf["macd_line"].shift(1) <= rf["macd_signal"].shift(1))
        & (rf["macd_line"] > rf["macd_signal"])
        & (rf["close"] > rf["ema200"])
    )
    macd_short = (
        (rf["macd_line"].shift(1) >= rf["macd_signal"].shift(1))
        & (rf["macd_line"] < rf["macd_signal"])
        & (rf["close"] < rf["ema200"])
    )
    rf.loc[macd_long, "signal_macd"] = 1
    rf.loc[macd_short, "signal_macd"] = -1

    rf["signal_rsi"] = 0
    rsi_long = (rf["rsi"] < 30) & (rf["rsi"] > rf["rsi"].shift(1)) & (rf["close"] > rf["ema200"])
    rsi_short = (rf["rsi"] > 70) & (rf["rsi"] < rf["rsi"].shift(1)) & (rf["close"] < rf["ema200"])
    rf.loc[rsi_long, "signal_rsi"] = 1
    rf.loc[rsi_short, "signal_rsi"] = -1

    rf["signal_stoch"] = 0
    stoch_long = (
        (rf["stoch_k"].shift(1) <= rf["stoch_d"].shift(1))
        & (rf["stoch_k"] > rf["stoch_d"])
        & (rf["stoch_k"] < 25)
        & (rf["close"] > rf["ema200"])
    )
    stoch_short = (
        (rf["stoch_k"].shift(1) >= rf["stoch_d"].shift(1))
        & (rf["stoch_k"] < rf["stoch_d"])
        & (rf["stoch_k"] > 75)
        & (rf["close"] < rf["ema200"])
    )
    rf.loc[stoch_long, "signal_stoch"] = 1
    rf.loc[stoch_short, "signal_stoch"] = -1

    rf["signal_bollinger"] = 0
    boll_long = (rf["close"] < rf["bb_lower"]) & (rf["close"] > rf["close"].shift(1))
    boll_short = (rf["close"] > rf["bb_upper"]) & (rf["close"] < rf["close"].shift(1))
    rf.loc[boll_long, "signal_bollinger"] = 1
    rf.loc[boll_short, "signal_bollinger"] = -1

    rf["signal_volume"] = 0
    volume_spike = (rf["volume"] > 2 * rf["volume_ma"]) & (rf["body_pct"] > 0.003)
    rf.loc[volume_spike & (rf["close"] > rf["open"]), "signal_volume"] = 1
    rf.loc[volume_spike & (rf["close"] < rf["open"]), "signal_volume"] = -1

    rf["signal_vwap"] = 0
    vwap_long = (rf["close"] > rf["vwap"]) & (rf["vwap"] > rf["vwap"].shift(1)) & (rf["close"] > rf["ema200"])
    vwap_short = (rf["close"] < rf["vwap"]) & (rf["vwap"] < rf["vwap"].shift(1)) & (rf["close"] < rf["ema200"])
    rf.loc[vwap_long, "signal_vwap"] = 1
    rf.loc[vwap_short, "signal_vwap"] = -1

    rf["breakout_prior_high"] = df["high"].rolling(config.RESEARCH_BREAKOUT_LOOKBACK).max().shift(1)
    rf["breakout_prior_low"] = df["low"].rolling(config.RESEARCH_BREAKOUT_LOOKBACK).min().shift(1)
    rf["range_prior_high"] = df["high"].rolling(config.RESEARCH_RANGE_LOOKBACK).max().shift(1)
    rf["range_prior_low"] = df["low"].rolling(config.RESEARCH_RANGE_LOOKBACK).min().shift(1)

    return rf


def _active_signal_names(row: pd.Series) -> tuple[str, ...]:
    if row["trend"] == reg.TRENDING:
        ordered = [
            ("EMA Cross", row["signal_ema"]),
            ("MACD", row["signal_macd"]),
            ("Bollinger", row["signal_bollinger"]),
            ("Volume", row["signal_volume"]),
            ("VWAP", row["signal_vwap"]),
        ]
    else:
        ordered = [
            ("RSI", row["signal_rsi"]),
            ("Stochastic", row["signal_stoch"]),
            ("Bollinger", row["signal_bollinger"]),
            ("Volume", row["signal_volume"]),
            ("VWAP", row["signal_vwap"]),
        ]
    return tuple(name for name, sig in ordered if sig != 0)


def _candidate_score_and_decision(row: pd.Series) -> tuple[float, float, str]:
    if not bool(row["regime_allows"]):
        return 0.0, 0.0, con.HOLD

    if row["trend"] == reg.TRENDING:
        score = (
            2.5 * float(row["signal_ema"])
            + 2.5 * float(row["signal_macd"])
            + 1.0 * float(row["signal_bollinger"])
            + 1.0 * float(row["signal_volume"])
            + 1.0 * float(row["signal_vwap"])
        )
        max_possible = 8.0
    else:
        score = (
            1.0 * float(row["signal_rsi"])
            + 0.0 * float(row["signal_stoch"])
            + 1.0 * float(row["signal_bollinger"])
            + 1.0 * float(row["signal_volume"])
            + 1.0 * float(row["signal_vwap"])
        )
        max_possible = 4.0

    ratio = score / max_possible if max_possible else 0.0
    if ratio >= config.CONSENSUS_THRESHOLD:
        return score, ratio, con.BUY
    if ratio <= -config.CONSENSUS_THRESHOLD:
        return score, ratio, con.SELL
    return score, ratio, con.HOLD


def _price_vs_vwap(close_price: float, vwap_value: float) -> str:
    if not math.isfinite(vwap_value):
        return ""
    if close_price > vwap_value:
        return "above"
    if close_price < vwap_value:
        return "below"
    return "at"


def _empty_research_context_fields() -> dict[str, object]:
    return {
        "trend_context_reason": "",
        "pullback_detected": False,
        "reclaim_detected": False,
        "pullback_depth_r": math.nan,
        "candles_since_impulse": -1,
        "range_high": math.nan,
        "range_low": math.nan,
        "range_mid": math.nan,
        "distance_from_range_boundary_r": math.nan,
        "rejection_detected": False,
        "candles_outside_range": -1,
    }


def _bucket_rank(bucket: str) -> int:
    order = {
        "close": 0,
        "low": 0,
        "medium": 1,
        "high": 2,
        "far": 2,
    }
    return order.get(bucket, -1)


def _count_consecutive_outside_range(
    df: pd.DataFrame,
    signal_pos: int,
    *,
    direction: str,
    boundary: float,
    lookback: int,
) -> int:
    count = 0
    start = max(0, signal_pos - lookback + 1)
    for pos in range(signal_pos, start - 1, -1):
        if direction == "LONG":
            outside = float(df["low"].iloc[pos]) < boundary
        else:
            outside = float(df["high"].iloc[pos]) > boundary
        if not outside:
            break
        count += 1
    return count


def _build_consensus_candidate_universe(df: pd.DataFrame, rf: pd.DataFrame) -> pd.DataFrame:
    warmup = _warmup_bars()
    rows: list[dict] = []

    for i in range(warmup, len(df) - 1):
        row = rf.iloc[i]
        score, ratio, decision = _candidate_score_and_decision(row)
        if decision not in (con.BUY, con.SELL):
            continue

        curr_close = float(df["close"].iloc[i])
        ema9_val = float(rf["ema9"].iloc[i])
        atr_val = float(rf["atr"].iloc[i])
        vwap_val = float(rf["vwap"].iloc[i])
        volume_ratio = float(rf["volume_ratio"].iloc[i])
        atr_pct = float(rf["atr_pct"].iloc[i])
        macd_hist = float(rf["macd_hist"].iloc[i])
        macd_slope = float(rf["macd_hist_slope"].iloc[i])

        needed = [curr_close, ema9_val, atr_val, vwap_val, volume_ratio, atr_pct]
        if any(not math.isfinite(v) for v in needed) or curr_close <= 0 or atr_val <= 0:
            continue
        if abs(curr_close - ema9_val) / curr_close >= 0.01:
            continue

        direction = "LONG" if decision == con.BUY else "SHORT"
        stop_distance = round(atr_val * config.ATR_STOP_MULTIPLIER, 2)
        if stop_distance <= 0:
            continue

        prev3 = float(df["close"].iloc[i - 3])
        prev6 = float(df["close"].iloc[i - 6])
        if direction == "LONG":
            move3_r = (curr_close - prev3) / stop_distance
            move6_r = (curr_close - prev6) / stop_distance
            dist_vwap_r = (curr_close - vwap_val) / stop_distance
            entry_price = float(df["open"].iloc[i + 1]) * (1.0 + config.SLIPPAGE)
        else:
            move3_r = (prev3 - curr_close) / stop_distance
            move6_r = (prev6 - curr_close) / stop_distance
            dist_vwap_r = (vwap_val - curr_close) / stop_distance
            entry_price = float(df["open"].iloc[i + 1]) * (1.0 - config.SLIPPAGE)

        macd_signal = float(row["signal_macd"])
        macd_agrees = bool(
            math.isfinite(macd_signal)
            and ((direction == "LONG" and macd_signal == 1.0) or (direction == "SHORT" and macd_signal == -1.0))
        )
        price_vs_vwap = _price_vs_vwap(curr_close, vwap_val)
        active_signals = _active_signal_names(row)

        rows.append({
            "signal_pos": i,
            "signal_time": df.index[i],
            "entry_time": df.index[i + 1],
            "candidate_family": "consensus",
            "direction": direction,
            "signals_fired": active_signals,
            "signal_tags": active_signals,
            "close_price": curr_close,
            "entry_price": entry_price,
            "vwap": vwap_val,
            "atr_value": atr_val,
            "atr_pct": atr_pct,
            "volume_ratio": volume_ratio,
            "distance_from_vwap_r": dist_vwap_r,
            "price_move_3r": move3_r,
            "price_move_6r": move6_r,
            "macd_histogram": macd_hist if math.isfinite(macd_hist) else 0.0,
            "macd_slope": macd_slope if math.isfinite(macd_slope) else 0.0,
            "trend": row["trend"],
            "vol_regime": row["vol_regime"],
            "halve_position": bool(row["halve_position"]),
            "score": score,
            "ratio": ratio,
            "breakout_level": math.nan,
            "prior_range_high": math.nan,
            "prior_range_low": math.nan,
            "range_width": math.nan,
            "breakout_distance_r": math.nan,
            "breakout_retest_confirmed": False,
            "macd_agrees": macd_agrees,
            "price_vs_vwap": price_vs_vwap,
            **_empty_research_context_fields(),
        })

    return pd.DataFrame(rows)


def _build_breakout_candidate_universe(df: pd.DataFrame, rf: pd.DataFrame) -> pd.DataFrame:
    warmup = max(_warmup_bars(), config.RESEARCH_BREAKOUT_LOOKBACK + 6)
    rows: list[dict] = []

    for i in range(warmup, len(df) - 1):
        row = rf.iloc[i]
        curr_close = float(df["close"].iloc[i])
        prev_close = float(df["close"].iloc[i - 1])
        atr_val = float(rf["atr"].iloc[i])
        vwap_val = float(rf["vwap"].iloc[i])
        volume_ratio = float(rf["volume_ratio"].iloc[i])
        atr_pct = float(rf["atr_pct"].iloc[i])
        prior_high = float(rf["breakout_prior_high"].iloc[i])
        prior_low = float(rf["breakout_prior_low"].iloc[i])
        macd_hist = float(rf["macd_hist"].iloc[i])
        macd_slope = float(rf["macd_hist_slope"].iloc[i])
        macd_signal = float(rf["signal_macd"].iloc[i])

        needed = [curr_close, atr_val, vwap_val, volume_ratio, atr_pct, prior_high, prior_low]
        if any(not math.isfinite(v) for v in needed) or curr_close <= 0 or atr_val <= 0:
            continue

        range_width = prior_high - prior_low
        if not math.isfinite(range_width) or range_width <= 0:
            continue

        long_breakout = curr_close > prior_high and prev_close <= prior_high
        short_breakout = curr_close < prior_low and prev_close >= prior_low
        if not long_breakout and not short_breakout:
            continue

        direction = "LONG" if long_breakout else "SHORT"
        breakout_level = prior_high if direction == "LONG" else prior_low
        stop_distance = round(atr_val * config.ATR_STOP_MULTIPLIER, 2)
        if stop_distance <= 0:
            continue

        prev3 = float(df["close"].iloc[i - 3])
        prev6 = float(df["close"].iloc[i - 6])
        price_vs_vwap = _price_vs_vwap(curr_close, vwap_val)

        if direction == "LONG":
            move3_r = (curr_close - prev3) / stop_distance
            move6_r = (curr_close - prev6) / stop_distance
            dist_vwap_r = (curr_close - vwap_val) / stop_distance
            breakout_distance_r = (curr_close - breakout_level) / stop_distance
            entry_price = float(df["open"].iloc[i + 1]) * (1.0 + config.SLIPPAGE)
            macd_agrees = math.isfinite(macd_signal) and macd_signal == 1.0
            breakout_retest_confirmed = float(df["low"].iloc[i]) <= breakout_level
        else:
            move3_r = (prev3 - curr_close) / stop_distance
            move6_r = (prev6 - curr_close) / stop_distance
            dist_vwap_r = (vwap_val - curr_close) / stop_distance
            breakout_distance_r = (breakout_level - curr_close) / stop_distance
            entry_price = float(df["open"].iloc[i + 1]) * (1.0 - config.SLIPPAGE)
            macd_agrees = math.isfinite(macd_signal) and macd_signal == -1.0
            breakout_retest_confirmed = float(df["high"].iloc[i]) >= breakout_level

        signal_tags = ["Breakout"]
        if volume_ratio >= config.RESEARCH_BREAKOUT_MIN_VOLUME_RATIO:
            signal_tags.append("VolumeExpansion")
        if macd_agrees:
            signal_tags.append("MACDAgree")
        if (direction == "LONG" and price_vs_vwap == "above") or (direction == "SHORT" and price_vs_vwap == "below"):
            signal_tags.append("VWAPAligned")

        rows.append({
            "signal_pos": i,
            "signal_time": df.index[i],
            "entry_time": df.index[i + 1],
            "candidate_family": "volume_breakout",
            "direction": direction,
            "signals_fired": ("Breakout",),
            "signal_tags": tuple(signal_tags),
            "close_price": curr_close,
            "entry_price": entry_price,
            "vwap": vwap_val,
            "atr_value": atr_val,
            "atr_pct": atr_pct,
            "volume_ratio": volume_ratio,
            "distance_from_vwap_r": dist_vwap_r,
            "price_move_3r": move3_r,
            "price_move_6r": move6_r,
            "macd_histogram": macd_hist if math.isfinite(macd_hist) else 0.0,
            "macd_slope": macd_slope if math.isfinite(macd_slope) else 0.0,
            "trend": row["trend"],
            "vol_regime": row["vol_regime"],
            "halve_position": bool(row["halve_position"]),
            "score": 0.0,
            "ratio": 0.0,
            "breakout_level": breakout_level,
            "prior_range_high": prior_high,
            "prior_range_low": prior_low,
            "range_width": range_width,
            "breakout_distance_r": breakout_distance_r,
            "breakout_retest_confirmed": breakout_retest_confirmed,
            "macd_agrees": macd_agrees,
            "price_vs_vwap": price_vs_vwap,
            **{
                **_empty_research_context_fields(),
                "range_high": prior_high,
                "range_low": prior_low,
                "range_mid": (prior_high + prior_low) / 2.0,
            },
        })

    return pd.DataFrame(rows)


def _structure_direction(df: pd.DataFrame, i: int) -> tuple[bool, bool]:
    recent_high = float(df["high"].iloc[i - 2: i + 1].max())
    prior_high = float(df["high"].iloc[i - 5: i - 2].max())
    recent_low = float(df["low"].iloc[i - 2: i + 1].min())
    prior_low = float(df["low"].iloc[i - 5: i - 2].min())
    structure_up = recent_high > prior_high and recent_low > prior_low
    structure_down = recent_high < prior_high and recent_low < prior_low
    return structure_up, structure_down


def _build_pullback_candidate_universe(df: pd.DataFrame, rf: pd.DataFrame) -> pd.DataFrame:
    warmup = max(
        _warmup_bars(),
        config.RESEARCH_PULLBACK_LOOKBACK + config.RESEARCH_RECLAIM_LOOKBACK + 6,
    )
    rows: list[dict] = []

    for i in range(warmup, len(df) - 1):
        curr_close = float(df["close"].iloc[i])
        curr_open = float(df["open"].iloc[i])
        atr_val = float(rf["atr"].iloc[i])
        atr_pct = float(rf["atr_pct"].iloc[i])
        vwap_val = float(rf["vwap"].iloc[i])
        volume_ratio = float(rf["volume_ratio"].iloc[i])
        ema_fast = float(rf["ema_fast"].iloc[i])
        ema_slow = float(rf["ema_slow"].iloc[i])
        ema200 = float(rf["ema200"].iloc[i])
        macd_hist = float(rf["macd_hist"].iloc[i])
        macd_slope = float(rf["macd_hist_slope"].iloc[i])
        macd_signal = float(rf["signal_macd"].iloc[i])

        needed = [curr_close, curr_open, atr_val, atr_pct, vwap_val, volume_ratio, ema_fast, ema_slow, ema200]
        if any(not math.isfinite(v) for v in needed) or curr_close <= 0 or atr_val <= 0:
            continue

        structure_up, structure_down = _structure_direction(df, i)
        ema_slow_prev = float(rf["ema_slow"].iloc[i - 3])
        ema_up = (
            math.isfinite(ema_slow_prev)
            and curr_close > ema200
            and ema_fast > ema_slow
            and ema_slow > ema_slow_prev
        )
        ema_down = (
            math.isfinite(ema_slow_prev)
            and curr_close < ema200
            and ema_fast < ema_slow
            and ema_slow < ema_slow_prev
        )

        long_trend = structure_up and (curr_close > vwap_val or ema_up)
        short_trend = structure_down and (curr_close < vwap_val or ema_down)
        if not long_trend and not short_trend:
            continue
        if long_trend and short_trend:
            continue

        direction = "LONG" if long_trend else "SHORT"
        stop_distance = round(atr_val * config.ATR_STOP_MULTIPLIER, 2)
        if stop_distance <= 0:
            continue

        prev3 = float(df["close"].iloc[i - 3])
        prev6 = float(df["close"].iloc[i - 6])
        price_vs_vwap = _price_vs_vwap(curr_close, vwap_val)
        price_window_start = i - config.RESEARCH_PULLBACK_LOOKBACK

        if direction == "LONG":
            move3_r = (curr_close - prev3) / stop_distance
            move6_r = (curr_close - prev6) / stop_distance
            dist_vwap_r = (curr_close - vwap_val) / stop_distance
            entry_price = float(df["open"].iloc[i + 1]) * (1.0 + config.SLIPPAGE)

            impulse_rel = int(df["high"].iloc[price_window_start:i].to_numpy().argmax())
            impulse_idx = price_window_start + impulse_rel
            impulse_price = float(df["high"].iloc[impulse_idx])
            post_impulse_lows = df["low"].iloc[impulse_idx + 1: i]
            post_impulse_vwap = rf["vwap"].iloc[impulse_idx + 1: i]
            post_impulse_ema = rf["ema_fast"].iloc[impulse_idx + 1: i]
            pullback_detected = False
            if not post_impulse_lows.empty:
                ref_hits = post_impulse_lows <= np.maximum(post_impulse_vwap.to_numpy(), post_impulse_ema.to_numpy())
                pullback_detected = bool(ref_hits.any())
                pullback_low = float(post_impulse_lows.min())
                pullback_depth_r = (impulse_price - pullback_low) / stop_distance
            else:
                pullback_depth_r = math.nan
            reclaim_floor = max(float(rf["vwap"].iloc[i]), float(rf["ema_fast"].iloc[i]))
            reclaim_detected = (
                curr_close > curr_open
                and curr_close > reclaim_floor
                and curr_close > float(df["close"].iloc[max(0, i - config.RESEARCH_RECLAIM_LOOKBACK): i].max())
            )
            trend_context_reason = "structure_up+" + ("vwap" if curr_close > vwap_val else "ema")
            macd_agrees = math.isfinite(macd_signal) and macd_signal == 1.0
        else:
            move3_r = (prev3 - curr_close) / stop_distance
            move6_r = (prev6 - curr_close) / stop_distance
            dist_vwap_r = (vwap_val - curr_close) / stop_distance
            entry_price = float(df["open"].iloc[i + 1]) * (1.0 - config.SLIPPAGE)

            impulse_rel = int(df["low"].iloc[price_window_start:i].to_numpy().argmin())
            impulse_idx = price_window_start + impulse_rel
            impulse_price = float(df["low"].iloc[impulse_idx])
            post_impulse_highs = df["high"].iloc[impulse_idx + 1: i]
            post_impulse_vwap = rf["vwap"].iloc[impulse_idx + 1: i]
            post_impulse_ema = rf["ema_fast"].iloc[impulse_idx + 1: i]
            pullback_detected = False
            if not post_impulse_highs.empty:
                ref_hits = post_impulse_highs >= np.minimum(post_impulse_vwap.to_numpy(), post_impulse_ema.to_numpy())
                pullback_detected = bool(ref_hits.any())
                pullback_high = float(post_impulse_highs.max())
                pullback_depth_r = (pullback_high - impulse_price) / stop_distance
            else:
                pullback_depth_r = math.nan
            reclaim_ceiling = min(float(rf["vwap"].iloc[i]), float(rf["ema_fast"].iloc[i]))
            reclaim_detected = (
                curr_close < curr_open
                and curr_close < reclaim_ceiling
                and curr_close < float(df["close"].iloc[max(0, i - config.RESEARCH_RECLAIM_LOOKBACK): i].min())
            )
            trend_context_reason = "structure_down+" + ("vwap" if curr_close < vwap_val else "ema")
            macd_agrees = math.isfinite(macd_signal) and macd_signal == -1.0

        candles_since_impulse = i - impulse_idx
        signal_tags = ["TrendContext"]
        if pullback_detected:
            signal_tags.append("Pullback")
        if reclaim_detected:
            signal_tags.append("Reclaim")
        if macd_agrees:
            signal_tags.append("MACDAgree")
        if volume_ratio >= 1.0:
            signal_tags.append("VolumeSupport")

        signals_fired = tuple(tag for tag in ("TrendContext", "Pullback", "Reclaim") if tag in signal_tags)

        rows.append({
            "signal_pos": i,
            "signal_time": df.index[i],
            "entry_time": df.index[i + 1],
            "candidate_family": "pullback_trend",
            "direction": direction,
            "signals_fired": signals_fired,
            "signal_tags": tuple(signal_tags),
            "close_price": curr_close,
            "entry_price": entry_price,
            "vwap": vwap_val,
            "atr_value": atr_val,
            "atr_pct": atr_pct,
            "volume_ratio": volume_ratio,
            "distance_from_vwap_r": dist_vwap_r,
            "price_move_3r": move3_r,
            "price_move_6r": move6_r,
            "macd_histogram": macd_hist if math.isfinite(macd_hist) else 0.0,
            "macd_slope": macd_slope if math.isfinite(macd_slope) else 0.0,
            "trend": rf["trend"].iloc[i],
            "vol_regime": rf["vol_regime"].iloc[i],
            "halve_position": bool(rf["halve_position"].iloc[i]),
            "score": 0.0,
            "ratio": 0.0,
            "breakout_level": math.nan,
            "prior_range_high": math.nan,
            "prior_range_low": math.nan,
            "range_width": math.nan,
            "breakout_distance_r": math.nan,
            "breakout_retest_confirmed": False,
            "macd_agrees": macd_agrees,
            "price_vs_vwap": price_vs_vwap,
            "trend_context_reason": trend_context_reason,
            "pullback_detected": pullback_detected,
            "reclaim_detected": reclaim_detected,
            "pullback_depth_r": pullback_depth_r,
            "candles_since_impulse": candles_since_impulse,
        })

    return pd.DataFrame(rows)


def _build_range_mean_reversion_candidate_universe(df: pd.DataFrame, rf: pd.DataFrame) -> pd.DataFrame:
    warmup = max(
        _warmup_bars(),
        config.RESEARCH_RANGE_LOOKBACK + config.RESEARCH_RECLAIM_LOOKBACK + 6,
    )
    rows: list[dict] = []

    for i in range(warmup, len(df) - 1):
        curr_open = float(df["open"].iloc[i])
        curr_high = float(df["high"].iloc[i])
        curr_low = float(df["low"].iloc[i])
        curr_close = float(df["close"].iloc[i])
        prev_close = float(df["close"].iloc[i - 1])
        atr_val = float(rf["atr"].iloc[i])
        atr_pct = float(rf["atr_pct"].iloc[i])
        vwap_val = float(rf["vwap"].iloc[i])
        volume_ratio = float(rf["volume_ratio"].iloc[i])
        prior_high = float(rf["range_prior_high"].iloc[i])
        prior_low = float(rf["range_prior_low"].iloc[i])
        macd_hist = float(rf["macd_hist"].iloc[i])
        macd_slope = float(rf["macd_hist_slope"].iloc[i])
        macd_signal = float(rf["signal_macd"].iloc[i])

        needed = [
            curr_open,
            curr_high,
            curr_low,
            curr_close,
            prev_close,
            atr_val,
            atr_pct,
            vwap_val,
            volume_ratio,
            prior_high,
            prior_low,
        ]
        if any(not math.isfinite(v) for v in needed) or curr_close <= 0 or atr_val <= 0:
            continue

        range_width = prior_high - prior_low
        if not math.isfinite(range_width) or range_width <= 0:
            continue

        stop_distance = round(atr_val * config.ATR_STOP_MULTIPLIER, 2)
        if stop_distance <= 0:
            continue

        prev3 = float(df["close"].iloc[i - 3])
        prev6 = float(df["close"].iloc[i - 6])
        price_vs_vwap = _price_vs_vwap(curr_close, vwap_val)
        range_mid = (prior_high + prior_low) / 2.0
        recent_closes = df["close"].iloc[i - config.RESEARCH_RECLAIM_LOOKBACK: i]
        lower_wick = min(curr_open, curr_close) - curr_low
        upper_wick = curr_high - max(curr_open, curr_close)
        body = abs(curr_close - curr_open)

        extended_below_vwap = curr_close < vwap_val or curr_low < vwap_val
        extended_above_vwap = curr_close > vwap_val or curr_high > vwap_val
        broke_range_low = curr_low < prior_low
        broke_range_high = curr_high > prior_high
        failed_breakout_long = broke_range_low and curr_close >= prior_low
        failed_breakout_short = broke_range_high and curr_close <= prior_high

        reclaim_up = (
            curr_close > curr_open
            and curr_close > float(recent_closes.max())
            and (failed_breakout_long or curr_close > prev_close)
        )
        reclaim_down = (
            curr_close < curr_open
            and curr_close < float(recent_closes.min())
            and (failed_breakout_short or curr_close < prev_close)
        )
        rejection_up = curr_close > curr_open and lower_wick >= body
        rejection_down = curr_close < curr_open and upper_wick >= body

        long_candidate = extended_below_vwap and (broke_range_low or curr_close <= range_mid) and (reclaim_up or rejection_up)
        short_candidate = extended_above_vwap and (broke_range_high or curr_close >= range_mid) and (reclaim_down or rejection_down)
        if long_candidate == short_candidate:
            continue

        direction = "LONG" if long_candidate else "SHORT"
        if direction == "LONG":
            move3_r = (curr_close - prev3) / stop_distance
            move6_r = (curr_close - prev6) / stop_distance
            dist_vwap_r = (curr_close - vwap_val) / stop_distance
            distance_from_range_boundary_r = (prior_low - curr_close) / stop_distance
            entry_price = float(df["open"].iloc[i + 1]) * (1.0 + config.SLIPPAGE)
            rejection_detected = rejection_up
            reclaim_detected = reclaim_up
            macd_agrees = math.isfinite(macd_signal) and macd_signal == 1.0
            candles_outside_range = _count_consecutive_outside_range(
                df,
                i,
                direction=direction,
                boundary=prior_low,
                lookback=config.RESEARCH_RECLAIM_LOOKBACK,
            )
            signal_tags = ["MeanReversion", "BelowVWAP"]
            if broke_range_low:
                signal_tags.append("RangeBreakLow")
            if failed_breakout_long:
                signal_tags.append("FailedBreakout")
        else:
            move3_r = (prev3 - curr_close) / stop_distance
            move6_r = (prev6 - curr_close) / stop_distance
            dist_vwap_r = (vwap_val - curr_close) / stop_distance
            distance_from_range_boundary_r = (curr_close - prior_high) / stop_distance
            entry_price = float(df["open"].iloc[i + 1]) * (1.0 - config.SLIPPAGE)
            rejection_detected = rejection_down
            reclaim_detected = reclaim_down
            macd_agrees = math.isfinite(macd_signal) and macd_signal == -1.0
            candles_outside_range = _count_consecutive_outside_range(
                df,
                i,
                direction=direction,
                boundary=prior_high,
                lookback=config.RESEARCH_RECLAIM_LOOKBACK,
            )
            signal_tags = ["MeanReversion", "AboveVWAP"]
            if broke_range_high:
                signal_tags.append("RangeBreakHigh")
            if failed_breakout_short:
                signal_tags.append("FailedBreakout")

        if reclaim_detected:
            signal_tags.append("Reclaim")
        if rejection_detected:
            signal_tags.append("Rejection")
        if macd_agrees:
            signal_tags.append("MACDAgree")
        if volume_ratio >= 1.0:
            signal_tags.append("VolumeSupport")
        if rf["trend"].iloc[i] == reg.RANGING:
            signal_tags.append("RangingContext")

        rows.append({
            "signal_pos": i,
            "signal_time": df.index[i],
            "entry_time": df.index[i + 1],
            "candidate_family": "range_mean_reversion",
            "direction": direction,
            "signals_fired": ("MeanReversion", "Reclaim" if reclaim_detected else "Rejection"),
            "signal_tags": tuple(signal_tags),
            "close_price": curr_close,
            "entry_price": entry_price,
            "vwap": vwap_val,
            "atr_value": atr_val,
            "atr_pct": atr_pct,
            "volume_ratio": volume_ratio,
            "distance_from_vwap_r": dist_vwap_r,
            "price_move_3r": move3_r,
            "price_move_6r": move6_r,
            "macd_histogram": macd_hist if math.isfinite(macd_hist) else 0.0,
            "macd_slope": macd_slope if math.isfinite(macd_slope) else 0.0,
            "trend": rf["trend"].iloc[i],
            "vol_regime": rf["vol_regime"].iloc[i],
            "halve_position": bool(rf["halve_position"].iloc[i]),
            "score": 0.0,
            "ratio": 0.0,
            "breakout_level": math.nan,
            "prior_range_high": prior_high,
            "prior_range_low": prior_low,
            "range_width": range_width,
            "breakout_distance_r": math.nan,
            "breakout_retest_confirmed": False,
            "macd_agrees": macd_agrees,
            "price_vs_vwap": price_vs_vwap,
            **_empty_research_context_fields(),
            "reclaim_detected": reclaim_detected,
            "rejection_detected": rejection_detected,
            "range_high": prior_high,
            "range_low": prior_low,
            "range_mid": range_mid,
            "distance_from_range_boundary_r": distance_from_range_boundary_r,
            "candles_outside_range": candles_outside_range,
        })

    return pd.DataFrame(rows)


def _build_candidate_universe(df: pd.DataFrame, rf: pd.DataFrame) -> pd.DataFrame:
    frames = [
        _build_consensus_candidate_universe(df, rf),
        _build_breakout_candidate_universe(df, rf),
        _build_pullback_candidate_universe(df, rf),
        _build_range_mean_reversion_candidate_universe(df, rf),
    ]
    non_empty = [frame for frame in frames if not frame.empty]
    if not non_empty:
        return pd.DataFrame()
    return pd.concat(non_empty, ignore_index=True)


def _build_regime_thresholds(candidate_df: pd.DataFrame) -> RegimeThresholds:
    return RegimeThresholds(
        atr_q1=float(candidate_df["atr_pct"].quantile(1 / 3)),
        atr_q2=float(candidate_df["atr_pct"].quantile(2 / 3)),
        volume_q1=float(candidate_df["volume_ratio"].quantile(1 / 3)),
        volume_q2=float(candidate_df["volume_ratio"].quantile(2 / 3)),
        vwap_q1=float(candidate_df["distance_from_vwap_r"].abs().quantile(1 / 3)),
        vwap_q2=float(candidate_df["distance_from_vwap_r"].abs().quantile(2 / 3)),
        move3_q1=float(candidate_df["price_move_3r"].quantile(1 / 3)),
        move3_q2=float(candidate_df["price_move_3r"].quantile(2 / 3)),
        move6_q1=float(candidate_df["price_move_6r"].quantile(1 / 3)),
        move6_q2=float(candidate_df["price_move_6r"].quantile(2 / 3)),
    )


def _apply_regime_buckets(candidate_df: pd.DataFrame, thresholds: RegimeThresholds) -> pd.DataFrame:
    work = candidate_df.copy()
    work["atr_bucket"] = work["atr_pct"].map(thresholds.atr_bucket)
    work["volume_bucket"] = work["volume_ratio"].map(thresholds.volume_bucket)
    work["vwap_distance_bucket"] = work["distance_from_vwap_r"].abs().map(thresholds.vwap_bucket)
    work["pre_entry_move_3c_bucket"] = work["price_move_3r"].map(thresholds.move3_bucket)
    work["pre_entry_move_6c_bucket"] = work["price_move_6r"].map(thresholds.move6_bucket)
    return work


def _candidate_context_from_row(row: pd.Series) -> CandidateContext:
    return CandidateContext(
        signal_pos=int(row["signal_pos"]),
        signal_time=row["signal_time"],
        entry_time=row["entry_time"],
        candidate_family=str(row.get("candidate_family", "consensus")),
        direction=str(row["direction"]),
        close_price=float(row["close_price"]),
        entry_price=float(row["entry_price"]),
        vwap=float(row["vwap"]),
        atr_value=float(row["atr_value"]),
        atr_pct=float(row["atr_pct"]),
        volume_ratio=float(row["volume_ratio"]),
        distance_from_vwap_r=float(row["distance_from_vwap_r"]),
        price_move_3r=float(row["price_move_3r"]),
        price_move_6r=float(row["price_move_6r"]),
        macd_histogram=float(row["macd_histogram"]),
        macd_slope=float(row["macd_slope"]),
        trend=str(row["trend"]),
        vol_regime=str(row["vol_regime"]),
        halve_position=bool(row["halve_position"]),
        signals_fired=tuple(row["signals_fired"]),
        signal_tags=tuple(row["signal_tags"]),
        atr_bucket=str(row["atr_bucket"]),
        volume_bucket=str(row["volume_bucket"]),
        vwap_distance_bucket=str(row["vwap_distance_bucket"]),
        pre_entry_move_3c_bucket=str(row["pre_entry_move_3c_bucket"]),
        pre_entry_move_6c_bucket=str(row["pre_entry_move_6c_bucket"]),
        breakout_level=float(row.get("breakout_level", math.nan)),
        prior_range_high=float(row.get("prior_range_high", math.nan)),
        prior_range_low=float(row.get("prior_range_low", math.nan)),
        range_width=float(row.get("range_width", math.nan)),
        breakout_distance_r=float(row.get("breakout_distance_r", math.nan)),
        breakout_retest_confirmed=bool(row.get("breakout_retest_confirmed", False)),
        macd_agrees=bool(row.get("macd_agrees", False)),
        price_vs_vwap=str(row.get("price_vs_vwap", "")),
        trend_context_reason=str(row.get("trend_context_reason", "")),
        pullback_detected=bool(row.get("pullback_detected", False)),
        reclaim_detected=bool(row.get("reclaim_detected", False)),
        pullback_depth_r=float(row.get("pullback_depth_r", math.nan)),
        candles_since_impulse=_int_or_default(row.get("candles_since_impulse", -1)),
        range_high=float(row.get("range_high", math.nan)),
        range_low=float(row.get("range_low", math.nan)),
        range_mid=float(row.get("range_mid", math.nan)),
        distance_from_range_boundary_r=float(row.get("distance_from_range_boundary_r", math.nan)),
        rejection_detected=bool(row.get("rejection_detected", False)),
        candles_outside_range=_int_or_default(row.get("candles_outside_range", -1)),
    )


def _print_regime_thresholds(thresholds: RegimeThresholds) -> None:
    print("\n  Research regime thresholds (quantile-derived from candidate universe):")
    print(f"    ATR pct buckets          : low <= {thresholds.atr_q1:.3f}% < medium <= {thresholds.atr_q2:.3f}% < high")
    print(f"    Volume ratio buckets     : low <= {thresholds.volume_q1:.3f}x < medium <= {thresholds.volume_q2:.3f}x < high")
    print(f"    VWAP distance buckets    : close <= {thresholds.vwap_q1:.3f}R < medium <= {thresholds.vwap_q2:.3f}R < far")
    print(f"    Pre-entry move 3c buckets: low <= {thresholds.move3_q1:.3f}R < medium <= {thresholds.move3_q2:.3f}R < high")
    print(f"    Pre-entry move 6c buckets: low <= {thresholds.move6_q1:.3f}R < medium <= {thresholds.move6_q2:.3f}R < high")


def _int_or_default(value: object, default: int = -1) -> int:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except TypeError:
        pass
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _setup_ok(name: str, ctx: CandidateContext) -> SetupDecision:
    return SetupDecision(
        allowed=True,
        setup_name=name,
        signal_tags=tuple(sorted(ctx.signal_tags)),
    )


def _setup_reject(name: str, ctx: CandidateContext, reason: str) -> SetupDecision:
    return SetupDecision(
        allowed=False,
        setup_name=name,
        rejection_reason=reason,
        signal_tags=tuple(sorted(ctx.signal_tags)),
    )


def setup_macd_vwap_base(ctx: CandidateContext) -> SetupDecision:
    if "MACD" not in ctx.signals_set or "VWAP" not in ctx.signals_set:
        return _setup_reject("MACD_VWAP_BASE", ctx, "requires MACD and VWAP")
    return _setup_ok("MACD_VWAP_BASE", ctx)


def setup_macd_vwap_short_only(ctx: CandidateContext) -> SetupDecision:
    base = setup_macd_vwap_base(ctx)
    if not base.allowed:
        return _setup_reject("MACD_VWAP_SHORT_ONLY", ctx, base.rejection_reason)
    if ctx.direction != "SHORT":
        return _setup_reject("MACD_VWAP_SHORT_ONLY", ctx, "short-only research setup")
    return _setup_ok("MACD_VWAP_SHORT_ONLY", ctx)


def setup_macd_vwap_no_medium_atr(ctx: CandidateContext) -> SetupDecision:
    base = setup_macd_vwap_base(ctx)
    if not base.allowed:
        return _setup_reject("MACD_VWAP_NO_MEDIUM_ATR", ctx, base.rejection_reason)
    if ctx.atr_bucket == "medium":
        return _setup_reject("MACD_VWAP_NO_MEDIUM_ATR", ctx, "medium ATR bucket blocked")
    return _setup_ok("MACD_VWAP_NO_MEDIUM_ATR", ctx)


def setup_macd_vwap_short_only_no_medium_atr(ctx: CandidateContext) -> SetupDecision:
    short_only = setup_macd_vwap_short_only(ctx)
    if not short_only.allowed:
        return _setup_reject("MACD_VWAP_SHORT_ONLY_NO_MEDIUM_ATR", ctx, short_only.rejection_reason)
    if ctx.atr_bucket == "medium":
        return _setup_reject("MACD_VWAP_SHORT_ONLY_NO_MEDIUM_ATR", ctx, "medium ATR bucket blocked")
    return _setup_ok("MACD_VWAP_SHORT_ONLY_NO_MEDIUM_ATR", ctx)


def setup_macd_vwap_volume(ctx: CandidateContext) -> SetupDecision:
    if "MACD" not in ctx.signals_set or "VWAP" not in ctx.signals_set or "Volume" not in ctx.signals_set:
        return _setup_reject("MACD_VWAP_VOLUME", ctx, "requires MACD, VWAP, and Volume")
    return _setup_ok("MACD_VWAP_VOLUME", ctx)


def setup_macd_volume_only(ctx: CandidateContext) -> SetupDecision:
    if "MACD" not in ctx.signals_set or "Volume" not in ctx.signals_set:
        return _setup_reject("MACD_VOLUME_ONLY", ctx, "requires MACD and Volume")
    if "VWAP" in ctx.signals_set:
        return _setup_reject("MACD_VOLUME_ONLY", ctx, "VWAP not allowed in MACD_VOLUME_ONLY")
    return _setup_ok("MACD_VOLUME_ONLY", ctx)


def setup_macd_only_reference(ctx: CandidateContext) -> SetupDecision:
    if "MACD" not in ctx.signals_set:
        return _setup_reject("MACD_ONLY_REFERENCE", ctx, "requires MACD")
    return _setup_ok("MACD_ONLY_REFERENCE", ctx)

def setup_volume_breakout_continuation(ctx: CandidateContext) -> SetupDecision:
    if not math.isfinite(ctx.breakout_level):
        return _setup_reject("VOLUME_BREAKOUT_CONTINUATION", ctx, "missing breakout reference level")
    if ctx.volume_ratio < config.RESEARCH_BREAKOUT_MIN_VOLUME_RATIO:
        return _setup_reject(
            "VOLUME_BREAKOUT_CONTINUATION",
            ctx,
            f"volume ratio below minimum {config.RESEARCH_BREAKOUT_MIN_VOLUME_RATIO:.2f}x",
        )
    if config.RESEARCH_BREAKOUT_REQUIRE_RETEST and not ctx.breakout_retest_confirmed:
        return _setup_reject("VOLUME_BREAKOUT_CONTINUATION", ctx, "retest required but not confirmed")
    return _setup_ok("VOLUME_BREAKOUT_CONTINUATION", ctx)


def setup_pullback_to_trend_continuation(ctx: CandidateContext) -> SetupDecision:
    if not ctx.trend_context_reason:
        return _setup_reject("PULLBACK_TO_TREND_CONTINUATION", ctx, "missing trend context")
    if ctx.candles_since_impulse < 2:
        return _setup_reject("PULLBACK_TO_TREND_CONTINUATION", ctx, "entry too close to initial impulse")
    if not ctx.pullback_detected:
        return _setup_reject("PULLBACK_TO_TREND_CONTINUATION", ctx, "pullback not detected")
    if not ctx.reclaim_detected:
        return _setup_reject("PULLBACK_TO_TREND_CONTINUATION", ctx, "reclaim not detected")
    if ctx.vwap_distance_bucket == "far":
        return _setup_reject("PULLBACK_TO_TREND_CONTINUATION", ctx, "entry too far from VWAP")
    return _setup_ok("PULLBACK_TO_TREND_CONTINUATION", ctx)


def setup_range_mean_reversion(ctx: CandidateContext) -> SetupDecision:
    if ctx.trend == reg.TRENDING:
        return _setup_reject("RANGE_MEAN_REVERSION", ctx, "strong trend regime blocked")
    if _bucket_rank(ctx.vwap_distance_bucket) < _bucket_rank(config.RESEARCH_RANGE_MIN_VWAP_BUCKET):
        return _setup_reject(
            "RANGE_MEAN_REVERSION",
            ctx,
            f"VWAP extension bucket below minimum {config.RESEARCH_RANGE_MIN_VWAP_BUCKET}",
        )
    if not ctx.reclaim_detected:
        return _setup_reject("RANGE_MEAN_REVERSION", ctx, "reclaim not detected")
    if not ctx.rejection_detected:
        return _setup_reject("RANGE_MEAN_REVERSION", ctx, "rejection not detected")
    return _setup_ok("RANGE_MEAN_REVERSION", ctx)


def _setup_range_mean_reversion_regime_variant(
    ctx: CandidateContext,
    *,
    name: str,
    allow_long: bool = True,
    allow_short: bool = True,
    block_far_vwap: bool = False,
    block_high_atr: bool = False,
    medium_atr_only: bool = False,
    require_allowed_range_context: bool = False,
    block_high_volume: bool = False,
) -> SetupDecision:
    """
    Research-only 2H range mean-reversion filter helper.

    These variants reuse the same broad entry idea and only tighten direction
    and regime buckets for diagnostics. They do not affect live trading.
    """
    base = setup_range_mean_reversion(ctx)
    if not base.allowed:
        return _setup_reject(name, ctx, base.rejection_reason)
    if ctx.direction == "LONG":
        if not allow_long:
            return _setup_reject(name, ctx, "longs blocked in this 2H regime variant")
    elif ctx.direction == "SHORT":
        if not allow_short:
            return _setup_reject(name, ctx, "shorts blocked in this 2H regime variant")
    else:
        return _setup_reject(name, ctx, "unsupported direction")
    if block_far_vwap and ctx.vwap_distance_bucket == "far":
        return _setup_reject(name, ctx, "far VWAP-distance bucket blocked")
    if block_high_atr and ctx.atr_bucket == "high":
        return _setup_reject(name, ctx, "high ATR bucket blocked")
    if medium_atr_only and ctx.atr_bucket != "medium":
        return _setup_reject(name, ctx, "medium ATR bucket required")
    if require_allowed_range_context and ctx.vol_regime == reg.HIGH_VOLATILITY:
        return _setup_reject(name, ctx, "high-volatility ranging regime blocked")
    if block_high_volume and ctx.volume_bucket == "high":
        return _setup_reject(name, ctx, "high volume bucket blocked")
    return _setup_ok(name, ctx)


def setup_rmr_2h_long_only(ctx: CandidateContext) -> SetupDecision:
    return _setup_range_mean_reversion_regime_variant(
        ctx,
        name="RMR_2H_LONG_ONLY",
        allow_long=True,
        allow_short=False,
    )


def setup_rmr_2h_no_far_vwap(ctx: CandidateContext) -> SetupDecision:
    return _setup_range_mean_reversion_regime_variant(
        ctx,
        name="RMR_2H_NO_FAR_VWAP",
        block_far_vwap=True,
    )


def setup_rmr_2h_no_high_atr(ctx: CandidateContext) -> SetupDecision:
    return _setup_range_mean_reversion_regime_variant(
        ctx,
        name="RMR_2H_NO_HIGH_ATR",
        block_high_atr=True,
    )


def setup_rmr_2h_no_far_vwap_no_high_atr(ctx: CandidateContext) -> SetupDecision:
    return _setup_range_mean_reversion_regime_variant(
        ctx,
        name="RMR_2H_NO_FAR_VWAP_NO_HIGH_ATR",
        block_far_vwap=True,
        block_high_atr=True,
    )


def setup_rmr_2h_long_only_no_far_vwap(ctx: CandidateContext) -> SetupDecision:
    return _setup_range_mean_reversion_regime_variant(
        ctx,
        name="RMR_2H_LONG_ONLY_NO_FAR_VWAP",
        allow_long=True,
        allow_short=False,
        block_far_vwap=True,
    )


def setup_rmr_2h_long_only_no_far_vwap_no_high_atr(ctx: CandidateContext) -> SetupDecision:
    return _setup_range_mean_reversion_regime_variant(
        ctx,
        name="RMR_2H_LONG_ONLY_NO_FAR_VWAP_NO_HIGH_ATR",
        allow_long=True,
        allow_short=False,
        block_far_vwap=True,
        block_high_atr=True,
    )


def setup_rmr_2h_long_only_no_far_vwap_no_high_vol(ctx: CandidateContext) -> SetupDecision:
    return _setup_range_mean_reversion_regime_variant(
        ctx,
        name="RMR_2H_LONG_ONLY_NO_FAR_VWAP_NO_HIGH_VOL",
        allow_long=True,
        allow_short=False,
        block_far_vwap=True,
        block_high_volume=True,
    )


def setup_rmr_2h_no_far_vwap_long_biased(ctx: CandidateContext) -> SetupDecision:
    """
    Research-only 2H bridge variant.

    Keep the broad no-far-VWAP sample but lean into the stronger long side by
    allowing shorts only outside the high-ATR bucket. This is intended to test
    whether a long-biased distribution is more stable than pure long-only.
    """
    base = _setup_range_mean_reversion_regime_variant(
        ctx,
        name="RMR_2H_NO_FAR_VWAP_LONG_BIASED",
        allow_long=True,
        allow_short=True,
        block_far_vwap=True,
    )
    if not base.allowed:
        return base
    if ctx.direction == "SHORT" and ctx.atr_bucket == "high":
        return _setup_reject("RMR_2H_NO_FAR_VWAP_LONG_BIASED", ctx, "shorts high ATR bucket blocked")
    return _setup_ok("RMR_2H_NO_FAR_VWAP_LONG_BIASED", ctx)


def setup_rmr_2h_no_far_vwap_long_biased_medvol_shorts(ctx: CandidateContext) -> SetupDecision:
    """
    Research-only 2H bridge variant.

    Keep all long entries from the no-far-VWAP profile, but allow shorts only
    in the cleaner medium-volume, non-high-ATR pocket. This is intended to test
    whether a narrower short filter improves distribution without collapsing the
    sample the way pure long-only variants do.
    """
    base = _setup_range_mean_reversion_regime_variant(
        ctx,
        name="RMR_2H_NO_FAR_VWAP_LONG_BIASED_MEDVOL_SHORTS",
        allow_long=True,
        allow_short=True,
        block_far_vwap=True,
    )
    if not base.allowed:
        return base
    if ctx.direction == "SHORT":
        if ctx.atr_bucket == "high":
            return _setup_reject(
                "RMR_2H_NO_FAR_VWAP_LONG_BIASED_MEDVOL_SHORTS",
                ctx,
                "shorts high ATR bucket blocked",
            )
        if ctx.volume_bucket != "medium":
            return _setup_reject(
                "RMR_2H_NO_FAR_VWAP_LONG_BIASED_MEDVOL_SHORTS",
                ctx,
                "shorts require medium volume bucket",
            )
    return _setup_ok("RMR_2H_NO_FAR_VWAP_LONG_BIASED_MEDVOL_SHORTS", ctx)


def setup_rmr_2h_no_far_vwap_long_biased_lowatr_medvol_shorts(ctx: CandidateContext) -> SetupDecision:
    """
    Research-only 2H bridge variant.

    Keep the broad no-far-VWAP long side intact, but allow shorts only in the
    lowest-friction pocket seen in the diagnostics: low ATR with medium volume.
    """
    base = _setup_range_mean_reversion_regime_variant(
        ctx,
        name="RMR_2H_NO_FAR_VWAP_LONG_BIASED_LOWATR_MEDVOL_SHORTS",
        allow_long=True,
        allow_short=True,
        block_far_vwap=True,
    )
    if not base.allowed:
        return base
    if ctx.direction == "SHORT":
        if ctx.atr_bucket != "low":
            return _setup_reject(
                "RMR_2H_NO_FAR_VWAP_LONG_BIASED_LOWATR_MEDVOL_SHORTS",
                ctx,
                "shorts require low ATR bucket",
            )
        if ctx.volume_bucket != "medium":
            return _setup_reject(
                "RMR_2H_NO_FAR_VWAP_LONG_BIASED_LOWATR_MEDVOL_SHORTS",
                ctx,
                "shorts require medium volume bucket",
            )
    return _setup_ok("RMR_2H_NO_FAR_VWAP_LONG_BIASED_LOWATR_MEDVOL_SHORTS", ctx)


def setup_rmr_2h_no_far_vwap_no_high_atr_no_high_vol(ctx: CandidateContext) -> SetupDecision:
    return _setup_range_mean_reversion_regime_variant(
        ctx,
        name="RMR_2H_NO_FAR_VWAP_NO_HIGH_ATR_NO_HIGH_VOL",
        allow_long=True,
        allow_short=True,
        block_far_vwap=True,
        block_high_atr=True,
        block_high_volume=True,
    )


def setup_rmr_2h_no_far_vwap_long_biased_no_high_atr_no_high_vol(ctx: CandidateContext) -> SetupDecision:
    """
    Research-only bridge variant.

    Keep all longs from the no-far-VWAP profile, but only allow shorts when
    they avoid the two weakest broad buckets: high ATR and high volume.
    """
    base = _setup_range_mean_reversion_regime_variant(
        ctx,
        name="RMR_2H_NO_FAR_VWAP_LONG_BIASED_NO_HIGH_ATR_NO_HIGH_VOL",
        allow_long=True,
        allow_short=True,
        block_far_vwap=True,
    )
    if not base.allowed:
        return base
    if ctx.direction == "SHORT":
        if ctx.atr_bucket == "high":
            return _setup_reject(
                "RMR_2H_NO_FAR_VWAP_LONG_BIASED_NO_HIGH_ATR_NO_HIGH_VOL",
                ctx,
                "shorts high ATR bucket blocked",
            )
        if ctx.volume_bucket == "high":
            return _setup_reject(
                "RMR_2H_NO_FAR_VWAP_LONG_BIASED_NO_HIGH_ATR_NO_HIGH_VOL",
                ctx,
                "shorts high volume bucket blocked",
            )
    return _setup_ok("RMR_2H_NO_FAR_VWAP_LONG_BIASED_NO_HIGH_ATR_NO_HIGH_VOL", ctx)


def setup_rmr_2h_no_far_vwap_long_biased_block_long_high_vol(ctx: CandidateContext) -> SetupDecision:
    """
    Research-only bridge variant.

    Keep the current long-biased short filter, but also remove the weakest long
    pocket from the diagnostics: high-volume longs.
    """
    base = setup_rmr_2h_no_far_vwap_long_biased(ctx)
    if not base.allowed:
        return base
    if ctx.direction == "LONG" and ctx.volume_bucket == "high":
        return _setup_reject(
            "RMR_2H_NO_FAR_VWAP_LONG_BIASED_BLOCK_LONG_HIGH_VOL",
            ctx,
            "longs high volume bucket blocked",
        )
    return _setup_ok("RMR_2H_NO_FAR_VWAP_LONG_BIASED_BLOCK_LONG_HIGH_VOL", ctx)


def setup_rmr_2h_no_far_vwap_long_biased_block_long_high_vol_short_lowatr(ctx: CandidateContext) -> SetupDecision:
    base = setup_rmr_2h_no_far_vwap_long_biased_block_long_high_vol(ctx)
    if not base.allowed:
        return base
    if ctx.direction == "SHORT" and ctx.atr_bucket != "low":
        return _setup_reject(
            "RMR_2H_NO_FAR_VWAP_LONG_BIASED_BLOCK_LONG_HIGH_VOL_SHORT_LOWATR",
            ctx,
            "shorts require low ATR bucket",
        )
    return _setup_ok("RMR_2H_NO_FAR_VWAP_LONG_BIASED_BLOCK_LONG_HIGH_VOL_SHORT_LOWATR", ctx)


def setup_rmr_2h_no_far_vwap_long_biased_block_long_high_vol_short_lowatr_medvol(ctx: CandidateContext) -> SetupDecision:
    base = setup_rmr_2h_no_far_vwap_long_biased_block_long_high_vol(ctx)
    if not base.allowed:
        return base
    if ctx.direction == "SHORT":
        if ctx.atr_bucket != "low":
            return _setup_reject(
                "RMR_2H_NO_FAR_VWAP_LONG_BIASED_BLOCK_LONG_HIGH_VOL_SHORT_LOWATR_MEDVOL",
                ctx,
                "shorts require low ATR bucket",
            )
        if ctx.volume_bucket != "medium":
            return _setup_reject(
                "RMR_2H_NO_FAR_VWAP_LONG_BIASED_BLOCK_LONG_HIGH_VOL_SHORT_LOWATR_MEDVOL",
                ctx,
                "shorts require medium volume bucket",
            )
    return _setup_ok("RMR_2H_NO_FAR_VWAP_LONG_BIASED_BLOCK_LONG_HIGH_VOL_SHORT_LOWATR_MEDVOL", ctx)


def setup_rmr_2h_no_far_vwap_long_biased_block_long_high_vol_short_no_high_vol(ctx: CandidateContext) -> SetupDecision:
    base = setup_rmr_2h_no_far_vwap_long_biased_block_long_high_vol(ctx)
    if not base.allowed:
        return base
    if ctx.direction == "SHORT" and ctx.volume_bucket == "high":
        return _setup_reject(
            "RMR_2H_NO_FAR_VWAP_LONG_BIASED_BLOCK_LONG_HIGH_VOL_SHORT_NO_HIGH_VOL",
            ctx,
            "shorts high volume bucket blocked",
        )
    return _setup_ok("RMR_2H_NO_FAR_VWAP_LONG_BIASED_BLOCK_LONG_HIGH_VOL_SHORT_NO_HIGH_VOL", ctx)


def setup_rmr_2h_no_far_vwap_long_biased_block_long_high_atr_or_high_vol(ctx: CandidateContext) -> SetupDecision:
    base = setup_rmr_2h_no_far_vwap_long_biased(ctx)
    if not base.allowed:
        return base
    if ctx.direction == "LONG" and (ctx.atr_bucket == "high" or ctx.volume_bucket == "high"):
        return _setup_reject(
            "RMR_2H_NO_FAR_VWAP_LONG_BIASED_BLOCK_LONG_HIGH_ATR_OR_HIGH_VOL",
            ctx,
            "longs high ATR or high volume bucket blocked",
        )
    return _setup_ok("RMR_2H_NO_FAR_VWAP_LONG_BIASED_BLOCK_LONG_HIGH_ATR_OR_HIGH_VOL", ctx)


def setup_rmr_2h_no_far_vwap_block_long_highatr_highvol(ctx: CandidateContext) -> SetupDecision:
    """
    Research-only broad 2H refinement.

    Keep the broad no-far-VWAP sample intact except for the clearest losing
    long pocket from the diagnostics: longs that arrive in both the high-ATR
    and high-volume buckets at once.
    """
    base = setup_rmr_2h_no_far_vwap(ctx)
    if not base.allowed:
        return base
    if ctx.direction == "LONG" and ctx.atr_bucket == "high" and ctx.volume_bucket == "high":
        return _setup_reject(
            "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL",
            ctx,
            "longs high ATR + high volume pocket blocked",
        )
    return _setup_ok("RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL", ctx)


def setup_rmr_2h_no_far_vwap_long_biased_short_lowatr(ctx: CandidateContext) -> SetupDecision:
    """
    Research-only bridge variant.

    Keep every long from the long-biased no-far-VWAP profile, but require the
    short side to stay in the low-ATR bucket. This tests whether the weakest
    remaining short drag can be removed without also tightening the long side.
    """
    base = setup_rmr_2h_no_far_vwap_long_biased(ctx)
    if not base.allowed:
        return base
    if ctx.direction == "SHORT" and ctx.atr_bucket != "low":
        return _setup_reject(
            "RMR_2H_NO_FAR_VWAP_LONG_BIASED_SHORT_LOWATR",
            ctx,
            "shorts require low ATR bucket",
        )
    return _setup_ok("RMR_2H_NO_FAR_VWAP_LONG_BIASED_SHORT_LOWATR", ctx)


def setup_rmr_2h_no_far_vwap_long_biased_block_long_highatr_highvol(ctx: CandidateContext) -> SetupDecision:
    """
    Research-only bridge variant.

    Keep the long-biased short filter, but only trim the single clearest weak
    long pocket: simultaneous high ATR and high volume.
    """
    base = setup_rmr_2h_no_far_vwap_long_biased(ctx)
    if not base.allowed:
        return base
    if ctx.direction == "LONG" and ctx.atr_bucket == "high" and ctx.volume_bucket == "high":
        return _setup_reject(
            "RMR_2H_NO_FAR_VWAP_LONG_BIASED_BLOCK_LONG_HIGHATR_HIGHVOL",
            ctx,
            "longs high ATR + high volume pocket blocked",
        )
    return _setup_ok("RMR_2H_NO_FAR_VWAP_LONG_BIASED_BLOCK_LONG_HIGHATR_HIGHVOL", ctx)


def setup_rmr_2h_no_far_vwap_block_long_highatr_highvol_short_lowatr_or_highatr_highvol(
    ctx: CandidateContext,
) -> SetupDecision:
    """
    Research-only broad refinement.

    Start from the best broad no-far-VWAP profile with the surgical long-side
    high-ATR/high-volume block, then allow shorts only in the two positive
    diagnostic pockets: low ATR, or the rarer high-ATR/high-volume squeeze.
    """
    base = setup_rmr_2h_no_far_vwap_block_long_highatr_highvol(ctx)
    if not base.allowed:
        return base
    if ctx.direction == "SHORT":
        short_ok = ctx.atr_bucket == "low" or (
            ctx.atr_bucket == "high" and ctx.volume_bucket == "high"
        )
        if not short_ok:
            return _setup_reject(
                "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_LOWATR_OR_HIGHATR_HIGHVOL",
                ctx,
                "shorts require low ATR or high ATR + high volume pocket",
            )
    return _setup_ok(
        "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_LOWATR_OR_HIGHATR_HIGHVOL",
        ctx,
    )


def setup_rmr_2h_no_far_vwap_block_long_highatr_highvol_short_lowatr_or_highvol(
    ctx: CandidateContext,
) -> SetupDecision:
    """
    Research-only broad refinement.

    Keep the surgical long-side block, then allow shorts in the full low-ATR
    pocket or any high-volume pocket. This is a milder version of the stricter
    short filter intended to preserve more sample and distribution.
    """
    base = setup_rmr_2h_no_far_vwap_block_long_highatr_highvol(ctx)
    if not base.allowed:
        return base
    if ctx.direction == "SHORT":
        short_ok = ctx.atr_bucket == "low" or ctx.volume_bucket == "high"
        if not short_ok:
            return _setup_reject(
                "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_LOWATR_OR_HIGHVOL",
                ctx,
                "shorts require low ATR or high volume pocket",
            )
    return _setup_ok(
        "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_LOWATR_OR_HIGHVOL",
        ctx,
    )


def setup_rmr_2h_no_far_vwap_block_long_highatr_highvol_short_block_highatr_lowmedvol(
    ctx: CandidateContext,
) -> SetupDecision:
    """
    Research-only broad refinement.

    Keep the mainline sample broad, but remove only the weakest high-ATR short
    pocket: high ATR combined with low or medium volume.
    """
    base = setup_rmr_2h_no_far_vwap_block_long_highatr_highvol(ctx)
    if not base.allowed:
        return base
    if ctx.direction == "SHORT" and ctx.atr_bucket == "high" and ctx.volume_bucket != "high":
        return _setup_reject(
            "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_HIGHATR_LOWMEDVOL",
            ctx,
            "shorts high ATR low/medium volume blocked",
        )
    return _setup_ok(
        "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_HIGHATR_LOWMEDVOL",
        ctx,
    )


def setup_rmr_2h_no_far_vwap_block_long_highatr_highvol_short_block_medatr_lowmedvol(
    ctx: CandidateContext,
) -> SetupDecision:
    """
    Research-only broad refinement.

    Keep the mainline sample broad, but remove only the weakest medium-ATR
    short pocket: medium ATR combined with low or medium volume.
    """
    base = setup_rmr_2h_no_far_vwap_block_long_highatr_highvol(ctx)
    if not base.allowed:
        return base
    if ctx.direction == "SHORT" and ctx.atr_bucket == "medium" and ctx.volume_bucket != "high":
        return _setup_reject(
            "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_MEDATR_LOWMEDVOL",
            ctx,
            "shorts medium ATR low/medium volume blocked",
        )
    return _setup_ok(
        "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_MEDATR_LOWMEDVOL",
        ctx,
    )


def setup_rmr_2h_no_far_vwap_block_long_highatr_highvol_short_block_medatr_lowvol(
    ctx: CandidateContext,
) -> SetupDecision:
    """
    Research-only broad refinement.

    A milder version of the medium-ATR short filter that removes only the
    medium-ATR, low-volume short bucket.
    """
    base = setup_rmr_2h_no_far_vwap_block_long_highatr_highvol(ctx)
    if not base.allowed:
        return base
    if ctx.direction == "SHORT" and ctx.atr_bucket == "medium" and ctx.volume_bucket == "low":
        return _setup_reject(
            "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_MEDATR_LOWVOL",
            ctx,
            "shorts medium ATR low volume blocked",
        )
    return _setup_ok(
        "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_MEDATR_LOWVOL",
        ctx,
    )


def setup_rmr_2h_no_far_vwap_block_long_highatr_highvol_short_block_nonlowatr_lowvol(
    ctx: CandidateContext,
) -> SetupDecision:
    """
    Research-only broad refinement.

    Keep the primary profile intact except for the broadest clearly weak short
    pocket: low-volume shorts when ATR is not low.
    """
    base = setup_rmr_2h_no_far_vwap_block_long_highatr_highvol(ctx)
    if not base.allowed:
        return base
    if ctx.direction == "SHORT" and ctx.volume_bucket == "low" and ctx.atr_bucket != "low":
        return _setup_reject(
            "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_NONLOWATR_LOWVOL",
            ctx,
            "shorts non-low ATR low volume blocked",
        )
    return _setup_ok(
        "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_NONLOWATR_LOWVOL",
        ctx,
    )


def setup_rmr_2h_no_far_vwap_block_long_highatr_highvol_short_block_highatr_medvol(
    ctx: CandidateContext,
) -> SetupDecision:
    """
    Research-only broad refinement.

    Keep the primary profile intact, but remove only the high-ATR medium-volume
    short pocket, which remains consistently weak in the diagnostics.
    """
    base = setup_rmr_2h_no_far_vwap_block_long_highatr_highvol(ctx)
    if not base.allowed:
        return base
    if ctx.direction == "SHORT" and ctx.atr_bucket == "high" and ctx.volume_bucket == "medium":
        return _setup_reject(
            "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_HIGHATR_MEDVOL",
            ctx,
            "shorts high ATR medium volume blocked",
        )
    return _setup_ok(
        "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_HIGHATR_MEDVOL",
        ctx,
    )


def setup_rmr_2h_no_far_vwap_block_long_highatr_highvol_short_block_highatr_medvol_and_medatr_lowvol(
    ctx: CandidateContext,
) -> SetupDecision:
    """
    Research-only broad refinement.

    Remove the two clearest weak short pockets while keeping the rest of the
    mainline profile intact: high-ATR medium-volume shorts and medium-ATR
    low-volume shorts.
    """
    base = setup_rmr_2h_no_far_vwap_block_long_highatr_highvol(ctx)
    if not base.allowed:
        return base
    if ctx.direction == "SHORT":
        if ctx.atr_bucket == "high" and ctx.volume_bucket == "medium":
            return _setup_reject(
                "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_HIGHATR_MEDVOL_AND_MEDATR_LOWVOL",
                ctx,
                "shorts high ATR medium volume blocked",
            )
        if ctx.atr_bucket == "medium" and ctx.volume_bucket == "low":
            return _setup_reject(
                "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_HIGHATR_MEDVOL_AND_MEDATR_LOWVOL",
                ctx,
                "shorts medium ATR low volume blocked",
            )
    return _setup_ok(
        "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_HIGHATR_MEDVOL_AND_MEDATR_LOWVOL",
        ctx,
    )


def setup_rmr_2h_no_far_vwap_block_long_highatr_highvol_block_long_lowatr_medvol_pre3_medium(
    ctx: CandidateContext,
) -> SetupDecision:
    """
    Research-only broad refinement.

    Keep the current leading mainline profile, but remove the narrow long
    timing pocket that remains weak: low ATR, medium volume, and medium
    3-candle pre-entry extension.
    """
    base = setup_rmr_2h_no_far_vwap_block_long_highatr_highvol_short_block_highatr_medvol(ctx)
    if not base.allowed:
        return base
    if (
        ctx.direction == "LONG"
        and ctx.atr_bucket == "low"
        and ctx.volume_bucket == "medium"
        and ctx.pre_entry_move_3c_bucket == "medium"
    ):
        return _setup_reject(
            "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_BLOCK_LONG_LOWATR_MEDVOL_PRE3_MEDIUM",
            ctx,
            "longs low ATR medium volume with medium 3-candle extension blocked",
        )
    return _setup_ok(
        "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_BLOCK_LONG_LOWATR_MEDVOL_PRE3_MEDIUM",
        ctx,
    )


def setup_rmr_2h_no_far_vwap_block_long_highatr_highvol_short_block_highatr_medvol_short_block_highatr_lowvol_pre3_medium(
    ctx: CandidateContext,
) -> SetupDecision:
    """
    Research-only broad refinement.

    Keep the current leading mainline profile, but remove the additional weak
    short timing pocket where ATR is high, volume is low, and the 3-candle
    pre-entry move is already extended.
    """
    base = setup_rmr_2h_no_far_vwap_block_long_highatr_highvol_short_block_highatr_medvol(ctx)
    if not base.allowed:
        return base
    if (
        ctx.direction == "SHORT"
        and ctx.atr_bucket == "high"
        and ctx.volume_bucket == "low"
        and ctx.pre_entry_move_3c_bucket == "medium"
    ):
        return _setup_reject(
            "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_HIGHATR_MEDVOL_SHORT_BLOCK_HIGHATR_LOWVOL_PRE3_MEDIUM",
            ctx,
            "shorts high ATR low volume with medium 3-candle extension blocked",
        )
    return _setup_ok(
        "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_HIGHATR_MEDVOL_SHORT_BLOCK_HIGHATR_LOWVOL_PRE3_MEDIUM",
        ctx,
    )


def setup_rmr_2h_no_far_vwap_block_long_highatr_highvol_short_block_highatr_medvol_block_long_lowatr_medvol_pre3_medium_pre6_low(
    ctx: CandidateContext,
) -> SetupDecision:
    """
    Research-only broad refinement.

    Keep the current leading mainline profile, but trim only the narrow long
    timing pocket that remained fully losing in diagnostics: low ATR, medium
    volume, medium 3-candle extension, and low 6-candle extension.
    """
    base = setup_rmr_2h_no_far_vwap_block_long_highatr_highvol_short_block_highatr_medvol(ctx)
    if not base.allowed:
        return base
    if (
        ctx.direction == "LONG"
        and ctx.atr_bucket == "low"
        and ctx.volume_bucket == "medium"
        and ctx.pre_entry_move_3c_bucket == "medium"
        and ctx.pre_entry_move_6c_bucket == "low"
    ):
        return _setup_reject(
            "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_HIGHATR_MEDVOL_BLOCK_LONG_LOWATR_MEDVOL_PRE3_MEDIUM_PRE6_LOW",
            ctx,
            "longs low ATR medium volume with medium 3-candle and low 6-candle extension blocked",
        )
    return _setup_ok(
        "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_HIGHATR_MEDVOL_BLOCK_LONG_LOWATR_MEDVOL_PRE3_MEDIUM_PRE6_LOW",
        ctx,
    )


def setup_rmr_2h_no_far_vwap_block_long_highatr_highvol_short_block_highatr_medvol_short_block_medatr_lowvol_pre3_medium(
    ctx: CandidateContext,
) -> SetupDecision:
    """
    Research-only broad refinement.

    Keep the current leading mainline profile, but remove only the medium-ATR
    low-volume short timing pocket when the 3-candle extension is already
    medium. This preserves the lone low-extension winner while trimming the
    repeated early failures.
    """
    base = setup_rmr_2h_no_far_vwap_block_long_highatr_highvol_short_block_highatr_medvol(ctx)
    if not base.allowed:
        return base
    if (
        ctx.direction == "SHORT"
        and ctx.atr_bucket == "medium"
        and ctx.volume_bucket == "low"
        and ctx.pre_entry_move_3c_bucket == "medium"
    ):
        return _setup_reject(
            "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_HIGHATR_MEDVOL_SHORT_BLOCK_MEDATR_LOWVOL_PRE3_MEDIUM",
            ctx,
            "shorts medium ATR low volume with medium 3-candle extension blocked",
        )
    return _setup_ok(
        "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_HIGHATR_MEDVOL_SHORT_BLOCK_MEDATR_LOWVOL_PRE3_MEDIUM",
        ctx,
    )


def setup_rmr_2h_no_far_vwap_block_long_highatr_highvol_short_block_highatr_medvol_block_long_lowatr_medvol_pre3_medium_pre6_low_short_block_medatr_lowvol_pre3_medium(
    ctx: CandidateContext,
) -> SetupDecision:
    """
    Research-only broad refinement.

    Combine the two most surgical timing fixes on the current mainline: trim
    the fully losing low-ATR medium-volume long pocket with short-term
    extension still compressed, and trim the repeated medium-ATR low-volume
    short failures when the 3-candle extension is already medium.
    """
    base = setup_rmr_2h_no_far_vwap_block_long_highatr_highvol_short_block_highatr_medvol(ctx)
    if not base.allowed:
        return base
    if (
        ctx.direction == "LONG"
        and ctx.atr_bucket == "low"
        and ctx.volume_bucket == "medium"
        and ctx.pre_entry_move_3c_bucket == "medium"
        and ctx.pre_entry_move_6c_bucket == "low"
    ):
        return _setup_reject(
            "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_HIGHATR_MEDVOL_BLOCK_LONG_LOWATR_MEDVOL_PRE3_MEDIUM_PRE6_LOW_SHORT_BLOCK_MEDATR_LOWVOL_PRE3_MEDIUM",
            ctx,
            "longs low ATR medium volume with medium 3-candle and low 6-candle extension blocked",
        )
    if (
        ctx.direction == "SHORT"
        and ctx.atr_bucket == "medium"
        and ctx.volume_bucket == "low"
        and ctx.pre_entry_move_3c_bucket == "medium"
    ):
        return _setup_reject(
            "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_HIGHATR_MEDVOL_BLOCK_LONG_LOWATR_MEDVOL_PRE3_MEDIUM_PRE6_LOW_SHORT_BLOCK_MEDATR_LOWVOL_PRE3_MEDIUM",
            ctx,
            "shorts medium ATR low volume with medium 3-candle extension blocked",
        )
    return _setup_ok(
        "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_HIGHATR_MEDVOL_BLOCK_LONG_LOWATR_MEDVOL_PRE3_MEDIUM_PRE6_LOW_SHORT_BLOCK_MEDATR_LOWVOL_PRE3_MEDIUM",
        ctx,
    )


def setup_rmr_2h_no_far_vwap_block_long_highatr_highvol_short_block_highatr_medvol_short_block_highatr_lowvol_pre6_low(
    ctx: CandidateContext,
) -> SetupDecision:
    """
    Research-only broad refinement.

    Keep the current leading mainline profile, but remove the remaining high-ATR
    low-volume short pocket when the 6-candle extension is still low. This
    targets the repeated failed downside fades without trimming the stronger
    higher-extension short rebounds.
    """
    base = setup_rmr_2h_no_far_vwap_block_long_highatr_highvol_short_block_highatr_medvol(ctx)
    if not base.allowed:
        return base
    if (
        ctx.direction == "SHORT"
        and ctx.atr_bucket == "high"
        and ctx.volume_bucket == "low"
        and ctx.pre_entry_move_6c_bucket == "low"
    ):
        return _setup_reject(
            "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_HIGHATR_MEDVOL_SHORT_BLOCK_HIGHATR_LOWVOL_PRE6_LOW",
            ctx,
            "shorts high ATR low volume with low 6-candle extension blocked",
        )
    return _setup_ok(
        "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_HIGHATR_MEDVOL_SHORT_BLOCK_HIGHATR_LOWVOL_PRE6_LOW",
        ctx,
    )


def setup_rmr_2h_no_far_vwap_block_long_highatr_highvol_short_block_highatr_medvol_short_block_lowvol_nonhighatr_pre6_low(
    ctx: CandidateContext,
) -> SetupDecision:
    """
    Research-only broad refinement.

    Keep the current leading mainline profile, but remove the remaining
    low-volume short pocket outside high ATR when the 6-candle extension is
    still low. This is meant to trim early weak fades without collapsing the
    full short sample into a tiny pocket.
    """
    base = setup_rmr_2h_no_far_vwap_block_long_highatr_highvol_short_block_highatr_medvol(ctx)
    if not base.allowed:
        return base
    if (
        ctx.direction == "SHORT"
        and ctx.volume_bucket == "low"
        and ctx.atr_bucket != "high"
        and ctx.pre_entry_move_6c_bucket == "low"
    ):
        return _setup_reject(
            "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_HIGHATR_MEDVOL_SHORT_BLOCK_LOWVOL_NONHIGHATR_PRE6_LOW",
            ctx,
            "shorts non-high ATR low volume with low 6-candle extension blocked",
        )
    return _setup_ok(
        "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_HIGHATR_MEDVOL_SHORT_BLOCK_LOWVOL_NONHIGHATR_PRE6_LOW",
        ctx,
    )


def setup_rmr_2h_no_far_vwap_block_long_highatr_highvol_short_block_highatr_medvol_short_block_lowvol_pre6_low(
    ctx: CandidateContext,
) -> SetupDecision:
    """
    Research-only broad refinement.

    Combine the current mainline with a full low-volume short timing block when
    the 6-candle extension is still low, regardless of ATR bucket.
    """
    base = setup_rmr_2h_no_far_vwap_block_long_highatr_highvol_short_block_highatr_medvol(ctx)
    if not base.allowed:
        return base
    if (
        ctx.direction == "SHORT"
        and ctx.volume_bucket == "low"
        and ctx.pre_entry_move_6c_bucket == "low"
    ):
        return _setup_reject(
            "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_HIGHATR_MEDVOL_SHORT_BLOCK_LOWVOL_PRE6_LOW",
            ctx,
            "shorts low volume with low 6-candle extension blocked",
        )
    return _setup_ok(
        "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_HIGHATR_MEDVOL_SHORT_BLOCK_LOWVOL_PRE6_LOW",
        ctx,
    )


def setup_rmr_2h_no_far_vwap_block_long_highatr_highvol_short_block_highatr_medvol_block_long_lowatr_medvol_pre3_medium_pre6_low_short_block_medatr_lowvol_pre6_low(
    ctx: CandidateContext,
) -> SetupDecision:
    """
    Research-only broad refinement.

    Keep the current leading mainline profile, but remove the medium-ATR
    low-volume short pocket when the 6-candle extension is still low.
    """
    base = setup_rmr_2h_no_far_vwap_block_long_highatr_highvol_short_block_highatr_medvol_block_long_lowatr_medvol_pre3_medium_pre6_low(ctx)
    if not base.allowed:
        return base
    if (
        ctx.direction == "SHORT"
        and ctx.atr_bucket == "medium"
        and ctx.volume_bucket == "low"
        and ctx.pre_entry_move_6c_bucket == "low"
    ):
        return _setup_reject(
            "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_HIGHATR_MEDVOL_BLOCK_LONG_LOWATR_MEDVOL_PRE3_MEDIUM_PRE6_LOW_SHORT_BLOCK_MEDATR_LOWVOL_PRE6_LOW",
            ctx,
            "shorts medium ATR low volume with low 6-candle extension blocked",
        )
    return _setup_ok(
        "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_HIGHATR_MEDVOL_BLOCK_LONG_LOWATR_MEDVOL_PRE3_MEDIUM_PRE6_LOW_SHORT_BLOCK_MEDATR_LOWVOL_PRE6_LOW",
        ctx,
    )


def setup_rmr_2h_no_far_vwap_block_long_highatr_highvol_short_block_highatr_medvol_block_long_lowatr_medvol_pre3_medium_pre6_low_block_long_medatr_medvol_pre3_low_pre6_low(
    ctx: CandidateContext,
) -> SetupDecision:
    """
    Research-only broad refinement.

    Keep the current leading mainline profile, but remove the remaining
    medium-ATR medium-volume long pocket when both the 3-candle and 6-candle
    extensions are still low.
    """
    base = setup_rmr_2h_no_far_vwap_block_long_highatr_highvol_short_block_highatr_medvol_block_long_lowatr_medvol_pre3_medium_pre6_low(ctx)
    if not base.allowed:
        return base
    if (
        ctx.direction == "LONG"
        and ctx.atr_bucket == "medium"
        and ctx.volume_bucket == "medium"
        and ctx.pre_entry_move_3c_bucket == "low"
        and ctx.pre_entry_move_6c_bucket == "low"
    ):
        return _setup_reject(
            "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_HIGHATR_MEDVOL_BLOCK_LONG_LOWATR_MEDVOL_PRE3_MEDIUM_PRE6_LOW_BLOCK_LONG_MEDATR_MEDVOL_PRE3_LOW_PRE6_LOW",
            ctx,
            "longs medium ATR medium volume with low 3-candle and low 6-candle extension blocked",
        )
    return _setup_ok(
        "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_HIGHATR_MEDVOL_BLOCK_LONG_LOWATR_MEDVOL_PRE3_MEDIUM_PRE6_LOW_BLOCK_LONG_MEDATR_MEDVOL_PRE3_LOW_PRE6_LOW",
        ctx,
    )


def setup_rmr_2h_no_far_vwap_block_long_highatr_highvol_short_block_highatr_medvol_block_long_lowatr_medvol_pre3_medium_pre6_low_short_block_medatr_highvol_pre6_low(
    ctx: CandidateContext,
) -> SetupDecision:
    """
    Research-only broad refinement.

    Keep the current best primary profile, but trim the narrow medium-ATR
    high-volume short pocket when the 6-candle extension is still low.
    """
    base = setup_rmr_2h_no_far_vwap_block_long_highatr_highvol_short_block_highatr_medvol_block_long_lowatr_medvol_pre3_medium_pre6_low(ctx)
    if not base.allowed:
        return base
    if (
        ctx.direction == "SHORT"
        and ctx.atr_bucket == "medium"
        and ctx.volume_bucket == "high"
        and ctx.pre_entry_move_6c_bucket == "low"
    ):
        return _setup_reject(
            "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_HIGHATR_MEDVOL_BLOCK_LONG_LOWATR_MEDVOL_PRE3_MEDIUM_PRE6_LOW_SHORT_BLOCK_MEDATR_HIGHVOL_PRE6_LOW",
            ctx,
            "shorts medium ATR high volume with low 6-candle extension blocked",
        )
    return _setup_ok(
        "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_HIGHATR_MEDVOL_BLOCK_LONG_LOWATR_MEDVOL_PRE3_MEDIUM_PRE6_LOW_SHORT_BLOCK_MEDATR_HIGHVOL_PRE6_LOW",
        ctx,
    )


def setup_rmr_2h_no_far_vwap_block_long_highatr_highvol_short_block_highatr_medvol_block_long_lowatr_medvol_pre3_medium_pre6_low_block_long_highatr_lowvol_pre3_medium_pre6_low(
    ctx: CandidateContext,
) -> SetupDecision:
    """
    Research-only broad refinement.

    Keep the current best primary profile, but trim the narrow high-ATR
    low-volume long pocket when the 3-candle extension is medium and the
    6-candle extension is still low.
    """
    base = setup_rmr_2h_no_far_vwap_block_long_highatr_highvol_short_block_highatr_medvol_block_long_lowatr_medvol_pre3_medium_pre6_low(ctx)
    if not base.allowed:
        return base
    if (
        ctx.direction == "LONG"
        and ctx.atr_bucket == "high"
        and ctx.volume_bucket == "low"
        and ctx.pre_entry_move_3c_bucket == "medium"
        and ctx.pre_entry_move_6c_bucket == "low"
    ):
        return _setup_reject(
            "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_HIGHATR_MEDVOL_BLOCK_LONG_LOWATR_MEDVOL_PRE3_MEDIUM_PRE6_LOW_BLOCK_LONG_HIGHATR_LOWVOL_PRE3_MEDIUM_PRE6_LOW",
            ctx,
            "longs high ATR low volume with medium 3-candle and low 6-candle extension blocked",
        )
    return _setup_ok(
        "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_HIGHATR_MEDVOL_BLOCK_LONG_LOWATR_MEDVOL_PRE3_MEDIUM_PRE6_LOW_BLOCK_LONG_HIGHATR_LOWVOL_PRE3_MEDIUM_PRE6_LOW",
        ctx,
    )


def setup_rmr_2h_no_far_vwap_block_long_highatr_highvol_short_block_highatr_medvol_block_long_lowatr_medvol_pre3_medium_pre6_low_block_long_highatr_lowvol_pre3_medium_pre6_low_short_block_medatr_highvol_pre6_low(
    ctx: CandidateContext,
) -> SetupDecision:
    """
    Research-only broad refinement.

    Combine the two remaining surgical timing cuts on the current best primary:
    the medium-ATR high-volume short pocket with low 6-candle extension and
    the high-ATR low-volume long pocket with medium 3-candle / low 6-candle
    extension.
    """
    base = setup_rmr_2h_no_far_vwap_block_long_highatr_highvol_short_block_highatr_medvol_block_long_lowatr_medvol_pre3_medium_pre6_low(ctx)
    if not base.allowed:
        return base
    if (
        ctx.direction == "SHORT"
        and ctx.atr_bucket == "medium"
        and ctx.volume_bucket == "high"
        and ctx.pre_entry_move_6c_bucket == "low"
    ):
        return _setup_reject(
            "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_HIGHATR_MEDVOL_BLOCK_LONG_LOWATR_MEDVOL_PRE3_MEDIUM_PRE6_LOW_BLOCK_LONG_HIGHATR_LOWVOL_PRE3_MEDIUM_PRE6_LOW_SHORT_BLOCK_MEDATR_HIGHVOL_PRE6_LOW",
            ctx,
            "shorts medium ATR high volume with low 6-candle extension blocked",
        )
    if (
        ctx.direction == "LONG"
        and ctx.atr_bucket == "high"
        and ctx.volume_bucket == "low"
        and ctx.pre_entry_move_3c_bucket == "medium"
        and ctx.pre_entry_move_6c_bucket == "low"
    ):
        return _setup_reject(
            "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_HIGHATR_MEDVOL_BLOCK_LONG_LOWATR_MEDVOL_PRE3_MEDIUM_PRE6_LOW_BLOCK_LONG_HIGHATR_LOWVOL_PRE3_MEDIUM_PRE6_LOW_SHORT_BLOCK_MEDATR_HIGHVOL_PRE6_LOW",
            ctx,
            "longs high ATR low volume with medium 3-candle and low 6-candle extension blocked",
        )
    return _setup_ok(
        "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_HIGHATR_MEDVOL_BLOCK_LONG_LOWATR_MEDVOL_PRE3_MEDIUM_PRE6_LOW_BLOCK_LONG_HIGHATR_LOWVOL_PRE3_MEDIUM_PRE6_LOW_SHORT_BLOCK_MEDATR_HIGHVOL_PRE6_LOW",
        ctx,
    )


def setup_rmr_2h_no_far_vwap_block_long_highatr_highvol_block_long_lowatr_medvol(
    ctx: CandidateContext,
) -> SetupDecision:
    """
    Research-only broad refinement.

    Keep the current best primary profile, but remove the weak long bucket
    where ATR is low and volume is medium.
    """
    base = setup_rmr_2h_no_far_vwap_block_long_highatr_highvol(ctx)
    if not base.allowed:
        return base
    if ctx.direction == "LONG" and ctx.atr_bucket == "low" and ctx.volume_bucket == "medium":
        return _setup_reject(
            "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_BLOCK_LONG_LOWATR_MEDVOL",
            ctx,
            "longs low ATR medium volume blocked",
        )
    return _setup_ok(
        "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_BLOCK_LONG_LOWATR_MEDVOL",
        ctx,
    )


def setup_rmr_2h_no_far_vwap_long_biased_block_long_highatr_highvol_short_lowatr(
    ctx: CandidateContext,
) -> SetupDecision:
    """
    Research-only bridge refinement.

    Combine the strongest current long-side filter with the strongest current
    short-side diagnostic: only keep low-ATR shorts after the surgical
    high-ATR/high-volume long block.
    """
    base = setup_rmr_2h_no_far_vwap_long_biased_block_long_highatr_highvol(ctx)
    if not base.allowed:
        return base
    if ctx.direction == "SHORT" and ctx.atr_bucket != "low":
        return _setup_reject(
            "RMR_2H_NO_FAR_VWAP_LONG_BIASED_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_LOWATR",
            ctx,
            "shorts require low ATR bucket",
        )
    return _setup_ok(
        "RMR_2H_NO_FAR_VWAP_LONG_BIASED_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_LOWATR",
        ctx,
    )


def setup_rmr_2h_no_far_vwap_long_biased_block_long_highatr_highvol_short_block_medatr_lowmedvol(
    ctx: CandidateContext,
) -> SetupDecision:
    """
    Research-only bridge refinement.

    Keep the surgical long-side block and the broad long-biased distribution,
    but remove only the clearly bad medium-ATR short trades when volume is not
    high. This is intentionally milder than forcing all shorts into low ATR.
    """
    base = setup_rmr_2h_no_far_vwap_long_biased_block_long_highatr_highvol(ctx)
    if not base.allowed:
        return base
    if ctx.direction == "SHORT" and ctx.atr_bucket == "medium" and ctx.volume_bucket != "high":
        return _setup_reject(
            "RMR_2H_NO_FAR_VWAP_LONG_BIASED_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_MEDATR_LOWMEDVOL",
            ctx,
            "shorts medium ATR low/medium volume blocked",
        )
    return _setup_ok(
        "RMR_2H_NO_FAR_VWAP_LONG_BIASED_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_MEDATR_LOWMEDVOL",
        ctx,
    )


def setup_rmr_2h_no_far_vwap_long_biased_block_long_highatr_highvol_short_block_medatr_lowvol(
    ctx: CandidateContext,
) -> SetupDecision:
    """
    Research-only bridge refinement.

    Keep the surgical long-side block and the broad long-biased distribution,
    but remove only the medium-ATR low-volume short pocket.
    """
    base = setup_rmr_2h_no_far_vwap_long_biased_block_long_highatr_highvol(ctx)
    if not base.allowed:
        return base
    if ctx.direction == "SHORT" and ctx.atr_bucket == "medium" and ctx.volume_bucket == "low":
        return _setup_reject(
            "RMR_2H_NO_FAR_VWAP_LONG_BIASED_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_MEDATR_LOWVOL",
            ctx,
            "shorts medium ATR low volume blocked",
        )
    return _setup_ok(
        "RMR_2H_NO_FAR_VWAP_LONG_BIASED_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_MEDATR_LOWVOL",
        ctx,
    )


def setup_rmr_2h_no_far_vwap_long_biased_block_long_highatr_highvol_block_long_lowatr_medvol(
    ctx: CandidateContext,
) -> SetupDecision:
    """
    Research-only bridge refinement.

    Keep the current best long-biased refinement, but remove the weak long
    bucket where ATR is low and volume is medium.
    """
    base = setup_rmr_2h_no_far_vwap_long_biased_block_long_highatr_highvol(ctx)
    if not base.allowed:
        return base
    if ctx.direction == "LONG" and ctx.atr_bucket == "low" and ctx.volume_bucket == "medium":
        return _setup_reject(
            "RMR_2H_NO_FAR_VWAP_LONG_BIASED_BLOCK_LONG_HIGHATR_HIGHVOL_BLOCK_LONG_LOWATR_MEDVOL",
            ctx,
            "longs low ATR medium volume blocked",
        )
    return _setup_ok(
        "RMR_2H_NO_FAR_VWAP_LONG_BIASED_BLOCK_LONG_HIGHATR_HIGHVOL_BLOCK_LONG_LOWATR_MEDVOL",
        ctx,
    )


def setup_rmr_2h_no_far_vwap_long_biased_block_long_highatr_highvol_block_long_lowatr_medvol_pre3_medium_pre6_low(
    ctx: CandidateContext,
) -> SetupDecision:
    """
    Research-only bridge refinement.

    Keep the best long-biased refinement intact except for the narrow long
    timing pocket that remained fully losing in diagnostics: low ATR, medium
    volume, medium 3-candle extension, and low 6-candle extension.
    """
    base = setup_rmr_2h_no_far_vwap_long_biased_block_long_highatr_highvol(ctx)
    if not base.allowed:
        return base
    if (
        ctx.direction == "LONG"
        and ctx.atr_bucket == "low"
        and ctx.volume_bucket == "medium"
        and ctx.pre_entry_move_3c_bucket == "medium"
        and ctx.pre_entry_move_6c_bucket == "low"
    ):
        return _setup_reject(
            "RMR_2H_NO_FAR_VWAP_LONG_BIASED_BLOCK_LONG_HIGHATR_HIGHVOL_BLOCK_LONG_LOWATR_MEDVOL_PRE3_MEDIUM_PRE6_LOW",
            ctx,
            "longs low ATR medium volume with medium 3-candle and low 6-candle extension blocked",
        )
    return _setup_ok(
        "RMR_2H_NO_FAR_VWAP_LONG_BIASED_BLOCK_LONG_HIGHATR_HIGHVOL_BLOCK_LONG_LOWATR_MEDVOL_PRE3_MEDIUM_PRE6_LOW",
        ctx,
    )


def setup_rmr_2h_no_far_vwap_long_biased_block_long_highatr_highvol_block_long_lowatr_medvol_pre3_medium_pre6_low_short_block_medatr_lowvol_pre6_low(
    ctx: CandidateContext,
) -> SetupDecision:
    """
    Research-only bridge refinement.

    Keep the best current long-biased refinement intact, but trim the remaining
    medium-ATR low-volume short pocket when the 6-candle extension is still
    low.
    """
    base = setup_rmr_2h_no_far_vwap_long_biased_block_long_highatr_highvol_block_long_lowatr_medvol_pre3_medium_pre6_low(ctx)
    if not base.allowed:
        return base
    if (
        ctx.direction == "SHORT"
        and ctx.atr_bucket == "medium"
        and ctx.volume_bucket == "low"
        and ctx.pre_entry_move_6c_bucket == "low"
    ):
        return _setup_reject(
            "RMR_2H_NO_FAR_VWAP_LONG_BIASED_BLOCK_LONG_HIGHATR_HIGHVOL_BLOCK_LONG_LOWATR_MEDVOL_PRE3_MEDIUM_PRE6_LOW_SHORT_BLOCK_MEDATR_LOWVOL_PRE6_LOW",
            ctx,
            "shorts medium ATR low volume with low 6-candle extension blocked",
        )
    return _setup_ok(
        "RMR_2H_NO_FAR_VWAP_LONG_BIASED_BLOCK_LONG_HIGHATR_HIGHVOL_BLOCK_LONG_LOWATR_MEDVOL_PRE3_MEDIUM_PRE6_LOW_SHORT_BLOCK_MEDATR_LOWVOL_PRE6_LOW",
        ctx,
    )


def setup_rmr_2h_no_far_vwap_long_biased_block_long_highatr_highvol_block_long_lowatr_medvol_pre3_medium_pre6_low_block_long_medatr_medvol_pre3_low_pre6_low(
    ctx: CandidateContext,
) -> SetupDecision:
    """
    Research-only bridge refinement.

    Keep the current best long-biased refinement, but remove the remaining
    medium-ATR medium-volume long pocket when both the 3-candle and 6-candle
    extensions are still low.
    """
    base = setup_rmr_2h_no_far_vwap_long_biased_block_long_highatr_highvol_block_long_lowatr_medvol_pre3_medium_pre6_low(ctx)
    if not base.allowed:
        return base
    if (
        ctx.direction == "LONG"
        and ctx.atr_bucket == "medium"
        and ctx.volume_bucket == "medium"
        and ctx.pre_entry_move_3c_bucket == "low"
        and ctx.pre_entry_move_6c_bucket == "low"
    ):
        return _setup_reject(
            "RMR_2H_NO_FAR_VWAP_LONG_BIASED_BLOCK_LONG_HIGHATR_HIGHVOL_BLOCK_LONG_LOWATR_MEDVOL_PRE3_MEDIUM_PRE6_LOW_BLOCK_LONG_MEDATR_MEDVOL_PRE3_LOW_PRE6_LOW",
            ctx,
            "longs medium ATR medium volume with low 3-candle and low 6-candle extension blocked",
        )
    return _setup_ok(
        "RMR_2H_NO_FAR_VWAP_LONG_BIASED_BLOCK_LONG_HIGHATR_HIGHVOL_BLOCK_LONG_LOWATR_MEDVOL_PRE3_MEDIUM_PRE6_LOW_BLOCK_LONG_MEDATR_MEDVOL_PRE3_LOW_PRE6_LOW",
        ctx,
    )


def setup_rmr_2h_no_far_vwap_long_biased_block_long_highatr_highvol_block_long_lowatr_medvol_pre3_medium_pre6_low_block_long_highatr_lowvol_pre3_medium_pre6_low(
    ctx: CandidateContext,
) -> SetupDecision:
    """
    Research-only bridge refinement.

    Keep the current best long-biased refinement, but trim the narrow high-ATR
    low-volume long pocket when the 3-candle extension is medium and the
    6-candle extension is still low.
    """
    base = setup_rmr_2h_no_far_vwap_long_biased_block_long_highatr_highvol_block_long_lowatr_medvol_pre3_medium_pre6_low(ctx)
    if not base.allowed:
        return base
    if (
        ctx.direction == "LONG"
        and ctx.atr_bucket == "high"
        and ctx.volume_bucket == "low"
        and ctx.pre_entry_move_3c_bucket == "medium"
        and ctx.pre_entry_move_6c_bucket == "low"
    ):
        return _setup_reject(
            "RMR_2H_NO_FAR_VWAP_LONG_BIASED_BLOCK_LONG_HIGHATR_HIGHVOL_BLOCK_LONG_LOWATR_MEDVOL_PRE3_MEDIUM_PRE6_LOW_BLOCK_LONG_HIGHATR_LOWVOL_PRE3_MEDIUM_PRE6_LOW",
            ctx,
            "longs high ATR low volume with medium 3-candle and low 6-candle extension blocked",
        )
    return _setup_ok(
        "RMR_2H_NO_FAR_VWAP_LONG_BIASED_BLOCK_LONG_HIGHATR_HIGHVOL_BLOCK_LONG_LOWATR_MEDVOL_PRE3_MEDIUM_PRE6_LOW_BLOCK_LONG_HIGHATR_LOWVOL_PRE3_MEDIUM_PRE6_LOW",
        ctx,
    )


def setup_rmr_2h_no_far_vwap_medium_atr(ctx: CandidateContext) -> SetupDecision:
    return _setup_range_mean_reversion_regime_variant(
        ctx,
        name="RMR_2H_NO_FAR_VWAP_MEDIUM_ATR",
        block_far_vwap=True,
        medium_atr_only=True,
    )


def setup_rmr_2h_no_far_vwap_allowed_range(ctx: CandidateContext) -> SetupDecision:
    return _setup_range_mean_reversion_regime_variant(
        ctx,
        name="RMR_2H_NO_FAR_VWAP_ALLOWED_RANGE",
        block_far_vwap=True,
        require_allowed_range_context=True,
    )


def _setup_far_vwap_mean_reversion(
    ctx: CandidateContext,
    *,
    name: str,
    allow_long: bool,
    allow_short: bool,
    exclude_high_volume: bool = False,
    medium_atr_only: bool = False,
    allow_high_volume: bool = True,
) -> SetupDecision:
    base = setup_range_mean_reversion(ctx)
    if not base.allowed:
        return _setup_reject(name, ctx, base.rejection_reason)
    if ctx.vwap_distance_bucket != "far":
        return _setup_reject(name, ctx, "requires far VWAP-distance bucket")
    if ctx.direction == "LONG":
        if not allow_long:
            return _setup_reject(name, ctx, "longs not allowed in this far-VWAP variant")
        if ctx.price_vs_vwap != "below":
            return _setup_reject(name, ctx, "long far-VWAP entry must be below VWAP")
    elif ctx.direction == "SHORT":
        if not allow_short:
            return _setup_reject(name, ctx, "shorts not allowed in this far-VWAP variant")
        if ctx.price_vs_vwap != "above":
            return _setup_reject(name, ctx, "short far-VWAP entry must be above VWAP")
    else:
        return _setup_reject(name, ctx, "unsupported direction")
    if exclude_high_volume and ctx.volume_bucket == "high":
        return _setup_reject(name, ctx, "high-volume bucket excluded")
    if not allow_high_volume and ctx.volume_bucket == "high":
        return _setup_reject(name, ctx, "high-volume bucket blocked")
    if medium_atr_only and ctx.atr_bucket != "medium":
        return _setup_reject(name, ctx, "medium ATR bucket required")
    return _setup_ok(name, ctx)


_FAR_VWAP_VARIANT_DESCRIPTIONS: dict[str, str] = {
    "FV_MR_0": "Far-VWAP mean reversion, both directions",
    "FV_MR_1": "Far-VWAP mean reversion, long-only",
    "FV_MR_2": "Far-VWAP long-only, exclude high-volume bucket",
    "FV_MR_3": "Far-VWAP long-only, medium ATR bucket only",
    "FV_MR_4": "Far-VWAP long-only, low/medium-volume only",
    "FV_MR_5": "Far-VWAP both directions, exclude high-volume bucket",
    "FV_MR_6": "Far-VWAP short-only reference",
}
_FAR_VWAP_SETUP_FAMILY = "FAR_VWAP_MEAN_REVERSION"
_FAR_VWAP_VARIANT_ORDER = tuple(_FAR_VWAP_VARIANT_DESCRIPTIONS.keys())
_RANGE_MR_2H_VARIANT_DESCRIPTIONS: dict[str, str] = {
    "RMR_2H_LONG_ONLY": "2H broad range MR, long-only",
    "RMR_2H_NO_FAR_VWAP": "2H broad range MR, far VWAP bucket blocked",
    "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL": (
        "2H broad range MR, far VWAP blocked and longs avoid the high ATR + high volume pocket"
    ),
    "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_LOWATR_OR_HIGHATR_HIGHVOL": (
        "2H broad range MR, surgical long block and shorts limited to low ATR or high ATR + high volume"
    ),
    "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_LOWATR_OR_HIGHVOL": (
        "2H broad range MR, surgical long block and shorts limited to low ATR or any high-volume pocket"
    ),
    "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_HIGHATR_LOWMEDVOL": (
        "2H broad range MR, surgical long block and shorts avoid high ATR low/medium volume"
    ),
    "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_MEDATR_LOWMEDVOL": (
        "2H broad range MR, surgical long block and shorts avoid medium ATR low/medium volume"
    ),
    "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_MEDATR_LOWVOL": (
        "2H broad range MR, surgical long block and shorts avoid medium ATR low volume"
    ),
    "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_NONLOWATR_LOWVOL": (
        "2H broad range MR, surgical long block and shorts avoid low volume unless ATR is low"
    ),
    "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_HIGHATR_MEDVOL": (
        "2H broad range MR, surgical long block and shorts avoid high ATR medium volume"
    ),
    "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_HIGHATR_MEDVOL_AND_MEDATR_LOWVOL": (
        "2H broad range MR, surgical long block and shorts avoid high ATR medium volume plus medium ATR low volume"
    ),
    "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_BLOCK_LONG_LOWATR_MEDVOL_PRE3_MEDIUM": (
        "2H broad range MR, mainline plus low ATR medium-volume longs with medium 3-candle extension blocked"
    ),
    "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_HIGHATR_MEDVOL_SHORT_BLOCK_HIGHATR_LOWVOL_PRE3_MEDIUM": (
        "2H broad range MR, mainline plus high ATR low-volume shorts with medium 3-candle extension blocked"
    ),
    "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_HIGHATR_MEDVOL_BLOCK_LONG_LOWATR_MEDVOL_PRE3_MEDIUM_PRE6_LOW": (
        "2H broad range MR, mainline plus low ATR medium-volume longs with medium 3-candle and low 6-candle extension blocked"
    ),
    "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_HIGHATR_MEDVOL_SHORT_BLOCK_MEDATR_LOWVOL_PRE3_MEDIUM": (
        "2H broad range MR, mainline plus medium ATR low-volume shorts with medium 3-candle extension blocked"
    ),
    "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_HIGHATR_MEDVOL_BLOCK_LONG_LOWATR_MEDVOL_PRE3_MEDIUM_PRE6_LOW_SHORT_BLOCK_MEDATR_LOWVOL_PRE3_MEDIUM": (
        "2H broad range MR, mainline plus the surgical low ATR medium-volume long pocket block and medium ATR low-volume short timing block"
    ),
    "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_HIGHATR_MEDVOL_SHORT_BLOCK_HIGHATR_LOWVOL_PRE6_LOW": (
        "2H broad range MR, mainline plus high ATR low-volume shorts with low 6-candle extension blocked"
    ),
    "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_HIGHATR_MEDVOL_SHORT_BLOCK_LOWVOL_NONHIGHATR_PRE6_LOW": (
        "2H broad range MR, mainline plus non-high ATR low-volume shorts with low 6-candle extension blocked"
    ),
    "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_HIGHATR_MEDVOL_SHORT_BLOCK_LOWVOL_PRE6_LOW": (
        "2H broad range MR, mainline plus all low-volume shorts with low 6-candle extension blocked"
    ),
    "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_HIGHATR_MEDVOL_BLOCK_LONG_LOWATR_MEDVOL_PRE3_MEDIUM_PRE6_LOW_SHORT_BLOCK_MEDATR_LOWVOL_PRE6_LOW": (
        "2H broad range MR, mainline plus medium ATR low-volume shorts with low 6-candle extension blocked"
    ),
    "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_HIGHATR_MEDVOL_BLOCK_LONG_LOWATR_MEDVOL_PRE3_MEDIUM_PRE6_LOW_BLOCK_LONG_MEDATR_MEDVOL_PRE3_LOW_PRE6_LOW": (
        "2H broad range MR, mainline plus medium ATR medium-volume longs with low 3-candle and low 6-candle extension blocked"
    ),
    "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_HIGHATR_MEDVOL_BLOCK_LONG_LOWATR_MEDVOL_PRE3_MEDIUM_PRE6_LOW_SHORT_BLOCK_MEDATR_HIGHVOL_PRE6_LOW": (
        "2H broad range MR, mainline plus medium ATR high-volume shorts with low 6-candle extension blocked"
    ),
    "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_HIGHATR_MEDVOL_BLOCK_LONG_LOWATR_MEDVOL_PRE3_MEDIUM_PRE6_LOW_BLOCK_LONG_HIGHATR_LOWVOL_PRE3_MEDIUM_PRE6_LOW": (
        "2H broad range MR, mainline plus high ATR low-volume longs with medium 3-candle and low 6-candle extension blocked"
    ),
    "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_HIGHATR_MEDVOL_BLOCK_LONG_LOWATR_MEDVOL_PRE3_MEDIUM_PRE6_LOW_BLOCK_LONG_HIGHATR_LOWVOL_PRE3_MEDIUM_PRE6_LOW_SHORT_BLOCK_MEDATR_HIGHVOL_PRE6_LOW": (
        "2H broad range MR, mainline plus the surgical high ATR low-volume long pocket block and medium ATR high-volume short block"
    ),
    "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_BLOCK_LONG_LOWATR_MEDVOL": (
        "2H broad range MR, surgical long block and longs avoid low ATR + medium volume"
    ),
    "RMR_2H_NO_FAR_VWAP_LONG_BIASED": "2H broad range MR, far VWAP blocked and shorts high ATR blocked",
    "RMR_2H_NO_FAR_VWAP_LONG_BIASED_SHORT_LOWATR": (
        "2H broad range MR, far VWAP blocked and shorts limited to low ATR"
    ),
    "RMR_2H_NO_FAR_VWAP_LONG_BIASED_BLOCK_LONG_HIGHATR_HIGHVOL": (
        "2H broad range MR, shorts high ATR blocked and longs avoid the high ATR + high volume pocket"
    ),
    "RMR_2H_NO_FAR_VWAP_LONG_BIASED_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_LOWATR": (
        "2H broad range MR, surgical long block and shorts limited to low ATR"
    ),
    "RMR_2H_NO_FAR_VWAP_LONG_BIASED_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_MEDATR_LOWMEDVOL": (
        "2H broad range MR, surgical long block and shorts avoid medium ATR low/medium volume"
    ),
    "RMR_2H_NO_FAR_VWAP_LONG_BIASED_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_MEDATR_LOWVOL": (
        "2H broad range MR, surgical long block and shorts avoid medium ATR low volume"
    ),
    "RMR_2H_NO_FAR_VWAP_LONG_BIASED_BLOCK_LONG_HIGHATR_HIGHVOL_BLOCK_LONG_LOWATR_MEDVOL": (
        "2H broad range MR, surgical long block and longs avoid low ATR + medium volume"
    ),
    "RMR_2H_NO_FAR_VWAP_LONG_BIASED_BLOCK_LONG_HIGHATR_HIGHVOL_BLOCK_LONG_LOWATR_MEDVOL_PRE3_MEDIUM_PRE6_LOW": (
        "2H broad range MR, long-biased surgical long block plus low ATR medium-volume longs with medium 3-candle and low 6-candle extension blocked"
    ),
    "RMR_2H_NO_FAR_VWAP_LONG_BIASED_BLOCK_LONG_HIGHATR_HIGHVOL_BLOCK_LONG_LOWATR_MEDVOL_PRE3_MEDIUM_PRE6_LOW_SHORT_BLOCK_MEDATR_LOWVOL_PRE6_LOW": (
        "2H broad range MR, long-biased surgical long block plus medium ATR low-volume shorts with low 6-candle extension blocked"
    ),
    "RMR_2H_NO_FAR_VWAP_LONG_BIASED_BLOCK_LONG_HIGHATR_HIGHVOL_BLOCK_LONG_LOWATR_MEDVOL_PRE3_MEDIUM_PRE6_LOW_BLOCK_LONG_MEDATR_MEDVOL_PRE3_LOW_PRE6_LOW": (
        "2H broad range MR, long-biased surgical long block plus medium ATR medium-volume longs with low 3-candle and low 6-candle extension blocked"
    ),
    "RMR_2H_NO_FAR_VWAP_LONG_BIASED_BLOCK_LONG_HIGHATR_HIGHVOL_BLOCK_LONG_LOWATR_MEDVOL_PRE3_MEDIUM_PRE6_LOW_BLOCK_LONG_HIGHATR_LOWVOL_PRE3_MEDIUM_PRE6_LOW": (
        "2H broad range MR, long-biased surgical long block plus high ATR low-volume longs with medium 3-candle and low 6-candle extension blocked"
    ),
    "RMR_2H_NO_FAR_VWAP_LONG_BIASED_MEDVOL_SHORTS": (
        "2H broad range MR, far VWAP blocked and shorts limited to medium volume outside high ATR"
    ),
    "RMR_2H_NO_FAR_VWAP_LONG_BIASED_LOWATR_MEDVOL_SHORTS": (
        "2H broad range MR, far VWAP blocked and shorts limited to low ATR + medium volume"
    ),
    "RMR_2H_NO_FAR_VWAP_NO_HIGH_ATR_NO_HIGH_VOL": (
        "2H broad range MR, far VWAP blocked and high ATR/high volume blocked"
    ),
    "RMR_2H_NO_FAR_VWAP_LONG_BIASED_NO_HIGH_ATR_NO_HIGH_VOL": (
        "2H broad range MR, far VWAP blocked and shorts avoid high ATR/high volume"
    ),
    "RMR_2H_NO_FAR_VWAP_LONG_BIASED_BLOCK_LONG_HIGH_VOL": (
        "2H broad range MR, far VWAP blocked, shorts high ATR blocked, and longs avoid high volume"
    ),
    "RMR_2H_NO_FAR_VWAP_LONG_BIASED_BLOCK_LONG_HIGH_VOL_SHORT_LOWATR": (
        "2H broad range MR, longs avoid high volume and shorts require low ATR"
    ),
    "RMR_2H_NO_FAR_VWAP_LONG_BIASED_BLOCK_LONG_HIGH_VOL_SHORT_LOWATR_MEDVOL": (
        "2H broad range MR, longs avoid high volume and shorts require low ATR + medium volume"
    ),
    "RMR_2H_NO_FAR_VWAP_LONG_BIASED_BLOCK_LONG_HIGH_VOL_SHORT_NO_HIGH_VOL": (
        "2H broad range MR, longs avoid high volume and shorts avoid high volume"
    ),
    "RMR_2H_NO_FAR_VWAP_LONG_BIASED_BLOCK_LONG_HIGH_ATR_OR_HIGH_VOL": (
        "2H broad range MR, shorts high ATR blocked and longs avoid high ATR/high volume"
    ),
    "RMR_2H_NO_HIGH_ATR": "2H broad range MR, high ATR bucket blocked",
    "RMR_2H_NO_FAR_VWAP_NO_HIGH_ATR": "2H broad range MR, far VWAP and high ATR blocked",
    "RMR_2H_LONG_ONLY_NO_FAR_VWAP": "2H broad range MR, long-only with far VWAP blocked",
    "RMR_2H_LONG_ONLY_NO_FAR_VWAP_NO_HIGH_ATR": (
        "2H broad range MR, long-only with far VWAP and high ATR blocked"
    ),
    "RMR_2H_LONG_ONLY_NO_FAR_VWAP_NO_HIGH_VOL": (
        "2H broad range MR, long-only with far VWAP and high volume blocked"
    ),
}
_RANGE_MR_2H_VARIANT_ORDER = tuple(_RANGE_MR_2H_VARIANT_DESCRIPTIONS.keys())


def _setup_family_name(setup_name: str) -> str:
    if setup_name in _FAR_VWAP_VARIANT_DESCRIPTIONS:
        return _FAR_VWAP_SETUP_FAMILY
    if setup_name in _RANGE_MR_2H_VARIANT_DESCRIPTIONS:
        return "RANGE_MEAN_REVERSION"
    return setup_name


def _setup_variant_name(setup_name: str) -> str:
    if setup_name in _FAR_VWAP_VARIANT_DESCRIPTIONS:
        return setup_name
    if setup_name in _RANGE_MR_2H_VARIANT_DESCRIPTIONS:
        return setup_name
    return ""


def _entry_combo_name(setup_name: str, entry_profile_name: str) -> str:
    if not entry_profile_name:
        return setup_name
    return f"{setup_name}::{entry_profile_name}"


def setup_fv_mr_0(ctx: CandidateContext) -> SetupDecision:
    return _setup_far_vwap_mean_reversion(
        ctx,
        name="FV_MR_0",
        allow_long=True,
        allow_short=True,
    )


def setup_fv_mr_1(ctx: CandidateContext) -> SetupDecision:
    return _setup_far_vwap_mean_reversion(
        ctx,
        name="FV_MR_1",
        allow_long=True,
        allow_short=False,
    )


def setup_fv_mr_2(ctx: CandidateContext) -> SetupDecision:
    return _setup_far_vwap_mean_reversion(
        ctx,
        name="FV_MR_2",
        allow_long=True,
        allow_short=False,
        exclude_high_volume=True,
    )


def setup_fv_mr_3(ctx: CandidateContext) -> SetupDecision:
    return _setup_far_vwap_mean_reversion(
        ctx,
        name="FV_MR_3",
        allow_long=True,
        allow_short=False,
        medium_atr_only=True,
    )


def setup_fv_mr_4(ctx: CandidateContext) -> SetupDecision:
    return _setup_far_vwap_mean_reversion(
        ctx,
        name="FV_MR_4",
        allow_long=True,
        allow_short=False,
        allow_high_volume=False,
    )


def setup_fv_mr_5(ctx: CandidateContext) -> SetupDecision:
    return _setup_far_vwap_mean_reversion(
        ctx,
        name="FV_MR_5",
        allow_long=True,
        allow_short=True,
        exclude_high_volume=True,
    )


def setup_fv_mr_6(ctx: CandidateContext) -> SetupDecision:
    return _setup_far_vwap_mean_reversion(
        ctx,
        name="FV_MR_6",
        allow_long=False,
        allow_short=True,
    )


SETUP_REGISTRY: dict[str, ResearchSetup] = {
    "MACD_VWAP_BASE": ResearchSetup("MACD_VWAP_BASE", "consensus", setup_macd_vwap_base),
    "MACD_VWAP_SHORT_ONLY": ResearchSetup("MACD_VWAP_SHORT_ONLY", "consensus", setup_macd_vwap_short_only),
    "MACD_VWAP_NO_MEDIUM_ATR": ResearchSetup("MACD_VWAP_NO_MEDIUM_ATR", "consensus", setup_macd_vwap_no_medium_atr),
    "MACD_VWAP_SHORT_ONLY_NO_MEDIUM_ATR": ResearchSetup(
        "MACD_VWAP_SHORT_ONLY_NO_MEDIUM_ATR",
        "consensus",
        setup_macd_vwap_short_only_no_medium_atr,
    ),
    "MACD_VWAP_VOLUME": ResearchSetup("MACD_VWAP_VOLUME", "consensus", setup_macd_vwap_volume),
    "MACD_VOLUME_ONLY": ResearchSetup("MACD_VOLUME_ONLY", "consensus", setup_macd_volume_only),
    "MACD_ONLY_REFERENCE": ResearchSetup("MACD_ONLY_REFERENCE", "consensus", setup_macd_only_reference),
    "VOLUME_BREAKOUT_CONTINUATION": ResearchSetup(
        "VOLUME_BREAKOUT_CONTINUATION",
        "volume_breakout",
        setup_volume_breakout_continuation,
    ),
    "PULLBACK_TO_TREND_CONTINUATION": ResearchSetup(
        "PULLBACK_TO_TREND_CONTINUATION",
        "pullback_trend",
        setup_pullback_to_trend_continuation,
    ),
    "RANGE_MEAN_REVERSION": ResearchSetup(
        "RANGE_MEAN_REVERSION",
        "range_mean_reversion",
        setup_range_mean_reversion,
    ),
    "RMR_2H_LONG_ONLY": ResearchSetup("RMR_2H_LONG_ONLY", "range_mean_reversion", setup_rmr_2h_long_only),
    "RMR_2H_NO_FAR_VWAP": ResearchSetup(
        "RMR_2H_NO_FAR_VWAP",
        "range_mean_reversion",
        setup_rmr_2h_no_far_vwap,
    ),
    "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL": ResearchSetup(
        "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL",
        "range_mean_reversion",
        setup_rmr_2h_no_far_vwap_block_long_highatr_highvol,
    ),
    "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_LOWATR_OR_HIGHATR_HIGHVOL": ResearchSetup(
        "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_LOWATR_OR_HIGHATR_HIGHVOL",
        "range_mean_reversion",
        setup_rmr_2h_no_far_vwap_block_long_highatr_highvol_short_lowatr_or_highatr_highvol,
    ),
    "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_LOWATR_OR_HIGHVOL": ResearchSetup(
        "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_LOWATR_OR_HIGHVOL",
        "range_mean_reversion",
        setup_rmr_2h_no_far_vwap_block_long_highatr_highvol_short_lowatr_or_highvol,
    ),
    "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_HIGHATR_LOWMEDVOL": ResearchSetup(
        "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_HIGHATR_LOWMEDVOL",
        "range_mean_reversion",
        setup_rmr_2h_no_far_vwap_block_long_highatr_highvol_short_block_highatr_lowmedvol,
    ),
    "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_MEDATR_LOWMEDVOL": ResearchSetup(
        "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_MEDATR_LOWMEDVOL",
        "range_mean_reversion",
        setup_rmr_2h_no_far_vwap_block_long_highatr_highvol_short_block_medatr_lowmedvol,
    ),
    "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_MEDATR_LOWVOL": ResearchSetup(
        "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_MEDATR_LOWVOL",
        "range_mean_reversion",
        setup_rmr_2h_no_far_vwap_block_long_highatr_highvol_short_block_medatr_lowvol,
    ),
    "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_NONLOWATR_LOWVOL": ResearchSetup(
        "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_NONLOWATR_LOWVOL",
        "range_mean_reversion",
        setup_rmr_2h_no_far_vwap_block_long_highatr_highvol_short_block_nonlowatr_lowvol,
    ),
    "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_HIGHATR_MEDVOL": ResearchSetup(
        "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_HIGHATR_MEDVOL",
        "range_mean_reversion",
        setup_rmr_2h_no_far_vwap_block_long_highatr_highvol_short_block_highatr_medvol,
    ),
    "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_HIGHATR_MEDVOL_AND_MEDATR_LOWVOL": ResearchSetup(
        "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_HIGHATR_MEDVOL_AND_MEDATR_LOWVOL",
        "range_mean_reversion",
        setup_rmr_2h_no_far_vwap_block_long_highatr_highvol_short_block_highatr_medvol_and_medatr_lowvol,
    ),
    "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_BLOCK_LONG_LOWATR_MEDVOL_PRE3_MEDIUM": ResearchSetup(
        "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_BLOCK_LONG_LOWATR_MEDVOL_PRE3_MEDIUM",
        "range_mean_reversion",
        setup_rmr_2h_no_far_vwap_block_long_highatr_highvol_block_long_lowatr_medvol_pre3_medium,
    ),
    "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_HIGHATR_MEDVOL_SHORT_BLOCK_HIGHATR_LOWVOL_PRE3_MEDIUM": ResearchSetup(
        "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_HIGHATR_MEDVOL_SHORT_BLOCK_HIGHATR_LOWVOL_PRE3_MEDIUM",
        "range_mean_reversion",
        setup_rmr_2h_no_far_vwap_block_long_highatr_highvol_short_block_highatr_medvol_short_block_highatr_lowvol_pre3_medium,
    ),
    "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_HIGHATR_MEDVOL_BLOCK_LONG_LOWATR_MEDVOL_PRE3_MEDIUM_PRE6_LOW": ResearchSetup(
        "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_HIGHATR_MEDVOL_BLOCK_LONG_LOWATR_MEDVOL_PRE3_MEDIUM_PRE6_LOW",
        "range_mean_reversion",
        setup_rmr_2h_no_far_vwap_block_long_highatr_highvol_short_block_highatr_medvol_block_long_lowatr_medvol_pre3_medium_pre6_low,
    ),
    "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_HIGHATR_MEDVOL_SHORT_BLOCK_MEDATR_LOWVOL_PRE3_MEDIUM": ResearchSetup(
        "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_HIGHATR_MEDVOL_SHORT_BLOCK_MEDATR_LOWVOL_PRE3_MEDIUM",
        "range_mean_reversion",
        setup_rmr_2h_no_far_vwap_block_long_highatr_highvol_short_block_highatr_medvol_short_block_medatr_lowvol_pre3_medium,
    ),
    "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_HIGHATR_MEDVOL_BLOCK_LONG_LOWATR_MEDVOL_PRE3_MEDIUM_PRE6_LOW_SHORT_BLOCK_MEDATR_LOWVOL_PRE3_MEDIUM": ResearchSetup(
        "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_HIGHATR_MEDVOL_BLOCK_LONG_LOWATR_MEDVOL_PRE3_MEDIUM_PRE6_LOW_SHORT_BLOCK_MEDATR_LOWVOL_PRE3_MEDIUM",
        "range_mean_reversion",
        setup_rmr_2h_no_far_vwap_block_long_highatr_highvol_short_block_highatr_medvol_block_long_lowatr_medvol_pre3_medium_pre6_low_short_block_medatr_lowvol_pre3_medium,
    ),
    "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_HIGHATR_MEDVOL_SHORT_BLOCK_HIGHATR_LOWVOL_PRE6_LOW": ResearchSetup(
        "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_HIGHATR_MEDVOL_SHORT_BLOCK_HIGHATR_LOWVOL_PRE6_LOW",
        "range_mean_reversion",
        setup_rmr_2h_no_far_vwap_block_long_highatr_highvol_short_block_highatr_medvol_short_block_highatr_lowvol_pre6_low,
    ),
    "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_HIGHATR_MEDVOL_SHORT_BLOCK_LOWVOL_NONHIGHATR_PRE6_LOW": ResearchSetup(
        "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_HIGHATR_MEDVOL_SHORT_BLOCK_LOWVOL_NONHIGHATR_PRE6_LOW",
        "range_mean_reversion",
        setup_rmr_2h_no_far_vwap_block_long_highatr_highvol_short_block_highatr_medvol_short_block_lowvol_nonhighatr_pre6_low,
    ),
    "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_HIGHATR_MEDVOL_SHORT_BLOCK_LOWVOL_PRE6_LOW": ResearchSetup(
        "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_HIGHATR_MEDVOL_SHORT_BLOCK_LOWVOL_PRE6_LOW",
        "range_mean_reversion",
        setup_rmr_2h_no_far_vwap_block_long_highatr_highvol_short_block_highatr_medvol_short_block_lowvol_pre6_low,
    ),
    "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_HIGHATR_MEDVOL_BLOCK_LONG_LOWATR_MEDVOL_PRE3_MEDIUM_PRE6_LOW_SHORT_BLOCK_MEDATR_LOWVOL_PRE6_LOW": ResearchSetup(
        "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_HIGHATR_MEDVOL_BLOCK_LONG_LOWATR_MEDVOL_PRE3_MEDIUM_PRE6_LOW_SHORT_BLOCK_MEDATR_LOWVOL_PRE6_LOW",
        "range_mean_reversion",
        setup_rmr_2h_no_far_vwap_block_long_highatr_highvol_short_block_highatr_medvol_block_long_lowatr_medvol_pre3_medium_pre6_low_short_block_medatr_lowvol_pre6_low,
    ),
    "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_HIGHATR_MEDVOL_BLOCK_LONG_LOWATR_MEDVOL_PRE3_MEDIUM_PRE6_LOW_BLOCK_LONG_MEDATR_MEDVOL_PRE3_LOW_PRE6_LOW": ResearchSetup(
        "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_HIGHATR_MEDVOL_BLOCK_LONG_LOWATR_MEDVOL_PRE3_MEDIUM_PRE6_LOW_BLOCK_LONG_MEDATR_MEDVOL_PRE3_LOW_PRE6_LOW",
        "range_mean_reversion",
        setup_rmr_2h_no_far_vwap_block_long_highatr_highvol_short_block_highatr_medvol_block_long_lowatr_medvol_pre3_medium_pre6_low_block_long_medatr_medvol_pre3_low_pre6_low,
    ),
    "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_HIGHATR_MEDVOL_BLOCK_LONG_LOWATR_MEDVOL_PRE3_MEDIUM_PRE6_LOW_SHORT_BLOCK_MEDATR_HIGHVOL_PRE6_LOW": ResearchSetup(
        "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_HIGHATR_MEDVOL_BLOCK_LONG_LOWATR_MEDVOL_PRE3_MEDIUM_PRE6_LOW_SHORT_BLOCK_MEDATR_HIGHVOL_PRE6_LOW",
        "range_mean_reversion",
        setup_rmr_2h_no_far_vwap_block_long_highatr_highvol_short_block_highatr_medvol_block_long_lowatr_medvol_pre3_medium_pre6_low_short_block_medatr_highvol_pre6_low,
    ),
    "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_HIGHATR_MEDVOL_BLOCK_LONG_LOWATR_MEDVOL_PRE3_MEDIUM_PRE6_LOW_BLOCK_LONG_HIGHATR_LOWVOL_PRE3_MEDIUM_PRE6_LOW": ResearchSetup(
        "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_HIGHATR_MEDVOL_BLOCK_LONG_LOWATR_MEDVOL_PRE3_MEDIUM_PRE6_LOW_BLOCK_LONG_HIGHATR_LOWVOL_PRE3_MEDIUM_PRE6_LOW",
        "range_mean_reversion",
        setup_rmr_2h_no_far_vwap_block_long_highatr_highvol_short_block_highatr_medvol_block_long_lowatr_medvol_pre3_medium_pre6_low_block_long_highatr_lowvol_pre3_medium_pre6_low,
    ),
    "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_HIGHATR_MEDVOL_BLOCK_LONG_LOWATR_MEDVOL_PRE3_MEDIUM_PRE6_LOW_BLOCK_LONG_HIGHATR_LOWVOL_PRE3_MEDIUM_PRE6_LOW_SHORT_BLOCK_MEDATR_HIGHVOL_PRE6_LOW": ResearchSetup(
        "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_HIGHATR_MEDVOL_BLOCK_LONG_LOWATR_MEDVOL_PRE3_MEDIUM_PRE6_LOW_BLOCK_LONG_HIGHATR_LOWVOL_PRE3_MEDIUM_PRE6_LOW_SHORT_BLOCK_MEDATR_HIGHVOL_PRE6_LOW",
        "range_mean_reversion",
        setup_rmr_2h_no_far_vwap_block_long_highatr_highvol_short_block_highatr_medvol_block_long_lowatr_medvol_pre3_medium_pre6_low_block_long_highatr_lowvol_pre3_medium_pre6_low_short_block_medatr_highvol_pre6_low,
    ),
    "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_BLOCK_LONG_LOWATR_MEDVOL": ResearchSetup(
        "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_BLOCK_LONG_LOWATR_MEDVOL",
        "range_mean_reversion",
        setup_rmr_2h_no_far_vwap_block_long_highatr_highvol_block_long_lowatr_medvol,
    ),
    "RMR_2H_NO_FAR_VWAP_LONG_BIASED": ResearchSetup(
        "RMR_2H_NO_FAR_VWAP_LONG_BIASED",
        "range_mean_reversion",
        setup_rmr_2h_no_far_vwap_long_biased,
    ),
    "RMR_2H_NO_FAR_VWAP_LONG_BIASED_SHORT_LOWATR": ResearchSetup(
        "RMR_2H_NO_FAR_VWAP_LONG_BIASED_SHORT_LOWATR",
        "range_mean_reversion",
        setup_rmr_2h_no_far_vwap_long_biased_short_lowatr,
    ),
    "RMR_2H_NO_FAR_VWAP_LONG_BIASED_BLOCK_LONG_HIGHATR_HIGHVOL": ResearchSetup(
        "RMR_2H_NO_FAR_VWAP_LONG_BIASED_BLOCK_LONG_HIGHATR_HIGHVOL",
        "range_mean_reversion",
        setup_rmr_2h_no_far_vwap_long_biased_block_long_highatr_highvol,
    ),
    "RMR_2H_NO_FAR_VWAP_LONG_BIASED_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_LOWATR": ResearchSetup(
        "RMR_2H_NO_FAR_VWAP_LONG_BIASED_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_LOWATR",
        "range_mean_reversion",
        setup_rmr_2h_no_far_vwap_long_biased_block_long_highatr_highvol_short_lowatr,
    ),
    "RMR_2H_NO_FAR_VWAP_LONG_BIASED_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_MEDATR_LOWMEDVOL": ResearchSetup(
        "RMR_2H_NO_FAR_VWAP_LONG_BIASED_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_MEDATR_LOWMEDVOL",
        "range_mean_reversion",
        setup_rmr_2h_no_far_vwap_long_biased_block_long_highatr_highvol_short_block_medatr_lowmedvol,
    ),
    "RMR_2H_NO_FAR_VWAP_LONG_BIASED_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_MEDATR_LOWVOL": ResearchSetup(
        "RMR_2H_NO_FAR_VWAP_LONG_BIASED_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_MEDATR_LOWVOL",
        "range_mean_reversion",
        setup_rmr_2h_no_far_vwap_long_biased_block_long_highatr_highvol_short_block_medatr_lowvol,
    ),
    "RMR_2H_NO_FAR_VWAP_LONG_BIASED_BLOCK_LONG_HIGHATR_HIGHVOL_BLOCK_LONG_LOWATR_MEDVOL": ResearchSetup(
        "RMR_2H_NO_FAR_VWAP_LONG_BIASED_BLOCK_LONG_HIGHATR_HIGHVOL_BLOCK_LONG_LOWATR_MEDVOL",
        "range_mean_reversion",
        setup_rmr_2h_no_far_vwap_long_biased_block_long_highatr_highvol_block_long_lowatr_medvol,
    ),
    "RMR_2H_NO_FAR_VWAP_LONG_BIASED_BLOCK_LONG_HIGHATR_HIGHVOL_BLOCK_LONG_LOWATR_MEDVOL_PRE3_MEDIUM_PRE6_LOW": ResearchSetup(
        "RMR_2H_NO_FAR_VWAP_LONG_BIASED_BLOCK_LONG_HIGHATR_HIGHVOL_BLOCK_LONG_LOWATR_MEDVOL_PRE3_MEDIUM_PRE6_LOW",
        "range_mean_reversion",
        setup_rmr_2h_no_far_vwap_long_biased_block_long_highatr_highvol_block_long_lowatr_medvol_pre3_medium_pre6_low,
    ),
    "RMR_2H_NO_FAR_VWAP_LONG_BIASED_BLOCK_LONG_HIGHATR_HIGHVOL_BLOCK_LONG_LOWATR_MEDVOL_PRE3_MEDIUM_PRE6_LOW_SHORT_BLOCK_MEDATR_LOWVOL_PRE6_LOW": ResearchSetup(
        "RMR_2H_NO_FAR_VWAP_LONG_BIASED_BLOCK_LONG_HIGHATR_HIGHVOL_BLOCK_LONG_LOWATR_MEDVOL_PRE3_MEDIUM_PRE6_LOW_SHORT_BLOCK_MEDATR_LOWVOL_PRE6_LOW",
        "range_mean_reversion",
        setup_rmr_2h_no_far_vwap_long_biased_block_long_highatr_highvol_block_long_lowatr_medvol_pre3_medium_pre6_low_short_block_medatr_lowvol_pre6_low,
    ),
    "RMR_2H_NO_FAR_VWAP_LONG_BIASED_BLOCK_LONG_HIGHATR_HIGHVOL_BLOCK_LONG_LOWATR_MEDVOL_PRE3_MEDIUM_PRE6_LOW_BLOCK_LONG_MEDATR_MEDVOL_PRE3_LOW_PRE6_LOW": ResearchSetup(
        "RMR_2H_NO_FAR_VWAP_LONG_BIASED_BLOCK_LONG_HIGHATR_HIGHVOL_BLOCK_LONG_LOWATR_MEDVOL_PRE3_MEDIUM_PRE6_LOW_BLOCK_LONG_MEDATR_MEDVOL_PRE3_LOW_PRE6_LOW",
        "range_mean_reversion",
        setup_rmr_2h_no_far_vwap_long_biased_block_long_highatr_highvol_block_long_lowatr_medvol_pre3_medium_pre6_low_block_long_medatr_medvol_pre3_low_pre6_low,
    ),
    "RMR_2H_NO_FAR_VWAP_LONG_BIASED_BLOCK_LONG_HIGHATR_HIGHVOL_BLOCK_LONG_LOWATR_MEDVOL_PRE3_MEDIUM_PRE6_LOW_BLOCK_LONG_HIGHATR_LOWVOL_PRE3_MEDIUM_PRE6_LOW": ResearchSetup(
        "RMR_2H_NO_FAR_VWAP_LONG_BIASED_BLOCK_LONG_HIGHATR_HIGHVOL_BLOCK_LONG_LOWATR_MEDVOL_PRE3_MEDIUM_PRE6_LOW_BLOCK_LONG_HIGHATR_LOWVOL_PRE3_MEDIUM_PRE6_LOW",
        "range_mean_reversion",
        setup_rmr_2h_no_far_vwap_long_biased_block_long_highatr_highvol_block_long_lowatr_medvol_pre3_medium_pre6_low_block_long_highatr_lowvol_pre3_medium_pre6_low,
    ),
    "RMR_2H_NO_FAR_VWAP_LONG_BIASED_MEDVOL_SHORTS": ResearchSetup(
        "RMR_2H_NO_FAR_VWAP_LONG_BIASED_MEDVOL_SHORTS",
        "range_mean_reversion",
        setup_rmr_2h_no_far_vwap_long_biased_medvol_shorts,
    ),
    "RMR_2H_NO_FAR_VWAP_LONG_BIASED_LOWATR_MEDVOL_SHORTS": ResearchSetup(
        "RMR_2H_NO_FAR_VWAP_LONG_BIASED_LOWATR_MEDVOL_SHORTS",
        "range_mean_reversion",
        setup_rmr_2h_no_far_vwap_long_biased_lowatr_medvol_shorts,
    ),
    "RMR_2H_NO_FAR_VWAP_NO_HIGH_ATR_NO_HIGH_VOL": ResearchSetup(
        "RMR_2H_NO_FAR_VWAP_NO_HIGH_ATR_NO_HIGH_VOL",
        "range_mean_reversion",
        setup_rmr_2h_no_far_vwap_no_high_atr_no_high_vol,
    ),
    "RMR_2H_NO_FAR_VWAP_LONG_BIASED_NO_HIGH_ATR_NO_HIGH_VOL": ResearchSetup(
        "RMR_2H_NO_FAR_VWAP_LONG_BIASED_NO_HIGH_ATR_NO_HIGH_VOL",
        "range_mean_reversion",
        setup_rmr_2h_no_far_vwap_long_biased_no_high_atr_no_high_vol,
    ),
    "RMR_2H_NO_FAR_VWAP_LONG_BIASED_BLOCK_LONG_HIGH_VOL": ResearchSetup(
        "RMR_2H_NO_FAR_VWAP_LONG_BIASED_BLOCK_LONG_HIGH_VOL",
        "range_mean_reversion",
        setup_rmr_2h_no_far_vwap_long_biased_block_long_high_vol,
    ),
    "RMR_2H_NO_FAR_VWAP_LONG_BIASED_BLOCK_LONG_HIGH_VOL_SHORT_LOWATR": ResearchSetup(
        "RMR_2H_NO_FAR_VWAP_LONG_BIASED_BLOCK_LONG_HIGH_VOL_SHORT_LOWATR",
        "range_mean_reversion",
        setup_rmr_2h_no_far_vwap_long_biased_block_long_high_vol_short_lowatr,
    ),
    "RMR_2H_NO_FAR_VWAP_LONG_BIASED_BLOCK_LONG_HIGH_VOL_SHORT_LOWATR_MEDVOL": ResearchSetup(
        "RMR_2H_NO_FAR_VWAP_LONG_BIASED_BLOCK_LONG_HIGH_VOL_SHORT_LOWATR_MEDVOL",
        "range_mean_reversion",
        setup_rmr_2h_no_far_vwap_long_biased_block_long_high_vol_short_lowatr_medvol,
    ),
    "RMR_2H_NO_FAR_VWAP_LONG_BIASED_BLOCK_LONG_HIGH_VOL_SHORT_NO_HIGH_VOL": ResearchSetup(
        "RMR_2H_NO_FAR_VWAP_LONG_BIASED_BLOCK_LONG_HIGH_VOL_SHORT_NO_HIGH_VOL",
        "range_mean_reversion",
        setup_rmr_2h_no_far_vwap_long_biased_block_long_high_vol_short_no_high_vol,
    ),
    "RMR_2H_NO_FAR_VWAP_LONG_BIASED_BLOCK_LONG_HIGH_ATR_OR_HIGH_VOL": ResearchSetup(
        "RMR_2H_NO_FAR_VWAP_LONG_BIASED_BLOCK_LONG_HIGH_ATR_OR_HIGH_VOL",
        "range_mean_reversion",
        setup_rmr_2h_no_far_vwap_long_biased_block_long_high_atr_or_high_vol,
    ),
    "RMR_2H_NO_HIGH_ATR": ResearchSetup(
        "RMR_2H_NO_HIGH_ATR",
        "range_mean_reversion",
        setup_rmr_2h_no_high_atr,
    ),
    "RMR_2H_NO_FAR_VWAP_NO_HIGH_ATR": ResearchSetup(
        "RMR_2H_NO_FAR_VWAP_NO_HIGH_ATR",
        "range_mean_reversion",
        setup_rmr_2h_no_far_vwap_no_high_atr,
    ),
    "RMR_2H_LONG_ONLY_NO_FAR_VWAP": ResearchSetup(
        "RMR_2H_LONG_ONLY_NO_FAR_VWAP",
        "range_mean_reversion",
        setup_rmr_2h_long_only_no_far_vwap,
    ),
    "RMR_2H_LONG_ONLY_NO_FAR_VWAP_NO_HIGH_ATR": ResearchSetup(
        "RMR_2H_LONG_ONLY_NO_FAR_VWAP_NO_HIGH_ATR",
        "range_mean_reversion",
        setup_rmr_2h_long_only_no_far_vwap_no_high_atr,
    ),
    "RMR_2H_LONG_ONLY_NO_FAR_VWAP_NO_HIGH_VOL": ResearchSetup(
        "RMR_2H_LONG_ONLY_NO_FAR_VWAP_NO_HIGH_VOL",
        "range_mean_reversion",
        setup_rmr_2h_long_only_no_far_vwap_no_high_vol,
    ),
    "FV_MR_0": ResearchSetup("FV_MR_0", "range_mean_reversion", setup_fv_mr_0),
    "FV_MR_1": ResearchSetup("FV_MR_1", "range_mean_reversion", setup_fv_mr_1),
    "FV_MR_2": ResearchSetup("FV_MR_2", "range_mean_reversion", setup_fv_mr_2),
    "FV_MR_3": ResearchSetup("FV_MR_3", "range_mean_reversion", setup_fv_mr_3),
    "FV_MR_4": ResearchSetup("FV_MR_4", "range_mean_reversion", setup_fv_mr_4),
    "FV_MR_5": ResearchSetup("FV_MR_5", "range_mean_reversion", setup_fv_mr_5),
    "FV_MR_6": ResearchSetup("FV_MR_6", "range_mean_reversion", setup_fv_mr_6),
}


def _entry_profile_allow(name: str, *tags: str) -> EntryDecision:
    return EntryDecision(True, name, entry_tags=tuple(tag for tag in tags if tag))


def _entry_profile_reject(name: str, reason: str, *tags: str) -> EntryDecision:
    return EntryDecision(
        False,
        name,
        rejection_reason=reason,
        entry_tags=tuple(tag for tag in tags if tag),
    )


def _entry_context_tags(ctx: CandidateContext, *tags: str) -> tuple[str, ...]:
    trend_tag = str(ctx.trend).lower() if ctx.trend else "unknown"
    return tuple(
        tag
        for tag in (
            *tags,
            f"family_{ctx.candidate_family}",
            f"atr_{ctx.atr_bucket or 'na'}",
            f"vol_{ctx.volume_bucket or 'na'}",
            f"vwap_{ctx.vwap_distance_bucket or 'na'}",
            f"move3_{ctx.pre_entry_move_3c_bucket or 'na'}",
            f"move6_{ctx.pre_entry_move_6c_bucket or 'na'}",
            f"trend_{trend_tag}",
        )
        if tag
    )


def _entry_profile_allow_ctx(ctx: CandidateContext, name: str, *tags: str) -> EntryDecision:
    return _entry_profile_allow(name, *_entry_context_tags(ctx, *tags))


def _entry_profile_reject_ctx(ctx: CandidateContext, name: str, reason: str, *tags: str) -> EntryDecision:
    return _entry_profile_reject(name, reason, *_entry_context_tags(ctx, *tags))


def _vwap_aligned_with_direction(ctx: CandidateContext) -> bool:
    if ctx.direction == "LONG":
        return ctx.price_vs_vwap == "above"
    return ctx.price_vs_vwap == "below"


def _vwap_opposed_to_direction(ctx: CandidateContext) -> bool:
    if ctx.direction == "LONG":
        return ctx.price_vs_vwap == "below"
    return ctx.price_vs_vwap == "above"


def _has_signal_tag(ctx: CandidateContext, tag: str) -> bool:
    return tag in ctx.signal_tags or tag in ctx.signals_set


def _anti_chase_core_decision(ctx: CandidateContext, name: str) -> EntryDecision:
    if ctx.candles_since_impulse >= 0 and ctx.candles_since_impulse < 2:
        return _entry_profile_reject_ctx(
            ctx,
            name,
            f"entry too close to initial impulse (candles_since_impulse={ctx.candles_since_impulse})",
            "anti_chase",
            "early_impulse",
        )
    if ctx.candidate_family in {"consensus", "volume_breakout", "pullback_trend"} and ctx.vwap_distance_bucket == "far":
        return _entry_profile_reject_ctx(
            ctx,
            name,
            (
                "trend-style entry too far from VWAP "
                f"(bucket={ctx.vwap_distance_bucket}, dist_r={ctx.distance_from_vwap_r:+.3f})"
            ),
            "anti_chase",
            "far_from_vwap",
        )
    if ctx.pre_entry_move_3c_bucket == "high" and ctx.pre_entry_move_6c_bucket == "high":
        return _entry_profile_reject_ctx(
            ctx,
            name,
            (
                "multi-candle pre-entry move already extended "
                f"(move3={ctx.price_move_3r:+.3f}R/{ctx.pre_entry_move_3c_bucket}, "
                f"move6={ctx.price_move_6r:+.3f}R/{ctx.pre_entry_move_6c_bucket})"
            ),
            "anti_chase",
            "multi_candle_extension",
        )
    return _entry_profile_allow_ctx(ctx, name, "anti_chase", "not_extended")


def _anti_chase_long_only_gate(ctx: CandidateContext, name: str) -> EntryDecision | None:
    base = _anti_chase_core_decision(ctx, name)
    if not base.allowed:
        return base
    if ctx.direction != "LONG":
        return _entry_profile_reject_ctx(
            ctx,
            name,
            "short entries blocked in long-only anti-chase diagnostic",
            "anti_chase",
            "long_only",
        )
    return None


def _anti_chase_long_only_research_gate(
    ctx: CandidateContext,
    name: str,
    *,
    allow_far_vwap: bool = False,
    allow_high_high_extension: bool = False,
    reject_high_atr: bool = False,
    reject_low_3c: bool = False,
    reject_medium_3c_outside_medium_atr: bool = False,
) -> EntryDecision | None:
    if ctx.direction != "LONG":
        return _entry_profile_reject_ctx(
            ctx,
            name,
            "short entries blocked in long-only anti-chase diagnostic",
            "anti_chase",
            "long_only",
        )

    if ctx.candles_since_impulse >= 0 and ctx.candles_since_impulse < 2:
        return _entry_profile_reject_ctx(
            ctx,
            name,
            f"entry too close to initial impulse (candles_since_impulse={ctx.candles_since_impulse})",
            "anti_chase",
            "early_impulse",
            "long_only",
        )

    if ctx.candidate_family in {"consensus", "volume_breakout", "pullback_trend"} and ctx.vwap_distance_bucket == "far":
        far_allowed = (
            allow_far_vwap
            and ctx.atr_bucket == "medium"
            and ctx.pre_entry_move_3c_bucket == "high"
            and ctx.pre_entry_move_6c_bucket != "high"
        )
        if not far_allowed:
            return _entry_profile_reject_ctx(
                ctx,
                name,
                (
                    "trend-style entry too far from VWAP "
                    f"(bucket={ctx.vwap_distance_bucket}, dist_r={ctx.distance_from_vwap_r:+.3f})"
                ),
                "anti_chase",
                "far_from_vwap",
                "long_only",
            )

    if ctx.pre_entry_move_3c_bucket == "high" and ctx.pre_entry_move_6c_bucket == "high":
        extension_allowed = allow_high_high_extension and ctx.atr_bucket == "medium"
        if not extension_allowed:
            return _entry_profile_reject_ctx(
                ctx,
                name,
                (
                    "multi-candle pre-entry move already extended "
                    f"(move3={ctx.price_move_3r:+.3f}R/{ctx.pre_entry_move_3c_bucket}, "
                    f"move6={ctx.price_move_6r:+.3f}R/{ctx.pre_entry_move_6c_bucket})"
                ),
                "anti_chase",
                "multi_candle_extension",
                "long_only",
            )

    if reject_high_atr and ctx.atr_bucket == "high":
        return _entry_profile_reject_ctx(
            ctx,
            name,
            f"high ATR breakout continuation rejected (atr_bucket={ctx.atr_bucket})",
            "anti_chase",
            "long_only",
            "high_atr_blocked",
        )

    if reject_low_3c and ctx.pre_entry_move_3c_bucket == "low":
        return _entry_profile_reject_ctx(
            ctx,
            name,
            (
                "recent move too weak for continuation timing "
                f"(move3={ctx.price_move_3r:+.3f}R/{ctx.pre_entry_move_3c_bucket})"
            ),
            "anti_chase",
            "long_only",
            "weak_3c_impulse",
        )

    if reject_medium_3c_outside_medium_atr and ctx.pre_entry_move_3c_bucket == "medium" and ctx.atr_bucket != "medium":
        return _entry_profile_reject_ctx(
            ctx,
            name,
            (
                "middling impulse rejected outside medium ATR regime "
                f"(move3={ctx.price_move_3r:+.3f}R/{ctx.pre_entry_move_3c_bucket}, atr={ctx.atr_bucket})"
            ),
            "anti_chase",
            "long_only",
            "mid_impulse_bad_atr_mix",
        )

    return None


def entry_profile_baseline(
    ctx: CandidateContext,
    setup_spec: ResearchSetup,
    setup_decision: SetupDecision,
) -> EntryDecision:
    """
    Research-only pass-through baseline.

    Future entry-timing overlays should be evaluated here without changing
    setup-family definitions or any live-trading behavior.
    """
    _ = (ctx, setup_spec, setup_decision)
    return _entry_profile_allow("ENTRY_BASELINE", "setup_native_timing")


def entry_profile_anti_chase(
    ctx: CandidateContext,
    setup_spec: ResearchSetup,
    setup_decision: SetupDecision,
) -> EntryDecision:
    """
    Generic timing overlay that rejects entries that already look extended or
    too close to the initial impulse.
    """
    _ = (setup_spec, setup_decision)
    return _anti_chase_core_decision(ctx, "ENTRY_ANTI_CHASE")


def entry_profile_anti_chase_long_only(
    ctx: CandidateContext,
    setup_spec: ResearchSetup,
    setup_decision: SetupDecision,
) -> EntryDecision:
    _ = (setup_spec, setup_decision)
    name = "ENTRY_ANTI_CHASE_LONG_ONLY"
    gate = _anti_chase_long_only_gate(ctx, name)
    if gate is not None:
        return gate
    return _entry_profile_allow_ctx(ctx, name, "anti_chase", "long_only", "not_extended")


def entry_profile_anti_chase_volume_confirmed(
    ctx: CandidateContext,
    setup_spec: ResearchSetup,
    setup_decision: SetupDecision,
) -> EntryDecision:
    _ = (setup_spec, setup_decision)
    name = "ENTRY_ANTI_CHASE_VOLUME_CONFIRMED"
    base = _anti_chase_core_decision(ctx, name)
    if not base.allowed:
        return base
    if ctx.volume_bucket == "low":
        return _entry_profile_reject_ctx(
            ctx,
            name,
            f"volume participation too weak (bucket={ctx.volume_bucket}, ratio={ctx.volume_ratio:.3f}x)",
            "anti_chase",
            "volume_confirmed",
            "low_volume",
        )
    return _entry_profile_allow_ctx(ctx, name, "anti_chase", "volume_confirmed", f"vol_{ctx.volume_bucket}")


def entry_profile_anti_chase_long_lookback(
    ctx: CandidateContext,
    setup_spec: ResearchSetup,
    setup_decision: SetupDecision,
) -> EntryDecision:
    _ = (setup_spec, setup_decision)
    name = "ENTRY_ANTI_CHASE_LONG_LOOKBACK"
    base = _anti_chase_core_decision(ctx, name)
    if not base.allowed:
        return base
    if ctx.pre_entry_move_6c_bucket == "high":
        return _entry_profile_reject_ctx(
            ctx,
            name,
            (
                "longer lookback still shows extended move "
                f"(move6={ctx.price_move_6r:+.3f}R/{ctx.pre_entry_move_6c_bucket})"
            ),
            "anti_chase",
            "long_lookback",
            "extended_6c",
        )
    return _entry_profile_allow_ctx(ctx, name, "anti_chase", "long_lookback", "not_extended_6c")


def entry_profile_anti_chase_long_only_impulse(
    ctx: CandidateContext,
    setup_spec: ResearchSetup,
    setup_decision: SetupDecision,
) -> EntryDecision:
    """
    Research-only continuation timing profile.

    Hypothesis: the long-only anti-chase idea degrades when the recent move is
    neither clearly impulsive nor clearly reset. Require the short-term thrust
    bucket to be strong while still avoiding the original anti-chase failures.
    """
    _ = (setup_spec, setup_decision)
    name = "ENTRY_ANTI_CHASE_LONG_ONLY_IMPULSE"
    gate = _anti_chase_long_only_gate(ctx, name)
    if gate is not None:
        return gate
    if ctx.pre_entry_move_3c_bucket != "high":
        return _entry_profile_reject_ctx(
            ctx,
            name,
            (
                "recent impulse not strong enough for continuation timing "
                f"(move3={ctx.price_move_3r:+.3f}R/{ctx.pre_entry_move_3c_bucket})"
            ),
            "anti_chase",
            "long_only",
            "requires_impulse",
            "not_high_3c",
        )
    return _entry_profile_allow_ctx(ctx, name, "anti_chase", "long_only", "requires_impulse", "high_3c")


def entry_profile_anti_chase_long_only_6c_guard(
    ctx: CandidateContext,
    setup_spec: ResearchSetup,
    setup_decision: SetupDecision,
) -> EntryDecision:
    """
    Research-only continuation timing profile.

    Hypothesis: the long-only anti-chase idea fails when the broader 6-candle
    path is already too extended even if the shorter 3-candle move looks okay.
    """
    _ = (setup_spec, setup_decision)
    name = "ENTRY_ANTI_CHASE_LONG_ONLY_6C_GUARD"
    gate = _anti_chase_long_only_gate(ctx, name)
    if gate is not None:
        return gate
    if ctx.pre_entry_move_6c_bucket == "high":
        return _entry_profile_reject_ctx(
            ctx,
            name,
            (
                "six-candle extension still too stretched for continuation timing "
                f"(move6={ctx.price_move_6r:+.3f}R/{ctx.pre_entry_move_6c_bucket})"
            ),
            "anti_chase",
            "long_only",
            "six_candle_guard",
            "extended_6c",
        )
    return _entry_profile_allow_ctx(ctx, name, "anti_chase", "long_only", "six_candle_guard", "not_extended_6c")


def entry_profile_anti_chase_long_only_medium_atr(
    ctx: CandidateContext,
    setup_spec: ResearchSetup,
    setup_decision: SetupDecision,
) -> EntryDecision:
    """
    Research-only continuation timing profile.

    Hypothesis: the long-only anti-chase edge is only present in the middle
    volatility tercile, while low/high ATR windows create whipsaw or exhaustion.
    """
    _ = (setup_spec, setup_decision)
    name = "ENTRY_ANTI_CHASE_LONG_ONLY_MEDIUM_ATR"
    gate = _anti_chase_long_only_gate(ctx, name)
    if gate is not None:
        return gate
    if ctx.atr_bucket != "medium":
        return _entry_profile_reject_ctx(
            ctx,
            name,
            f"ATR regime not in preferred medium bucket (atr_bucket={ctx.atr_bucket})",
            "anti_chase",
            "long_only",
            "medium_atr_only",
            "atr_filtered",
        )
    return _entry_profile_allow_ctx(ctx, name, "anti_chase", "long_only", "medium_atr_only")


def entry_profile_anti_chase_long_only_balanced(
    ctx: CandidateContext,
    setup_spec: ResearchSetup,
    setup_decision: SetupDecision,
) -> EntryDecision:
    """
    Research-only continuation timing profile.

    Hypothesis: stable continuation entries either occur in a medium ATR regime
    or arrive with a clearly strong 3-candle impulse. The weakest pocket was
    middling 3-candle extension outside medium ATR, so block that zone.
    """
    _ = (setup_spec, setup_decision)
    name = "ENTRY_ANTI_CHASE_LONG_ONLY_BALANCED"
    gate = _anti_chase_long_only_gate(ctx, name)
    if gate is not None:
        return gate
    if ctx.pre_entry_move_3c_bucket == "low":
        return _entry_profile_reject_ctx(
            ctx,
            name,
            (
                "recent move too weak for breakout continuation "
                f"(move3={ctx.price_move_3r:+.3f}R/{ctx.pre_entry_move_3c_bucket})"
            ),
            "anti_chase",
            "long_only",
            "balanced",
            "weak_3c_impulse",
        )
    if ctx.pre_entry_move_3c_bucket == "medium" and ctx.atr_bucket != "medium":
        return _entry_profile_reject_ctx(
            ctx,
            name,
            (
                "middling impulse rejected outside medium ATR regime "
                f"(move3={ctx.price_move_3r:+.3f}R/{ctx.pre_entry_move_3c_bucket}, atr={ctx.atr_bucket})"
            ),
            "anti_chase",
            "long_only",
            "balanced",
            "mid_impulse_bad_atr_mix",
        )
    return _entry_profile_allow_ctx(
        ctx,
        name,
        "anti_chase",
        "long_only",
        "balanced",
        f"atr_{ctx.atr_bucket}",
        f"move3_{ctx.pre_entry_move_3c_bucket}",
    )


def entry_profile_anti_chase_long_only_relaxed_vwap(
    ctx: CandidateContext,
    setup_spec: ResearchSetup,
    setup_decision: SetupDecision,
) -> EntryDecision:
    """
    Research-only continuation timing profile.

    Hypothesis: the long-only anti-chase filter may be throwing away too much
    sample by hard-blocking every far-from-VWAP entry. Allow only the narrow
    far-VWAP pocket that still has a strong short-term impulse and has not
    already stretched over the broader 6-candle path.
    """
    _ = (setup_spec, setup_decision)
    name = "ENTRY_ANTI_CHASE_LONG_ONLY_RELAXED_VWAP"
    gate = _anti_chase_long_only_research_gate(
        ctx,
        name,
        allow_far_vwap=True,
    )
    if gate is not None:
        return gate
    tags = ["anti_chase", "long_only", "relaxed_vwap"]
    if ctx.vwap_distance_bucket == "far":
        tags.extend(("far_vwap_allowed", "medium_atr", "high_3c", "not_high_6c"))
    else:
        tags.append("non_far_vwap")
    return _entry_profile_allow_ctx(ctx, name, *tags)


def entry_profile_anti_chase_long_only_relaxed_extension(
    ctx: CandidateContext,
    setup_spec: ResearchSetup,
    setup_decision: SetupDecision,
) -> EntryDecision:
    """
    Research-only continuation timing profile.

    Hypothesis: some continuation entries can survive a strong 3c+6c extension
    if volatility is in the medium regime. Relax the hard extension block only
    in that bucket so sample can rise without reopening the worst high-ATR and
    low-ATR exhaustion zones.
    """
    _ = (setup_spec, setup_decision)
    name = "ENTRY_ANTI_CHASE_LONG_ONLY_RELAXED_EXTENSION"
    gate = _anti_chase_long_only_research_gate(
        ctx,
        name,
        allow_high_high_extension=True,
    )
    if gate is not None:
        return gate
    tags = ["anti_chase", "long_only", "relaxed_extension"]
    if ctx.pre_entry_move_3c_bucket == "high" and ctx.pre_entry_move_6c_bucket == "high":
        tags.extend(("high_high_extension_allowed", f"atr_{ctx.atr_bucket}"))
    else:
        tags.append("base_extension_guard")
    return _entry_profile_allow_ctx(ctx, name, *tags)


def entry_profile_anti_chase_long_only_adaptive(
    ctx: CandidateContext,
    setup_spec: ResearchSetup,
    setup_decision: SetupDecision,
) -> EntryDecision:
    """
    Research-only continuation timing profile.

    Hypothesis: the surviving continuation pocket is narrower than the original
    long-only overlay. Keep the better 1460-day buckets by:
      - rejecting high ATR,
      - rejecting weak 3-candle impulse,
      - rejecting middling 3-candle impulse outside medium ATR,
      - allowing far-VWAP and high/high extension only in a controlled
        medium-ATR / high-impulse pocket.
    """
    _ = (setup_spec, setup_decision)
    name = "ENTRY_ANTI_CHASE_LONG_ONLY_ADAPTIVE"
    gate = _anti_chase_long_only_research_gate(
        ctx,
        name,
        allow_far_vwap=True,
        allow_high_high_extension=True,
        reject_high_atr=True,
        reject_low_3c=True,
        reject_medium_3c_outside_medium_atr=True,
    )
    if gate is not None:
        return gate
    return _entry_profile_allow_ctx(
        ctx,
        name,
        "anti_chase",
        "long_only",
        "adaptive",
        f"atr_{ctx.atr_bucket}",
        f"move3_{ctx.pre_entry_move_3c_bucket}",
        f"move6_{ctx.pre_entry_move_6c_bucket}",
        f"vwap_{ctx.vwap_distance_bucket}",
    )


def entry_profile_anti_chase_trend_only(
    ctx: CandidateContext,
    setup_spec: ResearchSetup,
    setup_decision: SetupDecision,
) -> EntryDecision:
    _ = (setup_spec, setup_decision)
    name = "ENTRY_ANTI_CHASE_TREND_ONLY"
    base = _anti_chase_core_decision(ctx, name)
    if not base.allowed:
        return base
    if ctx.candidate_family == "range_mean_reversion" or _has_signal_tag(ctx, "RangingContext") or ctx.trend != reg.TRENDING:
        return _entry_profile_reject_ctx(
            ctx,
            name,
            f"range-bound or mean-reversion context blocked (family={ctx.candidate_family}, trend={ctx.trend})",
            "anti_chase",
            "trend_only",
            "range_blocked",
        )
    return _entry_profile_allow_ctx(ctx, name, "anti_chase", "trend_only", "not_ranging")


def entry_profile_anti_chase_trend_volume(
    ctx: CandidateContext,
    setup_spec: ResearchSetup,
    setup_decision: SetupDecision,
) -> EntryDecision:
    _ = (setup_spec, setup_decision)
    name = "ENTRY_ANTI_CHASE_TREND_VOLUME"
    base = _anti_chase_core_decision(ctx, name)
    if not base.allowed:
        return base
    if ctx.candidate_family == "range_mean_reversion" or _has_signal_tag(ctx, "RangingContext") or ctx.trend != reg.TRENDING:
        return _entry_profile_reject_ctx(
            ctx,
            name,
            f"range-bound or mean-reversion context blocked (family={ctx.candidate_family}, trend={ctx.trend})",
            "anti_chase",
            "trend_volume",
            "range_blocked",
        )
    if ctx.volume_bucket == "low":
        return _entry_profile_reject_ctx(
            ctx,
            name,
            f"volume participation too weak (bucket={ctx.volume_bucket}, ratio={ctx.volume_ratio:.3f}x)",
            "anti_chase",
            "trend_volume",
            "low_volume",
        )
    if ctx.pre_entry_move_6c_bucket == "high":
        return _entry_profile_reject_ctx(
            ctx,
            name,
            (
                "longer lookback still shows extended move "
                f"(move6={ctx.price_move_6r:+.3f}R/{ctx.pre_entry_move_6c_bucket})"
            ),
            "anti_chase",
            "trend_volume",
            "extended_6c",
        )
    return _entry_profile_allow_ctx(ctx, name, "anti_chase", "trend_volume", f"vol_{ctx.volume_bucket}", "not_ranging")


def entry_profile_momentum_confirmed(
    ctx: CandidateContext,
    setup_spec: ResearchSetup,
    setup_decision: SetupDecision,
) -> EntryDecision:
    """
    Trend-following overlay that asks for directional alignment plus some
    confirmation, while still avoiding obvious chase entries.
    """
    _ = (setup_spec, setup_decision)
    name = "ENTRY_MOMENTUM_CONFIRMED"
    if ctx.trend != reg.TRENDING and not ctx.trend_context_reason:
        return _entry_profile_reject(name, "requires trend context", "momentum", "needs_trend")
    if not _vwap_aligned_with_direction(ctx):
        return _entry_profile_reject(name, "price not aligned with VWAP direction", "momentum", "vwap_misaligned")
    if ctx.vwap_distance_bucket == "far":
        return _entry_profile_reject(name, "momentum entry already too far from VWAP", "momentum", "far_from_vwap")
    if ctx.volume_bucket == "low":
        return _entry_profile_reject(name, "requires at least medium participation", "momentum", "low_volume")
    if ctx.candles_since_impulse >= 0 and ctx.candles_since_impulse < 2:
        return _entry_profile_reject(name, "impulse not yet tested", "momentum", "early_impulse")
    if ctx.pullback_detected and not ctx.reclaim_detected:
        return _entry_profile_reject(name, "pullback not reclaimed yet", "momentum", "await_reclaim")
    if not (
        ctx.macd_agrees
        or _has_signal_tag(ctx, "MACDAgree")
        or _has_signal_tag(ctx, "Reclaim")
        or _has_signal_tag(ctx, "Breakout")
    ):
        return _entry_profile_reject(name, "missing momentum confirmation tag", "momentum", "no_confirmation")
    return _entry_profile_allow(name, "momentum", "confirmed", "vwap_aligned")


def entry_profile_pullback_reclaim(
    ctx: CandidateContext,
    setup_spec: ResearchSetup,
    setup_decision: SetupDecision,
) -> EntryDecision:
    """
    Wait for a pullback and reclaim instead of taking early continuation.
    """
    _ = (setup_spec, setup_decision)
    name = "ENTRY_PULLBACK_RECLAIM"
    if not ctx.pullback_detected:
        return _entry_profile_reject(name, "pullback not detected", "pullback", "needs_pullback")
    if not ctx.reclaim_detected:
        return _entry_profile_reject(name, "reclaim not detected", "pullback", "needs_reclaim")
    if ctx.candles_since_impulse >= 0 and ctx.candles_since_impulse < 2:
        return _entry_profile_reject(name, "reclaim too close to impulse", "pullback", "early_reclaim")
    if ctx.vwap_distance_bucket == "far":
        return _entry_profile_reject(name, "pullback reclaim still too far from VWAP", "pullback", "far_from_vwap")
    if not _vwap_aligned_with_direction(ctx):
        return _entry_profile_reject(name, "reclaim not back on trend side of VWAP", "pullback", "vwap_misaligned")
    return _entry_profile_allow(name, "pullback", "reclaim", "vwap_aligned")


def entry_profile_mean_reversion(
    ctx: CandidateContext,
    setup_spec: ResearchSetup,
    setup_decision: SetupDecision,
) -> EntryDecision:
    """
    Fade entries only when price is stretched away from VWAP and a reversal cue
    is present. This intentionally avoids high-volume continuation.
    """
    _ = (setup_spec, setup_decision)
    name = "ENTRY_MEAN_REVERSION"
    if _bucket_rank(ctx.vwap_distance_bucket) < _bucket_rank("medium"):
        return _entry_profile_reject(name, "requires medium/far VWAP stretch", "mean_reversion", "too_close_to_mean")
    if not _vwap_opposed_to_direction(ctx):
        return _entry_profile_reject(name, "price not stretched against entry direction", "mean_reversion", "wrong_side_of_vwap")
    if not (ctx.rejection_detected or ctx.reclaim_detected or _has_signal_tag(ctx, "FailedBreakout")):
        return _entry_profile_reject(name, "missing reversal trigger", "mean_reversion", "no_reversal_trigger")
    if ctx.volume_bucket == "high":
        return _entry_profile_reject(name, "high-volume continuation blocked", "mean_reversion", "high_volume")
    return _entry_profile_allow(name, "mean_reversion", "stretched", "reversal_trigger")


def entry_profile_range_bound_rejection(
    ctx: CandidateContext,
    setup_spec: ResearchSetup,
    setup_decision: SetupDecision,
) -> EntryDecision:
    """
    Research-only range/chop overlay. Avoids strong trend context and asks for
    a rejection or failed-breakout style signal near an existing range edge.
    """
    _ = (setup_spec, setup_decision)
    name = "ENTRY_RANGE_BOUND_REJECTION"
    if ctx.trend != reg.RANGING and not _has_signal_tag(ctx, "RangingContext"):
        return _entry_profile_reject(name, "requires ranging context", "range_bound", "needs_range")
    if _bucket_rank(ctx.vwap_distance_bucket) < _bucket_rank("medium"):
        return _entry_profile_reject(name, "range fade too close to VWAP", "range_bound", "too_close_to_mean")
    if not (ctx.rejection_detected or _has_signal_tag(ctx, "FailedBreakout")):
        return _entry_profile_reject(name, "requires rejection or failed breakout", "range_bound", "no_rejection")
    if ctx.volume_bucket == "high":
        return _entry_profile_reject(name, "range fade blocks high-volume expansion", "range_bound", "high_volume")
    if ctx.candles_outside_range > config.RESEARCH_RECLAIM_LOOKBACK:
        return _entry_profile_reject(name, "move persisted outside range too long", "range_bound", "persistent_expansion")
    return _entry_profile_allow(name, "range_bound", "rejection", "fade")


ENTRY_PROFILE_REGISTRY: dict[str, ResearchEntryProfile] = {
    "ENTRY_BASELINE": ResearchEntryProfile(
        name="ENTRY_BASELINE",
        description="No extra entry filter. Mirrors the setup's native timing.",
        evaluator=entry_profile_baseline,
    ),
    "ENTRY_ANTI_CHASE": ResearchEntryProfile(
        name="ENTRY_ANTI_CHASE",
        description="Reject early-impulse and already-extended entries.",
        evaluator=entry_profile_anti_chase,
    ),
    "ENTRY_ANTI_CHASE_LONG_ONLY": ResearchEntryProfile(
        name="ENTRY_ANTI_CHASE_LONG_ONLY",
        description="Anti-chase overlay with long-only directional filtering.",
        evaluator=entry_profile_anti_chase_long_only,
    ),
    "ENTRY_ANTI_CHASE_VOLUME_CONFIRMED": ResearchEntryProfile(
        name="ENTRY_ANTI_CHASE_VOLUME_CONFIRMED",
        description="Anti-chase overlay plus minimum medium participation.",
        evaluator=entry_profile_anti_chase_volume_confirmed,
    ),
    "ENTRY_ANTI_CHASE_LONG_LOOKBACK": ResearchEntryProfile(
        name="ENTRY_ANTI_CHASE_LONG_LOOKBACK",
        description="Anti-chase overlay with a stricter 6-candle extension check.",
        evaluator=entry_profile_anti_chase_long_lookback,
    ),
    "ENTRY_ANTI_CHASE_LONG_ONLY_IMPULSE": ResearchEntryProfile(
        name="ENTRY_ANTI_CHASE_LONG_ONLY_IMPULSE",
        description="Long-only anti-chase overlay that keeps only strong 3-candle impulse entries.",
        evaluator=entry_profile_anti_chase_long_only_impulse,
    ),
    "ENTRY_ANTI_CHASE_LONG_ONLY_6C_GUARD": ResearchEntryProfile(
        name="ENTRY_ANTI_CHASE_LONG_ONLY_6C_GUARD",
        description="Long-only anti-chase overlay that blocks high 6-candle extension entries.",
        evaluator=entry_profile_anti_chase_long_only_6c_guard,
    ),
    "ENTRY_ANTI_CHASE_LONG_ONLY_MEDIUM_ATR": ResearchEntryProfile(
        name="ENTRY_ANTI_CHASE_LONG_ONLY_MEDIUM_ATR",
        description="Long-only anti-chase overlay restricted to the medium ATR regime.",
        evaluator=entry_profile_anti_chase_long_only_medium_atr,
    ),
    "ENTRY_ANTI_CHASE_LONG_ONLY_BALANCED": ResearchEntryProfile(
        name="ENTRY_ANTI_CHASE_LONG_ONLY_BALANCED",
        description="Long-only anti-chase overlay that accepts medium ATR or clearly impulsive 3-candle entries.",
        evaluator=entry_profile_anti_chase_long_only_balanced,
    ),
    "ENTRY_ANTI_CHASE_LONG_ONLY_RELAXED_VWAP": ResearchEntryProfile(
        name="ENTRY_ANTI_CHASE_LONG_ONLY_RELAXED_VWAP",
        description="Long-only anti-chase overlay that conditionally allows far-VWAP entries in the medium-ATR, high-impulse pocket.",
        evaluator=entry_profile_anti_chase_long_only_relaxed_vwap,
    ),
    "ENTRY_ANTI_CHASE_LONG_ONLY_RELAXED_EXTENSION": ResearchEntryProfile(
        name="ENTRY_ANTI_CHASE_LONG_ONLY_RELAXED_EXTENSION",
        description="Long-only anti-chase overlay that conditionally allows high 3c/high 6c extension only in medium ATR.",
        evaluator=entry_profile_anti_chase_long_only_relaxed_extension,
    ),
    "ENTRY_ANTI_CHASE_LONG_ONLY_ADAPTIVE": ResearchEntryProfile(
        name="ENTRY_ANTI_CHASE_LONG_ONLY_ADAPTIVE",
        description="Long-only anti-chase overlay that rejects high ATR and weak impulse while conditionally reopening narrow far-VWAP / extension pockets.",
        evaluator=entry_profile_anti_chase_long_only_adaptive,
    ),
    "ENTRY_ANTI_CHASE_TREND_ONLY": ResearchEntryProfile(
        name="ENTRY_ANTI_CHASE_TREND_ONLY",
        description="Anti-chase overlay that blocks range-bound and mean-reversion contexts.",
        evaluator=entry_profile_anti_chase_trend_only,
    ),
    "ENTRY_ANTI_CHASE_TREND_VOLUME": ResearchEntryProfile(
        name="ENTRY_ANTI_CHASE_TREND_VOLUME",
        description="Anti-chase overlay with trend-only context, medium participation, and longer-lookback guard.",
        evaluator=entry_profile_anti_chase_trend_volume,
    ),
    "ENTRY_MOMENTUM_CONFIRMED": ResearchEntryProfile(
        name="ENTRY_MOMENTUM_CONFIRMED",
        description="Trend-following entries with directional confirmation and less chase.",
        evaluator=entry_profile_momentum_confirmed,
    ),
    "ENTRY_PULLBACK_RECLAIM": ResearchEntryProfile(
        name="ENTRY_PULLBACK_RECLAIM",
        description="Wait for a pullback plus reclaim before entry.",
        evaluator=entry_profile_pullback_reclaim,
    ),
    "ENTRY_MEAN_REVERSION": ResearchEntryProfile(
        name="ENTRY_MEAN_REVERSION",
        description="Fade stretched price back toward mean when reversal cues exist.",
        evaluator=entry_profile_mean_reversion,
    ),
    "ENTRY_RANGE_BOUND_REJECTION": ResearchEntryProfile(
        name="ENTRY_RANGE_BOUND_REJECTION",
        description="Range-bound rejection profile that avoids strong expansion.",
        evaluator=entry_profile_range_bound_rejection,
    ),
}

# Research-only exit experiments for RANGE_MEAN_REVERSION.
# These do not affect live trading or any non-range setup.
_MEAN_REVERSION_EXIT_SPECS: list[MeanReversionExitSpec] = [
    MeanReversionExitSpec("MR_EXIT_0", "Current F2 baseline"),
    MeanReversionExitSpec("MR_EXIT_1", "Full exit at VWAP touch"),
    MeanReversionExitSpec("MR_EXIT_2", "50% at VWAP touch, BE stop, Stage-C on remainder"),
    MeanReversionExitSpec("MR_EXIT_3", "Full exit at range midpoint touch"),
    MeanReversionExitSpec("MR_EXIT_4", "50% at range midpoint, 50% at VWAP touch"),
    MeanReversionExitSpec("MR_EXIT_5", "Full exit at min(VWAP touch, 1.0R)"),
    MeanReversionExitSpec("MR_EXIT_6", "Full exit at min(VWAP touch, 0.75R)"),
    MeanReversionExitSpec("MR_EXIT_7", "ATR trail after 0.5R (1.0xATR)"),
    MeanReversionExitSpec("MR_EXIT_8", "ATR trail after 1.0R (1.0xATR)"),
    MeanReversionExitSpec("MR_EXIT_9", "Full exit on 2-bar momentum fade after 0.75R"),
    MeanReversionExitSpec("MR_EXIT_10", "50% at 1.0R, BE stop, 1.0xATR trail remainder"),
    MeanReversionExitSpec("MR_EXIT_11", "50% at 1.0R, BE stop, full exit on 2-bar momentum fade"),
    MeanReversionExitSpec("MR_EXIT_12", "ATR trail after 0.75R (1.5xATR) — wider, later activation"),
    MeanReversionExitSpec("MR_EXIT_13", "50% at 0.5R, BE stop, 1.5xATR trail remainder"),
    MeanReversionExitSpec("MR_EXIT_14", "RSI(7) cross-50 exit after 0.5R profit secured"),
]
_RANGE_MR_HORIZON_SETUPS = (
    "RANGE_MEAN_REVERSION",
    "FV_MR_0",
    "FV_MR_1",
    "FV_MR_6",
)
_RANGE_MR_2H_FILTER_SETUPS = _RANGE_MR_2H_VARIANT_ORDER
_RANGE_MR_HORIZON_EXITS = (
    "MR_EXIT_0",
    "MR_EXIT_1",
    "MR_EXIT_3",
)
_RANGE_MR_DYNAMIC_EXIT_CODES = (
    "MR_EXIT_7",
    "MR_EXIT_8",
    "MR_EXIT_9",
    "MR_EXIT_10",
    "MR_EXIT_11",
    "MR_EXIT_12",
    "MR_EXIT_13",
    "MR_EXIT_14",
)
_RANGE_MR_DYNAMIC_EXIT_SETUPS = (
    # Promoted live setup — test MR_EXIT_7/8/9 against it
    "RMR_2H_NO_FAR_VWAP_LONG_BIASED_BLOCK_LONG_HIGHATR_HIGHVOL",
    "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL_SHORT_BLOCK_HIGHATR_MEDVOL_BLOCK_LONG_LOWATR_MEDVOL_PRE3_MEDIUM_PRE6_LOW_BLOCK_LONG_MEDATR_MEDVOL_PRE3_LOW_PRE6_LOW",
    "RMR_2H_NO_FAR_VWAP_LONG_BIASED_BLOCK_LONG_HIGHATR_HIGHVOL_BLOCK_LONG_LOWATR_MEDVOL_PRE3_MEDIUM_PRE6_LOW",
    "RMR_2H_NO_FAR_VWAP_BLOCK_LONG_HIGHATR_HIGHVOL",
    "RMR_2H_NO_FAR_VWAP",
)


# ── Dynamic stop helper ───────────────────────────────────────────────────────

def _advance_stop(trade: SimTrade, profit_R: float) -> None:
    """
    Advance current_stop toward price based on the active stage.

    Stage A  (profit_R < stage_b_r)       : hard stop only — no movement.
    Stage B  (stage_b_r ≤ profit_R, pre-TP): trail at stage_b_atr_mult×ATR behind
                                             highest/lowest close; floor at entry.
    Stage C  (after partial TP)            : trail at 0.8×ATR; floor at BE+0.1R.

    current_stop only ever moves in favour (max for LONG, min for SHORT).
    """
    atr = trade.atr_at_entry
    sd  = trade.stop_distance

    if trade.direction == "LONG":
        if not trade.tp_hit:
            if profit_R >= trade.stage_b_r:
                raw   = trade.highest_close - atr * trade.stage_b_atr_mult
                floor = trade.entry_price                          # breakeven
                trade.current_stop = max(trade.current_stop, round(max(raw, floor), 2))
        else:
            raw   = trade.highest_close - atr * config.STAGE_C_ATR_MULT
            floor = trade.entry_price + trade.stage_c_floor_offset_r * sd
            trade.current_stop = max(trade.current_stop, round(max(raw, floor), 2))
    else:  # SHORT
        if not trade.tp_hit:
            if profit_R >= trade.stage_b_r:
                raw    = trade.lowest_close + atr * trade.stage_b_atr_mult
                ceil_  = trade.entry_price                         # breakeven
                trade.current_stop = min(trade.current_stop, round(min(raw, ceil_), 2))
        else:
            raw    = trade.lowest_close + atr * config.STAGE_C_ATR_MULT
            ceil_  = trade.entry_price - trade.stage_c_floor_offset_r * sd
            trade.current_stop = min(trade.current_stop, round(min(raw, ceil_), 2))


# ── Core simulation loop ──────────────────────────────────────────────────────

_BASELINE_BLOCKED_ONLY_SIGS = {"VWAP", "Volume", "Stochastic"}
_STRONG_CONFIRMATIONS = {"MACD", "EMA Cross", "Volume"}


def _entry_filter_allows(signals_fired: list[str], signal_experiment: str = "S0") -> tuple[bool, str]:
    sigs = set(signals_fired)

    if len(signals_fired) < 2:
        return False, f"weak consensus (only {len(signals_fired)} signal)"

    if sigs <= _BASELINE_BLOCKED_ONLY_SIGS:
        return False, "VWAP/Volume/Stoch only — no trend or MACD confirmation"

    if config.REQUIRE_MACD and "MACD" not in sigs:
        return False, f"MACD required but absent (signals: {signals_fired})"

    if signal_experiment == "S0":
        return True, ""
    if signal_experiment == "S1":
        if "Volume" not in sigs:
            return False, f"S1 reject: require MACD+Volume (signals: {signals_fired})"
        return True, ""
    if signal_experiment == "S2":
        if "Volume" not in sigs or "VWAP" not in sigs:
            return False, f"S2 reject: require MACD+Volume+VWAP (signals: {signals_fired})"
        return True, ""
    if signal_experiment == "S3":
        if "EMA Cross" not in sigs:
            return False, f"S3 reject: require MACD+EMA Cross (signals: {signals_fired})"
        return True, ""
    if signal_experiment == "S4":
        if "EMA Cross" not in sigs or "Volume" not in sigs:
            return False, f"S4 reject: require MACD+EMA Cross+Volume (signals: {signals_fired})"
        return True, ""
    if signal_experiment == "S5":
        if "EMA Cross" not in sigs and "Volume" not in sigs:
            return False, f"S5 reject: MACD+VWAP-only is too weak (signals: {signals_fired})"
        return True, ""
    if signal_experiment == "S6":
        strong_count = len(_STRONG_CONFIRMATIONS & sigs)
        if strong_count < 2:
            return False, (
                "S6 reject: need at least two strong confirmations "
                f"(MACD/EMA Cross/Volume) (signals: {signals_fired})"
            )
        return True, ""

    raise ValueError(f"Unknown signal experiment: {signal_experiment}")


def _run_simulation(
    df: pd.DataFrame,
    initial_balance: float,
    *,
    signal_experiment: str = "S0",
    log_skips: bool = True,
) -> tuple[list[SimTrade], list[float]]:
    """
    Iterate through *df* candle-by-candle, applying the full strategy stack.

    Entry : next candle open after signal (anti-chase: |close−EMA9|/close < 1%).
    Exit  : staged model —
              Stage A  (< stage_b_r R)  : hard stop only.
              Stage B  (≥ stage_b_r R)  : trail at stage_b_atr_mult×ATR, floor BE.
              Partial TP (partial_tp_r R): close 50%, stop → BE+0.1R.
              Stage C  (after TP)        : trail at 0.8×ATR, floor BE+0.1R.
            Early exits: stall (>6 candles, <0.3R) / time (>30 candles, <0.5R).
            Winner extension: if trending + ATR expanding → delay trail 2 candles.
            Conservative rule: hard-stop always wins on the same candle.

    Returns (trades, equity_curve).
    """
    warmup = _warmup_bars()

    trades: list[SimTrade] = []
    balance = initial_balance
    peak_balance = initial_balance
    equity_curve: list[float] = [balance] * warmup

    open_trade: Optional[SimTrade] = None

    for i in range(warmup, len(df) - 1):
        candle      = df.iloc[i]
        next_candle = df.iloc[i + 1]
        history     = df.iloc[: i + 1]

        lo         = float(candle["low"])
        hi         = float(candle["high"])
        curr_close = float(candle["close"])

        # ── Manage open position ──────────────────────────────────────────────
        if open_trade is not None:
            open_trade.candles_in_trade += 1
            sd = open_trade.stop_distance

            # Profit in R units (close-based) for stage and guard checks
            if open_trade.direction == "LONG":
                profit_R = (curr_close - open_trade.entry_price) / sd if sd > 0 else 0.0
            else:
                profit_R = (open_trade.entry_price - curr_close) / sd if sd > 0 else 0.0

            # ── Path tracking (before exits so exit candle is captured) ────────
            if sd > 0:
                ep = open_trade.entry_price
                cit = open_trade.candles_in_trade
                if open_trade.direction == "LONG":
                    if hi > open_trade.mfe_price:
                        open_trade.mfe_price = hi
                        open_trade.candle_at_mfe = cit
                    if lo < open_trade.mae_price:
                        open_trade.mae_price = lo
                        open_trade.candle_at_mae = cit
                    r_hi = (hi - ep) / sd
                    r_lo = (ep - lo) / sd
                else:
                    if lo < open_trade.mfe_price:
                        open_trade.mfe_price = lo
                        open_trade.candle_at_mfe = cit
                    if hi > open_trade.mae_price:
                        open_trade.mae_price = hi
                        open_trade.candle_at_mae = cit
                    r_hi = (ep - lo) / sd
                    r_lo = (hi - ep) / sd
                if not open_trade.reached_0_5r and r_hi >= 0.5:
                    open_trade.reached_0_5r = True
                    open_trade.candles_to_0_5r = cit
                if not open_trade.reached_1_0r and r_hi >= 1.0:
                    open_trade.reached_1_0r = True
                    open_trade.candles_to_1_0r = cit
                if not open_trade.reached_1_5r and r_hi >= 1.5:
                    open_trade.reached_1_5r = True
                    open_trade.candles_to_1_5r = cit
                if not open_trade.reached_2_0r and r_hi >= 2.0:
                    open_trade.reached_2_0r = True
                if open_trade.candle_first_neg_1r == -1 and r_lo >= 1.0:
                    open_trade.candle_first_neg_1r = cit

            # ── Classify exits (in strict priority order) ─────────────────────
            stop_hit       = False
            partial_tp     = False
            trail_hit      = False
            stall_exit     = False
            time_exit      = False
            partial_tp_price = 0.0

            if open_trade.direction == "LONG":
                stop_hit = lo <= open_trade.current_stop
                if not stop_hit:
                    if not open_trade.tp_hit:
                        partial_tp_price = (
                            open_trade.entry_price + sd * open_trade.partial_tp_r
                        )
                        partial_tp = hi >= partial_tp_price
                        if not partial_tp:
                            stall_exit = (
                                config.STALL_EXIT_ENABLED
                                and open_trade.candles_in_trade > config.STALL_CANDLES
                                and profit_R < config.STALL_R_THRESHOLD
                            )
                            time_exit = (
                                open_trade.candles_in_trade > config.TIME_CANDLES
                                and profit_R < config.TIME_R_THRESHOLD
                            )
                    else:
                        trail_hit = lo <= open_trade.current_stop
            else:  # SHORT
                stop_hit = hi >= open_trade.current_stop
                if not stop_hit:
                    if not open_trade.tp_hit:
                        partial_tp_price = (
                            open_trade.entry_price - sd * open_trade.partial_tp_r
                        )
                        partial_tp = lo <= partial_tp_price
                        if not partial_tp:
                            stall_exit = (
                                config.STALL_EXIT_ENABLED
                                and open_trade.candles_in_trade > config.STALL_CANDLES
                                and profit_R < config.STALL_R_THRESHOLD
                            )
                            time_exit = (
                                open_trade.candles_in_trade > config.TIME_CANDLES
                                and profit_R < config.TIME_R_THRESHOLD
                            )
                    else:
                        trail_hit = hi >= open_trade.current_stop

            # ── Execute exits ─────────────────────────────────────────────────
            def _close_pnl(size: float, exit_px: float) -> float:
                if open_trade.direction == "LONG":
                    return risk.net_pnl(size, open_trade.entry_price, exit_px)
                gross     = size * (open_trade.entry_price - exit_px)
                entry_fee = size * open_trade.entry_price * config.MAKER_FEE
                exit_fee  = size * exit_px * config.MAKER_FEE
                return round(gross - entry_fee - exit_fee, 6)

            def _finalise(exit_px: float, reason: str) -> None:
                nonlocal balance, peak_balance, open_trade
                pnl = open_trade.partial_pnl + _close_pnl(open_trade.remaining_size, exit_px)
                balance      += pnl
                peak_balance  = max(peak_balance, balance)
                open_trade.exit_time   = candle.name
                open_trade.exit_price  = round(exit_px, 2)
                open_trade.pnl_net     = pnl
                open_trade.result      = "WIN" if pnl > 0 else "LOSS"
                open_trade.exit_reason = reason
                denom = open_trade.size * open_trade.stop_distance
                open_trade.exit_r = round(pnl / denom, 3) if denom > 0 else 0.0
                trades.append(open_trade)
                open_trade = None

            if stop_hit:
                _finalise(open_trade.current_stop, "STOP")
            elif stall_exit:
                _finalise(curr_close, "STALL")
            elif time_exit:
                _finalise(curr_close, "TIME")

            elif partial_tp:
                half = open_trade.remaining_size * 0.5
                open_trade.partial_pnl   += _close_pnl(half, partial_tp_price)
                open_trade.remaining_size = half
                open_trade.tp_hit         = True
                # Move stop to breakeven + BE_OFFSET_R × R
                if open_trade.direction == "LONG":
                    be = round(open_trade.entry_price + open_trade.stage_c_floor_offset_r * sd, 2)
                    open_trade.current_stop = max(open_trade.current_stop, be)
                else:
                    be = round(open_trade.entry_price - open_trade.stage_c_floor_offset_r * sd, 2)
                    open_trade.current_stop = min(open_trade.current_stop, be)

            elif trail_hit:
                _finalise(open_trade.current_stop, "TRAIL")

            # ── Post-candle updates (trade still open) ────────────────────────
            if open_trade is not None:
                # Update close-based extremes
                if open_trade.direction == "LONG":
                    open_trade.highest_close = max(open_trade.highest_close, curr_close)
                else:
                    open_trade.lowest_close = min(open_trade.lowest_close, curr_close)

                # Winner-extension check: trending price structure + expanding ATR
                if i >= 2:
                    if open_trade.direction == "LONG":
                        trending = (
                            float(df.iloc[i    ]["high"]) >
                            float(df.iloc[i - 1]["high"]) >
                            float(df.iloc[i - 2]["high"])
                        )
                    else:
                        trending = (
                            float(df.iloc[i    ]["low"]) <
                            float(df.iloc[i - 1]["low"]) <
                            float(df.iloc[i - 2]["low"])
                        )
                    if trending and (hi - lo) > open_trade.atr_at_entry * 0.9:
                        open_trade.trail_delay = max(open_trade.trail_delay, 2)

                # Advance dynamic stop (skip if winner-extension delay active)
                if open_trade.trail_delay > 0:
                    open_trade.trail_delay -= 1
                else:
                    _advance_stop(open_trade, profit_R)

            equity_curve.append(balance)
            continue

        # ── Compute regime ─────────────────────────────────────────────────────
        adx_df     = history.ta.adx(length=config.ADX_PERIOD)
        atr_series = history.ta.atr(length=config.ATR_PERIOD)

        adx_val = float(adx_df[f"ADX_{config.ADX_PERIOD}"].iloc[-1])
        atr_val = float(atr_series.iloc[-1])
        atr_pct = (atr_val / curr_close) * 100.0

        trend = reg.TRENDING if adx_val > config.ADX_TREND_THRESHOLD else reg.RANGING
        vol   = reg.HIGH_VOLATILITY if atr_pct > config.ATR_HIGH_VOL_THRESHOLD_PCT else reg.NORMAL

        if not reg.regime_allows_trade(trend, vol):
            equity_curve.append(balance)
            continue

        # ── Compute consensus ──────────────────────────────────────────────────
        result = con.compute(history, trend, vol)

        if result.decision not in (con.BUY, con.SELL):
            equity_curve.append(balance)
            continue

        # ── Anti-chase filter ──────────────────────────────────────────────────
        ema9_series = history.ta.ema(length=config.EMA_FAST)
        if ema9_series is None or ema9_series.iloc[-1] != ema9_series.iloc[-1]:
            equity_curve.append(balance)
            continue
        ema9_val = float(ema9_series.iloc[-1])
        if abs(curr_close - ema9_val) / curr_close >= 0.01:
            equity_curve.append(balance)
            continue

        # ── Calculate trade parameters ─────────────────────────────────────────
        direction = "LONG" if result.decision == con.BUY else "SHORT"
        halve     = reg.should_halve_position(trend, vol)

        if direction == "LONG":
            entry_price = float(next_candle["open"]) * (1.0 + config.SLIPPAGE)
        else:
            entry_price = float(next_candle["open"]) * (1.0 - config.SLIPPAGE)

        params = risk.calculate(
            df=history,
            entry_price=entry_price,
            account_balance=balance,
            halve=halve,
        )

        if params.position_size <= 0:
            equity_curve.append(balance)
            continue

        if direction == "LONG":
            stop_price = entry_price - params.stop_distance
            tp_price   = entry_price + (params.stop_distance * 2.0)   # reference only
        else:
            stop_price = entry_price + params.stop_distance
            tp_price   = entry_price - (params.stop_distance * 2.0)

        signals_fired = [s.name for s in result.breakdown if s.signal != 0]

        # ── Quality guards ────────────────────────────────────────────────────
        allow_trade, skip_reason = _entry_filter_allows(
            signals_fired,
            signal_experiment=signal_experiment,
        )
        if not allow_trade:
            if log_skips:
                logger.log_info(f"SKIP: {skip_reason}")
            equity_curve.append(balance)
            continue

        # ── Regime-based exit modulation ──────────────────────────────────────
        base_b_r   = config.DEFAULT_STAGE_B_R
        base_mult  = config.DEFAULT_STAGE_B_ATR_MULT
        base_tp_r  = config.DEFAULT_PARTIAL_TP_R

        if "Volume" in signals_fired:
            stage_b_r    = max(0.0, base_b_r - 0.2)   # faster activation
            stage_b_mult = max(0.5, base_mult - 0.2)  # tighter trail
            partial_tp_r = base_tp_r
        else:                                           # MACD/EMA-confirmed
            stage_b_r    = base_b_r
            stage_b_mult = base_mult
            partial_tp_r = base_tp_r

        open_trade = SimTrade(
            trade_num=len(trades) + 1,
            entry_time=next_candle.name,
            entry_price=entry_price,
            stop_price=round(stop_price, 2),
            tp_price=round(tp_price, 2),
            size=params.position_size,
            direction=direction,
            signals_fired=signals_fired,
            atr_at_entry=atr_val,
            stop_distance=round(params.stop_distance, 2),
            remaining_size=params.position_size,
            current_stop=round(stop_price, 2),
            stage_b_r=stage_b_r,
            stage_b_atr_mult=stage_b_mult,
            partial_tp_r=partial_tp_r,
            highest_close=entry_price,
            lowest_close=entry_price,
            mfe_price=entry_price,
            mae_price=entry_price,
        )

        equity_curve.append(balance)

    # Force-close any still-open trade at the final candle's close price
    if open_trade is not None:
        last_close = float(df["close"].iloc[-1])
        if open_trade.direction == "LONG":
            close_pnl = risk.net_pnl(open_trade.remaining_size, open_trade.entry_price, last_close)
        else:
            gross     = open_trade.remaining_size * (open_trade.entry_price - last_close)
            entry_fee = open_trade.remaining_size * open_trade.entry_price * config.MAKER_FEE
            exit_fee  = open_trade.remaining_size * last_close * config.MAKER_FEE
            close_pnl = round(gross - entry_fee - exit_fee, 6)
        total_pnl = open_trade.partial_pnl + close_pnl
        balance += total_pnl
        open_trade.exit_time   = df.index[-1]
        open_trade.exit_price  = last_close
        open_trade.pnl_net     = total_pnl
        open_trade.result      = "WIN" if total_pnl > 0 else "LOSS"
        open_trade.exit_reason = "FORCE_CLOSE"
        denom = open_trade.size * open_trade.stop_distance
        open_trade.exit_r = round(total_pnl / denom, 3) if denom > 0 else 0.0
        trades.append(open_trade)
        equity_curve.append(balance)

    return trades, equity_curve


# ── Metrics calculation ───────────────────────────────────────────────────────

@dataclass
class BacktestMetrics:
    label: str
    initial_balance: float
    final_balance: float
    total_trades: int
    wins: int
    losses: int
    win_rate_pct: float
    avg_win_pct: float
    avg_loss_pct: float
    profit_factor: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    total_return_pct: float


def _compute_metrics(
    trades: list[SimTrade],
    equity_curve: list[float],
    initial_balance: float,
    label: str,
) -> BacktestMetrics:
    if not trades:
        return BacktestMetrics(
            label=label,
            initial_balance=initial_balance,
            final_balance=initial_balance,
            total_trades=0,
            wins=0,
            losses=0,
            win_rate_pct=0.0,
            avg_win_pct=0.0,
            avg_loss_pct=0.0,
            profit_factor=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown_pct=0.0,
            total_return_pct=0.0,
        )

    wins = [t for t in trades if t.result == "WIN"]
    losses = [t for t in trades if t.result == "LOSS"]
    final_balance = equity_curve[-1]

    win_rate = len(wins) / len(trades) * 100

    avg_win_pct = (
        float(np.mean([t.pnl_net / (t.entry_price * t.size) * 100 for t in wins]))
        if wins else 0.0
    )
    avg_loss_pct = (
        float(np.mean([abs(t.pnl_net) / (t.entry_price * t.size) * 100 for t in losses]))
        if losses else 0.0
    )

    gross_profit = sum(t.pnl_net for t in wins) if wins else 0.0
    gross_loss = abs(sum(t.pnl_net for t in losses)) if losses else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    # Risk-adjusted ratios use per-candle equity returns and a horizon-aware
    # annualisation factor so 2H/4H research does not inherit 1H scaling.
    eq = np.array(equity_curve, dtype=float)
    returns = np.diff(eq) / eq[:-1]
    returns = returns[np.isfinite(returns)]
    annual_periods = 8_760.0
    label_lc = label.lower()
    if "4h" in label_lc:
        annual_periods = 24.0 / 4.0 * 365.0
    elif "2h" in label_lc:
        annual_periods = 24.0 / 2.0 * 365.0
    elif "1d" in label_lc or "daily" in label_lc:
        annual_periods = 365.0

    if len(returns) > 1 and returns.std() > 0:
        sharpe = float(returns.mean() / returns.std() * math.sqrt(annual_periods))
    else:
        sharpe = 0.0
    downside = returns[returns < 0]
    if len(returns) > 1 and len(downside) > 0 and downside.std() > 0:
        sortino = float(returns.mean() / downside.std() * math.sqrt(annual_periods))
    else:
        sortino = 0.0

    # Max drawdown (peak-to-trough %)
    peak = eq[0]
    max_dd = 0.0
    for val in eq:
        if val > peak:
            peak = val
        dd = (peak - val) / peak * 100
        if dd > max_dd:
            max_dd = dd

    total_return = (final_balance - initial_balance) / initial_balance * 100

    return BacktestMetrics(
        label=label,
        initial_balance=initial_balance,
        final_balance=final_balance,
        total_trades=len(trades),
        wins=len(wins),
        losses=len(losses),
        win_rate_pct=round(win_rate, 2),
        avg_win_pct=round(avg_win_pct, 4),
        avg_loss_pct=round(avg_loss_pct, 4),
        profit_factor=round(profit_factor, 4),
        sharpe_ratio=round(sharpe, 4),
        sortino_ratio=round(sortino, 4),
        max_drawdown_pct=round(max_dd, 4),
        total_return_pct=round(total_return, 4),
    )


# ── ASCII equity chart ────────────────────────────────────────────────────────

def _ascii_chart(equity: list[float], width: int = 60, height: int = 12) -> str:
    if len(equity) < 2:
        return "(not enough data)"

    eq = np.array(equity, dtype=float)
    min_val = eq.min()
    max_val = eq.max()
    span = max_val - min_val if max_val != min_val else 1.0

    # Downsample to *width* columns
    indices = np.linspace(0, len(eq) - 1, width).astype(int)
    sampled = eq[indices]

    # Map each sample to a row (0 = top, height-1 = bottom)
    def to_row(val: float) -> int:
        normalised = (val - min_val) / span
        return height - 1 - int(normalised * (height - 1))

    rows_idx = [to_row(v) for v in sampled]

    # Build grid
    grid = [[" " for _ in range(width)] for _ in range(height)]
    for col, row in enumerate(rows_idx):
        grid[row][col] = "█"

    lines = []
    for r, row in enumerate(grid):
        # Y-axis label at first and last row
        if r == 0:
            label = f"${max_val:>8,.0f} │"
        elif r == height - 1:
            label = f"${min_val:>8,.0f} │"
        else:
            label = " " * 9 + "│"
        lines.append(label + "".join(row))

    x_axis = " " * 10 + "└" + "─" * width
    lines.append(x_axis)
    lines.append(
        " " * 10 + f"  t=0{' ' * (width - 12)}t={len(equity)}"
    )
    return "\n".join(lines)


# ── CSV export ────────────────────────────────────────────────────────────────

def _write_csv(trades: list[SimTrade], equity_curve: list[float], initial_balance: float) -> None:
    peak = initial_balance
    with open(EQUITY_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_EQUITY_HEADERS)
        writer.writeheader()
        balance = initial_balance
        for t in trades:
            balance += t.pnl_net or 0.0
            peak = max(peak, balance)
            dd = (peak - balance) / peak * 100
            writer.writerow({
                "timestamp": t.exit_time.strftime("%Y-%m-%d %H:%M:%S") if t.exit_time else "",
                "trade_num": t.trade_num,
                "direction": t.direction,
                "entry": round(t.entry_price, 2),
                "exit": round(t.exit_price or 0.0, 2),
                "pnl_net": round(t.pnl_net or 0.0, 4),
                "balance": round(balance, 2),
                "drawdown_pct": round(dd, 4),
            })
    logger.log_info(f"Equity curve saved to {EQUITY_CSV}")


# ── Report printer ────────────────────────────────────────────────────────────

def _print_metrics(m: BacktestMetrics) -> None:
    sep = "─" * 52
    print(f"\n{'━' * 52}")
    print(f"  {m.label}")
    print(f"{'━' * 52}")
    print(f"  Initial balance : ${m.initial_balance:>12,.2f}")
    print(f"  Final balance   : ${m.final_balance:>12,.2f}")
    print(f"  Total return    : {m.total_return_pct:>+.2f}%")
    print(sep)
    print(f"  Total trades    : {m.total_trades}")
    print(f"  Wins / Losses   : {m.wins} / {m.losses}")
    print(f"  Win rate        : {m.win_rate_pct:.2f}%")
    print(f"  Avg win         : +{m.avg_win_pct:.4f}%")
    print(f"  Avg loss        : -{m.avg_loss_pct:.4f}%")
    print(sep)
    print(f"  Profit factor   : {m.profit_factor:.4f}")
    print(f"  Sharpe ratio    : {m.sharpe_ratio:.4f}  (annualised)")
    print(f"  Sortino ratio   : {m.sortino_ratio:.4f}  (annualised)")
    print(f"  Max drawdown    : {m.max_drawdown_pct:.4f}%")
    print()


# ── Attribution printer ───────────────────────────────────────────────────────

def _r_stats(subset: list[SimTrade]) -> tuple[float, float, float]:
    """Return (avg_R, median_R, profit_factor_R) for a set of trades."""
    if not subset:
        return 0.0, 0.0, 0.0
    rs = [t.exit_r for t in subset]
    avg_r = sum(rs) / len(rs)
    med_r = statistics.median(rs)
    wins_r  = sum(r for r in rs if r > 0)
    losses_r = abs(sum(r for r in rs if r < 0))
    pf_r = wins_r / losses_r if losses_r > 0 else float("inf")
    return avg_r, med_r, pf_r


def _trade_mfe_r(trade: SimTrade) -> float:
    if trade.stop_distance <= 0:
        return 0.0
    if trade.direction == "LONG":
        return (trade.mfe_price - trade.entry_price) / trade.stop_distance
    return (trade.entry_price - trade.mfe_price) / trade.stop_distance


@dataclass
class SignalSplitStats:
    trades: int
    win_rate_pct: float
    profit_factor: float
    avg_r: float
    median_r: float
    avg_win_r: float
    avg_loss_r: float
    return_pct: float
    partial_tp_hit_rate_pct: float
    partial_tp_hits: int
    no_tp_avg_mfe_r: float
    no_tp_median_mfe_r: float
    never_reached_0_5r: int
    combo_breakdown: list[tuple[str, int, float, float]]


@dataclass
class SignalExperimentResult:
    label: str
    description: str
    in_metrics: BacktestMetrics
    out_metrics: BacktestMetrics
    in_stats: SignalSplitStats
    out_stats: SignalSplitStats


def _safe_mean(vals: list[float]) -> float:
    return float(statistics.mean(vals)) if vals else 0.0


def _safe_median(vals: list[float]) -> float:
    return float(statistics.median(vals)) if vals else 0.0


def _build_signal_split_stats(trades: list[SimTrade], metrics: BacktestMetrics) -> SignalSplitStats:
    wins = [t for t in trades if t.exit_r > 0]
    losses = [t for t in trades if t.exit_r <= 0]
    no_tp = [t for t in trades if not t.tp_hit]

    combo_map: dict[str, list[SimTrade]] = {}
    for trade in trades:
        combo = "+".join(sorted(trade.signals_fired)) if trade.signals_fired else "(none)"
        combo_map.setdefault(combo, []).append(trade)

    combo_rows: list[tuple[str, int, float, float]] = []
    for combo, combo_trades in sorted(combo_map.items(), key=lambda item: (-len(item[1]), item[0])):
        tp_hits = sum(1 for t in combo_trades if t.tp_hit)
        wins_n = sum(1 for t in combo_trades if t.exit_r > 0)
        combo_rows.append((
            combo,
            len(combo_trades),
            tp_hits / len(combo_trades) * 100.0,
            wins_n / len(combo_trades) * 100.0,
        ))

    partial_hits = sum(1 for t in trades if t.tp_hit)
    no_tp_mfes = [_trade_mfe_r(t) for t in no_tp]
    return SignalSplitStats(
        trades=len(trades),
        win_rate_pct=metrics.win_rate_pct,
        profit_factor=metrics.profit_factor,
        avg_r=_safe_mean([t.exit_r for t in trades]),
        median_r=_safe_median([t.exit_r for t in trades]),
        avg_win_r=_safe_mean([t.exit_r for t in wins]),
        avg_loss_r=_safe_mean([t.exit_r for t in losses]),
        return_pct=metrics.total_return_pct,
        partial_tp_hit_rate_pct=(partial_hits / len(trades) * 100.0) if trades else 0.0,
        partial_tp_hits=partial_hits,
        no_tp_avg_mfe_r=_safe_mean(no_tp_mfes),
        no_tp_median_mfe_r=_safe_median(no_tp_mfes),
        never_reached_0_5r=sum(1 for t in trades if not t.reached_0_5r),
        combo_breakdown=combo_rows,
    )


def _print_attribution(trades: list[SimTrade], label: str) -> None:
    """Print per-strategy and exit-reason attribution for a set of trades."""
    if not trades:
        return

    closed = [t for t in trades if t.result in ("WIN", "LOSS")]
    if not closed:
        return

    W = 70

    # ── Per-strategy table ────────────────────────────────────────────────────
    all_names: list[str] = []
    for t in closed:
        for n in t.signals_fired:
            if n not in all_names:
                all_names.append(n)
    all_names.sort()

    print(f"\n  {'━' * W}")
    print(f"  STRATEGY ATTRIBUTION — {label}")
    print(f"  {'━' * W}")
    hdr = f"  {'Strategy':<16} {'N':>4} {'Wins':>5} {'WR%':>6} {'Avg-R':>7} {'Med-R':>7} {'PF-R':>6}"
    print(hdr)
    print(f"  {'─' * W}")

    for name in all_names:
        grp  = [t for t in closed if name in t.signals_fired]
        wins = [t for t in grp if t.result == "WIN"]
        wr   = len(wins) / len(grp) * 100 if grp else 0.0
        avg_r, med_r, pf_r = _r_stats(grp)
        pf_str = f"{pf_r:.2f}" if pf_r != float("inf") else "  ∞"
        print(
            f"  {name:<16} {len(grp):>4} {len(wins):>5} {wr:>5.1f}% {avg_r:>7.3f} {med_r:>7.3f} {pf_str:>6}"
        )

    # ── Exit reason breakdown ─────────────────────────────────────────────────
    print(f"\n  {'─' * W}")
    print(f"  EXIT REASON BREAKDOWN")
    print(f"  {'─' * W}")
    hdr2 = f"  {'Reason':<12} {'N':>4} {'Wins':>5} {'WR%':>6} {'Avg-R':>7} {'Med-R':>7} {'PF-R':>6}"
    print(hdr2)
    print(f"  {'─' * W}")

    for reason in ["STOP", "STALL", "TIME", "TRAIL", "FORCE_CLOSE"]:
        grp = [t for t in closed if t.exit_reason == reason]
        if not grp:
            continue
        wins = [t for t in grp if t.result == "WIN"]
        wr   = len(wins) / len(grp) * 100
        avg_r, med_r, pf_r = _r_stats(grp)
        pf_str = f"{pf_r:.2f}" if pf_r != float("inf") else "  ∞"
        print(
            f"  {reason:<12} {len(grp):>4} {len(wins):>5} {wr:>5.1f}% {avg_r:>7.3f} {med_r:>7.3f} {pf_str:>6}"
        )

    # ── Signal presence breakdown ─────────────────────────────────────────────
    print(f"\n  {'─' * W}")
    print(f"  SIGNAL PRESENCE BREAKDOWN")
    print(f"  {'─' * W}")
    hdr3 = f"  {'Group':<26} {'N':>4} {'WR%':>6} {'Avg-R':>7} {'Med-R':>7} {'PF-R':>6}"
    print(hdr3)
    print(f"  {'─' * W}")

    groups = [
        ("MACD present",          [t for t in closed if "MACD"        in t.signals_fired]),
        ("MACD absent",           [t for t in closed if "MACD"    not in t.signals_fired]),
        ("Stoch present",         [t for t in closed if "Stochastic"   in t.signals_fired]),
        ("Stoch absent",          [t for t in closed if "Stochastic" not in t.signals_fired]),
        ("EMA present",           [t for t in closed if "EMA Cross"    in t.signals_fired]),
        ("RSI present",           [t for t in closed if "RSI"          in t.signals_fired]),
        ("No RSI & no Stoch",     [t for t in closed if "RSI" not in t.signals_fired
                                                    and "Stochastic" not in t.signals_fired]),
        ("tp_hit (partial TP)",   [t for t in closed if t.tp_hit]),
        ("no partial TP",         [t for t in closed if not t.tp_hit]),
    ]

    for desc, grp in groups:
        if not grp:
            continue
        wins = [t for t in grp if t.result == "WIN"]
        wr   = len(wins) / len(grp) * 100
        avg_r, med_r, pf_r = _r_stats(grp)
        pf_str = f"{pf_r:.2f}" if pf_r != float("inf") else "  ∞"
        print(
            f"  {desc:<26} {len(grp):>4} {wr:>5.1f}% {avg_r:>7.3f} {med_r:>7.3f} {pf_str:>6}"
        )

    print()


# ── Experiment framework ──────────────────────────────────────────────────────

_EXPERIMENT_CONFIGS: list[tuple[str, dict]] = [
    ("A: Baseline",                  {}),
    ("B: No stall exit",             {"STALL_EXIT_ENABLED": False}),
    ("C1: TP at 1.5R",               {"DEFAULT_PARTIAL_TP_R": 1.5}),
    ("C2: TP at 2.0R",               {"DEFAULT_PARTIAL_TP_R": 2.0}),
    ("D1: Stage-C 1.0×ATR",          {"STAGE_C_ATR_MULT": 1.0}),
    ("D2: Stage-C 1.2×ATR",          {"STAGE_C_ATR_MULT": 1.2}),
    ("D3: Stage-C 1.5×ATR",          {"STAGE_C_ATR_MULT": 1.5}),
    ("D4: Stage-C 2.0×ATR",          {"STAGE_C_ATR_MULT": 2.0}),
    ("E1: No stall + C 1.2×",        {"STALL_EXIT_ENABLED": False, "STAGE_C_ATR_MULT": 1.2}),
    ("E2: No stall + C 1.5×",        {"STALL_EXIT_ENABLED": False, "STAGE_C_ATR_MULT": 1.5}),
    ("E3: No stall + C 2.0×",        {"STALL_EXIT_ENABLED": False, "STAGE_C_ATR_MULT": 2.0}),
    ("F1: NoStall+TP1.5R+C1.2×",     {"STALL_EXIT_ENABLED": False, "DEFAULT_PARTIAL_TP_R": 1.5, "STAGE_C_ATR_MULT": 1.2}),
    ("F2: NoStall+TP1.5R+C1.5×",     {"STALL_EXIT_ENABLED": False, "DEFAULT_PARTIAL_TP_R": 1.5, "STAGE_C_ATR_MULT": 1.5}),
    ("F3: NoStall+TP2.0R+C1.5×",     {"STALL_EXIT_ENABLED": False, "DEFAULT_PARTIAL_TP_R": 2.0, "STAGE_C_ATR_MULT": 1.5}),
    ("G: Flat 3.5×ATR trail",        {"STALL_EXIT_ENABLED": False,
                                       "DEFAULT_STAGE_B_R": 0.0,
                                       "DEFAULT_STAGE_B_ATR_MULT": 3.5,
                                       "DEFAULT_PARTIAL_TP_R": 999.0}),
]

_SIGNAL_EXPERIMENT_CONFIGS: list[tuple[str, str, str]] = [
    ("S0", "S0 baseline", "Current candidate: MACD required, stoch disabled, VWAP/Volume/Stoch-only blocked."),
    ("S1", "S1 MACD+Volume", "Require MACD + Volume. VWAP can support but cannot replace Volume."),
    ("S2", "S2 MACD+Volume+VWAP", "Require MACD + Volume + VWAP."),
    ("S3", "S3 MACD+EMA", "Require MACD + EMA trend confirmation. Volume/VWAP optional support."),
    ("S4", "S4 MACD+EMA+Volume", "Require MACD + EMA trend confirmation + Volume."),
    ("S5", "S5 Reject MACD+VWAP-only", "Reject MACD+VWAP-only trades; require MACD plus EMA or Volume."),
    ("S6", "S6 Two strong confirmations", "Need at least two strong confirmations among MACD, EMA Cross, Volume."),
]


def _run_experiment(
    df_in: pd.DataFrame,
    df_out: pd.DataFrame,
    initial_balance: float,
    label: str,
    **overrides,
) -> tuple[BacktestMetrics, BacktestMetrics]:
    """Run IS+OOS on pre-split data with temporary config overrides."""
    old = {k: getattr(config, k) for k in overrides}
    for k, v in overrides.items():
        setattr(config, k, v)
    try:
        in_trades, in_eq   = _run_simulation(df_in,  initial_balance)
        out_trades, out_eq = _run_simulation(df_out, initial_balance)
        in_m  = _compute_metrics(in_trades,  in_eq,  initial_balance, label + " IS")
        out_m = _compute_metrics(out_trades, out_eq, initial_balance, label + " OOS")
    finally:
        for k, v in old.items():
            setattr(config, k, v)
    return in_m, out_m


def _run_signal_experiment(
    df_in: pd.DataFrame,
    df_out: pd.DataFrame,
    initial_balance: float,
    code: str,
    label: str,
    description: str,
) -> SignalExperimentResult:
    in_trades, in_eq = _run_simulation(
        df_in,
        initial_balance,
        signal_experiment=code,
        log_skips=False,
    )
    in_metrics = _compute_metrics(in_trades, in_eq, initial_balance, label + " IS")

    out_start_balance = in_eq[-1] if in_eq else initial_balance
    out_trades, out_eq = _run_simulation(
        df_out,
        out_start_balance,
        signal_experiment=code,
        log_skips=False,
    )
    out_metrics = _compute_metrics(out_trades, out_eq, out_start_balance, label + " OOS")

    return SignalExperimentResult(
        label=label,
        description=description,
        in_metrics=in_metrics,
        out_metrics=out_metrics,
        in_stats=_build_signal_split_stats(in_trades, in_metrics),
        out_stats=_build_signal_split_stats(out_trades, out_metrics),
    )


def _print_experiment_table(results: list[tuple[str, BacktestMetrics, BacktestMetrics]]) -> None:
    W = 96
    print(f"\n{'━' * W}")
    print("  EXPERIMENT COMPARISON TABLE  (IS = in-sample 70%, OOS = out-of-sample 30%)")
    print(f"{'━' * W}")
    print(
        f"  {'Experiment':<32} "
        f"{'IS-Tr':>5} {'IS-WR':>6} {'IS-PF':>6} {'IS-Ret':>7}  "
        f"{'OOS-Tr':>6} {'OOS-WR':>7} {'OOS-PF':>7} {'OOS-Ret':>8}"
    )
    print(f"  {'─' * (W - 2)}")
    for label, im, om in results:
        pf_str_is  = f"{im.profit_factor:.3f}"  if im.profit_factor  != float("inf") else "   ∞"
        pf_str_oos = f"{om.profit_factor:.3f}"  if om.profit_factor  != float("inf") else "   ∞"
        print(
            f"  {label:<32} "
            f"{im.total_trades:>5} {im.win_rate_pct:>5.1f}% {pf_str_is:>6} {im.total_return_pct:>+6.1f}%  "
            f"{om.total_trades:>6} {om.win_rate_pct:>6.1f}% {pf_str_oos:>7} {om.total_return_pct:>+7.1f}%"
        )
    print(f"  {'─' * (W - 2)}")

    valid = [(l, im, om) for l, im, om in results if om.total_trades >= 5]
    if valid:
        best = max(valid, key=lambda r: r[2].profit_factor if r[2].profit_factor != float("inf") else 0.0)
        print(
            f"\n  Best OOS profit factor: [{best[0]}]"
            f"  PF={best[2].profit_factor:.3f}  WR={best[2].win_rate_pct:.1f}%"
            f"  Ret={best[2].total_return_pct:+.1f}%"
        )
    print()


def run_experiments(
    df_in: pd.DataFrame,
    df_out: pd.DataFrame,
    initial_balance: float = 10_000.0,
) -> None:
    """Run all exit experiments on the same pre-split data and print comparison table."""
    W = 96
    print(f"\n{'━' * W}")
    print(f"  RUNNING {len(_EXPERIMENT_CONFIGS)} EXIT EXPERIMENTS  (same data splits)")
    print(f"{'━' * W}")
    results = []
    for label, overrides in _EXPERIMENT_CONFIGS:
        print(f"  {label}…", end=" ", flush=True)
        in_m, out_m = _run_experiment(df_in, df_out, initial_balance, label, **overrides)
        print(f"IS PF={in_m.profit_factor:.3f}  OOS PF={out_m.profit_factor:.3f}")
        results.append((label, in_m, out_m))
    _print_experiment_table(results)


def _print_signal_experiment_summary(results: list[SignalExperimentResult]) -> None:
    width = 160
    print(f"\n{'━' * width}")
    print("  SIGNAL FILTER EXPERIMENTS  (IS = in-sample 70%, OOS = out-of-sample 30%)")
    print(f"{'━' * width}")
    print(
        f"  {'Experiment':<28} {'IS-Tr':>5} {'IS-PF':>6} {'IS-AvgR':>8} "
        f"{'OOS-Tr':>6} {'OOS-WR':>7} {'OOS-PF':>7} {'OOS-AvgR':>8} "
        f"{'OOS-MedR':>8} {'OOS-TP':>7} {'NoTP MedMFE':>11} {'Never<0.5R':>11} {'OOS Ret':>8}"
    )
    print(f"  {'─' * (width - 2)}")
    for result in results:
        oos = result.out_stats
        ins = result.in_stats
        pf_str_is = f"{result.in_metrics.profit_factor:.3f}" if result.in_metrics.profit_factor != float("inf") else "   ∞"
        pf_str_oos = f"{result.out_metrics.profit_factor:.3f}" if result.out_metrics.profit_factor != float("inf") else "   ∞"
        print(
            f"  {result.label:<28} {ins.trades:>5} {pf_str_is:>6} {ins.avg_r:>+8.3f} "
            f"{oos.trades:>6} {oos.win_rate_pct:>6.1f}% {pf_str_oos:>7} {oos.avg_r:>+8.3f} "
            f"{oos.median_r:>+8.3f} {oos.partial_tp_hit_rate_pct:>6.1f}% "
            f"{oos.no_tp_median_mfe_r:>+11.3f} {oos.never_reached_0_5r:>11} {oos.return_pct:>+7.2f}%"
        )
    print()


def _print_signal_experiment_details(results: list[SignalExperimentResult]) -> None:
    for result in results:
        print(f"\n{'─' * 96}")
        print(f"  {result.label}")
        print(f"  {result.description}")
        print(f"  {'─' * 96}")
        for split_name, stats in [("IS", result.in_stats), ("OOS", result.out_stats)]:
            pf_str = f"{stats.profit_factor:.3f}" if stats.profit_factor != float("inf") else "∞"
            print(
                f"  {split_name}: trades={stats.trades}  WR={stats.win_rate_pct:.1f}%  PF={pf_str}  "
                f"AvgR={stats.avg_r:+.3f}  MedR={stats.median_r:+.3f}  "
                f"AvgWin={stats.avg_win_r:+.3f}  AvgLoss={stats.avg_loss_r:+.3f}  "
                f"Ret={stats.return_pct:+.2f}%"
            )
            print(
                f"      TP-hit={stats.partial_tp_hits}/{stats.trades} ({stats.partial_tp_hit_rate_pct:.1f}%)  "
                f"No-TP AvgMFE={stats.no_tp_avg_mfe_r:+.3f}  "
                f"No-TP MedMFE={stats.no_tp_median_mfe_r:+.3f}  "
                f"Never<0.5R={stats.never_reached_0_5r}"
            )
            print("      Signal combos:")
            for combo, count, tp_rate, wr in stats.combo_breakdown:
                print(f"        {combo:<34} {count:>3} trades  TP={tp_rate:>5.1f}%  WR={wr:>5.1f}%")


def run_signal_experiments(
    df_in: pd.DataFrame,
    df_out: pd.DataFrame,
    initial_balance: float = 10_000.0,
) -> None:
    width = 96
    print(f"\n{'━' * width}")
    print(f"  RUNNING {len(_SIGNAL_EXPERIMENT_CONFIGS)} SIGNAL FILTER EXPERIMENTS  (same data splits)")
    print(f"{'━' * width}")
    results: list[SignalExperimentResult] = []
    for code, label, description in _SIGNAL_EXPERIMENT_CONFIGS:
        print(f"  {label}…", end=" ", flush=True)
        result = _run_signal_experiment(
            df_in=df_in,
            df_out=df_out,
            initial_balance=initial_balance,
            code=code,
            label=label,
            description=description,
        )
        print(
            f"OOS PF={result.out_metrics.profit_factor:.3f}  "
            f"OOS AvgR={result.out_stats.avg_r:+.3f}  "
            f"OOS TP={result.out_stats.partial_tp_hit_rate_pct:.1f}%"
        )
        results.append(result)

    _print_signal_experiment_summary(results)
    _print_signal_experiment_details(results)


@dataclass
class WalkForwardWindowResult:
    setup_name: str
    window_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    in_metrics: BacktestMetrics
    out_metrics: BacktestMetrics
    in_trades: list[SimTrade]
    out_trades: list[SimTrade]
    entry_profile_name: str = ""


def _research_close_pnl(trade: SimTrade, exit_px: float) -> float:
    return _research_leg_close_pnl(
        direction=trade.direction,
        size=trade.remaining_size,
        entry_price=trade.entry_price,
        exit_px=exit_px,
    )


def _research_leg_close_pnl(
    *,
    direction: str,
    size: float,
    entry_price: float,
    exit_px: float,
) -> float:
    if direction == "LONG":
        return risk.net_pnl(size, entry_price, exit_px)
    gross = size * (entry_price - exit_px)
    entry_fee = size * entry_price * config.MAKER_FEE
    exit_fee = size * exit_px * config.MAKER_FEE
    return round(gross - entry_fee - exit_fee, 6)


def _research_position_size(balance: float, entry_price: float, stop_distance: float, halve: bool) -> float:
    risk_amount = balance * config.RISK_PER_TRADE
    if halve:
        risk_amount /= 2.0
    position_size = risk_amount / stop_distance if stop_distance > 0 else 0.0
    max_value = balance * config.MAX_POSITION_PCT
    max_size = max_value / entry_price if entry_price > 0 else 0.0
    return round(min(position_size, max_size), 5)


def _stage_params_for_signals(signals_fired: tuple[str, ...]) -> tuple[float, float, float]:
    base_b_r = config.DEFAULT_STAGE_B_R
    base_mult = config.DEFAULT_STAGE_B_ATR_MULT
    base_tp_r = config.DEFAULT_PARTIAL_TP_R
    if "Volume" in signals_fired:
        return max(0.0, base_b_r - 0.2), max(0.5, base_mult - 0.2), base_tp_r
    return base_b_r, base_mult, base_tp_r


def _skip_setup_row(ctx: CandidateContext, decision: SetupDecision, phase: str, window_label: str) -> dict:
    breakout_level = round(ctx.breakout_level, 2) if math.isfinite(ctx.breakout_level) else None
    pullback_depth_r = round(ctx.pullback_depth_r, 4) if math.isfinite(ctx.pullback_depth_r) else None
    range_high = round(ctx.range_high, 2) if math.isfinite(ctx.range_high) else None
    range_low = round(ctx.range_low, 2) if math.isfinite(ctx.range_low) else None
    range_mid = round(ctx.range_mid, 2) if math.isfinite(ctx.range_mid) else None
    range_boundary_r = (
        round(ctx.distance_from_range_boundary_r, 4)
        if math.isfinite(ctx.distance_from_range_boundary_r) else None
    )
    raw_setup_name = decision.setup_name
    return {
        "phase": phase,
        "window_label": window_label,
        "timestamp": ctx.signal_time.isoformat(),
        "rejection_layer": "setup",
        "setup_name": _setup_family_name(raw_setup_name),
        "variant_name": _setup_variant_name(raw_setup_name),
        "setup_code": raw_setup_name,
        "entry_profile_name": "",
        "entry_profile_tags": "",
        "direction": ctx.direction,
        "rejection_reason": decision.rejection_reason,
        "signals_fired": "|".join(ctx.signals_fired),
        "signal_tags": "|".join(decision.signal_tags),
        "atr_bucket": ctx.atr_bucket,
        "volume_bucket": ctx.volume_bucket,
        "vwap_distance_bucket": ctx.vwap_distance_bucket,
        "pre_entry_move_bucket": ctx.pre_entry_move_3c_bucket,
        "pre_entry_move_3c_bucket": ctx.pre_entry_move_3c_bucket,
        "pre_entry_move_6c_bucket": ctx.pre_entry_move_6c_bucket,
        "range_high": range_high,
        "range_low": range_low,
        "range_mid": range_mid,
        "breakout_level": breakout_level,
        "close_price": round(ctx.close_price, 2),
        "vwap": round(ctx.vwap, 2),
        "distance_from_vwap_r": round(ctx.distance_from_vwap_r, 4),
        "distance_from_range_boundary_r": range_boundary_r,
        "atr_pct": round(ctx.atr_pct, 4),
        "volume_ratio": round(ctx.volume_ratio, 4),
        "macd_agreed": ctx.macd_agrees,
        "price_vs_vwap": ctx.price_vs_vwap,
        "trend_context_reason": ctx.trend_context_reason,
        "pullback_detected": ctx.pullback_detected,
        "reclaim_detected": ctx.reclaim_detected,
        "rejection_detected": ctx.rejection_detected,
        "pullback_depth_r": pullback_depth_r,
        "candles_since_impulse": ctx.candles_since_impulse,
        "candles_outside_range": ctx.candles_outside_range,
        "macd_histogram": round(ctx.macd_histogram, 6),
        "macd_slope": round(ctx.macd_slope, 6),
    }


def _skip_entry_profile_row(
    ctx: CandidateContext,
    setup_decision: SetupDecision,
    entry_decision: EntryDecision,
    phase: str,
    window_label: str,
) -> dict:
    row = _skip_setup_row(ctx, setup_decision, phase, window_label)
    row["rejection_layer"] = "entry_profile"
    row["entry_profile_name"] = entry_decision.entry_profile_name
    row["entry_profile_tags"] = "|".join(entry_decision.entry_tags)
    row["rejection_reason"] = entry_decision.rejection_reason
    return row


def _trade_setup_row(trade: SimTrade, phase: str, window_label: str) -> dict:
    breakout_level = round(trade.breakout_level, 2) if math.isfinite(trade.breakout_level) else None
    pullback_depth_r = round(trade.pullback_depth_r, 4) if math.isfinite(trade.pullback_depth_r) else None
    range_high = round(trade.range_high, 2) if math.isfinite(trade.range_high) else None
    range_low = round(trade.range_low, 2) if math.isfinite(trade.range_low) else None
    range_mid = round(trade.range_mid, 2) if math.isfinite(trade.range_mid) else None
    vwap_touch_r = round(trade.vwap_touch_r, 4) if math.isfinite(trade.vwap_touch_r) else None
    range_mid_touch_r = round(trade.range_mid_touch_r, 4) if math.isfinite(trade.range_mid_touch_r) else None
    range_boundary_r = (
        round(trade.distance_from_range_boundary_r, 4)
        if math.isfinite(trade.distance_from_range_boundary_r) else None
    )
    sd = trade.stop_distance
    if sd > 0:
        if trade.direction == "LONG":
            mfe_r = (trade.mfe_price - trade.entry_price) / sd
            mae_r = (trade.entry_price - trade.mae_price) / sd
        else:
            mfe_r = (trade.entry_price - trade.mfe_price) / sd
            mae_r = (trade.mae_price - trade.entry_price) / sd
    else:
        mfe_r = 0.0
        mae_r = 0.0
    if sd > 0:
        if trade.direction == "LONG":
            distance_from_vwap_r = (trade.entry_close_price - trade.entry_vwap) / sd
        else:
            distance_from_vwap_r = (trade.entry_vwap - trade.entry_close_price) / sd
    else:
        distance_from_vwap_r = 0.0
    raw_setup_name = trade.setup_name
    return {
        "phase": phase,
        "window_label": window_label,
        "setup_name": _setup_family_name(raw_setup_name),
        "variant_name": _setup_variant_name(raw_setup_name),
        "setup_code": raw_setup_name,
        "entry_profile_name": trade.entry_profile_name,
        "entry_profile_tags": "|".join(trade.entry_profile_tags),
        "signal_time": trade.signal_time.isoformat() if trade.signal_time else "",
        "entry_time": trade.entry_time.isoformat() if trade.entry_time else "",
        "exit_time": trade.exit_time.isoformat() if trade.exit_time else "",
        "direction": trade.direction,
        "signals_fired": "|".join(trade.signals_fired),
        "signal_tags": "|".join(trade.signal_tags),
        "atr_bucket": trade.atr_bucket,
        "volume_bucket": trade.volume_bucket,
        "vwap_distance_bucket": trade.vwap_distance_bucket,
        "pre_entry_move_bucket": trade.pre_entry_move_3c_bucket,
        "pre_entry_move_3c_bucket": trade.pre_entry_move_3c_bucket,
        "pre_entry_move_6c_bucket": trade.pre_entry_move_6c_bucket,
        "range_high": range_high,
        "range_low": range_low,
        "range_mid": range_mid,
        "breakout_level": breakout_level,
        "close_price": round(trade.entry_close_price, 2),
        "vwap": round(trade.entry_vwap, 2),
        "distance_from_vwap_r": round(distance_from_vwap_r, 4),
        "distance_from_range_boundary_r": range_boundary_r,
        "atr_pct": round(trade.entry_atr_pct, 4),
        "volume_ratio": round(trade.entry_volume_ratio, 4),
        "macd_agreed": trade.macd_agrees,
        "entry_trend": trade.entry_trend,
        "entry_vol_regime": trade.entry_vol_regime,
        "price_vs_vwap": trade.price_vs_vwap,
        "trend_context_reason": trade.trend_context_reason,
        "pullback_detected": trade.pullback_detected,
        "reclaim_detected": trade.reclaim_detected,
        "rejection_detected": trade.rejection_detected,
        "pullback_depth_r": pullback_depth_r,
        "candles_since_impulse": trade.candles_since_impulse,
        "candles_outside_range": trade.candles_outside_range,
        "macd_histogram": round(trade.entry_macd_histogram, 6),
        "macd_slope": round(trade.entry_macd_slope, 6),
        "mfe_r": round(mfe_r, 4),
        "mae_r": round(mae_r, 4),
        "reached_0_5r": trade.reached_0_5r,
        "reached_1_0r": trade.reached_1_0r,
        "reached_1_5r": trade.reached_1_5r,
        "reached_2_0r": trade.reached_2_0r,
        "candles_to_0_5r": trade.candles_to_0_5r,
        "candles_to_1_0r": trade.candles_to_1_0r,
        "candles_to_1_5r": trade.candles_to_1_5r,
        "candle_first_neg_1r": trade.candle_first_neg_1r,
        "touched_range_mid_after_entry": trade.touched_range_mid_after_entry,
        "candles_to_range_mid_touch": trade.candles_to_range_mid_touch,
        "range_mid_touch_r": range_mid_touch_r,
        "touched_vwap_after_entry": trade.touched_vwap_after_entry,
        "candles_to_vwap_touch": trade.candles_to_vwap_touch,
        "vwap_touch_r": vwap_touch_r,
        "max_continuation_away_from_vwap_before_reversion_r": round(
            trade.max_continuation_away_from_vwap_before_reversion_r,
            4,
        ),
        "research_exit_code": trade.research_exit_code,
        "partial_tp_hit": trade.tp_hit,
        "exit_reason": trade.exit_reason,
        "exit_r": round(trade.exit_r, 4),
        "pnl_net": round(trade.pnl_net or 0.0, 4),
    }


def _build_candidate_context_map(candidate_df: pd.DataFrame) -> dict[int, list[CandidateContext]]:
    contexts: dict[int, list[CandidateContext]] = {}
    for _, row in candidate_df.iterrows():
        ctx = _candidate_context_from_row(row)
        contexts.setdefault(ctx.signal_pos, []).append(ctx)
    return contexts


def _prepare_research_context(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, RegimeThresholds | None, dict[int, list[CandidateContext]]]:
    research_frame = _compute_research_frame(df)
    candidate_df = _build_candidate_universe(df, research_frame)
    if candidate_df.empty:
        return research_frame, candidate_df, None, {}
    thresholds = _build_regime_thresholds(candidate_df)
    candidate_df = _apply_regime_buckets(candidate_df, thresholds)
    candidate_map = _build_candidate_context_map(candidate_df)
    return research_frame, candidate_df, thresholds, candidate_map


def _run_research_setup_simulation(
    df: pd.DataFrame,
    research_frame: pd.DataFrame,
    candidate_map: dict[int, list[CandidateContext]],
    start_pos: int,
    end_pos: int,
    initial_balance: float,
    setup_spec: ResearchSetup,
    *,
    phase: str,
    window_label: str,
    entry_profile_spec: ResearchEntryProfile | None = None,
    research_exit_code: str | None = None,
    diagnostic_target_code: str | None = None,
    skip_rows: list[dict] | None = None,
    regime_series: "pd.Series | None" = None,
) -> tuple[list[SimTrade], list[float]]:
    trades: list[SimTrade] = []
    balance = initial_balance
    equity_curve: list[float] = [balance]
    open_trade: Optional[SimTrade] = None
    range_exit_research = (
        setup_spec.candidate_family == "range_mean_reversion" and research_exit_code is not None
    )
    continuation_target_research = (
        diagnostic_target_code is not None
        and setup_spec.candidate_family in {"volume_breakout", "pullback_trend"}
    )
    active_entry_profile = entry_profile_spec or ENTRY_PROFILE_REGISTRY["ENTRY_BASELINE"]

    start = max(start_pos, _warmup_bars())
    stop = min(end_pos, len(df))
    if stop - start < 2:
        return trades, equity_curve

    for i in range(start, stop - 1):
        candle = df.iloc[i]
        lo = float(candle["low"])
        hi = float(candle["high"])
        curr_close = float(candle["close"])

        if open_trade is not None:
            open_trade.candles_in_trade += 1
            sd = open_trade.stop_distance
            curr_vwap = float(research_frame["vwap"].iloc[i])

            if open_trade.direction == "LONG":
                profit_R = (curr_close - open_trade.entry_price) / sd if sd > 0 else 0.0
            else:
                profit_R = (open_trade.entry_price - curr_close) / sd if sd > 0 else 0.0

            if sd > 0:
                ep = open_trade.entry_price
                cit = open_trade.candles_in_trade
                if open_trade.direction == "LONG":
                    if hi > open_trade.mfe_price:
                        open_trade.mfe_price = hi
                        open_trade.candle_at_mfe = cit
                    if lo < open_trade.mae_price:
                        open_trade.mae_price = lo
                        open_trade.candle_at_mae = cit
                    r_hi = (hi - ep) / sd
                    r_lo = (ep - lo) / sd
                else:
                    if lo < open_trade.mfe_price:
                        open_trade.mfe_price = lo
                        open_trade.candle_at_mfe = cit
                    if hi > open_trade.mae_price:
                        open_trade.mae_price = hi
                        open_trade.candle_at_mae = cit
                    r_hi = (ep - lo) / sd
                    r_lo = (hi - ep) / sd
                if not open_trade.reached_0_5r and r_hi >= 0.5:
                    open_trade.reached_0_5r = True
                    open_trade.candles_to_0_5r = cit
                if not open_trade.reached_1_0r and r_hi >= 1.0:
                    open_trade.reached_1_0r = True
                    open_trade.candles_to_1_0r = cit
                if not open_trade.reached_1_5r and r_hi >= 1.5:
                    open_trade.reached_1_5r = True
                    open_trade.candles_to_1_5r = cit
                if not open_trade.reached_2_0r and r_hi >= 2.0:
                    open_trade.reached_2_0r = True
                if open_trade.candle_first_neg_1r == -1 and r_lo >= 1.0:
                    open_trade.candle_first_neg_1r = cit
                if math.isfinite(curr_vwap):
                    if open_trade.direction == "LONG":
                        away_from_vwap_r = max(0.0, (curr_vwap - lo) / sd)
                        open_trade.max_continuation_away_from_vwap_before_reversion_r = max(
                            open_trade.max_continuation_away_from_vwap_before_reversion_r,
                            away_from_vwap_r,
                        )
                        if not open_trade.touched_vwap_after_entry and hi >= curr_vwap:
                            open_trade.touched_vwap_after_entry = True
                            open_trade.candles_to_vwap_touch = cit
                            open_trade.vwap_touch_r = (curr_vwap - ep) / sd
                    else:
                        away_from_vwap_r = max(0.0, (hi - curr_vwap) / sd)
                        open_trade.max_continuation_away_from_vwap_before_reversion_r = max(
                            open_trade.max_continuation_away_from_vwap_before_reversion_r,
                            away_from_vwap_r,
                        )
                        if not open_trade.touched_vwap_after_entry and lo <= curr_vwap:
                            open_trade.touched_vwap_after_entry = True
                            open_trade.candles_to_vwap_touch = cit
                            open_trade.vwap_touch_r = (ep - curr_vwap) / sd
                if math.isfinite(open_trade.range_mid):
                    if open_trade.direction == "LONG":
                        if not open_trade.touched_range_mid_after_entry and hi >= open_trade.range_mid:
                            open_trade.touched_range_mid_after_entry = True
                            open_trade.candles_to_range_mid_touch = cit
                            open_trade.range_mid_touch_r = (open_trade.range_mid - ep) / sd
                    else:
                        if not open_trade.touched_range_mid_after_entry and lo <= open_trade.range_mid:
                            open_trade.touched_range_mid_after_entry = True
                            open_trade.candles_to_range_mid_touch = cit
                            open_trade.range_mid_touch_r = (ep - open_trade.range_mid) / sd

            vwap_touched_this_candle = _price_touches_level(open_trade.direction, lo, hi, curr_vwap)
            range_mid_touched_this_candle = _price_touches_level(
                open_trade.direction,
                lo,
                hi,
                open_trade.range_mid,
            )

            stop_hit = False
            partial_tp = False
            trail_hit = False
            stall_exit = False
            time_exit = False
            partial_tp_price = 0.0
            custom_full_exit_price: float | None = None
            custom_full_exit_reason = ""
            custom_partial_exit_price: float | None = None
            custom_partial_exit_reason = ""
            custom_follow_on_exit_price: float | None = None
            custom_follow_on_exit_reason = ""

            if open_trade.direction == "LONG":
                stop_hit = lo <= open_trade.current_stop
                if not stop_hit:
                    if not open_trade.tp_hit:
                        partial_tp_price = open_trade.entry_price + sd * open_trade.partial_tp_r
                        partial_tp = hi >= partial_tp_price
                        if not partial_tp:
                            stall_exit = (
                                config.STALL_EXIT_ENABLED
                                and open_trade.candles_in_trade > config.STALL_CANDLES
                                and profit_R < config.STALL_R_THRESHOLD
                            )
                            time_exit = (
                                open_trade.candles_in_trade > config.TIME_CANDLES
                                and profit_R < config.TIME_R_THRESHOLD
                            )
                    else:
                        trail_hit = lo <= open_trade.current_stop
            else:
                stop_hit = hi >= open_trade.current_stop
                if not stop_hit:
                    if not open_trade.tp_hit:
                        partial_tp_price = open_trade.entry_price - sd * open_trade.partial_tp_r
                        partial_tp = lo <= partial_tp_price
                        if not partial_tp:
                            stall_exit = (
                                config.STALL_EXIT_ENABLED
                                and open_trade.candles_in_trade > config.STALL_CANDLES
                                and profit_R < config.STALL_R_THRESHOLD
                            )
                            time_exit = (
                                open_trade.candles_in_trade > config.TIME_CANDLES
                                and profit_R < config.TIME_R_THRESHOLD
                            )
                    else:
                        trail_hit = hi >= open_trade.current_stop

            if range_exit_research and research_exit_code != "MR_EXIT_0":
                partial_tp = False

                if research_exit_code == "MR_EXIT_1":
                    if not stop_hit and vwap_touched_this_candle:
                        custom_full_exit_price = curr_vwap
                        custom_full_exit_reason = "VWAP_TOUCH"
                elif research_exit_code == "MR_EXIT_2":
                    if (
                        not stop_hit
                        and open_trade.research_exit_progress == 0
                        and vwap_touched_this_candle
                    ):
                        custom_partial_exit_price = curr_vwap
                        custom_partial_exit_reason = "VWAP_TOUCH_PARTIAL"
                elif research_exit_code == "MR_EXIT_3":
                    if not stop_hit and range_mid_touched_this_candle:
                        custom_full_exit_price = open_trade.range_mid
                        custom_full_exit_reason = "RANGE_MID_TOUCH"
                elif research_exit_code == "MR_EXIT_4":
                    if not stop_hit and open_trade.research_exit_progress == 0 and range_mid_touched_this_candle:
                        custom_partial_exit_price = open_trade.range_mid
                        custom_partial_exit_reason = "RANGE_MID_PARTIAL"
                        if vwap_touched_this_candle and math.isfinite(curr_vwap):
                            if (
                                open_trade.direction == "LONG"
                                and curr_vwap >= open_trade.range_mid
                            ) or (
                                open_trade.direction == "SHORT"
                                and curr_vwap <= open_trade.range_mid
                            ):
                                custom_follow_on_exit_price = curr_vwap
                                custom_follow_on_exit_reason = "VWAP_TOUCH_AFTER_RANGE_MID"
                    elif not stop_hit and open_trade.research_exit_progress >= 1 and vwap_touched_this_candle:
                        custom_full_exit_price = curr_vwap
                        custom_full_exit_reason = "VWAP_TOUCH_AFTER_RANGE_MID"
                elif research_exit_code == "MR_EXIT_5":
                    fixed_r_price = (
                        open_trade.entry_price + sd
                        if open_trade.direction == "LONG"
                        else open_trade.entry_price - sd
                    )
                    custom_full_exit_price = _earliest_profit_target_price(
                        direction=open_trade.direction,
                        entry_price=open_trade.entry_price,
                        lo=lo,
                        hi=hi,
                        dynamic_target_price=curr_vwap,
                        fixed_r_target_price=fixed_r_price,
                    )
                    if custom_full_exit_price is not None and not stop_hit:
                        custom_full_exit_reason = "VWAP_OR_1_0R"
                elif research_exit_code == "MR_EXIT_6":
                    fixed_r_price = (
                        open_trade.entry_price + (sd * 0.75)
                        if open_trade.direction == "LONG"
                        else open_trade.entry_price - (sd * 0.75)
                    )
                    custom_full_exit_price = _earliest_profit_target_price(
                        direction=open_trade.direction,
                        entry_price=open_trade.entry_price,
                        lo=lo,
                        hi=hi,
                        dynamic_target_price=curr_vwap,
                        fixed_r_target_price=fixed_r_price,
                    )
                    if custom_full_exit_price is not None and not stop_hit:
                        custom_full_exit_reason = "VWAP_OR_0_75R"
                elif research_exit_code == "MR_EXIT_7":
                    if not stop_hit:
                        custom_full_exit_price = _atr_trailing_exit_price(
                            trade=open_trade,
                            profit_r=profit_R,
                            activation_r=0.5,
                            atr_mult=1.0,
                            lo=lo,
                            hi=hi,
                        )
                        if custom_full_exit_price is not None:
                            custom_full_exit_reason = "ATR_TRAIL_0_5R"
                elif research_exit_code == "MR_EXIT_8":
                    if not stop_hit:
                        custom_full_exit_price = _atr_trailing_exit_price(
                            trade=open_trade,
                            profit_r=profit_R,
                            activation_r=1.0,
                            atr_mult=1.0,
                            lo=lo,
                            hi=hi,
                        )
                        if custom_full_exit_price is not None:
                            custom_full_exit_reason = "ATR_TRAIL_1_0R"
                elif research_exit_code == "MR_EXIT_9":
                    if not stop_hit and profit_R >= 0.75 and i >= 2:
                        prev_close = float(df["close"].iloc[i - 1])
                        prev2_close = float(df["close"].iloc[i - 2])
                        if open_trade.direction == "LONG":
                            momentum_fade = curr_close < prev_close < prev2_close
                        else:
                            momentum_fade = curr_close > prev_close > prev2_close
                        if momentum_fade:
                            custom_full_exit_price = curr_close
                            custom_full_exit_reason = "MOMENTUM_FADE_0_75R"
                elif research_exit_code == "MR_EXIT_10":
                    fixed_r_price = (
                        open_trade.entry_price + sd
                        if open_trade.direction == "LONG"
                        else open_trade.entry_price - sd
                    )
                    if (
                        not stop_hit
                        and open_trade.research_exit_progress == 0
                        and _price_touches_level(open_trade.direction, lo, hi, fixed_r_price)
                    ):
                        custom_partial_exit_price = fixed_r_price
                        custom_partial_exit_reason = "PARTIAL_1_0R_ATR_TRAIL"
                    elif not stop_hit and open_trade.research_exit_progress >= 1:
                        custom_full_exit_price = _atr_trailing_exit_price(
                            trade=open_trade,
                            profit_r=max(profit_R, 0.0),
                            activation_r=0.0,
                            atr_mult=1.0,
                            lo=lo,
                            hi=hi,
                        )
                        if custom_full_exit_price is not None:
                            custom_full_exit_reason = "ATR_TRAIL_AFTER_PARTIAL_1_0R"
                elif research_exit_code == "MR_EXIT_11":
                    fixed_r_price = (
                        open_trade.entry_price + sd
                        if open_trade.direction == "LONG"
                        else open_trade.entry_price - sd
                    )
                    if (
                        not stop_hit
                        and open_trade.research_exit_progress == 0
                        and _price_touches_level(open_trade.direction, lo, hi, fixed_r_price)
                    ):
                        custom_partial_exit_price = fixed_r_price
                        custom_partial_exit_reason = "PARTIAL_1_0R_MOMENTUM_FADE"
                    elif not stop_hit and open_trade.research_exit_progress >= 1 and i >= 2:
                        prev_close = float(df["close"].iloc[i - 1])
                        prev2_close = float(df["close"].iloc[i - 2])
                        if open_trade.direction == "LONG":
                            momentum_fade = curr_close < prev_close < prev2_close
                        else:
                            momentum_fade = curr_close > prev_close > prev2_close
                        if momentum_fade:
                            custom_full_exit_price = curr_close
                            custom_full_exit_reason = "MOMENTUM_FADE_AFTER_PARTIAL_1_0R"
                elif research_exit_code == "MR_EXIT_12":
                    # ATR trail activated at 0.75R with a wider 1.5× ATR band.
                    if not stop_hit:
                        custom_full_exit_price = _atr_trailing_exit_price(
                            trade=open_trade,
                            profit_r=profit_R,
                            activation_r=0.75,
                            atr_mult=1.5,
                            lo=lo,
                            hi=hi,
                        )
                        if custom_full_exit_price is not None:
                            custom_full_exit_reason = "ATR_TRAIL_0_75R_1_5X"
                elif research_exit_code == "MR_EXIT_13":
                    # Partial 50% exit at 0.5R → move stop to BE → 1.5× ATR trail on remainder.
                    fixed_r_price = (
                        open_trade.entry_price + sd * 0.5
                        if open_trade.direction == "LONG"
                        else open_trade.entry_price - sd * 0.5
                    )
                    if (
                        not stop_hit
                        and open_trade.research_exit_progress == 0
                        and _price_touches_level(open_trade.direction, lo, hi, fixed_r_price)
                    ):
                        custom_partial_exit_price = fixed_r_price
                        custom_partial_exit_reason = "PARTIAL_0_5R_ATR_TRAIL_1_5X"
                    elif not stop_hit and open_trade.research_exit_progress >= 1:
                        custom_full_exit_price = _atr_trailing_exit_price(
                            trade=open_trade,
                            profit_r=max(profit_R, 0.0),
                            activation_r=0.0,
                            atr_mult=1.5,
                            lo=lo,
                            hi=hi,
                        )
                        if custom_full_exit_price is not None:
                            custom_full_exit_reason = "ATR_TRAIL_1_5X_AFTER_PARTIAL_0_5R"
                elif research_exit_code == "MR_EXIT_14":
                    # RSI(7) crosses back through 50 after 0.5R profit secured.
                    if not stop_hit and profit_R >= 0.5 and i >= 1:
                        rsi7_curr = float(research_frame["rsi7"].iloc[i])
                        rsi7_prev = float(research_frame["rsi7"].iloc[i - 1])
                        if math.isfinite(rsi7_curr) and math.isfinite(rsi7_prev):
                            if open_trade.direction == "LONG":
                                rsi_cross = rsi7_curr < 50.0 and rsi7_prev >= 50.0
                            else:
                                rsi_cross = rsi7_curr > 50.0 and rsi7_prev <= 50.0
                            if rsi_cross:
                                custom_full_exit_price = curr_close
                                custom_full_exit_reason = "RSI7_CROSS_50"
            elif continuation_target_research and diagnostic_target_code != "CONT_TARGET_BASELINE":
                partial_tp = False
                if diagnostic_target_code == "CONT_TARGET_1_0R":
                    fixed_target = (
                        open_trade.entry_price + sd
                        if open_trade.direction == "LONG"
                        else open_trade.entry_price - sd
                    )
                    if not stop_hit and _price_touches_level(open_trade.direction, lo, hi, fixed_target):
                        custom_full_exit_price = fixed_target
                        custom_full_exit_reason = "TARGET_1_0R"
                elif diagnostic_target_code == "CONT_TARGET_0_75R":
                    fixed_target = (
                        open_trade.entry_price + (sd * 0.75)
                        if open_trade.direction == "LONG"
                        else open_trade.entry_price - (sd * 0.75)
                    )
                    if not stop_hit and _price_touches_level(open_trade.direction, lo, hi, fixed_target):
                        custom_full_exit_price = fixed_target
                        custom_full_exit_reason = "TARGET_0_75R"
                elif diagnostic_target_code == "CONT_TARGET_VWAP_TOUCH":
                    if not stop_hit and vwap_touched_this_candle and math.isfinite(curr_vwap):
                        custom_full_exit_price = curr_vwap
                        custom_full_exit_reason = "VWAP_TOUCH"
                elif diagnostic_target_code == "CONT_TARGET_RANGE_MID":
                    if (
                        not stop_hit
                        and math.isfinite(open_trade.range_mid)
                        and range_mid_touched_this_candle
                    ):
                        custom_full_exit_price = open_trade.range_mid
                        custom_full_exit_reason = "RANGE_MID_TOUCH"

            def _finalise(exit_px: float, reason: str) -> None:
                nonlocal balance, open_trade
                pnl = open_trade.partial_pnl + _research_close_pnl(open_trade, exit_px)
                balance += pnl
                open_trade.exit_time = candle.name
                open_trade.exit_price = round(exit_px, 2)
                open_trade.pnl_net = pnl
                open_trade.result = "WIN" if pnl > 0 else "LOSS"
                open_trade.exit_reason = reason
                denom = open_trade.size * open_trade.stop_distance
                open_trade.exit_r = round(pnl / denom, 3) if denom > 0 else 0.0
                trades.append(open_trade)
                open_trade = None

            def _apply_partial_exit(
                exit_px: float,
                *,
                move_stop_to_entry: bool,
                activate_stage_c: bool,
                progress_value: int,
            ) -> None:
                half = open_trade.remaining_size * 0.5
                open_trade.partial_pnl += _research_leg_close_pnl(
                    direction=open_trade.direction,
                    size=half,
                    entry_price=open_trade.entry_price,
                    exit_px=exit_px,
                )
                open_trade.remaining_size = half
                open_trade.research_exit_progress = progress_value
                if activate_stage_c:
                    open_trade.tp_hit = True
                    open_trade.stage_c_floor_offset_r = 0.0
                if move_stop_to_entry:
                    if open_trade.direction == "LONG":
                        open_trade.current_stop = max(
                            open_trade.current_stop,
                            round(open_trade.entry_price, 2),
                        )
                    else:
                        open_trade.current_stop = min(
                            open_trade.current_stop,
                            round(open_trade.entry_price, 2),
                        )

            if stop_hit:
                _finalise(open_trade.current_stop, "STOP")
            elif custom_full_exit_price is not None and custom_full_exit_reason:
                _finalise(custom_full_exit_price, custom_full_exit_reason)
            elif custom_partial_exit_price is not None and custom_partial_exit_reason:
                if research_exit_code == "MR_EXIT_2":
                    _apply_partial_exit(
                        custom_partial_exit_price,
                        move_stop_to_entry=True,
                        activate_stage_c=True,
                        progress_value=1,
                    )
                elif research_exit_code == "MR_EXIT_4":
                    _apply_partial_exit(
                        custom_partial_exit_price,
                        move_stop_to_entry=False,
                        activate_stage_c=False,
                        progress_value=1,
                    )
                    if (
                        open_trade is not None
                        and custom_follow_on_exit_price is not None
                        and custom_follow_on_exit_reason
                    ):
                        _finalise(custom_follow_on_exit_price, custom_follow_on_exit_reason)
                elif research_exit_code in {"MR_EXIT_10", "MR_EXIT_11", "MR_EXIT_13"}:
                    _apply_partial_exit(
                        custom_partial_exit_price,
                        move_stop_to_entry=True,
                        activate_stage_c=True,
                        progress_value=1,
                    )
            elif stall_exit:
                _finalise(curr_close, "STALL")
            elif time_exit:
                _finalise(curr_close, "TIME")
            elif partial_tp:
                half = open_trade.remaining_size * 0.5
                open_trade.partial_pnl += _research_leg_close_pnl(
                    direction=open_trade.direction,
                    size=half,
                    entry_price=open_trade.entry_price,
                    exit_px=partial_tp_price,
                )
                open_trade.remaining_size = half
                open_trade.tp_hit = True
                if open_trade.direction == "LONG":
                    be = round(open_trade.entry_price + open_trade.stage_c_floor_offset_r * sd, 2)
                    open_trade.current_stop = max(open_trade.current_stop, be)
                else:
                    be = round(open_trade.entry_price - open_trade.stage_c_floor_offset_r * sd, 2)
                    open_trade.current_stop = min(open_trade.current_stop, be)
            elif trail_hit:
                _finalise(open_trade.current_stop, "TRAIL")

            if open_trade is not None:
                if open_trade.direction == "LONG":
                    open_trade.highest_close = max(open_trade.highest_close, curr_close)
                else:
                    open_trade.lowest_close = min(open_trade.lowest_close, curr_close)

                if i >= 2:
                    if open_trade.direction == "LONG":
                        trending = (
                            float(df.iloc[i]["high"]) >
                            float(df.iloc[i - 1]["high"]) >
                            float(df.iloc[i - 2]["high"])
                        )
                    else:
                        trending = (
                            float(df.iloc[i]["low"]) <
                            float(df.iloc[i - 1]["low"]) <
                            float(df.iloc[i - 2]["low"])
                        )
                    if trending and (hi - lo) > open_trade.atr_at_entry * 0.9:
                        open_trade.trail_delay = max(open_trade.trail_delay, 2)

                if open_trade.trail_delay > 0:
                    open_trade.trail_delay -= 1
                else:
                    _advance_stop(open_trade, profit_R)

            equity_curve.append(balance)
            continue

        contexts = candidate_map.get(i, [])
        matching_contexts = [
            ctx for ctx in contexts
            if ctx.candidate_family == setup_spec.candidate_family and (ctx.signal_pos + 1) < stop
        ]
        if not matching_contexts:
            equity_curve.append(balance)
            continue

        chosen_ctx: CandidateContext | None = None
        chosen_decision: SetupDecision | None = None
        for ctx in matching_contexts:
            decision = setup_spec.evaluator(ctx)
            if decision.allowed:
                chosen_ctx = ctx
                chosen_decision = decision
                break
            if skip_rows is not None:
                skip_rows.append(_skip_setup_row(ctx, decision, phase, window_label))

        if chosen_ctx is None or chosen_decision is None:
            equity_curve.append(balance)
            continue

        ctx = chosen_ctx
        decision = chosen_decision
        entry_decision = active_entry_profile.evaluator(ctx, setup_spec, decision)
        if not entry_decision.allowed:
            if skip_rows is not None:
                skip_rows.append(
                    _skip_entry_profile_row(
                        ctx,
                        decision,
                        entry_decision,
                        phase,
                        window_label,
                    )
                )
            equity_curve.append(balance)
            continue

        # ── Regime gate (additive, opt-in via regime_series) ──────────────────
        # Allowed regimes for range-MR:
        #   RANGING always allowed.
        #   HIGH_VOLATILITY allowed when REGIME_GATE_ALLOW_HIGH_VOL is True,
        #     but SHORT entries are blocked in HIGH_VOL (mean-reversion short
        #     into high-vol spike is high-risk).
        #   TRENDING_UP / TRENDING_DOWN: always blocked for range-MR.
        #   UNKNOWN: pass through (insufficient data — let setup filter decide).
        if regime_series is not None and setup_spec.candidate_family == "range_mean_reversion":
            try:
                bar_regime = str(regime_series.iloc[i])
            except (IndexError, KeyError):
                bar_regime = "UNKNOWN"
            _allowed_regimes = {"RANGING", "UNKNOWN"}
            if config.REGIME_GATE_ALLOW_HIGH_VOL:
                _allowed_regimes.add("HIGH_VOLATILITY")
            _regime_blocked = False
            _regime_skip_reason = ""
            if bar_regime not in _allowed_regimes:
                _regime_blocked = True
                _regime_skip_reason = f"regime_gate: {bar_regime}"
            elif bar_regime == "HIGH_VOLATILITY" and ctx.direction == "SHORT":
                _regime_blocked = True
                _regime_skip_reason = "regime_gate: SHORT blocked in HIGH_VOLATILITY"
            if _regime_blocked:
                if skip_rows is not None:
                    skip_row = _skip_setup_row(ctx, decision, phase, window_label)
                    skip_row["skip_reason"] = _regime_skip_reason
                    skip_row["bar_regime"] = bar_regime
                    skip_rows.append(skip_row)
                equity_curve.append(balance)
                continue
        else:
            bar_regime = ""

        # ── Anti-chase filter (additive, opt-in via ANTI_CHASE_ENABLED) ───────
        # Blocks entries where the pre-entry 3-candle AND 6-candle moves are
        # both in the high extension bucket — price has already run too far.
        # Uses same logic as _anti_chase_core_decision (3c+6c both "high").
        if config.ANTI_CHASE_ENABLED and setup_spec.candidate_family == "range_mean_reversion":
            _3c = ctx.pre_entry_move_3c_bucket
            _6c = ctx.pre_entry_move_6c_bucket
            if _3c == config.ANTI_CHASE_3C_LEVEL and _6c == config.ANTI_CHASE_6C_LEVEL:
                if skip_rows is not None:
                    skip_row = _skip_setup_row(ctx, decision, phase, window_label)
                    skip_row["skip_reason"] = (
                        f"anti_chase: 3c={_3c}({ctx.price_move_3r:+.3f}R) "
                        f"6c={_6c}({ctx.price_move_6r:+.3f}R)"
                    )
                    skip_rows.append(skip_row)
                equity_curve.append(balance)
                continue

        # ── Entry buffer check (additive, opt-in via ENTRY_BUFFER_RESEARCH) ───
        # Blocks entries where the N+1 open drifted > ENTRY_BUFFER_PCT from the
        # signal-bar close.  Uses EntryConfirmationBuffer (module-level import).
        if config.ENTRY_BUFFER_RESEARCH and setup_spec.candidate_family == "range_mean_reversion":
            _signal_close = ctx.close_price
            _entry_open = ctx.entry_price
            if _signal_close > 0:
                _drift = abs(_entry_open - _signal_close) / _signal_close
                if _drift > config.ENTRY_BUFFER_PCT:
                    if skip_rows is not None:
                        skip_row = _skip_setup_row(ctx, decision, phase, window_label)
                        skip_row["skip_reason"] = (
                            f"entry_buffer: drift={_drift * 100:.3f}% "
                            f"(max={config.ENTRY_BUFFER_PCT * 100:.3f}%)"
                        )
                        skip_rows.append(skip_row)
                    equity_curve.append(balance)
                    continue

        stop_distance = round(ctx.atr_value * config.ATR_STOP_MULTIPLIER, 2)
        position_size = _research_position_size(
            balance=balance,
            entry_price=ctx.entry_price,
            stop_distance=stop_distance,
            halve=ctx.halve_position,
        )
        if position_size <= 0:
            equity_curve.append(balance)
            continue

        if ctx.direction == "LONG":
            stop_price = ctx.entry_price - stop_distance
            tp_price = ctx.entry_price + (stop_distance * 2.0)
        else:
            stop_price = ctx.entry_price + stop_distance
            tp_price = ctx.entry_price - (stop_distance * 2.0)

        stage_b_r, stage_b_atr_mult, partial_tp_r = _stage_params_for_signals(ctx.signals_fired)

        open_trade = SimTrade(
            trade_num=len(trades) + 1,
            entry_time=ctx.entry_time,
            entry_price=ctx.entry_price,
            stop_price=round(stop_price, 2),
            tp_price=round(tp_price, 2),
            size=position_size,
            direction=ctx.direction,
            signals_fired=list(ctx.signals_fired),
            atr_at_entry=ctx.atr_value,
            stop_distance=stop_distance,
            remaining_size=position_size,
            current_stop=round(stop_price, 2),
            stage_b_r=stage_b_r,
            stage_b_atr_mult=stage_b_atr_mult,
            partial_tp_r=partial_tp_r,
            stage_c_floor_offset_r=config.BE_OFFSET_R,
            highest_close=ctx.entry_price,
            lowest_close=ctx.entry_price,
            mfe_price=ctx.entry_price,
            mae_price=ctx.entry_price,
            setup_name=decision.setup_name,
            entry_profile_name=entry_decision.entry_profile_name,
            signal_time=ctx.signal_time,
            signal_tags=list(decision.signal_tags),
            entry_profile_tags=list(entry_decision.entry_tags),
            atr_bucket=ctx.atr_bucket,
            volume_bucket=ctx.volume_bucket,
            vwap_distance_bucket=ctx.vwap_distance_bucket,
            pre_entry_move_3c_bucket=ctx.pre_entry_move_3c_bucket,
            pre_entry_move_6c_bucket=ctx.pre_entry_move_6c_bucket,
            entry_close_price=ctx.close_price,
            entry_vwap=ctx.vwap,
            entry_atr_pct=ctx.atr_pct,
            entry_volume_ratio=ctx.volume_ratio,
            entry_macd_histogram=ctx.macd_histogram,
            entry_macd_slope=ctx.macd_slope,
            entry_trend=ctx.trend,
            entry_vol_regime=ctx.vol_regime,
            breakout_level=ctx.breakout_level,
            macd_agrees=ctx.macd_agrees,
            price_vs_vwap=ctx.price_vs_vwap,
            trend_context_reason=ctx.trend_context_reason,
            pullback_detected=ctx.pullback_detected,
            reclaim_detected=ctx.reclaim_detected,
            pullback_depth_r=ctx.pullback_depth_r,
            candles_since_impulse=ctx.candles_since_impulse,
            range_high=ctx.range_high,
            range_low=ctx.range_low,
            range_mid=ctx.range_mid,
            distance_from_range_boundary_r=ctx.distance_from_range_boundary_r,
            rejection_detected=ctx.rejection_detected,
            candles_outside_range=ctx.candles_outside_range,
            research_exit_code=research_exit_code or diagnostic_target_code or "",
        )

        equity_curve.append(balance)

    if open_trade is not None:
        last_close = float(df["close"].iloc[stop - 1])
        pnl = open_trade.partial_pnl + _research_close_pnl(open_trade, last_close)
        balance += pnl
        open_trade.exit_time = df.index[stop - 1]
        open_trade.exit_price = round(last_close, 2)
        open_trade.pnl_net = pnl
        open_trade.result = "WIN" if pnl > 0 else "LOSS"
        open_trade.exit_reason = "FORCE_CLOSE"
        denom = open_trade.size * open_trade.stop_distance
        open_trade.exit_r = round(pnl / denom, 3) if denom > 0 else 0.0
        trades.append(open_trade)
        equity_curve.append(balance)

    return trades, equity_curve


def _direction_breakdown(trades: list[SimTrade], direction: str) -> dict[str, float]:
    subset = [t for t in trades if t.direction == direction]
    wins = [t for t in subset if t.result == "WIN"]
    rs = [t.exit_r for t in subset]
    pf = _score_profit_factor(rs)
    _, avg_win, avg_loss = _trade_win_loss_stats(subset)
    return {
        "trades": len(subset),
        "win_rate_pct": (len(wins) / len(subset) * 100.0) if subset else 0.0,
        "avg_r": (sum(rs) / len(rs)) if rs else 0.0,
        "median_r": statistics.median(rs) if rs else 0.0,
        "profit_factor": pf,
        "avg_win_r": avg_win,
        "avg_loss_r": avg_loss,
    }


def _tp_hit_rate(trades: list[SimTrade]) -> float:
    return (sum(1 for t in trades if t.tp_hit) / len(trades) * 100.0) if trades else 0.0


def _vwap_touch_rate(trades: list[SimTrade]) -> float:
    return (sum(1 for t in trades if t.touched_vwap_after_entry) / len(trades) * 100.0) if trades else 0.0


def _range_mid_touch_rate(trades: list[SimTrade]) -> float:
    return (sum(1 for t in trades if t.touched_range_mid_after_entry) / len(trades) * 100.0) if trades else 0.0


def _price_touches_level(direction: str, lo: float, hi: float, level: float) -> bool:
    if not math.isfinite(level):
        return False
    return hi >= level if direction == "LONG" else lo <= level


def _earliest_profit_target_price(
    *,
    direction: str,
    entry_price: float,
    lo: float,
    hi: float,
    dynamic_target_price: float | None = None,
    fixed_r_target_price: float | None = None,
) -> float | None:
    hit_prices: list[float] = []
    for price in (dynamic_target_price, fixed_r_target_price):
        if price is None or not math.isfinite(price):
            continue
        if _price_touches_level(direction, lo, hi, price):
            hit_prices.append(price)
    if not hit_prices:
        return None
    if direction == "LONG":
        return min(hit_prices)
    return max(hit_prices)


def _atr_trailing_exit_price(
    *,
    trade: SimTrade,
    profit_r: float,
    activation_r: float,
    atr_mult: float,
    lo: float,
    hi: float,
) -> float | None:
    if profit_r < activation_r:
        return None
    if trade.direction == "LONG":
        trail = max(trade.entry_price, trade.highest_close - (trade.atr_at_entry * atr_mult))
    else:
        trail = min(trade.entry_price, trade.lowest_close + (trade.atr_at_entry * atr_mult))
    trail = round(trail, 2)
    if _price_touches_level(trade.direction, lo, hi, trail):
        return trail
    return None


def _write_dict_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_dict_csv(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _range_regime_summary_rows(trade_rows: list[dict]) -> list[dict]:
    rows: list[dict] = []
    grouped: dict[tuple[str, str, str, str], list[dict]] = {}

    def _truthy(value: object) -> bool:
        return str(value).strip().lower() in {"1", "true", "yes", "y"}

    for row in trade_rows:
        if row.get("research_horizon") != "2h":
            continue
        setup_name = row.get("setup_code") or row.get("setup_name", "")
        exit_code = row.get("research_exit_code", "")
        if setup_name not in _RANGE_MR_DYNAMIC_EXIT_SETUPS or exit_code not in {"MR_EXIT_0", "MR_EXIT_9", "MR_EXIT_10", "MR_EXIT_11", "MR_EXIT_12", "MR_EXIT_13", "MR_EXIT_14"}:
            continue
        trend = row.get("entry_trend", "") or "UNKNOWN"
        vol_regime = row.get("entry_vol_regime", "") or "UNKNOWN"
        trend_hint = "trend_hint" if str(row.get("trend_context_reason", "")).strip() else "no_trend_hint"
        key = (setup_name, exit_code, f"{trend}|{vol_regime}|{trend_hint}", row.get("direction", ""))
        grouped.setdefault(key, []).append(row)

    for (setup_name, exit_code, regime_label, direction), subset in grouped.items():
        rs = [_parse_float(r.get("exit_r")) for r in subset]
        rows.append({
            "setup_name": setup_name,
            "research_exit_code": exit_code,
            "regime_label": regime_label,
            "direction": direction,
            "trades": len(subset),
            "win_rate_pct": round(sum(1 for r in rs if r > 0) / len(rs) * 100.0, 3) if rs else 0.0,
            "avg_r": round(float(statistics.mean(rs)) if rs else 0.0, 4),
            "median_r": round(float(statistics.median(rs)) if rs else 0.0, 4),
            "profit_factor": round(_score_profit_factor(rs), 4) if rs else 0.0,
            "tp_hit_rate_pct": round(sum(_truthy(r.get("partial_tp_hit")) for r in subset) / len(subset) * 100.0, 3) if subset else 0.0,
        })
    rows.sort(key=lambda row: (row["setup_name"], row["research_exit_code"], row["regime_label"], row["direction"]))
    return rows


def _parse_float(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    lower = text.lower()
    if lower == "inf":
        return float("inf")
    if lower == "-inf":
        return float("-inf")
    try:
        return float(text)
    except ValueError:
        return default


def _parse_int(value: object, default: int = 0) -> int:
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    try:
        return int(float(text))
    except ValueError:
        return default


def _walk_forward_windows(df: pd.DataFrame) -> list[tuple[int, int, int, int, int]]:
    windows: list[tuple[int, int, int, int, int]] = []
    start_time = df.index[0]
    final_time = df.index[-1]
    window_id = 1
    offset_days = 0

    while True:
        train_start = start_time + timedelta(days=offset_days)
        train_end = train_start + timedelta(days=RESEARCH_TRAIN_DAYS)
        test_end = train_end + timedelta(days=RESEARCH_TEST_DAYS)
        if test_end > final_time:
            break

        train_start_pos = int(df.index.searchsorted(train_start, side="left"))
        train_end_pos = int(df.index.searchsorted(train_end, side="left"))
        test_end_pos = int(df.index.searchsorted(test_end, side="left"))
        windows.append((window_id, train_start_pos, train_end_pos, train_end_pos, test_end_pos))
        offset_days += RESEARCH_STEP_DAYS
        window_id += 1

    return windows


def _walk_forward_windows_with_offset(
    df: pd.DataFrame,
    *,
    start_offset_days: int,
    window_id_start: int,
) -> list[tuple[int, int, int, int, int]]:
    windows: list[tuple[int, int, int, int, int]] = []
    start_time = df.index[0]
    final_time = df.index[-1]
    window_id = window_id_start
    offset_days = start_offset_days

    while True:
        train_start = start_time + timedelta(days=offset_days)
        train_end = train_start + timedelta(days=RESEARCH_TRAIN_DAYS)
        test_end = train_end + timedelta(days=RESEARCH_TEST_DAYS)
        if test_end > final_time:
            break

        train_start_pos = int(df.index.searchsorted(train_start, side="left"))
        train_end_pos = int(df.index.searchsorted(train_end, side="left"))
        test_end_pos = int(df.index.searchsorted(test_end, side="left"))
        windows.append((window_id, train_start_pos, train_end_pos, train_end_pos, test_end_pos))
        offset_days += RESEARCH_STEP_DAYS
        window_id += 1

    return windows


def _range_research_windows(df: pd.DataFrame) -> list[tuple[int, int, int, int, int]]:
    """
    Research-only dual-offset walk-forward windows for range MR.

    This keeps the existing rolling walk-forward view and adds a half-step
    offset pass so we validate the same idea across more unseen regime
    boundaries without changing live behavior.
    """
    windows = _walk_forward_windows(df)
    half_step = max(1, RESEARCH_STEP_DAYS // 2)
    windows.extend(
        _walk_forward_windows_with_offset(
            df,
            start_offset_days=half_step,
            window_id_start=len(windows) + 1,
        )
    )
    return windows


def _window_result_row(result: WalkForwardWindowResult) -> dict:
    long_break = _direction_breakdown(result.out_trades, "LONG")
    short_break = _direction_breakdown(result.out_trades, "SHORT")
    combo_name = _entry_combo_name(result.setup_name, result.entry_profile_name)
    return {
        "setup_name": result.setup_name,
        "entry_profile_name": result.entry_profile_name,
        "combo_name": combo_name,
        "window_id": result.window_id,
        "train_start": result.train_start.isoformat(),
        "train_end": result.train_end.isoformat(),
        "test_start": result.test_start.isoformat(),
        "test_end": result.test_end.isoformat(),
        "is_trades": result.in_metrics.total_trades,
        "is_win_rate_pct": result.in_metrics.win_rate_pct,
        "is_profit_factor": result.in_metrics.profit_factor,
        "is_avg_r": (sum(t.exit_r for t in result.in_trades) / len(result.in_trades)) if result.in_trades else 0.0,
        "is_median_r": statistics.median([t.exit_r for t in result.in_trades]) if result.in_trades else 0.0,
        "is_tp_hit_rate_pct": _tp_hit_rate(result.in_trades),
        "is_sharpe_ratio": result.in_metrics.sharpe_ratio,
        "is_sortino_ratio": result.in_metrics.sortino_ratio,
        "is_max_drawdown_pct": result.in_metrics.max_drawdown_pct,
        "is_return_pct": result.in_metrics.total_return_pct,
        "oos_trades": result.out_metrics.total_trades,
        "oos_win_rate_pct": result.out_metrics.win_rate_pct,
        "oos_profit_factor": result.out_metrics.profit_factor,
        "oos_avg_r": (sum(t.exit_r for t in result.out_trades) / len(result.out_trades)) if result.out_trades else 0.0,
        "oos_median_r": statistics.median([t.exit_r for t in result.out_trades]) if result.out_trades else 0.0,
        "oos_tp_hit_rate_pct": _tp_hit_rate(result.out_trades),
        "oos_sharpe_ratio": result.out_metrics.sharpe_ratio,
        "oos_sortino_ratio": result.out_metrics.sortino_ratio,
        "oos_max_drawdown_pct": result.out_metrics.max_drawdown_pct,
        "oos_return_pct": result.out_metrics.total_return_pct,
        "oos_long_trades": long_break["trades"],
        "oos_long_win_rate_pct": long_break["win_rate_pct"],
        "oos_long_avg_r": long_break["avg_r"],
        "oos_long_profit_factor": long_break["profit_factor"],
        "oos_short_trades": short_break["trades"],
        "oos_short_win_rate_pct": short_break["win_rate_pct"],
        "oos_short_avg_r": short_break["avg_r"],
        "oos_short_profit_factor": short_break["profit_factor"],
    }


def _promotion_summary(
    setup_name: str,
    window_results: list[WalkForwardWindowResult],
    *,
    entry_profile_name: str = "",
) -> dict:
    _PROMOTION_PF_CAP = 10.0
    combo_name = _entry_combo_name(setup_name, entry_profile_name)
    oos_windows = [w for w in window_results]
    all_oos_trades = [trade for result in oos_windows for trade in result.out_trades]
    # Cap infinite PF values to avoid spurious IS/OOS agreement failures
    pfs = [
        min(w.out_metrics.profit_factor, _PROMOTION_PF_CAP)
        if math.isfinite(w.out_metrics.profit_factor)
        else _PROMOTION_PF_CAP
        for w in oos_windows
    ]
    avg_pfs = float(statistics.mean(pfs)) if pfs else 0.0
    med_pfs = float(statistics.median(pfs)) if pfs else 0.0
    # Worst-quarter estimate: trade-count-weighted bottom-25% mean PF.
    # Weighting by trade count gives more influence to windows with real sample
    # mass, reducing noise from 1-trade windows where PF = 0 or PF = inf.
    # Minimum weight per window = 1 (so 0-trade windows don't vanish entirely).
    _MIN_TRADES_FOR_WQ = 5
    pf_with_weight = [
        (pf, max(1, w.out_metrics.total_trades))
        for pf, w in zip(pfs, oos_windows)
    ]
    # Sort by PF ascending; take the bottom 25%
    pf_with_weight_sorted = sorted(pf_with_weight, key=lambda x: x[0])
    bottom_n = max(1, len(pf_with_weight_sorted) // 4)
    bottom_quarter_items = pf_with_weight_sorted[:bottom_n]
    total_wq_weight = sum(w for _, w in bottom_quarter_items)
    bottom_quarter_mean = (
        sum(pf * w for pf, w in bottom_quarter_items) / total_wq_weight
        if total_wq_weight > 0 else 0.0
    )
    total_oos_trades = sum(w.out_metrics.total_trades for w in oos_windows)
    avg_trades_per_window = total_oos_trades / len(oos_windows) if oos_windows else 0.0
    oos_avg_rs = [
        (sum(t.exit_r for t in w.out_trades) / len(w.out_trades)) if w.out_trades else 0.0
        for w in oos_windows
    ]
    oos_med_rs = [
        statistics.median([t.exit_r for t in w.out_trades]) if w.out_trades else 0.0
        for w in oos_windows
    ]
    tp_rates = [_tp_hit_rate(w.out_trades) for w in oos_windows if w.out_trades]
    avg_tp = float(statistics.mean(tp_rates)) if tp_rates else 0.0
    tp_std = float(statistics.pstdev(tp_rates)) if len(tp_rates) > 1 else 0.0
    profitable_windows = sum(1 for w in oos_windows if w.out_metrics.total_return_pct > 0)
    losing_windows = sum(1 for w in oos_windows if w.out_metrics.total_return_pct <= 0)
    avg_is_pf = float(statistics.mean([
        min(w.in_metrics.profit_factor, _PROMOTION_PF_CAP)
        if math.isfinite(w.in_metrics.profit_factor)
        else _PROMOTION_PF_CAP
        for w in oos_windows
    ])) if oos_windows else 0.0
    avg_is_sharpe = float(statistics.mean([w.in_metrics.sharpe_ratio for w in oos_windows])) if oos_windows else 0.0
    avg_is_sortino = float(statistics.mean([w.in_metrics.sortino_ratio for w in oos_windows])) if oos_windows else 0.0
    avg_is_r = float(statistics.mean([
        (sum(t.exit_r for t in w.in_trades) / len(w.in_trades)) if w.in_trades else 0.0
        for w in oos_windows
    ])) if oos_windows else 0.0
    avg_oos_sharpe = float(statistics.mean([w.out_metrics.sharpe_ratio for w in oos_windows])) if oos_windows else 0.0
    avg_oos_sortino = float(statistics.mean([w.out_metrics.sortino_ratio for w in oos_windows])) if oos_windows else 0.0
    oos_win_rate_pct, oos_avg_win_r, oos_avg_loss_r = _trade_win_loss_stats(all_oos_trades)
    positive_returns = [max(w.out_metrics.total_return_pct, 0.0) for w in oos_windows]
    total_positive_return = sum(positive_returns)
    small_cluster_positive = sum(
        max(w.out_metrics.total_return_pct, 0.0)
        for w in oos_windows
        if w.out_metrics.total_trades <= 3
    )
    low_trade_profit_share = (
        small_cluster_positive / total_positive_return
        if total_positive_return > 0 else 0.0
    )
    # IS/OOS agreement ratio: OOS PF / IS PF (< 1 = degradation, 1 = stable)
    is_oos_pf_ratio = round(avg_pfs / avg_is_pf, 4) if avg_is_pf > 0 else 0.0
    # Bootstrap Monte Carlo confidence intervals on all OOS R-multiples
    _mc = mc.bootstrap([t.exit_r for t in all_oos_trades])
    # Cluster share: % of OOS trades that fell inside profitable windows
    trades_in_winning_windows = sum(
        len(w.out_trades) for w in oos_windows if w.out_metrics.total_return_pct > 0
    )
    cluster_share_winning_pct = round(
        trades_in_winning_windows / total_oos_trades * 100.0 if total_oos_trades > 0 else 0.0,
        3,
    )

    # Three-tier adaptive thresholds based on avg trades per OOS window.
    #
    # Rationale for tiers:
    #   normal    (≥10 trades/window): statistical power is sufficient — tight gate.
    #   low       (5–9 trades/window): moderate relaxation for sparser signals.
    #   very_low  (<5 trades/window):  further relaxation; with 3–4 trades/window:
    #     • worst_quarter_pf: a single-loss window in the bottom quarter forces
    #       bottom_quarter_mean ≈ 0 regardless of overall strategy quality.
    #     • tp_hit_stable: binary TP outcomes (0 % or 100 % per window) produce
    #       ~33 % std by coin-flip math, exceeding the low-sample 30 % gate.
    #     • no_small_cluster_dependency: when nearly all windows have ≤3 trades,
    #       "small cluster" windows are the entire dataset — the check is vacuous.
    #     • is_oos_agree: avg_is_r is diluted by many 0-trade windows, making the
    #       sign unreliable; fall back to PF-direction agreement instead.
    _LOW_SAMPLE_THRESHOLD      = 10   # < 10  → low_sample
    _VERY_LOW_SAMPLE_THRESHOLD = 5    # < 5   → very_low_sample
    _low_sample      = avg_trades_per_window < _LOW_SAMPLE_THRESHOLD
    _very_low_sample = avg_trades_per_window < _VERY_LOW_SAMPLE_THRESHOLD

    # worst_quarter_pf — progressive floor
    _worst_quarter_threshold = (
        0.30 if _very_low_sample   # single-loss windows are inevitable at this density
        else 0.60 if _low_sample
        else 0.80
    )

    # tp_hit_stable — binary TP outcomes at <5 trades/window are pure noise;
    # relax the cap slightly for low-sample too (30 % → 35 %).
    _tp_std_threshold = (
        float("inf") if _very_low_sample  # skip: theoretical floor ~33 % for binary TP
        else 35.0 if _low_sample
        else 25.0
    )

    # no_small_cluster_dependency — when avg trades/window < 5, all windows are
    # "small clusters" (≤3 trades) by definition; relax cap to 0.75.
    _small_cluster_cap = (
        0.75 if _very_low_sample
        else 0.50
    )

    # is_oos_agree — for very sparse setups avg_is_r is diluted by many 0-trade
    # windows and the sign is unreliable; use PF-direction agreement instead.
    _oos_avg_r = statistics.mean(oos_avg_rs) if oos_avg_rs else 0.0
    if _very_low_sample:
        _is_oos_agree = (avg_is_pf > 1.0) == (avg_pfs > 1.0)
    else:
        _is_oos_agree = (
            avg_is_r == 0.0
            or math.copysign(1, avg_is_r) == math.copysign(1, _oos_avg_r)
        ) and (
            _low_sample  # direction check alone is sufficient when n < _LOW_SAMPLE_THRESHOLD
            or abs(avg_is_pf - avg_pfs) <= 0.75
        )

    criteria = {
        "min_total_oos_trades": total_oos_trades >= 50,
        "median_window_pf": med_pfs > 1.05,
        "average_window_pf": avg_pfs > 1.10,
        # Conservative heuristic for "worst 25% not catastrophic" (three-tier adaptive).
        "worst_quarter_pf": bottom_quarter_mean >= _worst_quarter_threshold,
        "avg_r_positive": _oos_avg_r > 0,
        # "Not deeply negative" — fail if the typical window median R is materially below zero.
        "median_r_not_deeply_negative": (statistics.median(oos_med_rs) if oos_med_rs else 0.0) > -0.25,
        # Stable enough if TP-hit volatility across active windows is not extreme (three-tier adaptive).
        "tp_hit_stable": tp_std <= _tp_std_threshold,
        # IS/OOS should agree in direction (three-tier adaptive — see comment above).
        "is_oos_agree": _is_oos_agree,
        # Guard against tiny windows driving most of the upside (relaxed for very sparse).
        "no_small_cluster_dependency": low_trade_profit_share <= _small_cluster_cap,
    }
    passed = all(criteria.values())
    failed_checks = [name for name, ok in criteria.items() if not ok]
    return {
        "setup_name": setup_name,
        "entry_profile_name": entry_profile_name,
        "combo_name": combo_name,
        "passed_promotion": passed,
        "failed_checks": "|".join(failed_checks),
        "average_oos_pf": round(avg_pfs, 4),
        "median_oos_pf": round(med_pfs, 4),
        "profitable_windows": profitable_windows,
        "losing_windows": losing_windows,
        "worst_window_pf": round(min(pfs), 4) if pfs else 0.0,
        "best_window_pf": round(max(pfs), 4) if pfs else 0.0,
        "bottom_quarter_mean_pf": round(bottom_quarter_mean, 4),
        "total_oos_trades": total_oos_trades,
        "avg_trades_per_window": round(avg_trades_per_window, 3),
        "average_oos_avg_r": round(float(statistics.mean(oos_avg_rs)) if oos_avg_rs else 0.0, 4),
        "median_oos_median_r": round(float(statistics.median(oos_med_rs)) if oos_med_rs else 0.0, 4),
        "oos_win_rate_pct": round(oos_win_rate_pct, 3),
        "oos_avg_win_r": round(oos_avg_win_r, 4),
        "oos_avg_loss_r": round(oos_avg_loss_r, 4),
        "average_tp_hit_rate_pct": round(avg_tp, 3),
        "tp_hit_rate_std_pct": round(tp_std, 3),
        "average_is_pf": round(avg_is_pf, 4),
        "average_is_sharpe": round(avg_is_sharpe, 4),
        "average_is_sortino": round(avg_is_sortino, 4),
        "average_is_avg_r": round(avg_is_r, 4),
        "average_oos_sharpe": round(avg_oos_sharpe, 4),
        "average_oos_sortino": round(avg_oos_sortino, 4),
        "small_cluster_profit_share": round(low_trade_profit_share, 4),
        "is_oos_pf_ratio": is_oos_pf_ratio,
        "cluster_share_winning_pct": cluster_share_winning_pct,
        # Bootstrap MC confidence intervals (all OOS trades pooled)
        "mc_pf_p05": _mc.pf_p05,
        "mc_pf_p50": _mc.pf_p50,
        "mc_pf_p95": _mc.pf_p95,
        "mc_avg_r_p05": _mc.avg_r_p05,
        "mc_avg_r_p50": _mc.avg_r_p50,
        "mc_avg_r_p95": _mc.avg_r_p95,
        "mc_win_rate_p05": round(_mc.win_rate_p05 * 100, 3),
        "mc_win_rate_p50": round(_mc.win_rate_p50 * 100, 3),
        "mc_win_rate_p95": round(_mc.win_rate_p95 * 100, 3),
        "mc_prob_pf_above_1": _mc.prob_pf_above_1,
        "mc_prob_avg_r_positive": _mc.prob_avg_r_positive,
        "mc_low_sample_adaptive": "very_low" if _very_low_sample else ("low" if _low_sample else False),
    }


def _trade_r_stats(trades: list[SimTrade]) -> tuple[float, float]:
    if not trades:
        return 0.0, 0.0
    rs = [t.exit_r for t in trades]
    return float(statistics.mean(rs)), float(statistics.median(rs))


def _trade_win_loss_stats(trades: list[SimTrade]) -> tuple[float, float, float]:
    if not trades:
        return 0.0, 0.0, 0.0
    wins = [t.exit_r for t in trades if t.exit_r > 0]
    losses = [t.exit_r for t in trades if t.exit_r < 0]
    win_rate = sum(1 for t in trades if t.exit_r > 0) / len(trades) * 100.0
    avg_win = float(statistics.mean(wins)) if wins else 0.0
    avg_loss = float(statistics.mean(losses)) if losses else 0.0
    return win_rate, avg_win, avg_loss


def _bucket_trade_stats(trades: list[SimTrade], attr_name: str, bucket_value: str) -> dict[str, float]:
    subset = [trade for trade in trades if getattr(trade, attr_name) == bucket_value]
    avg_r, med_r = _trade_r_stats(subset)
    win_rate, avg_win, avg_loss = _trade_win_loss_stats(subset)
    return {
        "trades": len(subset),
        "profit_factor": _score_profit_factor([t.exit_r for t in subset]),
        "avg_r": avg_r,
        "median_r": med_r,
        "win_rate_pct": win_rate,
        "avg_win_r": avg_win,
        "avg_loss_r": avg_loss,
        "tp_hit_rate_pct": _tp_hit_rate(subset),
    }


def _setup_outcome_label(summary_row: dict) -> str:
    if summary_row["passed_promotion"]:
        return "pass"
    if summary_row["total_oos_trades"] == 0:
        return "retire immediately"
    if summary_row["total_oos_trades"] < 50 and summary_row["average_oos_pf"] >= 1.0:
        return "promising but sample-limited"
    if summary_row["average_oos_pf"] < 0.75 and summary_row["median_oos_pf"] < 0.75:
        return "retire immediately"
    return "fail"


def _report_outcome_label(summary_row: dict, *, retire_label: str = "retire immediately") -> str:
    outcome = _setup_outcome_label(summary_row)
    if outcome == "retire immediately":
        return retire_label
    return outcome


def _range_exit_window_row(exit_spec: MeanReversionExitSpec, result: WalkForwardWindowResult) -> dict:
    row = _window_result_row(result)
    row["exit_code"] = exit_spec.code
    row["exit_description"] = exit_spec.description
    row["oos_vwap_touch_rate_pct"] = round(_vwap_touch_rate(result.out_trades), 3)
    row["oos_range_mid_touch_rate_pct"] = round(_range_mid_touch_rate(result.out_trades), 3)
    return row


def _range_exit_summary_row(
    exit_spec: MeanReversionExitSpec,
    window_results: list[WalkForwardWindowResult],
) -> dict:
    row = _promotion_summary(exit_spec.code, window_results)
    all_oos_trades = [trade for result in window_results for trade in result.out_trades]
    avg_r, med_r = _trade_r_stats(all_oos_trades)
    win_rate, avg_win, avg_loss = _trade_win_loss_stats(all_oos_trades)
    row.update({
        "exit_code": exit_spec.code,
        "exit_description": exit_spec.description,
        "win_rate_pct": round(win_rate, 3),
        "avg_r_all_oos": round(avg_r, 4),
        "median_r_all_oos": round(med_r, 4),
        "avg_win_r": round(avg_win, 4),
        "avg_loss_r": round(avg_loss, 4),
        "vwap_touch_capture_rate_pct": round(_vwap_touch_rate(all_oos_trades), 3),
        "range_mid_capture_rate_pct": round(_range_mid_touch_rate(all_oos_trades), 3),
    })
    return row


def _entry_research_summary_row(
    setup_name: str,
    entry_profile_name: str,
    window_results: list[WalkForwardWindowResult],
    **extra: object,
) -> dict:
    row = _promotion_summary(
        setup_name,
        window_results,
        entry_profile_name=entry_profile_name,
    )
    all_oos_trades = [trade for result in window_results for trade in result.out_trades]
    avg_r, med_r = _trade_r_stats(all_oos_trades)
    win_rate, avg_win, avg_loss = _trade_win_loss_stats(all_oos_trades)
    row.update({
        "win_rate_pct": round(win_rate, 3),
        "avg_r_all_oos": round(avg_r, 4),
        "median_r_all_oos": round(med_r, 4),
        "avg_win_r": round(avg_win, 4),
        "avg_loss_r": round(avg_loss, 4),
        "tp_hit_rate_pct": round(_tp_hit_rate(all_oos_trades), 3),
        "vwap_touch_rate_pct": round(_vwap_touch_rate(all_oos_trades), 3),
        "range_mid_touch_rate_pct": round(_range_mid_touch_rate(all_oos_trades), 3),
    })
    row.update(extra)
    return row


def _entry_window_row(
    result: WalkForwardWindowResult,
    **extra: object,
) -> dict:
    row = _window_result_row(result)
    row.update(extra)
    return row


def _print_range_exit_experiment_report(
    summary_rows: list[dict],
    oos_trade_map: dict[str, list[SimTrade]],
) -> None:
    print(f"\n{'━' * 112}")
    print("  RANGE_MEAN_REVERSION EXIT RESEARCH  (entries fixed, backtest-only)")
    print(f"{'━' * 112}")
    print("  Setup-specific exit comparisons for RANGE_MEAN_REVERSION only.")

    for row in summary_rows:
        outcome = _setup_outcome_label(row)
        print(
            f"    {row['exit_code']:<9} {outcome:<23} "
            f"PF avg/med={row['average_oos_pf'] if math.isfinite(row['average_oos_pf']) else float('inf'):.3f}/"
            f"{row['median_oos_pf']:.3f}  Trades={row['total_oos_trades']:>4}  "
            f"AvgR={row['avg_r_all_oos']:+.3f}  MedR={row['median_r_all_oos']:+.3f}  "
            f"VWAP-touch={row['vwap_touch_capture_rate_pct']:.1f}%  "
            f"Range-mid={row['range_mid_capture_rate_pct']:.1f}%"
        )

    best_row = max(
        summary_rows,
        key=lambda row: (
            row["median_oos_pf"],
            row["average_oos_pf"] if math.isfinite(row["average_oos_pf"]) else 9999.0,
            row["avg_r_all_oos"],
        ),
    ) if summary_rows else None

    for row in summary_rows:
        trades = oos_trade_map.get(row["exit_code"], [])
        avg_r, med_r = _trade_r_stats(trades)
        win_rate, avg_win, avg_loss = _trade_win_loss_stats(trades)
        long_stats = _direction_breakdown(trades, "LONG")
        short_stats = _direction_breakdown(trades, "SHORT")
        atr_breakdown = {
            bucket: _bucket_trade_stats(trades, "atr_bucket", bucket)
            for bucket in ("low", "medium", "high")
        }
        vwap_breakdown = {
            bucket: _bucket_trade_stats(trades, "vwap_distance_bucket", bucket)
            for bucket in ("close", "medium", "far")
        }
        vol_breakdown = {
            bucket: _bucket_trade_stats(trades, "volume_bucket", bucket)
            for bucket in ("low", "medium", "high")
        }

        print(f"\n  {row['exit_code']}  {row['exit_description']}")
        print(f"    Outcome            : {_setup_outcome_label(row)}")
        print(f"    Total OOS trades   : {row['total_oos_trades']}")
        print(
            f"    Avg / med OOS PF   : "
            f"{row['average_oos_pf'] if math.isfinite(row['average_oos_pf']) else float('inf'):.3f} / "
            f"{row['median_oos_pf']:.3f}"
        )
        print(
            f"    Profitable windows : {row['profitable_windows']}   "
            f"Losing windows: {row['losing_windows']}"
        )
        print(f"    Worst window PF    : {row['worst_window_pf']:.3f}")
        print(f"    Avg / median R     : {avg_r:+.3f} / {med_r:+.3f}")
        print(
            f"    Win rate           : {win_rate:.1f}%   "
            f"Avg win / loss: {avg_win:+.3f} / {avg_loss:+.3f}"
        )
        print(
            f"    VWAP-touch capture : {_vwap_touch_rate(trades):.1f}%   "
            f"Range-mid capture: {_range_mid_touch_rate(trades):.1f}%"
        )

        print("    Long / short breakdown:")
        print(
            f"      LONG  trades={int(long_stats['trades'])}  WR={long_stats['win_rate_pct']:.1f}%  "
            f"PF={long_stats['profit_factor']:.3f}  AvgR={long_stats['avg_r']:+.3f}  "
            f"AvgWin={long_stats['avg_win_r']:+.3f}  AvgLoss={long_stats['avg_loss_r']:+.3f}"
        )
        print(
            f"      SHORT trades={int(short_stats['trades'])}  WR={short_stats['win_rate_pct']:.1f}%  "
            f"PF={short_stats['profit_factor']:.3f}  AvgR={short_stats['avg_r']:+.3f}  "
            f"AvgWin={short_stats['avg_win_r']:+.3f}  AvgLoss={short_stats['avg_loss_r']:+.3f}"
        )

        print("    ATR bucket breakdown:")
        for bucket in ("low", "medium", "high"):
            stats = atr_breakdown[bucket]
            print(
                f"      {bucket:<6} trades={int(stats['trades'])}  WR={stats['win_rate_pct']:.1f}%  "
                f"PF={stats['profit_factor']:.3f}  AvgR={stats['avg_r']:+.3f}"
            )

        print("    VWAP distance bucket breakdown:")
        for bucket in ("close", "medium", "far"):
            stats = vwap_breakdown[bucket]
            print(
                f"      {bucket:<6} trades={int(stats['trades'])}  WR={stats['win_rate_pct']:.1f}%  "
                f"PF={stats['profit_factor']:.3f}  AvgR={stats['avg_r']:+.3f}"
            )

        print("    Volume bucket breakdown:")
        for bucket in ("low", "medium", "high"):
            stats = vol_breakdown[bucket]
            print(
                f"      {bucket:<6} trades={int(stats['trades'])}  WR={stats['win_rate_pct']:.1f}%  "
                f"PF={stats['profit_factor']:.3f}  AvgR={stats['avg_r']:+.3f}"
            )

    if best_row:
        best_trades = oos_trade_map.get(best_row["exit_code"], [])
        best_long = _direction_breakdown(best_trades, "LONG")
        best_short = _direction_breakdown(best_trades, "SHORT")
        best_far = _bucket_trade_stats(best_trades, "vwap_distance_bucket", "far")
        best_low_vol = _bucket_trade_stats(best_trades, "volume_bucket", "low")
        baseline = next((row for row in summary_rows if row["exit_code"] == "MR_EXIT_0"), None)

        print(f"\n{'━' * 112}")
        print("  RANGE EXIT DIAGNOSTIC ANSWERS")
        print(f"{'━' * 112}")
        if baseline:
            entry_validity = (
                best_row["median_oos_pf"] > baseline["median_oos_pf"]
                and best_row["avg_r_all_oos"] > baseline["avg_r_all_oos"]
            )
            print(
                "  Q: Was the entry idea valid but exit wrong?\n"
                f"     {'Partially yes' if entry_validity else 'Not convincingly'}: "
                f"best exit {best_row['exit_code']} changed median PF from "
                f"{baseline['median_oos_pf']:.3f} to {best_row['median_oos_pf']:.3f} "
                f"and AvgR from {baseline['avg_r_all_oos']:+.3f} to {best_row['avg_r_all_oos']:+.3f}."
            )
            print(
                "  Q: Is VWAP touch a better profit target than 1.5R?\n"
                f"     {'Yes, directionally' if best_row['median_oos_pf'] > baseline['median_oos_pf'] else 'No clear edge'}: "
                f"MR_EXIT_0 median PF={baseline['median_oos_pf']:.3f}, best median PF={best_row['median_oos_pf']:.3f}."
            )
        print(
            "  Q: Should shorts be excluded?\n"
            f"     {'Probably yes' if best_short['avg_r'] < best_long['avg_r'] - 0.10 else 'Not clearly'}: "
            f"LONG AvgR={best_long['avg_r']:+.3f}, SHORT AvgR={best_short['avg_r']:+.3f} in the best exit run."
        )
        print(
            "  Q: Are far-VWAP and low-volume buckets the real edge?\n"
            f"     {'They look like the strongest pocket' if best_far['avg_r'] > 0 and best_low_vol['avg_r'] > 0 else 'Not consistently'}: "
            f"far-VWAP AvgR={best_far['avg_r']:+.3f}, PF={best_far['profit_factor']:.3f}; "
            f"low-volume AvgR={best_low_vol['avg_r']:+.3f}, PF={best_low_vol['profit_factor']:.3f}."
        )
        print(
            "  Q: Is RANGE_MEAN_REVERSION promising, failed, or retired?\n"
            f"     {'promising but sample-limited' if _setup_outcome_label(best_row) == 'promising but sample-limited' else _setup_outcome_label(best_row)}."
        )


def _run_range_mean_reversion_exit_research(
    *,
    df: pd.DataFrame,
    research_frame: pd.DataFrame,
    candidate_map: dict[int, list[CandidateContext]],
    windows: list[tuple[int, int, int, int, int]],
    initial_balance: float,
) -> None:
    setup_spec = SETUP_REGISTRY["RANGE_MEAN_REVERSION"]
    window_rows: list[dict] = []
    summary_rows: list[dict] = []
    oos_trade_map: dict[str, list[SimTrade]] = {}

    for exit_spec in _MEAN_REVERSION_EXIT_SPECS:
        results: list[WalkForwardWindowResult] = []
        oos_trade_map[exit_spec.code] = []
        for window_id, train_start_pos, train_end_pos, test_start_pos, test_end_pos in windows:
            train_trades, train_eq = _run_research_setup_simulation(
                df=df,
                research_frame=research_frame,
                candidate_map=candidate_map,
                start_pos=train_start_pos,
                end_pos=train_end_pos,
                initial_balance=initial_balance,
                setup_spec=setup_spec,
                phase="TRAIN",
                window_label=f"W{window_id:02d}",
                research_exit_code=exit_spec.code,
            )
            train_metrics = _compute_metrics(
                train_trades,
                train_eq,
                initial_balance,
                f"{setup_spec.name} {exit_spec.code} TRAIN W{window_id:02d}",
            )

            test_start_balance = train_eq[-1] if train_eq else initial_balance
            test_trades, test_eq = _run_research_setup_simulation(
                df=df,
                research_frame=research_frame,
                candidate_map=candidate_map,
                start_pos=test_start_pos,
                end_pos=test_end_pos,
                initial_balance=test_start_balance,
                setup_spec=setup_spec,
                phase="TEST",
                window_label=f"W{window_id:02d}",
                research_exit_code=exit_spec.code,
            )
            test_metrics = _compute_metrics(
                test_trades,
                test_eq,
                test_start_balance,
                f"{setup_spec.name} {exit_spec.code} TEST W{window_id:02d}",
            )

            result = WalkForwardWindowResult(
                setup_name=exit_spec.code,
                window_id=window_id,
                train_start=df.index[train_start_pos],
                train_end=df.index[train_end_pos - 1],
                test_start=df.index[test_start_pos],
                test_end=df.index[test_end_pos - 1],
                in_metrics=train_metrics,
                out_metrics=test_metrics,
                in_trades=train_trades,
                out_trades=test_trades,
            )
            results.append(result)
            oos_trade_map[exit_spec.code].extend(test_trades)
            window_rows.append(_range_exit_window_row(exit_spec, result))

        summary_rows.append(_range_exit_summary_row(exit_spec, results))

    _write_dict_csv(RANGE_EXIT_WINDOWS_CSV, window_rows)
    _write_dict_csv(RANGE_EXIT_SUMMARY_CSV, summary_rows)
    print(f"  Range exit window CSV      : {RANGE_EXIT_WINDOWS_CSV}")
    print(f"  Range exit summary CSV     : {RANGE_EXIT_SUMMARY_CSV}")
    _print_range_exit_experiment_report(summary_rows, oos_trade_map)


def _print_research_report(
    full_scan_rows: list[dict],
    summary_rows: list[dict],
    oos_trade_map: dict[str, list[SimTrade]],
) -> None:
    print(f"\n{'━' * 96}")
    print("  SETUP / REGIME RESEARCH REPORT  (research only, not live-ready)")
    print(f"{'━' * 96}")
    print(
        f"  Baseline frozen: {RESEARCH_BASELINE.name}  [{RESEARCH_BASELINE.status}]"
    )
    print(f"  Notes: {RESEARCH_BASELINE.notes}")

    print("\n  Setup summary:")
    for row in summary_rows:
        status = "PAPER-CANDIDATE" if row["passed_promotion"] else "FAILED"
        print(
            f"    {row['setup_name']:<32} {status:<15} "
            f"OOS PF avg/med={row['average_oos_pf']:.3f}/{row['median_oos_pf']:.3f}  "
            f"Trades={row['total_oos_trades']:>4}  Failed={row['failed_checks'] or '—'}"
        )

    failed_immediately = [
        row["setup_name"] for row in summary_rows
        if row["total_oos_trades"] == 0 or row["average_oos_pf"] == 0.0
    ]
    sample_limited = [
        row["setup_name"] for row in summary_rows
        if row["total_oos_trades"] < 50 and row["average_oos_pf"] >= 1.0
    ]
    robust = [row["setup_name"] for row in summary_rows if row["passed_promotion"]]

    print("\n  Immediate failures:")
    print(f"    {', '.join(failed_immediately) if failed_immediately else 'None'}")
    print("  Promising but sample-limited:")
    print(f"    {', '.join(sample_limited) if sample_limited else 'None'}")
    print("  Robust across windows:")
    print(f"    {', '.join(robust) if robust else 'None'}")

    summary_by_setup = {row["setup_name"]: row for row in summary_rows}
    base = summary_by_setup.get("MACD_VWAP_BASE")
    short_only = summary_by_setup.get("MACD_VWAP_SHORT_ONLY")
    no_medium = summary_by_setup.get("MACD_VWAP_NO_MEDIUM_ATR")
    short_no_medium = summary_by_setup.get("MACD_VWAP_SHORT_ONLY_NO_MEDIUM_ATR")
    volume_setup = summary_by_setup.get("MACD_VWAP_VOLUME")

    print("\n  Interpretation:")
    if base and base["average_oos_pf"] < 1.0:
        print("    MACD+VWAP should be retired as the primary setup; it remains weak across windows.")
    else:
        print("    MACD+VWAP is not clearly robust enough to promote.")

    if short_only and base and short_only["average_oos_pf"] > base["average_oos_pf"]:
        print("    Disabling longs improves the setup diagnostically, but it is not enough on its own unless promotion rules pass.")
    else:
        print("    Disabling longs does not create a clear promotion candidate by itself.")

    if no_medium and base and no_medium["average_oos_pf"] > base["average_oos_pf"]:
        print("    Blocking medium ATR looks directionally helpful in research, but it is still only a diagnostic until it passes promotion.")
    else:
        print("    Medium ATR is a known stress zone, but blocking it is not yet a robust answer.")

    if volume_setup and volume_setup["total_oos_trades"] < 50:
        print("    The Volume-confirmed setup is too rare right now; its results are sample-limited.")
    else:
        print("    The Volume-confirmed setup has enough activity to judge on its own.")

    if robust:
        print("    At least one setup passed the paper-trading promotion rules.")
    else:
        print("    No setup passed the promotion rules, so the current strategy family should not be advanced to paper trading or live trading.")

    breakout_summary = summary_by_setup.get("VOLUME_BREAKOUT_CONTINUATION")
    breakout_oos_trades = oos_trade_map.get("VOLUME_BREAKOUT_CONTINUATION", [])
    base_summary = summary_by_setup.get("MACD_VWAP_BASE")
    if breakout_summary:
        avg_r, med_r = _trade_r_stats(breakout_oos_trades)
        long_stats = _direction_breakdown(breakout_oos_trades, "LONG")
        short_stats = _direction_breakdown(breakout_oos_trades, "SHORT")
        atr_breakdown = {
            bucket: _bucket_trade_stats(breakout_oos_trades, "atr_bucket", bucket)
            for bucket in ("low", "medium", "high")
        }
        vol_breakdown = {
            bucket: _bucket_trade_stats(breakout_oos_trades, "volume_bucket", bucket)
            for bucket in ("low", "medium", "high")
        }

        print("\n  VOLUME_BREAKOUT_CONTINUATION walk-forward:")
        print(
            f"    Outcome            : {_setup_outcome_label(breakout_summary)}"
        )
        print(
            f"    Total OOS trades   : {breakout_summary['total_oos_trades']}"
        )
        print(
            f"    Avg / med OOS PF   : {breakout_summary['average_oos_pf']:.3f} / "
            f"{breakout_summary['median_oos_pf']:.3f}"
        )
        print(
            f"    Profitable windows : {breakout_summary['profitable_windows']}   "
            f"Losing windows: {breakout_summary['losing_windows']}"
        )
        print(
            f"    Worst window PF    : {breakout_summary['worst_window_pf']:.3f}"
        )
        print(
            f"    Avg / median R     : {avg_r:+.3f} / {med_r:+.3f}"
        )
        print(
            f"    TP hit rate        : {_tp_hit_rate(breakout_oos_trades):.1f}%"
        )

        print("    Long / short breakdown:")
        print(
            f"      LONG  trades={int(long_stats['trades'])}  PF={long_stats['profit_factor']:.3f}  "
            f"AvgR={long_stats['avg_r']:+.3f}  MedR={long_stats['median_r']:+.3f}"
        )
        print(
            f"      SHORT trades={int(short_stats['trades'])}  PF={short_stats['profit_factor']:.3f}  "
            f"AvgR={short_stats['avg_r']:+.3f}  MedR={short_stats['median_r']:+.3f}"
        )

        print("    ATR bucket breakdown:")
        for bucket in ("low", "medium", "high"):
            stats = atr_breakdown[bucket]
            print(
                f"      {bucket:<6} trades={int(stats['trades'])}  PF={stats['profit_factor']:.3f}  "
                f"AvgR={stats['avg_r']:+.3f}  TP={stats['tp_hit_rate_pct']:.1f}%"
            )

        print("    Volume bucket breakdown:")
        for bucket in ("low", "medium", "high"):
            stats = vol_breakdown[bucket]
            print(
                f"      {bucket:<6} trades={int(stats['trades'])}  PF={stats['profit_factor']:.3f}  "
                f"AvgR={stats['avg_r']:+.3f}  TP={stats['tp_hit_rate_pct']:.1f}%"
            )

        if base_summary:
            print("    Comparison vs failed MACD_VWAP_BASE:")
            print(
                f"      MACD_VWAP_BASE PF avg/med = {base_summary['average_oos_pf']:.3f} / "
                f"{base_summary['median_oos_pf']:.3f} across {base_summary['total_oos_trades']} OOS trades"
            )
            print(
                f"      Breakout delta PF avg    = "
                f"{breakout_summary['average_oos_pf'] - base_summary['average_oos_pf']:+.3f}"
            )

    pullback_summary = summary_by_setup.get("PULLBACK_TO_TREND_CONTINUATION")
    pullback_oos_trades = oos_trade_map.get("PULLBACK_TO_TREND_CONTINUATION", [])
    if pullback_summary:
        avg_r, med_r = _trade_r_stats(pullback_oos_trades)
        long_stats = _direction_breakdown(pullback_oos_trades, "LONG")
        short_stats = _direction_breakdown(pullback_oos_trades, "SHORT")
        atr_breakdown = {
            bucket: _bucket_trade_stats(pullback_oos_trades, "atr_bucket", bucket)
            for bucket in ("low", "medium", "high")
        }
        vwap_breakdown = {
            bucket: _bucket_trade_stats(pullback_oos_trades, "vwap_distance_bucket", bucket)
            for bucket in ("close", "medium", "far")
        }
        vol_breakdown = {
            bucket: _bucket_trade_stats(pullback_oos_trades, "volume_bucket", bucket)
            for bucket in ("low", "medium", "high")
        }

        print("\n  PULLBACK_TO_TREND_CONTINUATION walk-forward:")
        print(f"    Outcome            : {_setup_outcome_label(pullback_summary)}")
        print(f"    Total OOS trades   : {pullback_summary['total_oos_trades']}")
        print(
            f"    Avg / med OOS PF   : {pullback_summary['average_oos_pf']:.3f} / "
            f"{pullback_summary['median_oos_pf']:.3f}"
        )
        print(
            f"    Profitable windows : {pullback_summary['profitable_windows']}   "
            f"Losing windows: {pullback_summary['losing_windows']}"
        )
        print(f"    Worst window PF    : {pullback_summary['worst_window_pf']:.3f}")
        print(f"    Avg / median R     : {avg_r:+.3f} / {med_r:+.3f}")
        print(f"    TP hit rate        : {_tp_hit_rate(pullback_oos_trades):.1f}%")

        print("    Long / short breakdown:")
        print(
            f"      LONG  trades={int(long_stats['trades'])}  PF={long_stats['profit_factor']:.3f}  "
            f"AvgR={long_stats['avg_r']:+.3f}  MedR={long_stats['median_r']:+.3f}"
        )
        print(
            f"      SHORT trades={int(short_stats['trades'])}  PF={short_stats['profit_factor']:.3f}  "
            f"AvgR={short_stats['avg_r']:+.3f}  MedR={short_stats['median_r']:+.3f}"
        )

        print("    ATR bucket breakdown:")
        for bucket in ("low", "medium", "high"):
            stats = atr_breakdown[bucket]
            print(
                f"      {bucket:<6} trades={int(stats['trades'])}  PF={stats['profit_factor']:.3f}  "
                f"AvgR={stats['avg_r']:+.3f}  TP={stats['tp_hit_rate_pct']:.1f}%"
            )

        print("    VWAP distance bucket breakdown:")
        for bucket in ("close", "medium", "far"):
            stats = vwap_breakdown[bucket]
            print(
                f"      {bucket:<6} trades={int(stats['trades'])}  PF={stats['profit_factor']:.3f}  "
                f"AvgR={stats['avg_r']:+.3f}  TP={stats['tp_hit_rate_pct']:.1f}%"
            )

        print("    Volume bucket breakdown:")
        for bucket in ("low", "medium", "high"):
            stats = vol_breakdown[bucket]
            print(
                f"      {bucket:<6} trades={int(stats['trades'])}  PF={stats['profit_factor']:.3f}  "
                f"AvgR={stats['avg_r']:+.3f}  TP={stats['tp_hit_rate_pct']:.1f}%"
            )

        if base_summary:
            print("    Comparison vs retired MACD_VWAP_BASE:")
            print(
                f"      MACD_VWAP_BASE PF avg/med = {base_summary['average_oos_pf']:.3f} / "
                f"{base_summary['median_oos_pf']:.3f} across {base_summary['total_oos_trades']} OOS trades"
            )
            print(
                f"      Pullback delta PF avg    = "
                f"{pullback_summary['average_oos_pf'] - base_summary['average_oos_pf']:+.3f}"
            )
        if breakout_summary:
            print("    Comparison vs retired VOLUME_BREAKOUT_CONTINUATION:")
            print(
                f"      BREAKOUT PF avg/med      = {breakout_summary['average_oos_pf']:.3f} / "
                f"{breakout_summary['median_oos_pf']:.3f} across {breakout_summary['total_oos_trades']} OOS trades"
            )
            print(
                f"      Pullback delta PF avg    = "
                f"{pullback_summary['average_oos_pf'] - breakout_summary['average_oos_pf']:+.3f}"
            )

    range_summary = summary_by_setup.get("RANGE_MEAN_REVERSION")
    range_oos_trades = oos_trade_map.get("RANGE_MEAN_REVERSION", [])
    if range_summary:
        avg_r, med_r = _trade_r_stats(range_oos_trades)
        long_stats = _direction_breakdown(range_oos_trades, "LONG")
        short_stats = _direction_breakdown(range_oos_trades, "SHORT")
        atr_breakdown = {
            bucket: _bucket_trade_stats(range_oos_trades, "atr_bucket", bucket)
            for bucket in ("low", "medium", "high")
        }
        vwap_breakdown = {
            bucket: _bucket_trade_stats(range_oos_trades, "vwap_distance_bucket", bucket)
            for bucket in ("close", "medium", "far")
        }
        vol_breakdown = {
            bucket: _bucket_trade_stats(range_oos_trades, "volume_bucket", bucket)
            for bucket in ("low", "medium", "high")
        }

        print("\n  RANGE_MEAN_REVERSION walk-forward:")
        print(f"    Outcome            : {_setup_outcome_label(range_summary)}")
        print(f"    Total OOS trades   : {range_summary['total_oos_trades']}")
        print(
            f"    Avg / med OOS PF   : {range_summary['average_oos_pf']:.3f} / "
            f"{range_summary['median_oos_pf']:.3f}"
        )
        print(
            f"    Profitable windows : {range_summary['profitable_windows']}   "
            f"Losing windows: {range_summary['losing_windows']}"
        )
        print(f"    Worst window PF    : {range_summary['worst_window_pf']:.3f}")
        print(f"    Avg / median R     : {avg_r:+.3f} / {med_r:+.3f}")
        print(f"    TP hit rate        : {_tp_hit_rate(range_oos_trades):.1f}%")
        print(f"    VWAP-touch rate    : {_vwap_touch_rate(range_oos_trades):.1f}%")

        print("    Long / short breakdown:")
        print(
            f"      LONG  trades={int(long_stats['trades'])}  PF={long_stats['profit_factor']:.3f}  "
            f"AvgR={long_stats['avg_r']:+.3f}  MedR={long_stats['median_r']:+.3f}"
        )
        print(
            f"      SHORT trades={int(short_stats['trades'])}  PF={short_stats['profit_factor']:.3f}  "
            f"AvgR={short_stats['avg_r']:+.3f}  MedR={short_stats['median_r']:+.3f}"
        )

        print("    ATR bucket breakdown:")
        for bucket in ("low", "medium", "high"):
            stats = atr_breakdown[bucket]
            print(
                f"      {bucket:<6} trades={int(stats['trades'])}  PF={stats['profit_factor']:.3f}  "
                f"AvgR={stats['avg_r']:+.3f}  TP={stats['tp_hit_rate_pct']:.1f}%"
            )

        print("    VWAP distance bucket breakdown:")
        for bucket in ("close", "medium", "far"):
            stats = vwap_breakdown[bucket]
            print(
                f"      {bucket:<6} trades={int(stats['trades'])}  PF={stats['profit_factor']:.3f}  "
                f"AvgR={stats['avg_r']:+.3f}  TP={stats['tp_hit_rate_pct']:.1f}%"
            )

        print("    Volume bucket breakdown:")
        for bucket in ("low", "medium", "high"):
            stats = vol_breakdown[bucket]
            print(
                f"      {bucket:<6} trades={int(stats['trades'])}  PF={stats['profit_factor']:.3f}  "
                f"AvgR={stats['avg_r']:+.3f}  TP={stats['tp_hit_rate_pct']:.1f}%"
            )

        if base_summary:
            print("    Comparison vs retired MACD_VWAP_BASE:")
            print(
                f"      MACD_VWAP_BASE PF avg/med = {base_summary['average_oos_pf']:.3f} / "
                f"{base_summary['median_oos_pf']:.3f} across {base_summary['total_oos_trades']} OOS trades"
            )
            print(
                f"      Range delta PF avg       = "
                f"{range_summary['average_oos_pf'] - base_summary['average_oos_pf']:+.3f}"
            )
        if breakout_summary:
            print("    Comparison vs retired VOLUME_BREAKOUT_CONTINUATION:")
            print(
                f"      BREAKOUT PF avg/med      = {breakout_summary['average_oos_pf']:.3f} / "
                f"{breakout_summary['median_oos_pf']:.3f} across {breakout_summary['total_oos_trades']} OOS trades"
            )
            print(
                f"      Range delta PF avg       = "
                f"{range_summary['average_oos_pf'] - breakout_summary['average_oos_pf']:+.3f}"
            )
        if pullback_summary:
            print("    Comparison vs retired PULLBACK_TO_TREND_CONTINUATION:")
            print(
                f"      PULLBACK PF avg/med      = {pullback_summary['average_oos_pf']:.3f} / "
                f"{pullback_summary['median_oos_pf']:.3f} across {pullback_summary['total_oos_trades']} OOS trades"
            )
            print(
                f"      Range delta PF avg       = "
                f"{range_summary['average_oos_pf'] - pullback_summary['average_oos_pf']:+.3f}"
            )

    far_rows = {
        code: next((row for row in summary_rows if row["setup_name"] == code), None)
        for code in _FAR_VWAP_VARIANT_ORDER
    }
    if any(row is not None for row in far_rows.values()):
        broad_range_summary = summary_by_setup.get("RANGE_MEAN_REVERSION")
        broad_range_trades = oos_trade_map.get("RANGE_MEAN_REVERSION", [])
        broad_avg_r, broad_med_r = _trade_r_stats(broad_range_trades)

        print("\n  FAR_VWAP_MEAN_REVERSION walk-forward:")
        print("    Research-only far-VWAP variants using the current MR_EXIT_0 / F2 baseline exit.")

        for code in _FAR_VWAP_VARIANT_ORDER:
            row = far_rows.get(code)
            if row is None:
                continue
            trades = oos_trade_map.get(code, [])
            avg_r, med_r = _trade_r_stats(trades)
            win_rate, avg_win, avg_loss = _trade_win_loss_stats(trades)
            long_stats = _direction_breakdown(trades, "LONG")
            short_stats = _direction_breakdown(trades, "SHORT")
            atr_breakdown = {
                bucket: _bucket_trade_stats(trades, "atr_bucket", bucket)
                for bucket in ("low", "medium", "high")
            }
            vol_breakdown = {
                bucket: _bucket_trade_stats(trades, "volume_bucket", bucket)
                for bucket in ("low", "medium", "high")
            }

            print(f"\n  {code}  {_FAR_VWAP_VARIANT_DESCRIPTIONS[code]}")
            print(f"    Outcome            : {_report_outcome_label(row, retire_label='retire')}")
            print(f"    Total OOS trades   : {row['total_oos_trades']}")
            print(
                f"    Avg / med OOS PF   : {row['average_oos_pf']:.3f} / "
                f"{row['median_oos_pf']:.3f}"
            )
            print(
                f"    Profitable windows : {row['profitable_windows']}   "
                f"Losing windows: {row['losing_windows']}"
            )
            print(f"    Worst window PF    : {row['worst_window_pf']:.3f}")
            print(f"    Avg / median R     : {avg_r:+.3f} / {med_r:+.3f}")
            print(
                f"    Win rate           : {win_rate:.1f}%   "
                f"TP hit rate: {_tp_hit_rate(trades):.1f}%   "
                f"Avg win / loss: {avg_win:+.3f} / {avg_loss:+.3f}"
            )
            print(
                f"    VWAP-touch rate    : {_vwap_touch_rate(trades):.1f}%   "
                f"Range-mid touch: {_range_mid_touch_rate(trades):.1f}%"
            )

            print("    Long / short breakdown:")
            print(
                f"      LONG  trades={int(long_stats['trades'])}  WR={long_stats['win_rate_pct']:.1f}%  "
                f"PF={long_stats['profit_factor']:.3f}  AvgR={long_stats['avg_r']:+.3f}  "
                f"AvgWin={long_stats['avg_win_r']:+.3f}  AvgLoss={long_stats['avg_loss_r']:+.3f}"
            )
            print(
                f"      SHORT trades={int(short_stats['trades'])}  WR={short_stats['win_rate_pct']:.1f}%  "
                f"PF={short_stats['profit_factor']:.3f}  AvgR={short_stats['avg_r']:+.3f}  "
                f"AvgWin={short_stats['avg_win_r']:+.3f}  AvgLoss={short_stats['avg_loss_r']:+.3f}"
            )

            print("    ATR bucket breakdown:")
            for bucket in ("low", "medium", "high"):
                stats = atr_breakdown[bucket]
                print(
                    f"      {bucket:<6} trades={int(stats['trades'])}  WR={stats['win_rate_pct']:.1f}%  "
                    f"PF={stats['profit_factor']:.3f}  AvgR={stats['avg_r']:+.3f}"
                )

            print("    Volume bucket breakdown:")
            for bucket in ("low", "medium", "high"):
                stats = vol_breakdown[bucket]
                print(
                    f"      {bucket:<6} trades={int(stats['trades'])}  WR={stats['win_rate_pct']:.1f}%  "
                    f"PF={stats['profit_factor']:.3f}  AvgR={stats['avg_r']:+.3f}"
                )

            if broad_range_summary:
                print("    Comparison vs broad RANGE_MEAN_REVERSION:")
                print(
                    f"      RANGE PF avg/med        = {broad_range_summary['average_oos_pf']:.3f} / "
                    f"{broad_range_summary['median_oos_pf']:.3f} across "
                    f"{broad_range_summary['total_oos_trades']} OOS trades"
                )
                print(
                    f"      Variant delta PF avg    = "
                    f"{row['average_oos_pf'] - broad_range_summary['average_oos_pf']:+.3f}"
                )
                print(
                    f"      Variant delta AvgR      = {avg_r - broad_avg_r:+.3f}   "
                    f"Delta MedR = {med_r - broad_med_r:+.3f}"
                )

        best_far_row = max(
            (row for row in far_rows.values() if row is not None),
            key=lambda item: (
                item["median_oos_pf"],
                item["average_oos_pf"] if math.isfinite(item["average_oos_pf"]) else 9999.0,
                item["average_oos_avg_r"],
                item["total_oos_trades"],
            ),
        )
        best_far_code = best_far_row["setup_name"]
        best_far_trades = oos_trade_map.get(best_far_code, [])
        best_far_avg_r, _ = _trade_r_stats(best_far_trades)
        both_dir_row = far_rows.get("FV_MR_0")
        long_row = far_rows.get("FV_MR_1")
        long_ex_high_row = far_rows.get("FV_MR_2")
        medium_atr_row = far_rows.get("FV_MR_3")
        both_ex_high_row = far_rows.get("FV_MR_5")
        short_row = far_rows.get("FV_MR_6")

        print(f"\n{'━' * 112}")
        print("  FAR-VWAP DIAGNOSTIC ANSWERS")
        print(f"{'━' * 112}")
        if broad_range_summary:
            improves_broad = (
                best_far_row["median_oos_pf"] > broad_range_summary["median_oos_pf"]
                and best_far_avg_r > broad_avg_r
            )
            print(
                "  Q: Does far-VWAP actually improve broad mean reversion?\n"
                f"     {'Yes, directionally' if improves_broad else 'Not enough'}: "
                f"best variant {best_far_code} has PF avg/med "
                f"{best_far_row['average_oos_pf']:.3f}/{best_far_row['median_oos_pf']:.3f} "
                f"vs broad RANGE_MEAN_REVERSION {broad_range_summary['average_oos_pf']:.3f}/"
                f"{broad_range_summary['median_oos_pf']:.3f}."
            )
        if long_row and both_dir_row:
            long_only_better = (
                long_row["median_oos_pf"] > both_dir_row["median_oos_pf"]
                and long_row["average_oos_avg_r"] > both_dir_row["average_oos_avg_r"]
            )
            print(
                "  Q: Is long-only justified?\n"
                f"     {'Probably yes' if long_only_better else 'Not clearly'}: "
                f"FV_MR_1 PF avg/med {long_row['average_oos_pf']:.3f}/{long_row['median_oos_pf']:.3f} "
                f"vs FV_MR_0 {both_dir_row['average_oos_pf']:.3f}/{both_dir_row['median_oos_pf']:.3f}."
            )
        if long_row and long_ex_high_row and both_dir_row and both_ex_high_row:
            high_volume_exclusion_helped = (
                long_ex_high_row["median_oos_pf"] > long_row["median_oos_pf"]
                and both_ex_high_row["median_oos_pf"] >= both_dir_row["median_oos_pf"]
            )
            print(
                "  Q: Is excluding high-volume justified?\n"
                f"     {'Yes, directionally' if high_volume_exclusion_helped else 'Not consistently'}: "
                f"FV_MR_2 median PF {long_ex_high_row['median_oos_pf']:.3f} vs FV_MR_1 {long_row['median_oos_pf']:.3f}; "
                f"FV_MR_5 median PF {both_ex_high_row['median_oos_pf']:.3f} vs FV_MR_0 {both_dir_row['median_oos_pf']:.3f}."
            )
        if medium_atr_row:
            medium_atr_small = medium_atr_row["total_oos_trades"] < 50
            print(
                "  Q: Is medium ATR the true pocket or too sample-limited?\n"
                f"     {'Too sample-limited so far' if medium_atr_small else 'Large enough to judge'}: "
                f"FV_MR_3 traded {medium_atr_row['total_oos_trades']} OOS setups with "
                f"PF avg/med {medium_atr_row['average_oos_pf']:.3f}/{medium_atr_row['median_oos_pf']:.3f}."
            )
        if short_row:
            short_dead = (
                short_row["total_oos_trades"] == 0
                or (
                    short_row["average_oos_pf"] < 1.0
                    and short_row["median_oos_pf"] < 1.0
                    and short_row["average_oos_avg_r"] < 0.0
                )
            )
            print(
                "  Q: Is short-only dead?\n"
                f"     {'Probably yes' if short_dead else 'Not yet'}: "
                f"FV_MR_6 PF avg/med {short_row['average_oos_pf']:.3f}/{short_row['median_oos_pf']:.3f} "
                f"across {short_row['total_oos_trades']} OOS trades."
            )

        family_should_promote = any(
            row is not None and row["passed_promotion"] for row in far_rows.values()
        )
        family_sample_limited = any(
            row is not None
            and _report_outcome_label(row, retire_label="retire") == "promising but sample-limited"
            for row in far_rows.values()
        )
        family_improving = broad_range_summary is not None and (
            best_far_row["median_oos_pf"] > broad_range_summary["median_oos_pf"]
            or best_far_avg_r > broad_avg_r
        )
        family_status = "retired"
        if family_should_promote:
            family_status = "promoted"
        elif family_sample_limited or family_improving:
            family_status = "continued"
        print(
            "  Q: Should FAR_VWAP_MEAN_REVERSION be promoted, continued, or retired?\n"
            f"     {family_status}."
        )


def _print_entry_profile_research_report(summary_rows: list[dict]) -> None:
    print(f"\n{'━' * 112}")
    print("  ENTRY PROFILE RESEARCH LAYER  (research-only, setup overlays)")
    print(f"{'━' * 112}")
    print(
        "  This layer keeps setup-family selection fixed and evaluates entry-timing "
        "profiles on top of it."
    )

    if not summary_rows:
        print("  No entry-profile results were generated.")
        return

    for row in summary_rows:
        status = "PAPER-CANDIDATE" if row["passed_promotion"] else "FAILED"
        avg_pf = row["average_oos_pf"]
        pf_display = f"{avg_pf:.3f}" if math.isfinite(avg_pf) else "∞"
        print(
            f"    {row['combo_name']:<56} {status:<15} "
            f"PF avg/med={pf_display}/{row['median_oos_pf']:.3f}  "
            f"Trades={row['total_oos_trades']:>4}  Failed={row['failed_checks'] or '—'}"
        )

    profile_groups: dict[str, list[dict]] = {}
    for row in summary_rows:
        profile_groups.setdefault(row["entry_profile_name"], []).append(row)

    print("\n  Entry profile roll-up:")
    for profile_name, rows in profile_groups.items():
        best_row = max(
            rows,
            key=lambda item: (
                item["median_oos_pf"],
                item["average_oos_pf"] if math.isfinite(item["average_oos_pf"]) else 9999.0,
                item["average_oos_avg_r"],
                item["total_oos_trades"],
            ),
        )
        improving = sum(
            1
            for row in rows
            if row["average_oos_pf"] >= 1.0
            or row["median_oos_pf"] >= 1.0
            or row["average_oos_avg_r"] > 0.0
        )
        print(
            f"    {profile_name:<32} combos={len(rows):>2}  "
            f"best={best_row['setup_name']:<32}  "
            f"best PF avg/med="
            f"{best_row['average_oos_pf'] if math.isfinite(best_row['average_oos_pf']) else float('inf'):.3f}/"
            f"{best_row['median_oos_pf']:.3f}  "
            f"best trades={best_row['total_oos_trades']:>4}  "
            f"improving combos={improving:>2}"
        )


def run_entry_profile_research(df: pd.DataFrame, initial_balance: float) -> None:
    """
    Research-only entry overlay framework.

    This does not create a new setup family. It evaluates named entry-timing
    profiles on top of the existing research setup registry so entry timing can
    be studied independently from setup-family definitions.
    """
    print(f"\n{'━' * 112}")
    print("  RUNNING ENTRY PROFILE RESEARCH  (backtest-only, setup overlays)")
    print(f"{'━' * 112}")

    research_frame, candidate_df, _thresholds, candidate_map = _prepare_research_context(df)
    if candidate_df.empty:
        print("  Candidate universe size   : 0 potential entries")
        print("  No candidates were generated, so entry-profile research cannot run.")
        _write_dict_csv(ENTRY_PROFILE_SKIPS_CSV, [])
        _write_dict_csv(ENTRY_PROFILE_TRADES_CSV, [])
        _write_dict_csv(ENTRY_PROFILE_WINDOWS_CSV, [])
        _write_dict_csv(ENTRY_PROFILE_SUMMARY_CSV, [])
        return

    print(f"  Candidate universe size   : {len(candidate_df)} potential entries")
    print(f"  Entry profiles registered : {len(ENTRY_PROFILE_REGISTRY)}")

    combos = [
        (setup_spec, entry_profile_spec)
        for setup_spec in SETUP_REGISTRY.values()
        for entry_profile_spec in ENTRY_PROFILE_REGISTRY.values()
    ]
    print(f"  Setup/profile combinations: {len(combos)}")

    skip_rows: list[dict] = []
    trade_rows: list[dict] = []
    full_label = f"FULL_{config.BACKTEST_DAYS}D"
    for setup_spec, entry_profile_spec in combos:
        trades, _eq = _run_research_setup_simulation(
            df=df,
            research_frame=research_frame,
            candidate_map=candidate_map,
            start_pos=0,
            end_pos=len(df),
            initial_balance=initial_balance,
            setup_spec=setup_spec,
            phase="FULL",
            window_label=full_label,
            entry_profile_spec=entry_profile_spec,
            skip_rows=skip_rows,
        )
        trade_rows.extend(_trade_setup_row(trade, "FULL", full_label) for trade in trades)

    _write_dict_csv(ENTRY_PROFILE_SKIPS_CSV, skip_rows)
    _write_dict_csv(ENTRY_PROFILE_TRADES_CSV, trade_rows)
    print(f"  Entry skips CSV saved     : {ENTRY_PROFILE_SKIPS_CSV}")
    print(f"  Entry trades CSV saved    : {ENTRY_PROFILE_TRADES_CSV}")

    windows = _walk_forward_windows(df)
    window_rows: list[dict] = []
    summary_rows: list[dict] = []
    for setup_spec, entry_profile_spec in combos:
        results: list[WalkForwardWindowResult] = []
        combo_name = _entry_combo_name(setup_spec.name, entry_profile_spec.name)
        for window_id, train_start_pos, train_end_pos, test_start_pos, test_end_pos in windows:
            train_trades, train_eq = _run_research_setup_simulation(
                df=df,
                research_frame=research_frame,
                candidate_map=candidate_map,
                start_pos=train_start_pos,
                end_pos=train_end_pos,
                initial_balance=initial_balance,
                setup_spec=setup_spec,
                phase="TRAIN",
                window_label=f"W{window_id:02d}",
                entry_profile_spec=entry_profile_spec,
            )
            train_metrics = _compute_metrics(
                train_trades,
                train_eq,
                initial_balance,
                f"{combo_name} TRAIN W{window_id:02d}",
            )

            test_start_balance = train_eq[-1] if train_eq else initial_balance
            test_trades, test_eq = _run_research_setup_simulation(
                df=df,
                research_frame=research_frame,
                candidate_map=candidate_map,
                start_pos=test_start_pos,
                end_pos=test_end_pos,
                initial_balance=test_start_balance,
                setup_spec=setup_spec,
                phase="TEST",
                window_label=f"W{window_id:02d}",
                entry_profile_spec=entry_profile_spec,
            )
            test_metrics = _compute_metrics(
                test_trades,
                test_eq,
                test_start_balance,
                f"{combo_name} TEST W{window_id:02d}",
            )

            result = WalkForwardWindowResult(
                setup_name=setup_spec.name,
                window_id=window_id,
                train_start=df.index[train_start_pos],
                train_end=df.index[train_end_pos - 1],
                test_start=df.index[test_start_pos],
                test_end=df.index[test_end_pos - 1],
                in_metrics=train_metrics,
                out_metrics=test_metrics,
                in_trades=train_trades,
                out_trades=test_trades,
                entry_profile_name=entry_profile_spec.name,
            )
            results.append(result)
            window_rows.append(_window_result_row(result))

        summary_rows.append(
            _promotion_summary(
                setup_spec.name,
                results,
                entry_profile_name=entry_profile_spec.name,
            )
        )

    _write_dict_csv(ENTRY_PROFILE_WINDOWS_CSV, window_rows)
    _write_dict_csv(ENTRY_PROFILE_SUMMARY_CSV, summary_rows)
    print(f"  Entry windows CSV saved   : {ENTRY_PROFILE_WINDOWS_CSV}")
    print(f"  Entry summary CSV saved   : {ENTRY_PROFILE_SUMMARY_CSV}")
    _write_entry_failure_diagnostics(trade_rows, window_rows)
    _print_entry_profile_research_report(summary_rows)


def _write_entry_failure_diagnostics(trade_rows: list[dict], window_rows: list[dict]) -> None:
    target_setup = "VOLUME_BREAKOUT_CONTINUATION"
    target_profile = "ENTRY_ANTI_CHASE_LONG_ONLY"
    full_rows = [
        row for row in trade_rows
        if row.get("setup_name") == target_setup
        and row.get("entry_profile_name") == target_profile
        and row.get("phase") == "FULL"
    ]
    filtered_windows = [
        row for row in window_rows
        if row.get("setup_name") == target_setup
        and row.get("entry_profile_name") == target_profile
    ]
    if not full_rows:
        _write_dict_csv(ENTRY_FAILURE_DIAGNOSTICS_CSV, [])
        return

    def _is_true(value: object) -> bool:
        return str(value).strip().lower() == "true"

    rows: list[dict] = []
    total = len(full_rows)
    losers = [row for row in full_rows if float(row.get("exit_r", 0.0)) <= 0.0]
    rows.extend([
        {"section": "headline", "metric": "total_trades", "value": total},
        {"section": "headline", "metric": "fail_before_0_5r_pct", "value": round(sum(not _is_true(r.get("reached_0_5r")) for r in full_rows) / total * 100.0, 3)},
        {"section": "headline", "metric": "fail_before_1_0r_pct", "value": round(sum(not _is_true(r.get("reached_1_0r")) for r in full_rows) / total * 100.0, 3)},
        {"section": "headline", "metric": "loser_fail_before_0_5r_pct", "value": round(sum(not _is_true(r.get("reached_0_5r")) for r in losers) / len(losers) * 100.0, 3) if losers else 0.0},
        {"section": "headline", "metric": "loser_fail_before_1_0r_pct", "value": round(sum(not _is_true(r.get("reached_1_0r")) for r in losers) / len(losers) * 100.0, 3) if losers else 0.0},
        {"section": "headline", "metric": "profitable_windows", "value": sum(float(r.get("oos_return_pct", 0.0)) > 0.0 for r in filtered_windows)},
        {"section": "headline", "metric": "losing_windows", "value": sum(float(r.get("oos_return_pct", 0.0)) <= 0.0 for r in filtered_windows)},
        {"section": "headline", "metric": "profitable_windows_le2_trades", "value": sum(float(r.get("oos_return_pct", 0.0)) > 0.0 and int(float(r.get("oos_trades", 0))) <= 2 for r in filtered_windows)},
    ])

    for dimension in (
        "atr_bucket",
        "volume_bucket",
        "vwap_distance_bucket",
        "pre_entry_move_3c_bucket",
        "pre_entry_move_6c_bucket",
    ):
        buckets = sorted({row.get(dimension, "") for row in full_rows if row.get(dimension, "")})
        for bucket in buckets:
            subset = [row for row in full_rows if row.get(dimension) == bucket]
            rs = [float(row.get("exit_r", 0.0)) for row in subset]
            rows.append({
                "section": "bucket",
                "dimension": dimension,
                "bucket": bucket,
                "trades": len(subset),
                "win_rate_pct": round(sum(r > 0.0 for r in rs) / len(rs) * 100.0, 3) if rs else 0.0,
                "avg_r": round(sum(rs) / len(rs), 4) if rs else 0.0,
                "median_r": round(statistics.median(rs), 4) if rs else 0.0,
                "tp_hit_rate_pct": round(sum(_is_true(row.get("partial_tp_hit")) for row in subset) / len(subset) * 100.0, 3) if subset else 0.0,
                "fail_before_0_5r_pct": round(sum(not _is_true(row.get("reached_0_5r")) for row in subset) / len(subset) * 100.0, 3) if subset else 0.0,
                "fail_before_1_0r_pct": round(sum(not _is_true(row.get("reached_1_0r")) for row in subset) / len(subset) * 100.0, 3) if subset else 0.0,
            })

    _write_dict_csv(ENTRY_FAILURE_DIAGNOSTICS_CSV, rows)


def _research_horizon_frames(df: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    frames: list[tuple[str, pd.DataFrame]] = [(str(config.INTERVAL).lower(), df.copy())]
    if str(config.INTERVAL).lower() != "1h":
        return frames
    for horizon in ENTRY_ALT_HORIZONS[1:]:
        resampled = _resample_ohlcv(df, horizon)
        if not resampled.empty:
            frames.append((horizon, resampled))
    return frames


def _entry_alt_combos() -> list[tuple[ResearchSetup, ResearchEntryProfile]]:
    return [
        (SETUP_REGISTRY[setup_name], ENTRY_PROFILE_REGISTRY[profile_name])
        for setup_name in ENTRY_ALT_SETUPS
        for profile_name in ENTRY_ALT_PROFILES
    ]


def _entry_target_combos() -> list[tuple[ResearchSetup, ResearchEntryProfile]]:
    return [
        (SETUP_REGISTRY[setup_name], ENTRY_PROFILE_REGISTRY[profile_name])
        for setup_name in ENTRY_TARGET_SETUPS
        for profile_name in ENTRY_TARGET_PROFILES
    ]


def run_entry_alternative_horizon_research(df: pd.DataFrame, initial_balance: float) -> None:
    """
    Research-only horizon comparison layer.

    Resamples the same market history to 2H/4H and re-runs a targeted subset of
    continuation entry profiles so we can test whether higher timeframe
    follow-through is materially better than 1H.
    """
    print(f"\n{'━' * 112}")
    print("  RUNNING ENTRY HORIZON RESEARCH  (research-only, 1H vs 2H vs 4H)")
    print(f"{'━' * 112}")
    if str(config.INTERVAL).lower() != "1h":
        print("  Skipping alternative horizon research: base interval is not 1H.")
        _write_dict_csv(ENTRY_ALT_HORIZON_SKIPS_CSV, [])
        _write_dict_csv(ENTRY_ALT_HORIZON_TRADES_CSV, [])
        _write_dict_csv(ENTRY_ALT_HORIZON_WINDOWS_CSV, [])
        _write_dict_csv(ENTRY_ALT_HORIZON_SUMMARY_CSV, [])
        return

    combos = _entry_alt_combos()
    horizon_skip_rows: list[dict] = []
    horizon_trade_rows: list[dict] = []
    horizon_window_rows: list[dict] = []
    horizon_summary_rows: list[dict] = []

    for horizon_label, horizon_df in _research_horizon_frames(df):
        if len(horizon_df) < (_warmup_bars() + 10):
            continue
        print(f"  {horizon_label.upper()} horizon…", flush=True)
        research_frame, candidate_df, _thresholds, candidate_map = _prepare_research_context(horizon_df)
        if candidate_df.empty:
            continue
        full_label = f"FULL_{config.BACKTEST_DAYS}D_{horizon_label.upper()}"
        for setup_spec, entry_profile_spec in combos:
            skip_start = len(horizon_skip_rows)
            trades, _eq = _run_research_setup_simulation(
                df=horizon_df,
                research_frame=research_frame,
                candidate_map=candidate_map,
                start_pos=0,
                end_pos=len(horizon_df),
                initial_balance=initial_balance,
                setup_spec=setup_spec,
                phase="FULL",
                window_label=full_label,
                entry_profile_spec=entry_profile_spec,
                skip_rows=horizon_skip_rows,
            )
            for row in horizon_skip_rows[skip_start:]:
                row["research_horizon"] = horizon_label
            for trade in trades:
                row = _trade_setup_row(trade, "FULL", full_label)
                row["research_horizon"] = horizon_label
                horizon_trade_rows.append(row)

        windows = _walk_forward_windows(horizon_df)
        for setup_spec, entry_profile_spec in combos:
            results: list[WalkForwardWindowResult] = []
            for window_id, train_start_pos, train_end_pos, test_start_pos, test_end_pos in windows:
                train_trades, train_eq = _run_research_setup_simulation(
                    df=horizon_df,
                    research_frame=research_frame,
                    candidate_map=candidate_map,
                    start_pos=train_start_pos,
                    end_pos=train_end_pos,
                    initial_balance=initial_balance,
                    setup_spec=setup_spec,
                    phase="TRAIN",
                    window_label=f"{horizon_label.upper()}_W{window_id:02d}",
                    entry_profile_spec=entry_profile_spec,
                )
                train_metrics = _compute_metrics(
                    train_trades,
                    train_eq,
                    initial_balance,
                    f"{setup_spec.name} {entry_profile_spec.name} {horizon_label} TRAIN W{window_id:02d}",
                )
                test_start_balance = train_eq[-1] if train_eq else initial_balance
                test_trades, test_eq = _run_research_setup_simulation(
                    df=horizon_df,
                    research_frame=research_frame,
                    candidate_map=candidate_map,
                    start_pos=test_start_pos,
                    end_pos=test_end_pos,
                    initial_balance=test_start_balance,
                    setup_spec=setup_spec,
                    phase="TEST",
                    window_label=f"{horizon_label.upper()}_W{window_id:02d}",
                    entry_profile_spec=entry_profile_spec,
                )
                test_metrics = _compute_metrics(
                    test_trades,
                    test_eq,
                    test_start_balance,
                    f"{setup_spec.name} {entry_profile_spec.name} {horizon_label} TEST W{window_id:02d}",
                )
                result = WalkForwardWindowResult(
                    setup_name=setup_spec.name,
                    window_id=window_id,
                    train_start=horizon_df.index[train_start_pos],
                    train_end=horizon_df.index[train_end_pos - 1],
                    test_start=horizon_df.index[test_start_pos],
                    test_end=horizon_df.index[test_end_pos - 1],
                    in_metrics=train_metrics,
                    out_metrics=test_metrics,
                    in_trades=train_trades,
                    out_trades=test_trades,
                    entry_profile_name=entry_profile_spec.name,
                )
                results.append(result)
                horizon_window_rows.append(
                    _entry_window_row(
                        result,
                        research_horizon=horizon_label,
                    )
                )

            horizon_summary_rows.append(
                _entry_research_summary_row(
                    setup_spec.name,
                    entry_profile_spec.name,
                    results,
                    research_horizon=horizon_label,
                )
            )

    _write_dict_csv(ENTRY_ALT_HORIZON_SKIPS_CSV, horizon_skip_rows)
    _write_dict_csv(ENTRY_ALT_HORIZON_TRADES_CSV, horizon_trade_rows)
    _write_dict_csv(ENTRY_ALT_HORIZON_WINDOWS_CSV, horizon_window_rows)
    _write_dict_csv(ENTRY_ALT_HORIZON_SUMMARY_CSV, horizon_summary_rows)
    print(f"  Horizon skips CSV saved   : {ENTRY_ALT_HORIZON_SKIPS_CSV}")
    print(f"  Horizon trades CSV saved  : {ENTRY_ALT_HORIZON_TRADES_CSV}")
    print(f"  Horizon windows CSV saved : {ENTRY_ALT_HORIZON_WINDOWS_CSV}")
    print(f"  Horizon summary CSV saved : {ENTRY_ALT_HORIZON_SUMMARY_CSV}")


def run_entry_target_diagnostics(df: pd.DataFrame, initial_balance: float) -> None:
    """
    Research-only target diagnostics for continuation setups.

    Entries remain fixed; only diagnostic full-exit targets change so we can
    measure whether the continuation problem is mostly entry quality or a reward
    model mismatch.
    """
    print(f"\n{'━' * 112}")
    print("  RUNNING CONTINUATION TARGET DIAGNOSTICS  (research-only)")
    print(f"{'━' * 112}")

    trade_rows: list[dict] = []
    window_rows: list[dict] = []
    summary_rows: list[dict] = []

    for horizon_label, horizon_df in _research_horizon_frames(df):
        if len(horizon_df) < (_warmup_bars() + 10):
            continue
        research_frame, candidate_df, _thresholds, candidate_map = _prepare_research_context(horizon_df)
        if candidate_df.empty:
            continue
        full_label = f"FULL_{config.BACKTEST_DAYS}D_{horizon_label.upper()}"
        for setup_spec, entry_profile_spec in _entry_target_combos():
            for target_spec in ENTRY_TARGET_SPECS:
                trades, _eq = _run_research_setup_simulation(
                    df=horizon_df,
                    research_frame=research_frame,
                    candidate_map=candidate_map,
                    start_pos=0,
                    end_pos=len(horizon_df),
                    initial_balance=initial_balance,
                    setup_spec=setup_spec,
                    phase="FULL",
                    window_label=full_label,
                    entry_profile_spec=entry_profile_spec,
                    diagnostic_target_code=target_spec.code,
                )
                for trade in trades:
                    row = _trade_setup_row(trade, "FULL", full_label)
                    row["research_horizon"] = horizon_label
                    row["target_code"] = target_spec.code
                    row["target_description"] = target_spec.description
                    trade_rows.append(row)

        windows = _walk_forward_windows(horizon_df)
        for setup_spec, entry_profile_spec in _entry_target_combos():
            for target_spec in ENTRY_TARGET_SPECS:
                results: list[WalkForwardWindowResult] = []
                for window_id, train_start_pos, train_end_pos, test_start_pos, test_end_pos in windows:
                    train_trades, train_eq = _run_research_setup_simulation(
                        df=horizon_df,
                        research_frame=research_frame,
                        candidate_map=candidate_map,
                        start_pos=train_start_pos,
                        end_pos=train_end_pos,
                        initial_balance=initial_balance,
                        setup_spec=setup_spec,
                        phase="TRAIN",
                        window_label=f"{horizon_label.upper()}_W{window_id:02d}",
                        entry_profile_spec=entry_profile_spec,
                        diagnostic_target_code=target_spec.code,
                    )
                    train_metrics = _compute_metrics(
                        train_trades,
                        train_eq,
                        initial_balance,
                        f"{setup_spec.name} {entry_profile_spec.name} {target_spec.code} {horizon_label} TRAIN W{window_id:02d}",
                    )
                    test_start_balance = train_eq[-1] if train_eq else initial_balance
                    test_trades, test_eq = _run_research_setup_simulation(
                        df=horizon_df,
                        research_frame=research_frame,
                        candidate_map=candidate_map,
                        start_pos=test_start_pos,
                        end_pos=test_end_pos,
                        initial_balance=test_start_balance,
                        setup_spec=setup_spec,
                        phase="TEST",
                        window_label=f"{horizon_label.upper()}_W{window_id:02d}",
                        entry_profile_spec=entry_profile_spec,
                        diagnostic_target_code=target_spec.code,
                    )
                    test_metrics = _compute_metrics(
                        test_trades,
                        test_eq,
                        test_start_balance,
                        f"{setup_spec.name} {entry_profile_spec.name} {target_spec.code} {horizon_label} TEST W{window_id:02d}",
                    )
                    result = WalkForwardWindowResult(
                        setup_name=setup_spec.name,
                        window_id=window_id,
                        train_start=horizon_df.index[train_start_pos],
                        train_end=horizon_df.index[train_end_pos - 1],
                        test_start=horizon_df.index[test_start_pos],
                        test_end=horizon_df.index[test_end_pos - 1],
                        in_metrics=train_metrics,
                        out_metrics=test_metrics,
                        in_trades=train_trades,
                        out_trades=test_trades,
                        entry_profile_name=entry_profile_spec.name,
                    )
                    results.append(result)
                    window_rows.append(
                        _entry_window_row(
                            result,
                            research_horizon=horizon_label,
                            target_code=target_spec.code,
                            target_description=target_spec.description,
                        )
                    )
                summary_rows.append(
                    _entry_research_summary_row(
                        setup_spec.name,
                        entry_profile_spec.name,
                        results,
                        research_horizon=horizon_label,
                        target_code=target_spec.code,
                        target_description=target_spec.description,
                    )
                )

    _write_dict_csv(ENTRY_TARGET_TRADES_CSV, trade_rows)
    _write_dict_csv(ENTRY_TARGET_WINDOWS_CSV, window_rows)
    _write_dict_csv(ENTRY_TARGET_SUMMARY_CSV, summary_rows)
    print(f"  Target trades CSV saved   : {ENTRY_TARGET_TRADES_CSV}")
    print(f"  Target windows CSV saved  : {ENTRY_TARGET_WINDOWS_CSV}")
    print(f"  Target summary CSV saved  : {ENTRY_TARGET_SUMMARY_CSV}")


def run_range_mean_reversion_research(df: pd.DataFrame, initial_balance: float) -> None:
    """
    Research-only range mean-reversion horizon runner.

    Evaluates the existing range mean-reversion family across 1H/2H/4H using
    the current baseline exit plus simple VWAP-touch and range-mid exits.
    """
    print(f"\n{'━' * 112}")
    print("  RUNNING RANGE MEAN-REVERSION RESEARCH  (research-only, 1H vs 2H vs 4H)")
    print(f"{'━' * 112}")

    baseline_entry_profile = ENTRY_PROFILE_REGISTRY["ENTRY_BASELINE"]
    base_setup_specs = [SETUP_REGISTRY[name] for name in _RANGE_MR_HORIZON_SETUPS]
    base_exit_specs = [spec for spec in _MEAN_REVERSION_EXIT_SPECS if spec.code in _RANGE_MR_HORIZON_EXITS]

    skip_rows: list[dict] = []
    trade_rows: list[dict] = []
    window_rows: list[dict] = []
    summary_rows: list[dict] = []

    # Pre-compute regime series once for all 2H range-MR setups when gate is enabled.
    # Lazy import to avoid circular dependency at module load time.
    _regime_series_cache: dict[str, "pd.Series | None"] = {}

    def _get_regime_series(hdf: "pd.DataFrame", label: str) -> "pd.Series | None":
        if label not in _regime_series_cache:
            if config.REGIME_GATE_MR:
                from regime_classifier import RegimeClassifier
                clf = RegimeClassifier()
                _regime_series_cache[label] = clf.classify_series(hdf)
            else:
                _regime_series_cache[label] = None
        return _regime_series_cache[label]

    for horizon_label, horizon_df in _research_horizon_frames(df):
        if len(horizon_df) < (_warmup_bars() + 10):
            continue
        print(f"  {horizon_label.upper()} horizon…", flush=True)
        setup_specs = list(base_setup_specs)
        if horizon_label == "2h":
            setup_specs.extend(SETUP_REGISTRY[name] for name in _RANGE_MR_2H_FILTER_SETUPS)
        research_frame, candidate_df, _thresholds, candidate_map = _prepare_research_context(horizon_df)
        if candidate_df.empty:
            continue

        windows = _range_research_windows(horizon_df)
        full_label = f"FULL_{config.BACKTEST_DAYS}D_{horizon_label.upper()}"
        horizon_regime_series = _get_regime_series(horizon_df, horizon_label)

        for setup_spec in setup_specs:
            exit_specs = list(base_exit_specs)
            if setup_spec.name in _RANGE_MR_2H_VARIANT_DESCRIPTIONS:
                exit_specs = [spec for spec in base_exit_specs if spec.code in ("MR_EXIT_0", "MR_EXIT_3")]
            if horizon_label == "2h" and setup_spec.name in _RANGE_MR_DYNAMIC_EXIT_SETUPS:
                dynamic_specs = [
                    spec for spec in _MEAN_REVERSION_EXIT_SPECS
                    if spec.code in _RANGE_MR_DYNAMIC_EXIT_CODES
                ]
                exit_specs = exit_specs + dynamic_specs
            # Pass regime gate only for 2H range-MR setups where it's meaningful
            sim_regime_series = (
                horizon_regime_series
                if horizon_label == "2h" and setup_spec.candidate_family == "range_mean_reversion"
                else None
            )
            for exit_spec in exit_specs:
                skip_start = len(skip_rows)
                trades, _eq = _run_research_setup_simulation(
                    df=horizon_df,
                    research_frame=research_frame,
                    candidate_map=candidate_map,
                    start_pos=0,
                    end_pos=len(horizon_df),
                    initial_balance=initial_balance,
                    setup_spec=setup_spec,
                    phase="FULL",
                    window_label=full_label,
                    entry_profile_spec=baseline_entry_profile,
                    research_exit_code=exit_spec.code,
                    skip_rows=skip_rows if exit_spec.code == "MR_EXIT_0" else None,
                    regime_series=sim_regime_series,
                )
                if exit_spec.code == "MR_EXIT_0":
                    for row in skip_rows[skip_start:]:
                        row["research_horizon"] = horizon_label
                        row["research_exit_code"] = exit_spec.code
                        row["research_exit_description"] = exit_spec.description
                for trade in trades:
                    row = _trade_setup_row(trade, "FULL", full_label)
                    row["research_horizon"] = horizon_label
                    row["research_exit_code"] = exit_spec.code
                    row["research_exit_description"] = exit_spec.description
                    trade_rows.append(row)

                results: list[WalkForwardWindowResult] = []
                for window_id, train_start_pos, train_end_pos, test_start_pos, test_end_pos in windows:
                    train_trades, train_eq = _run_research_setup_simulation(
                        df=horizon_df,
                        research_frame=research_frame,
                        candidate_map=candidate_map,
                        start_pos=train_start_pos,
                        end_pos=train_end_pos,
                        initial_balance=initial_balance,
                        setup_spec=setup_spec,
                        phase="TRAIN",
                        window_label=f"{horizon_label.upper()}_{exit_spec.code}_W{window_id:02d}",
                        entry_profile_spec=baseline_entry_profile,
                        research_exit_code=exit_spec.code,
                        regime_series=sim_regime_series,
                    )
                    train_metrics = _compute_metrics(
                        train_trades,
                        train_eq,
                        initial_balance,
                        f"{setup_spec.name} {exit_spec.code} {horizon_label} TRAIN W{window_id:02d}",
                    )
                    test_start_balance = train_eq[-1] if train_eq else initial_balance
                    test_trades, test_eq = _run_research_setup_simulation(
                        df=horizon_df,
                        research_frame=research_frame,
                        candidate_map=candidate_map,
                        start_pos=test_start_pos,
                        end_pos=test_end_pos,
                        initial_balance=test_start_balance,
                        setup_spec=setup_spec,
                        phase="TEST",
                        window_label=f"{horizon_label.upper()}_{exit_spec.code}_W{window_id:02d}",
                        entry_profile_spec=baseline_entry_profile,
                        research_exit_code=exit_spec.code,
                        regime_series=sim_regime_series,
                    )
                    test_metrics = _compute_metrics(
                        test_trades,
                        test_eq,
                        test_start_balance,
                        f"{setup_spec.name} {exit_spec.code} {horizon_label} TEST W{window_id:02d}",
                    )
                    result = WalkForwardWindowResult(
                        setup_name=setup_spec.name,
                        window_id=window_id,
                        train_start=horizon_df.index[train_start_pos],
                        train_end=horizon_df.index[train_end_pos - 1],
                        test_start=horizon_df.index[test_start_pos],
                        test_end=horizon_df.index[test_end_pos - 1],
                        in_metrics=train_metrics,
                        out_metrics=test_metrics,
                        in_trades=train_trades,
                        out_trades=test_trades,
                        entry_profile_name=baseline_entry_profile.name,
                    )
                    results.append(result)
                    row = _entry_window_row(result, research_horizon=horizon_label)
                    row["research_exit_code"] = exit_spec.code
                    row["research_exit_description"] = exit_spec.description
                    window_rows.append(row)

                summary = _entry_research_summary_row(
                    setup_spec.name,
                    baseline_entry_profile.name,
                    results,
                    research_horizon=horizon_label,
                )
                summary["research_exit_code"] = exit_spec.code
                summary["research_exit_description"] = exit_spec.description
                summary_rows.append(summary)

    _write_dict_csv(RANGE_MR_RESEARCH_SKIPS_CSV, skip_rows)
    _write_dict_csv(RANGE_MR_RESEARCH_TRADES_CSV, trade_rows)
    _write_dict_csv(RANGE_MR_RESEARCH_WINDOWS_CSV, window_rows)
    _write_dict_csv(RANGE_MR_RESEARCH_SUMMARY_CSV, summary_rows)
    regime_summary_rows = _range_regime_summary_rows(trade_rows)
    _write_dict_csv(RANGE_MR_RESEARCH_REGIME_SUMMARY_CSV, regime_summary_rows)

    continuation_rows = _read_dict_csv(ENTRY_ALT_HORIZON_SUMMARY_CSV)

    def _find_summary(rows: list[dict], *, setup: str, horizon: str, exit_code: str) -> dict | None:
        return next(
            (
                row for row in rows
                if row.get("setup_name") == setup
                and row.get("research_horizon") == horizon
                and row.get("research_exit_code") == exit_code
            ),
            None,
        )

    def _find_continuation(horizon: str, setup: str) -> dict | None:
        return next(
            (
                row for row in continuation_rows
                if row.get("setup_name") == setup
                and row.get("research_horizon") == horizon
                and row.get("entry_profile_name") == "ENTRY_BASELINE"
            ),
            None,
        )

    report_lines = [
        "# Range Mean-Reversion Research",
        "",
        "Research-only horizon and exit comparison for the existing range mean-reversion family.",
        "",
        f"Validation uses dual-offset walk-forward windows: {RESEARCH_TRAIN_DAYS}d train / {RESEARCH_TEST_DAYS}d test / {RESEARCH_STEP_DAYS}d step plus a {max(1, RESEARCH_STEP_DAYS // 2)}d offset pass.",
        "",
        "## Broad Range Mean-Reversion",
    ]
    for horizon in ("1h", "2h", "4h"):
        broad = _find_summary(summary_rows, setup="RANGE_MEAN_REVERSION", horizon=horizon, exit_code="MR_EXIT_0")
        breakout = _find_continuation(horizon, "VOLUME_BREAKOUT_CONTINUATION")
        pullback = _find_continuation(horizon, "PULLBACK_TO_TREND_CONTINUATION")
        if broad:
            line = (
                f"- {horizon.upper()} broad range MR: trades={broad['total_oos_trades']}, "
                f"median PF={broad['median_oos_pf']}, AvgR={broad['average_oos_avg_r']}, "
                f"WR={broad['oos_win_rate_pct']}%, TP={broad['average_tp_hit_rate_pct']}%, "
                f"windows={broad['profitable_windows']}/{broad['losing_windows']}."
            )
            report_lines.append(line)
            if breakout:
                report_lines.append(
                    f"  Compared with breakout baseline on {horizon.upper()}: "
                    f"breakout median PF={breakout['median_oos_pf']}, AvgR={breakout['average_oos_avg_r']}."
                )
            if pullback:
                report_lines.append(
                    f"  Compared with pullback baseline on {horizon.upper()}: "
                    f"pullback median PF={pullback['median_oos_pf']}, AvgR={pullback['average_oos_avg_r']}."
                )

    report_lines.extend([
        "",
        "## Far-VWAP Diagnostic Pocket",
    ])
    for horizon in ("1h", "2h", "4h"):
        for setup_name in ("FV_MR_0", "FV_MR_1", "FV_MR_6"):
            row = _find_summary(summary_rows, setup=setup_name, horizon=horizon, exit_code="MR_EXIT_0")
            if row:
                report_lines.append(
                    f"- {horizon.upper()} {setup_name}: trades={row['total_oos_trades']}, "
                    f"median PF={row['median_oos_pf']}, AvgR={row['average_oos_avg_r']}, "
                    f"windows={row['profitable_windows']}/{row['losing_windows']}."
                )

    report_lines.extend([
        "",
        "## 2H Regime Variants",
    ])
    for setup_name in _RANGE_MR_2H_FILTER_SETUPS:
        base_row = _find_summary(summary_rows, setup=setup_name, horizon="2h", exit_code="MR_EXIT_0")
        mid_row = _find_summary(summary_rows, setup=setup_name, horizon="2h", exit_code="MR_EXIT_3")
        if base_row:
            line = (
                f"- {setup_name}: trades={base_row['total_oos_trades']}, "
                f"median PF={base_row['median_oos_pf']}, AvgR={base_row['average_oos_avg_r']}, "
                f"WR={base_row['oos_win_rate_pct']}%, TP={base_row['average_tp_hit_rate_pct']}%, "
                f"windows={base_row['profitable_windows']}/{base_row['losing_windows']}."
            )
            if mid_row:
                line += (
                    f" Range-mid exit median PF={mid_row['median_oos_pf']}, "
                    f"AvgR={mid_row['average_oos_avg_r']}."
                )
            report_lines.append(line)

    report_lines.extend([
        "",
        "## Exit Comparison",
    ])
    for horizon in ("1h", "2h", "4h"):
        broad_base = _find_summary(summary_rows, setup="RANGE_MEAN_REVERSION", horizon=horizon, exit_code="MR_EXIT_0")
        broad_vwap = _find_summary(summary_rows, setup="RANGE_MEAN_REVERSION", horizon=horizon, exit_code="MR_EXIT_1")
        broad_mid = _find_summary(summary_rows, setup="RANGE_MEAN_REVERSION", horizon=horizon, exit_code="MR_EXIT_3")
        if broad_base and broad_vwap and broad_mid:
            report_lines.append(
                f"- {horizon.upper()} exits: baseline PF={broad_base['median_oos_pf']} / AvgR={broad_base['average_oos_avg_r']}; "
                f"VWAP-touch PF={broad_vwap['median_oos_pf']} / AvgR={broad_vwap['average_oos_avg_r']}; "
                f"range-mid PF={broad_mid['median_oos_pf']} / AvgR={broad_mid['average_oos_avg_r']}."
            )

    report_lines.extend([
        "",
        "## 2H Dynamic Exit Diagnostics",
    ])
    for setup_name in _RANGE_MR_DYNAMIC_EXIT_SETUPS:
        for exit_code in ("MR_EXIT_0", "MR_EXIT_7", "MR_EXIT_8", "MR_EXIT_9", "MR_EXIT_10", "MR_EXIT_11"):
            row = _find_summary(summary_rows, setup=setup_name, horizon="2h", exit_code=exit_code)
            if row:
                report_lines.append(
                    f"- {setup_name} {exit_code}: trades={row['total_oos_trades']}, "
                    f"median PF={row['median_oos_pf']}, AvgR={row['average_oos_avg_r']}, "
                    f"Sharpe={row['average_oos_sharpe']}, Sortino={row['average_oos_sortino']}, "
                    f"windows={row['profitable_windows']}/{row['losing_windows']}."
                )

    report_lines.extend([
        "",
        "## 2H Regime Diagnostics",
    ])
    for setup_name in _RANGE_MR_DYNAMIC_EXIT_SETUPS:
        for exit_code in ("MR_EXIT_0", "MR_EXIT_9", "MR_EXIT_10", "MR_EXIT_11"):
            matching = [
                row for row in regime_summary_rows
                if row.get("setup_name") == setup_name
                and row.get("research_exit_code") == exit_code
                and row.get("trades", 0) not in (0, "0", "")
            ]
            if not matching:
                continue
            matching.sort(key=lambda row: (int(row["trades"]), _parse_float(row["avg_r"])), reverse=True)
            best = matching[0]
            worst = min(matching, key=lambda row: (_parse_float(row["avg_r"]), _parse_float(row["profit_factor"])))
            report_lines.append(
                f"- {setup_name} {exit_code}: best regime {best['regime_label']} {best['direction']} "
                f"(N={best['trades']}, AvgR={best['avg_r']}, PF={best['profit_factor']}, TP={best['tp_hit_rate_pct']}%), "
                f"worst regime {worst['regime_label']} {worst['direction']} "
                f"(N={worst['trades']}, AvgR={worst['avg_r']}, PF={worst['profit_factor']}, TP={worst['tp_hit_rate_pct']}%)."
            )

    # ── Profile ranking ───────────────────────────────────────────────────────
    # Rank all 2H setups by a composite robustness score.  Primary metric is
    # worst_window_pf (floor quality); secondary metrics break ties.
    rankable = [
        row for row in summary_rows
        if row.get("research_horizon") == "2h"
        and row.get("research_exit_code") == "MR_EXIT_0"
    ]
    rankable.sort(
        key=lambda row: (
            _parse_float(row.get("worst_window_pf", 0)),
            _parse_float(row.get("median_oos_pf", 0)),
            _parse_float(row.get("average_oos_avg_r", 0)),
            -_parse_float(row.get("tp_hit_rate_std_pct", 999)),
        ),
        reverse=True,
    )
    report_lines.extend([
        "",
        "## Profile Ranking (2H, MR_EXIT_0, by robustness)",
        "Ranked by worst-window PF → median OOS PF → avg OOS R → TP-hit stability (lower std = better).",
        "",
        "| Rank | Setup | Passed | Worst WPF | Med OOS PF | Avg R | TP% | TP Std | IS/OOS PF ratio | Cluster Win% |",
        "| ---- | ----- | ------ | --------- | ---------- | ----- | --- | ------ | --------------- | ------------ |",
    ])
    primary_cutoff = min(5, len(rankable))
    for rank, row in enumerate(rankable, 1):
        tag = "★ PRIMARY" if rank <= primary_cutoff and row.get("passed_promotion") else ("refinement" if row.get("passed_promotion") else "")
        report_lines.append(
            f"| {rank} | {row['setup_name']} | {'✓' if row.get('passed_promotion') else '✗'} {tag} | "
            f"{_parse_float(row.get('worst_window_pf', 0)):.3f} | "
            f"{_parse_float(row.get('median_oos_pf', 0)):.3f} | "
            f"{_parse_float(row.get('average_oos_avg_r', 0)):+.3f} | "
            f"{_parse_float(row.get('average_tp_hit_rate_pct', 0)):.1f}% | "
            f"{_parse_float(row.get('tp_hit_rate_std_pct', 0)):.1f}% | "
            f"{_parse_float(row.get('is_oos_pf_ratio', 0)):.3f} | "
            f"{_parse_float(row.get('cluster_share_winning_pct', 0)):.1f}% |"
        )

    report_lines.extend([
        "",
        "## Notes",
        "- Promotion still requires at least 50 OOS trades, median PF > 1.05, positive AvgR, and profitable windows outnumbering losing windows.",
        "- No result should be promoted from this report alone without meeting the full walk-forward rules.",
        f"- Regime gate (REGIME_GATE_MR): {'ON' if config.REGIME_GATE_MR else 'OFF'} — gate forces RANGING regime at entry for 2H range-MR setups.",
    ])

    RANGE_MR_RESEARCH_REPORT_MD.write_text("\n".join(report_lines) + "\n")
    print(f"  Range MR skips CSV      : {RANGE_MR_RESEARCH_SKIPS_CSV}")
    print(f"  Range MR trades CSV     : {RANGE_MR_RESEARCH_TRADES_CSV}")
    print(f"  Range MR windows CSV    : {RANGE_MR_RESEARCH_WINDOWS_CSV}")
    print(f"  Range MR summary CSV    : {RANGE_MR_RESEARCH_SUMMARY_CSV}")
    print(f"  Range MR regime CSV     : {RANGE_MR_RESEARCH_REGIME_SUMMARY_CSV}")
    print(f"  Range MR report         : {RANGE_MR_RESEARCH_REPORT_MD}")


def _continuation_summary_key(row: dict) -> tuple[str, str, str, str, str]:
    return (
        row.get("diagnostic_source", ""),
        row.get("setup_name", ""),
        row.get("entry_profile_name", ""),
        row.get("research_horizon", ""),
        row.get("target_code", ""),
    )


def _window_profitable(row: dict) -> bool:
    trades = _parse_int(row.get("oos_trades", 0))
    if trades <= 0:
        return False
    pf_text = str(row.get("oos_profit_factor", "")).strip().lower()
    if pf_text == "inf":
        return True
    return _parse_float(row.get("oos_profit_factor", 0.0)) > 1.0


def _window_losing(row: dict) -> bool:
    trades = _parse_int(row.get("oos_trades", 0))
    if trades <= 0:
        return False
    pf_text = str(row.get("oos_profit_factor", "")).strip().lower()
    if pf_text == "inf":
        return False
    return _parse_float(row.get("oos_profit_factor", 0.0)) < 1.0


def _normalise_summary_row(row: dict, diagnostic_source: str) -> dict:
    norm = dict(row)
    norm["diagnostic_source"] = diagnostic_source
    norm["research_horizon"] = norm.get("research_horizon", "") or "1h"
    norm["target_code"] = norm.get("target_code", "") or "CONT_TARGET_BASELINE"
    norm["target_description"] = norm.get("target_description", "") or "Current F2 baseline exit."
    return norm


def _normalise_window_row(row: dict, diagnostic_source: str) -> dict:
    norm = dict(row)
    norm["diagnostic_source"] = diagnostic_source
    norm["research_horizon"] = norm.get("research_horizon", "") or "1h"
    norm["target_code"] = norm.get("target_code", "") or "CONT_TARGET_BASELINE"
    norm["target_description"] = norm.get("target_description", "") or "Current F2 baseline exit."
    return norm


def _failure_stat_rows(rows: list[dict]) -> dict[str, float]:
    out: dict[str, float] = {}
    for row in rows:
        metric = str(row.get("metric", "")).strip()
        if metric:
            out[metric] = _parse_float(row.get("value", 0.0))
    return out


def _profitable_window_counts(window_rows: list[dict]) -> tuple[int, int, int]:
    profitable = [row for row in window_rows if _window_profitable(row)]
    losing = [row for row in window_rows if _window_losing(row)]
    profitable_le2 = sum(_parse_int(row.get("oos_trades", 0)) <= 2 for row in profitable)
    return len(profitable), len(losing), profitable_le2


def _selected_continuation_rows(summary_rows: list[dict], *, setup_name: str, profiles: tuple[str, ...]) -> list[dict]:
    return [
        row
        for row in summary_rows
        if row.get("setup_name") == setup_name
        and row.get("entry_profile_name") in profiles
        and row.get("target_code") == "CONT_TARGET_BASELINE"
    ]


def run_entry_diagnosis() -> None:
    """
    Research-only diagnostic layer.

    Consolidates entry profile, alternative horizon, and continuation target
    outputs into a single diagnosis summary without changing live behavior.
    """
    print(f"\n{'━' * 112}")
    print("  RUNNING ENTRY DIAGNOSIS  (research-only reporting)")
    print(f"{'━' * 112}")

    continuation_setups = {"VOLUME_BREAKOUT_CONTINUATION", "PULLBACK_TO_TREND_CONTINUATION"}

    summary_rows: list[dict] = []
    window_rows: list[dict] = []

    summary_rows.extend(
        _normalise_summary_row(row, "entry_profile_1h")
        for row in _read_dict_csv(ENTRY_PROFILE_SUMMARY_CSV)
        if row.get("setup_name") in continuation_setups
    )
    summary_rows.extend(
        _normalise_summary_row(row, "alt_horizon")
        for row in _read_dict_csv(ENTRY_ALT_HORIZON_SUMMARY_CSV)
        if row.get("setup_name") in continuation_setups
    )
    summary_rows.extend(
        _normalise_summary_row(row, "target_diagnostics")
        for row in _read_dict_csv(ENTRY_TARGET_SUMMARY_CSV)
        if row.get("setup_name") in continuation_setups
    )

    window_rows.extend(
        _normalise_window_row(row, "entry_profile_1h")
        for row in _read_dict_csv(ENTRY_PROFILE_WINDOWS_CSV)
        if row.get("setup_name") in continuation_setups
    )
    window_rows.extend(
        _normalise_window_row(row, "alt_horizon")
        for row in _read_dict_csv(ENTRY_ALT_HORIZON_WINDOWS_CSV)
        if row.get("setup_name") in continuation_setups
    )
    window_rows.extend(
        _normalise_window_row(row, "target_diagnostics")
        for row in _read_dict_csv(ENTRY_TARGET_WINDOWS_CSV)
        if row.get("setup_name") in continuation_setups
    )

    if not summary_rows:
        print("  No continuation diagnosis inputs were found.")
        _write_dict_csv(ENTRY_DIAGNOSIS_SUMMARY_CSV, [])
        _write_dict_csv(ENTRY_DIAGNOSIS_WINDOWS_CSV, [])
        ENTRY_DIAGNOSIS_REPORT_MD.write_text(
            "# Entry Diagnosis\n\nNo continuation diagnosis inputs were available.\n"
        )
        return

    windows_by_key: dict[tuple[str, str, str, str, str], list[dict]] = {}
    for row in window_rows:
        windows_by_key.setdefault(_continuation_summary_key(row), []).append(row)

    diagnosis_summary_rows: list[dict] = []
    for row in summary_rows:
        key = _continuation_summary_key(row)
        related_windows = windows_by_key.get(key, [])
        profitable_windows, losing_windows, profitable_le2 = _profitable_window_counts(related_windows)
        zero_trade_windows = sum(_parse_int(win.get("oos_trades", 0)) == 0 for win in related_windows)
        merged = dict(row)
        merged["profitable_windows_counted"] = profitable_windows
        merged["losing_windows_counted"] = losing_windows
        merged["profitable_windows_le2_trades"] = profitable_le2
        merged["zero_trade_windows"] = zero_trade_windows
        diagnosis_summary_rows.append(merged)

    _write_dict_csv(ENTRY_DIAGNOSIS_SUMMARY_CSV, diagnosis_summary_rows)
    _write_dict_csv(ENTRY_DIAGNOSIS_WINDOWS_CSV, window_rows)

    failure_stats = _failure_stat_rows(_read_dict_csv(ENTRY_FAILURE_DIAGNOSTICS_CSV))

    continuation_full_rows = [
        row
        for row in _read_dict_csv(ENTRY_PROFILE_TRADES_CSV)
        if row.get("phase") == "FULL"
        and row.get("entry_profile_name") == "ENTRY_BASELINE"
        and row.get("setup_name") in continuation_setups
    ]
    breakout_full_rows = [row for row in continuation_full_rows if row.get("setup_name") == "VOLUME_BREAKOUT_CONTINUATION"]
    pullback_full_rows = [row for row in continuation_full_rows if row.get("setup_name") == "PULLBACK_TO_TREND_CONTINUATION"]

    def _reach_failure_pct(rows: list[dict], field: str) -> float:
        if not rows:
            return 0.0
        misses = sum(str(row.get(field, "")).strip().lower() != "true" for row in rows)
        return round(misses / len(rows) * 100.0, 3)

    baseline_failure_05 = _reach_failure_pct(continuation_full_rows, "reached_0_5r")
    baseline_failure_10 = _reach_failure_pct(continuation_full_rows, "reached_1_0r")

    selected_profiles = (
        "ENTRY_BASELINE",
        "ENTRY_ANTI_CHASE_LONG_ONLY",
        "ENTRY_ANTI_CHASE_LONG_ONLY_RELAXED_VWAP",
        "ENTRY_ANTI_CHASE_LONG_ONLY_RELAXED_EXTENSION",
        "ENTRY_ANTI_CHASE_LONG_ONLY_ADAPTIVE",
    )
    breakout_horizon_rows = _selected_continuation_rows(
        diagnosis_summary_rows,
        setup_name="VOLUME_BREAKOUT_CONTINUATION",
        profiles=selected_profiles,
    )
    pullback_horizon_rows = _selected_continuation_rows(
        diagnosis_summary_rows,
        setup_name="PULLBACK_TO_TREND_CONTINUATION",
        profiles=("ENTRY_BASELINE", "ENTRY_ANTI_CHASE_LONG_ONLY", "ENTRY_ANTI_CHASE_LONG_ONLY_ADAPTIVE"),
    )
    breakout_horizon_rows = [
        row for row in breakout_horizon_rows
        if row.get("diagnostic_source") == "alt_horizon"
    ]
    pullback_horizon_rows = [
        row for row in pullback_horizon_rows
        if row.get("diagnostic_source") == "alt_horizon"
    ]

    target_rows = [
        row for row in diagnosis_summary_rows
        if row.get("diagnostic_source") == "target_diagnostics"
        and row.get("setup_name") == "VOLUME_BREAKOUT_CONTINUATION"
        and row.get("entry_profile_name") in (
            "ENTRY_BASELINE",
            "ENTRY_ANTI_CHASE_LONG_ONLY",
            "ENTRY_ANTI_CHASE_LONG_ONLY_RELAXED_VWAP",
            "ENTRY_ANTI_CHASE_LONG_ONLY_ADAPTIVE",
        )
    ]

    def _find_row(rows: list[dict], *, setup: str, profile: str, horizon: str, target: str = "CONT_TARGET_BASELINE") -> dict | None:
        return next(
            (
                row
                for row in rows
                if row.get("setup_name") == setup
                and row.get("entry_profile_name") == profile
                and row.get("research_horizon") == horizon
                and row.get("target_code") == target
            ),
            None,
        )

    one_h_long_only = _find_row(
        diagnosis_summary_rows,
        setup="VOLUME_BREAKOUT_CONTINUATION",
        profile="ENTRY_ANTI_CHASE_LONG_ONLY",
        horizon="1h",
    )
    one_h_adaptive = _find_row(
        diagnosis_summary_rows,
        setup="VOLUME_BREAKOUT_CONTINUATION",
        profile="ENTRY_ANTI_CHASE_LONG_ONLY_ADAPTIVE",
        horizon="1h",
    )
    four_h_baseline = _find_row(
        diagnosis_summary_rows,
        setup="VOLUME_BREAKOUT_CONTINUATION",
        profile="ENTRY_BASELINE",
        horizon="4h",
    )

    report_lines = [
        "# Entry Diagnosis",
        "",
        "Research-only continuation diagnosis across the current BTC dataset. No live trading logic was changed.",
        "",
        "## Structural Summary",
        f"- Broad 1H continuation baseline remains weak. Breakout baseline was {(_find_row(diagnosis_summary_rows, setup='VOLUME_BREAKOUT_CONTINUATION', profile='ENTRY_BASELINE', horizon='1h') or {}).get('median_oos_pf', 'n/a')} median PF with {(_find_row(diagnosis_summary_rows, setup='VOLUME_BREAKOUT_CONTINUATION', profile='ENTRY_BASELINE', horizon='1h') or {}).get('total_oos_trades', 'n/a')} OOS trades.",
        f"- Across 1H baseline continuation trades (`VOLUME_BREAKOUT_CONTINUATION` + `PULLBACK_TO_TREND_CONTINUATION`), {baseline_failure_05:.1f}% never reached 0.5R and {baseline_failure_10:.1f}% never reached 1.0R.",
        f"- The original `ENTRY_ANTI_CHASE_LONG_ONLY` failure mode stayed real: {failure_stats.get('fail_before_0_5r_pct', 0.0):.1f}% of its 1H full trades failed before 0.5R, and {failure_stats.get('loser_fail_before_0_5r_pct', 0.0):.1f}% of losers never reached 0.5R.",
        "",
        "## Entry Profile Research",
    ]

    if one_h_long_only and one_h_adaptive:
        report_lines.extend([
            f"- 1H breakout `ENTRY_ANTI_CHASE_LONG_ONLY`: {one_h_long_only['total_oos_trades']} OOS trades, median PF {one_h_long_only['median_oos_pf']}, AvgR {one_h_long_only['average_oos_avg_r']}, profitable/losing windows {one_h_long_only['profitable_windows']}/{one_h_long_only['losing_windows']}.",
            f"- 1H breakout `ENTRY_ANTI_CHASE_LONG_ONLY_ADAPTIVE`: {one_h_adaptive['total_oos_trades']} OOS trades, median PF {one_h_adaptive['median_oos_pf']}, AvgR {one_h_adaptive['average_oos_avg_r']}, profitable/losing windows {one_h_adaptive['profitable_windows']}/{one_h_adaptive['losing_windows']}, cluster profit share {one_h_adaptive['small_cluster_profit_share']}.",
            "- Adaptive anti-chase helped by reopening some medium/far VWAP sample and cutting early outright failures, but it still failed because median PF stayed below 1.05, AvgR stayed negative, worst-window PF stayed at 0.0, and profit still leaned on small windows.",
        ])

    report_lines.extend([
        "",
        "## Time Horizon Effect",
    ])
    for horizon in ("1h", "2h", "4h"):
        breakout_base = _find_row(diagnosis_summary_rows, setup="VOLUME_BREAKOUT_CONTINUATION", profile="ENTRY_BASELINE", horizon=horizon)
        if breakout_base:
            report_lines.append(
                f"- Breakout baseline {horizon.upper()}: trades={breakout_base['total_oos_trades']}, "
                f"median PF={breakout_base['median_oos_pf']}, AvgR={breakout_base['average_oos_avg_r']}, "
                f"windows={breakout_base['profitable_windows']}/{breakout_base['losing_windows']}."
            )
    report_lines.extend([
        "- Higher timeframes improved follow-through directionally: breakout baseline median PF rose from 0.634 on 1H to 0.875 on 2H and 0.938 on 4H.",
        "- That improvement was not enough to create a robust edge. Even 4H baseline still failed median PF > 1.05 and stable-window requirements.",
        "- Refined long-only overlays on 4H produced better-looking trade quality, but sample stayed too small and profitable windows were still dominated by one- or two-trade pockets.",
        "",
        "## Reward / Exit Effect",
        "- Lower continuation targets did not reliably fix the problem.",
        "- On 1H breakout baseline, 1.0R and 0.75R increased win rate but reduced median PF versus the current baseline. VWAP-touch and range-mid exits were decisively bad.",
        "- On 1H adaptive anti-chase, the baseline target still beat 1.0R and 0.75R on median PF. Smaller targets improved hit rate but did not create stable expectancy.",
        "- Conclusion: the reward model is demanding, but it is not the only blocker. Easier targets alone do not rescue BTC continuation.",
        "",
        "## Structural Constraints",
        f"- Breakout baseline 1H: {len(breakout_full_rows)} full trades, median R {statistics.median(float(r['exit_r']) for r in breakout_full_rows):+.3f}, fail-before-0.5R {_reach_failure_pct(breakout_full_rows, 'reached_0_5r'):.1f}%.",
        f"- Pullback baseline 1H: {len(pullback_full_rows)} full trades, median R {statistics.median(float(r['exit_r']) for r in pullback_full_rows):+.3f}, fail-before-0.5R {_reach_failure_pct(pullback_full_rows, 'reached_0_5r'):.1f}%.",
        "- Low ATR is consistently the weakest broad bucket. Volume is not a strong separator for breakout continuation because accepted breakout trades are almost entirely high-volume already.",
        "- Multi-candle extension filters help avoid some obvious chases, but simply tightening or relaxing them mostly trades sample size against a still-weak underlying edge.",
        "",
        "## Recommendations",
        "- Do not promote any continuation profile to paper trading.",
        "- If continuation research continues, prefer higher timeframes before more 1H refinement because 2H/4H improved follow-through more than further 1H filter tweaks.",
        "- Consider testing a different asset with cleaner directional follow-through if continuation remains the goal.",
        "- If staying on BTC, prioritize alternative entry concepts such as range mean-reversion or range-trading, because continuation still clusters around median losses near the stop.",
        "- Any future reward-model work should be paired with a setup that first proves stable entry quality. Smaller targets by themselves were not enough.",
    ])

    ENTRY_DIAGNOSIS_REPORT_MD.write_text("\n".join(report_lines) + "\n")

    print(f"  Entry diagnosis summary CSV : {ENTRY_DIAGNOSIS_SUMMARY_CSV}")
    print(f"  Entry diagnosis windows CSV : {ENTRY_DIAGNOSIS_WINDOWS_CSV}")
    print(f"  Entry diagnosis report      : {ENTRY_DIAGNOSIS_REPORT_MD}")


def run_walk_forward_research(df: pd.DataFrame, initial_balance: float) -> None:
    """
    Research-only framework:
      - freeze the current candidate as a failed baseline,
      - build a named setup registry,
      - export skipped candidates with regime tags,
      - evaluate each setup on rolling walk-forward windows,
      - apply conservative promotion rules.
    """
    print(f"\n{'━' * 96}")
    print("  RUNNING WALK-FORWARD SETUP / REGIME RESEARCH  (backtest-only)")
    print(f"{'━' * 96}")

    research_frame, candidate_df, thresholds, candidate_map = _prepare_research_context(df)
    if candidate_df.empty:
        print("  Candidate universe size   : 0 potential entries")
        print("  No research candidates were generated, so walk-forward research cannot run.")
        _write_dict_csv(SKIPPED_SETUPS_CSV, [])
        _write_dict_csv(SETUP_TRADES_CSV, [])
        _write_dict_csv(WALK_FORWARD_WINDOWS_CSV, [])
        _write_dict_csv(WALK_FORWARD_SUMMARY_CSV, [])
        _write_dict_csv(RANGE_EXIT_WINDOWS_CSV, [])
        _write_dict_csv(RANGE_EXIT_SUMMARY_CSV, [])
        _write_dict_csv(ENTRY_PROFILE_SKIPS_CSV, [])
        _write_dict_csv(ENTRY_PROFILE_TRADES_CSV, [])
        _write_dict_csv(ENTRY_PROFILE_WINDOWS_CSV, [])
        _write_dict_csv(ENTRY_PROFILE_SUMMARY_CSV, [])
        return

    _print_regime_thresholds(thresholds)
    print(f"  Candidate universe size   : {len(candidate_df)} potential entries")

    skip_rows: list[dict] = []
    trade_rows: list[dict] = []
    full_scan_summary_rows: list[dict] = []
    full_label = f"FULL_{config.BACKTEST_DAYS}D"
    baseline_entry_profile = ENTRY_PROFILE_REGISTRY["ENTRY_BASELINE"]

    for setup_spec in SETUP_REGISTRY.values():
        setup_name = setup_spec.name
        trades, eq = _run_research_setup_simulation(
            df=df,
            research_frame=research_frame,
            candidate_map=candidate_map,
            start_pos=0,
            end_pos=len(df),
            initial_balance=initial_balance,
            setup_spec=setup_spec,
            phase="FULL",
            window_label=full_label,
            entry_profile_spec=baseline_entry_profile,
            skip_rows=skip_rows,
        )
        metrics = _compute_metrics(trades, eq, initial_balance, setup_name)
        full_scan_summary_rows.append({
            "setup_name": setup_name,
            "trades": metrics.total_trades,
            "win_rate_pct": metrics.win_rate_pct,
            "profit_factor": metrics.profit_factor,
            "avg_r": round((sum(t.exit_r for t in trades) / len(trades)) if trades else 0.0, 4),
            "median_r": round(statistics.median([t.exit_r for t in trades]) if trades else 0.0, 4),
            "tp_hit_rate_pct": round(_tp_hit_rate(trades), 4),
            "return_pct": metrics.total_return_pct,
        })
        trade_rows.extend(_trade_setup_row(trade, "FULL", full_label) for trade in trades)

    _write_dict_csv(SKIPPED_SETUPS_CSV, skip_rows)
    _write_dict_csv(SETUP_TRADES_CSV, trade_rows)
    print(f"  Skipped setups CSV saved  : {SKIPPED_SETUPS_CSV}")
    print(f"  Accepted trades CSV saved : {SETUP_TRADES_CSV}")

    window_rows: list[dict] = []
    summary_rows: list[dict] = []
    oos_trade_map: dict[str, list[SimTrade]] = {}
    windows = _walk_forward_windows(df)
    print(f"  Walk-forward windows      : {len(windows)}  ({RESEARCH_TRAIN_DAYS}d train / {RESEARCH_TEST_DAYS}d test / {RESEARCH_STEP_DAYS}d step)")

    for setup_spec in SETUP_REGISTRY.values():
        setup_name = setup_spec.name
        results: list[WalkForwardWindowResult] = []
        oos_trade_map[setup_name] = []
        for window_id, train_start_pos, train_end_pos, test_start_pos, test_end_pos in windows:
            train_trades, train_eq = _run_research_setup_simulation(
                df=df,
                research_frame=research_frame,
                candidate_map=candidate_map,
                start_pos=train_start_pos,
                end_pos=train_end_pos,
                initial_balance=initial_balance,
                setup_spec=setup_spec,
                phase="TRAIN",
                window_label=f"W{window_id:02d}",
                entry_profile_spec=baseline_entry_profile,
            )
            train_metrics = _compute_metrics(
                train_trades,
                train_eq,
                initial_balance,
                f"{setup_name} TRAIN W{window_id:02d}",
            )

            test_start_balance = train_eq[-1] if train_eq else initial_balance
            test_trades, test_eq = _run_research_setup_simulation(
                df=df,
                research_frame=research_frame,
                candidate_map=candidate_map,
                start_pos=test_start_pos,
                end_pos=test_end_pos,
                initial_balance=test_start_balance,
                setup_spec=setup_spec,
                phase="TEST",
                window_label=f"W{window_id:02d}",
                entry_profile_spec=baseline_entry_profile,
            )
            test_metrics = _compute_metrics(
                test_trades,
                test_eq,
                test_start_balance,
                f"{setup_name} TEST W{window_id:02d}",
            )

            result = WalkForwardWindowResult(
                setup_name=setup_name,
                window_id=window_id,
                train_start=df.index[train_start_pos],
                train_end=df.index[train_end_pos - 1],
                test_start=df.index[test_start_pos],
                test_end=df.index[test_end_pos - 1],
                in_metrics=train_metrics,
                out_metrics=test_metrics,
                in_trades=train_trades,
                out_trades=test_trades,
            )
            results.append(result)
            oos_trade_map[setup_name].extend(test_trades)
            window_rows.append(_window_result_row(result))

        summary_rows.append(_promotion_summary(setup_name, results))

    _write_dict_csv(WALK_FORWARD_WINDOWS_CSV, window_rows)
    _write_dict_csv(WALK_FORWARD_SUMMARY_CSV, summary_rows)
    print(f"  Walk-forward window CSV   : {WALK_FORWARD_WINDOWS_CSV}")
    print(f"  Walk-forward summary CSV  : {WALK_FORWARD_SUMMARY_CSV}")
    _print_research_report(full_scan_summary_rows, summary_rows, oos_trade_map)
    if config.RUN_RANGE_EXIT_RESEARCH:
        _run_range_mean_reversion_exit_research(
            df=df,
            research_frame=research_frame,
            candidate_map=candidate_map,
            windows=windows,
            initial_balance=initial_balance,
        )


# ── Entry point ───────────────────────────────────────────────────────────────

def run(client: Client, initial_balance: float = 10_000.0) -> None:
    """
    Full backtest run:
      1. Fetch *BACKTEST_DAYS* of 1H history
      2. 70/30 walk-forward split
      3. Run simulation on each split
      4. Print metrics + ASCII chart
      5. Save equity.csv
    """
    df = fetch_historical(client, config.BACKTEST_DAYS)

    split_idx = int(len(df) * 0.70)
    df_in = df.iloc[:split_idx].copy()
    df_out = df.iloc[split_idx:].copy()

    print(f"\n[Backtest] Total candles : {len(df)}")
    print(f"[Backtest] In-sample     : {len(df_in)} candles  "
          f"({df_in.index[0].date()} → {df_in.index[-1].date()})")
    print(f"[Backtest] Out-of-sample : {len(df_out)} candles  "
          f"({df_out.index[0].date()} → {df_out.index[-1].date()})")

    # ── In-sample run ─────────────────────────────────────────────────────────
    in_trades, in_equity = _run_simulation(df_in, initial_balance)
    in_metrics = _compute_metrics(in_trades, in_equity, initial_balance, "IN-SAMPLE (70%)")
    _print_metrics(in_metrics)

    print("  Equity Curve (in-sample):")
    print(_ascii_chart(in_equity))
    _print_attribution(in_trades, "IN-SAMPLE (70%)")

    # ── Out-of-sample run ─────────────────────────────────────────────────────
    # Start with the ending balance of the in-sample run
    out_start_balance = in_equity[-1] if in_equity else initial_balance
    out_trades, out_equity = _run_simulation(df_out, out_start_balance)
    out_metrics = _compute_metrics(out_trades, out_equity, out_start_balance, "OUT-OF-SAMPLE (30%)")
    _print_metrics(out_metrics)

    print("  Equity Curve (out-of-sample):")
    print(_ascii_chart(out_equity))
    _print_attribution(out_trades, "OUT-OF-SAMPLE (30%)")

    # ── Overfitting indicator ─────────────────────────────────────────────────
    if in_metrics.total_trades > 0 and out_metrics.total_trades > 0:
        wr_diff = abs(in_metrics.win_rate_pct - out_metrics.win_rate_pct)
        pf_diff = abs(in_metrics.profit_factor - out_metrics.profit_factor)
        print("  Walk-forward health check:")
        print(f"    Win-rate delta     : {wr_diff:.2f}%  "
              f"({'OK' if wr_diff < 15 else 'POSSIBLE OVERFIT'})")
        print(f"    Profit-factor delta: {pf_diff:.4f}  "
              f"({'OK' if pf_diff < 0.5 else 'POSSIBLE OVERFIT'})")
        print()

    # ── Write CSV (all trades combined) ──────────────────────────────────────
    all_trades = in_trades + [
        SimTrade(
            trade_num=t.trade_num + len(in_trades),
            entry_time=t.entry_time,
            entry_price=t.entry_price,
            stop_price=t.stop_price,
            tp_price=t.tp_price,
            size=t.size,
            direction=t.direction,
            signals_fired=t.signals_fired,
            stop_distance=t.stop_distance,
            current_stop=t.current_stop,
            tp_hit=t.tp_hit,
            partial_pnl=t.partial_pnl,
            exit_time=t.exit_time,
            exit_price=t.exit_price,
            pnl_net=t.pnl_net,
            result=t.result,
            exit_reason=t.exit_reason,
            exit_r=t.exit_r,
        )
        for t in out_trades
    ]
    combined_equity = in_equity + out_equity[1:]  # skip duplicate boundary point
    _write_csv(all_trades, combined_equity, initial_balance)

    # ── Trade forensics ───────────────────────────────────────────────────────
    import forensics as foren
    foren.run_forensics(df, in_trades, out_trades)

    # Research tooling is backtest-only and never affects live execution.
    if config.RUN_WALK_FORWARD_RESEARCH:
        run_walk_forward_research(df, initial_balance)

    # Entry-profile overlays are a separate research layer on top of setups.
    if config.RUN_ENTRY_PROFILE_RESEARCH:
        run_entry_profile_research(df, initial_balance)
    if config.RUN_ENTRY_ALTERNATIVE_HORIZONS:
        run_entry_alternative_horizon_research(df, initial_balance)
        run_entry_target_diagnostics(df, initial_balance)
    if config.RUN_ENTRY_DIAGNOSIS:
        run_entry_diagnosis()
    if config.RUN_RANGE_MEAN_REVERSION:
        run_range_mean_reversion_research(df, initial_balance)

    if config.RUN_SIGNAL_EXPERIMENTS:
        run_signal_experiments(df_in, df_out, initial_balance)

    # Keep optimization passes opt-in so the default backtest flow can stop at
    # the current-candidate validation and forensics report.
    if config.RUN_EXIT_EXPERIMENTS:
        run_experiments(df_in, df_out, initial_balance)

    # ROBUSTNESS walk-forward — regime-stratified, all 9 structural improvements.
    if config.RUN_ROBUSTNESS_WALK_FORWARD:
        from walk_forward_by_regime import run_regime_walk_forward
        from tiered_exit import ExitConfig
        exit_cfg = ExitConfig(
            enable_partial_tp=config.ENABLE_PARTIAL_TP,
            enable_time_stop=config.ENABLE_TIME_STOP,
            enable_momentum_exit=config.ENABLE_MOMENTUM_EXIT,
        )
        run_regime_walk_forward(df, initial_balance=initial_balance, exit_config=exit_cfg)
