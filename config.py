"""
config.py — load and validate all environment variables.
All other modules import from here; no other module calls load_dotenv().
"""
from __future__ import annotations

import os
import sys
from dotenv import load_dotenv

load_dotenv()


# ── helpers ───────────────────────────────────────────────────────────────────

def _get(key: str, default: str | None = None) -> str:
    val = os.getenv(key, default)
    if val is None:
        sys.exit(f"[config] Required env var '{key}' is not set. See .env.example.")
    return val


def _float(key: str, default: float) -> float:
    raw = os.getenv(key, str(default))
    try:
        return float(raw)
    except ValueError:
        sys.exit(f"[config] '{key}' must be a float, got: {raw!r}")


def _int(key: str, default: int) -> int:
    raw = os.getenv(key, str(default))
    try:
        return int(raw)
    except ValueError:
        sys.exit(f"[config] '{key}' must be an integer, got: {raw!r}")


def _bool(key: str, default: bool) -> bool:
    return os.getenv(key, str(default)).strip().lower() in ("true", "1", "yes")


# ── Binance credentials (optional at import time; validated in main.py) ───────
BINANCE_API_KEY: str = os.getenv("BINANCE_API_KEY", "")
BINANCE_SECRET_KEY: str = os.getenv("BINANCE_SECRET_KEY", "")
TESTNET: bool = _bool("TESTNET", True)

# ── Trading ───────────────────────────────────────────────────────────────────
SYMBOL: str = os.getenv("SYMBOL", "BTCUSDT")
INTERVAL: str = os.getenv("INTERVAL", "1h")

# ── Strategy ─────────────────────────────────────────────────────────────────
CONSENSUS_THRESHOLD: float = _float("CONSENSUS_THRESHOLD", 0.65)

# ── Risk management ───────────────────────────────────────────────────────────
RISK_PER_TRADE: float = _float("RISK_PER_TRADE", 0.01)
MAX_POSITION_PCT: float = _float("MAX_POSITION_PCT", 0.05)

ATR_STOP_MULTIPLIER: float = 1.5   # stop_distance = ATR * multiplier
TP_RR_RATIO: float = 2.0           # TP = entry + stop_distance * ratio

# ── Fees / slippage ───────────────────────────────────────────────────────────
MAKER_FEE: float = 0.001    # 0.1 %
SLIPPAGE: float = 0.001     # 0.1 %

# ── Backtest ─────────────────────────────────────────────────────────────────
BACKTEST_DAYS: int = _int("BACKTEST_DAYS", 180)
RUN_EXIT_EXPERIMENTS: bool = _bool("RUN_EXIT_EXPERIMENTS", False)
RUN_SIGNAL_EXPERIMENTS: bool = _bool("RUN_SIGNAL_EXPERIMENTS", False)
RUN_WALK_FORWARD_RESEARCH: bool = _bool("RUN_WALK_FORWARD_RESEARCH", False)
RUN_RANGE_EXIT_RESEARCH: bool = _bool("RUN_RANGE_EXIT_RESEARCH", False)
RUN_ENTRY_PROFILE_RESEARCH: bool = _bool("RUN_ENTRY_PROFILE_RESEARCH", False)
RUN_ENTRY_ALTERNATIVE_HORIZONS: bool = _bool("RUN_ENTRY_ALTERNATIVE_HORIZONS", False)
RUN_ENTRY_DIAGNOSIS: bool = _bool("RUN_ENTRY_DIAGNOSIS", False)
RUN_RANGE_MEAN_REVERSION: bool = _bool("RUN_RANGE_MEAN_REVERSION", False)

# ── Research-only setup family config ────────────────────────────────────────
# These values are used only by backtest research helpers and never by live mode.
RESEARCH_BREAKOUT_LOOKBACK: int = _int("RESEARCH_BREAKOUT_LOOKBACK", 20)
RESEARCH_BREAKOUT_MIN_VOLUME_RATIO: float = _float("RESEARCH_BREAKOUT_MIN_VOLUME_RATIO", 1.5)
RESEARCH_BREAKOUT_REQUIRE_RETEST: bool = _bool("RESEARCH_BREAKOUT_REQUIRE_RETEST", False)
RESEARCH_PULLBACK_LOOKBACK: int = _int("RESEARCH_PULLBACK_LOOKBACK", 12)
RESEARCH_RECLAIM_LOOKBACK: int = _int("RESEARCH_RECLAIM_LOOKBACK", 3)
RESEARCH_RANGE_LOOKBACK: int = _int("RESEARCH_RANGE_LOOKBACK", 24)
RESEARCH_RANGE_MIN_VWAP_BUCKET: str = os.getenv("RESEARCH_RANGE_MIN_VWAP_BUCKET", "medium").strip().lower()
if RESEARCH_RANGE_MIN_VWAP_BUCKET not in ("close", "medium", "far"):
    sys.exit(
        "[config] 'RESEARCH_RANGE_MIN_VWAP_BUCKET' must be one of "
        "'close', 'medium', or 'far'."
    )

# ── Operational ──────────────────────────────────────────────────────────────
LOOKBACK_CANDLES: int = _int("LOOKBACK_CANDLES", 1000)  # 1000 1H -> ~500 2H bars after resampling, clears 202-bar warmup gate
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()

# ── Indicator periods (not user-configurable but centralised here) ─────────--
ADX_PERIOD: int = 14
ATR_PERIOD: int = 14
EMA_FAST: int = 9
EMA_SLOW: int = 21
MACD_FAST: int = 12
MACD_SLOW: int = 26
MACD_SIGNAL: int = 9
RSI_PERIOD: int = 14
STOCH_K: int = 14
STOCH_D: int = 3
STOCH_SMOOTH_K: int = 3
BB_PERIOD: int = 20
BB_STD: float = 2.0
VOLUME_MA_PERIOD: int = 20

# ── Regime thresholds ─────────────────────────────────────────────────────────
ADX_TREND_THRESHOLD: float = 25.0
ATR_HIGH_VOL_THRESHOLD_PCT: float = 3.0   # ATR as % of close

# ── Staged exit parameters (switchable for experiments) ───────────────────────
STALL_EXIT_ENABLED: bool = False      # F2: stall exit disabled
STALL_CANDLES: int = 6                # stall exit if stuck this many candles
STALL_R_THRESHOLD: float = 0.3        # ...and profit_R below this
TIME_CANDLES: int = 30                # time exit after N candles
TIME_R_THRESHOLD: float = 0.5         # ...and profit_R below this
DEFAULT_STAGE_B_R: float = 0.8        # profit_R to activate Stage-B trail
DEFAULT_STAGE_B_ATR_MULT: float = 1.2  # Stage-B trail width (×ATR)
DEFAULT_PARTIAL_TP_R: float = 1.5     # F2: partial TP target (×R)
STAGE_C_ATR_MULT: float = 1.5         # F2: Stage-C trail width after partial TP
BE_OFFSET_R: float = 0.1              # stop moves to entry + BE_OFFSET_R×R after TP

# ── Entry quality gates ───────────────────────────────────────────────────────
REQUIRE_MACD: bool = True             # MACD must fire to approve any entry


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  [ROBUSTNESS]  — 9 structural improvements, all additive / opt-in           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# 1. Regime filter ─────────────────────────────────────────────────────────────
ADX_RANGING_THRESHOLD: float = _float("ADX_RANGING_THRESHOLD", 20.0)
# ADX_TREND_THRESHOLD=25 already defined above; used as "trending" boundary
ATR_HIGH_VOL_MULTIPLIER: float = _float("ATR_HIGH_VOL_MULTIPLIER", 1.5)
ATR_HIGH_VOL_PERIOD: int = _int("ATR_HIGH_VOL_PERIOD", 20)

# 2. Volatility-scaled TP/SL ───────────────────────────────────────────────────
TP_ATR_MULTIPLIER: float = _float("TP_ATR_MULTIPLIER", 1.5)
SL_ATR_MULTIPLIER: float = _float("SL_ATR_MULTIPLIER", 1.0)
MIN_RR_RATIO: float = _float("MIN_RR_RATIO", 1.3)

# 3. Walk-forward by regime ────────────────────────────────────────────────────
WF_WINDOW_DAYS: int = _int("WF_WINDOW_DAYS", 90)
WF_BACKTEST_DAYS: int = _int("WF_BACKTEST_DAYS", 1460)
WF_MIN_TRADES: int = _int("WF_MIN_TRADES", 15)
WF_PASS_PCT: float = _float("WF_PASS_PCT", 0.70)
WF_MIN_PF: float = _float("WF_MIN_PF", 1.2)
RUN_ROBUSTNESS_WALK_FORWARD: bool = _bool("RUN_ROBUSTNESS_WALK_FORWARD", False)

# 4. Tiered exit ───────────────────────────────────────────────────────────────
MAX_TRADE_BARS: int = _int("MAX_TRADE_BARS", 12)
PARTIAL_TP_LEVEL: float = _float("PARTIAL_TP_LEVEL", 0.75)
PARTIAL_TP_CLOSE_PCT: float = _float("PARTIAL_TP_CLOSE_PCT", 0.40)
MOMENTUM_EXIT_RSI_PERIOD: int = _int("MOMENTUM_EXIT_RSI_PERIOD", 7)
MOMENTUM_EXIT_MIN_R: float = _float("MOMENTUM_EXIT_MIN_R", 0.5)
ENABLE_PARTIAL_TP: bool = _bool("ENABLE_PARTIAL_TP", False)
ENABLE_TIME_STOP: bool = _bool("ENABLE_TIME_STOP", False)
ENABLE_MOMENTUM_EXIT: bool = _bool("ENABLE_MOMENTUM_EXIT", False)
ENABLE_ATR_TRAIL: bool = _bool("ENABLE_ATR_TRAIL", False)
ATR_TRAIL_ACTIVATION_R: float = _float("ATR_TRAIL_ACTIVATION_R", 0.5)
ATR_TRAIL_MULTIPLIER: float = _float("ATR_TRAIL_MULTIPLIER", 1.0)
ENABLE_VOLUME_GATE_MOMENTUM: bool = _bool("ENABLE_VOLUME_GATE_MOMENTUM", False)
VOLUME_GATE_MIN_RATIO: float = _float("VOLUME_GATE_MIN_RATIO", 1.2)    # vol > N×avg to allow momentum exit
NATR_MEDIUM_MIN: float = _float("NATR_MEDIUM_MIN", 0.003)
NATR_MEDIUM_MAX: float = _float("NATR_MEDIUM_MAX", 0.007)
# Tier 1.9: 2-bar close momentum-fade exit (no RSI required, pure price action)
ENABLE_MOMENTUM_FADE_EXIT: bool = _bool("ENABLE_MOMENTUM_FADE_EXIT", False)
MOMENTUM_FADE_MIN_R: float = _float("MOMENTUM_FADE_MIN_R", 0.75)
# Partial exit at exactly 1.0R with remainder trailed by ATR
ENABLE_PARTIAL_1R: bool = _bool("ENABLE_PARTIAL_1R", False)
PARTIAL_1R_CLOSE_PCT: float = _float("PARTIAL_1R_CLOSE_PCT", 0.50)
# Anti-chase filter: block entries with extended 3c AND 6c pre-entry moves
ANTI_CHASE_ENABLED: bool = _bool("ANTI_CHASE_ENABLED", False)
ANTI_CHASE_3C_LEVEL: str = os.getenv("ANTI_CHASE_3C_LEVEL", "high").strip().lower()   # bucket threshold
ANTI_CHASE_6C_LEVEL: str = os.getenv("ANTI_CHASE_6C_LEVEL", "high").strip().lower()
# Regime gate: allow HIGH_VOLATILITY entries alongside RANGING
REGIME_GATE_MR: bool = _bool("REGIME_GATE_MR", False)   # gate range-MR entries on RANGING/HIGH_VOL
REGIME_GATE_ALLOW_HIGH_VOL: bool = _bool("REGIME_GATE_ALLOW_HIGH_VOL", True)  # include HIGH_VOLATILITY
# Entry confirmation buffer enabled in research simulations
ENTRY_BUFFER_RESEARCH: bool = _bool("ENTRY_BUFFER_RESEARCH", False)

# 5. Sample guard ──────────────────────────────────────────────────────────────
MIN_WINDOW_TRADES: int = _int("MIN_WINDOW_TRADES", 20)
VALID_WINDOW_PCT: float = _float("VALID_WINDOW_PCT", 0.70)

# 6. Entry confirmation buffer ─────────────────────────────────────────────────
ENTRY_BUFFER_PCT: float = _float("ENTRY_BUFFER_PCT", 0.0015)

# ── Range mean-reversion live strategy ────────────────────────────────────────
# Thresholds calibrated from 1460-day 2H walk-forward research (May 2026).
# ATR% = ATR / close × 100; volume_ratio = volume / 20-bar volume MA.
ENABLE_RANGE_MR: bool    = _bool("ENABLE_RANGE_MR",    False)
RMR_ATR_LOW_PCT: float   = _float("RMR_ATR_LOW_PCT",   0.30)   # ATR% below → low bucket
RMR_ATR_HIGH_PCT: float  = _float("RMR_ATR_HIGH_PCT",  0.60)   # ATR% above → high bucket
RMR_VOL_LOW: float       = _float("RMR_VOL_LOW",       1.30)   # vol ratio below → low bucket
RMR_VOL_HIGH: float      = _float("RMR_VOL_HIGH",      2.00)   # vol ratio above → high bucket
RMR_VWAP_FAR_R: float    = _float("RMR_VWAP_FAR_R",    1.00)   # VWAP dist (in R) above → far
RMR_TP_RR_RATIO: float   = _float("RMR_TP_RR_RATIO",   1.50)   # TP = entry + stop × ratio
# ADX gate for RMR entries: only fire mean-reversion when ADX < this threshold.
# Lower than ADX_TREND_THRESHOLD (25) to restrict RMR to more truly-ranging markets.
RMR_ADX_THRESHOLD: float = _float("RMR_ADX_THRESHOLD", 20.0)
# Trend-following path: when ADX > RMR_ADX_THRESHOLD, optionally fire a LONG
# trend entry using EMA crossover + RSI confirmation.  Disabled by default
# (not yet walk-forward validated).
RMR_TREND_ENTRY: bool       = _bool("RMR_TREND_ENTRY",       False)
RMR_TREND_RSI_MIN: float    = _float("RMR_TREND_RSI_MIN",    55.0)  # RSI floor for trend entries
RMR_TREND_ADX_MAX: float    = _float("RMR_TREND_ADX_MAX",    40.0)  # skip ultra-strong trends

# 7. Volatility band gate for shorts ───────────────────────────────────────────
SHORT_MAX_NATR: float = _float("SHORT_MAX_NATR", 0.008)
SHORT_MIN_NATR: float = _float("SHORT_MIN_NATR", 0.002)

# 8. Multi-timeframe confirmation ──────────────────────────────────────────────
MTF_ADX_THRESHOLD: float = _float("MTF_ADX_THRESHOLD", 22.0)
MTF_BB_SIGMA: float = _float("MTF_BB_SIGMA", 1.5)
MTF_CONFIRMATION: bool = _bool("MTF_CONFIRMATION", True)

# 9. Research mode targets ─────────────────────────────────────────────────────
# Comma-separated setup names to run in --mode research (empty = all)
_RESEARCH_TARGETS_RAW: str = os.getenv("RESEARCH_TARGETS", "")
RESEARCH_TARGETS: list[str] = (
    [t.strip() for t in _RESEARCH_TARGETS_RAW.split(",") if t.strip()]
    if _RESEARCH_TARGETS_RAW.strip()
    else []
)


# ── Auto-pause on drawdown ────────────────────────────────────────────────────
AUTO_PAUSE_ON_DRAWDOWN: bool = _bool("AUTO_PAUSE_ON_DRAWDOWN", False)
MAX_DAILY_LOSS: float = _float("MAX_DAILY_LOSS", 0.02)
MAX_WEEKLY_LOSS: float = _float("MAX_WEEKLY_LOSS", 0.05)
MAX_CONSECUTIVE_LOSSES: int = _int("MAX_CONSECUTIVE_LOSSES", 3)
AUTO_UNPAUSE_DAILY: bool = _bool("AUTO_UNPAUSE_DAILY", True)
AUTO_UNPAUSE_WEEKLY: bool = _bool("AUTO_UNPAUSE_WEEKLY", False)
AUTO_UNPAUSE_CONSECUTIVE_LOSSES: bool = _bool("AUTO_UNPAUSE_CONSECUTIVE_LOSSES", False)
PAUSE_UNTIL_UTC: str = os.getenv("PAUSE_UNTIL_UTC", "")

# ── Trade journal reporting ───────────────────────────────────────────────────
ENABLE_DAILY_REPORT: bool = _bool("ENABLE_DAILY_REPORT", False)
ENABLE_WEEKLY_REPORT: bool = _bool("ENABLE_WEEKLY_REPORT", False)
REPORT_HOUR_UTC: int = _int("REPORT_HOUR_UTC", 23)

# ── Telegram alerts ───────────────────────────────────────────────────────────
ENABLE_TELEGRAM_ALERTS: bool = _bool("ENABLE_TELEGRAM_ALERTS", False)
TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")

# ── Trade quality filter engine ──────────────────────────────────────────────
# Master switch — set false to bypass all filters (not recommended in LIVE)
ENABLE_TRADE_FILTERS: bool = _bool("ENABLE_TRADE_FILTERS", True)

# A. BTC trend alignment — reject alt longs if BTC strongly bearish
BTC_ALIGNMENT_FILTER: bool = _bool("BTC_ALIGNMENT_FILTER", True)

# B. Volume quality — reject low-volume or fake-spike setups
VOLUME_QUALITY_FILTER: bool = _bool("VOLUME_QUALITY_FILTER", True)
VOLUME_QUALITY_MIN_RATIO: float = _float("VOLUME_QUALITY_MIN_RATIO", 0.5)

# C. Candle extension — reject entries after parabolic candles
CANDLE_EXTENSION_FILTER: bool = _bool("CANDLE_EXTENSION_FILTER", True)
CANDLE_EXTENSION_ATR_MULTIPLE: float = _float("CANDLE_EXTENSION_ATR_MULTIPLE", 2.5)

# D. Spread filter — reject if bid/ask spread too wide
SPREAD_FILTER: bool = _bool("SPREAD_FILTER", True)
MAX_SPREAD_BPS: float = _float("MAX_SPREAD_BPS", 10.0)   # basis points

# E. Volatility compression / breakout filter
VOLATILITY_COMPRESSION_FILTER: bool = _bool("VOLATILITY_COMPRESSION_FILTER", True)

# F. Session filter — grade penalty for low-quality sessions (no hard block by default)
SESSION_FILTER_ENABLED: bool = _bool("SESSION_FILTER_ENABLED", True)

# G. News risk filter — block entries near macro events
NEWS_RISK_FILTER: bool = _bool("NEWS_RISK_FILTER", True)

# H. Symbol cooldown — candles to wait after a stop-loss (1 candle = 1H)
SYMBOL_COOLDOWN_CANDLES: int = _int("SYMBOL_COOLDOWN_CANDLES", 3)

# ── Trade grading ─────────────────────────────────────────────────────────────
# Minimum grade to execute a trade. Options: A+ A B C
# Default: B → A+, A, and B grades execute (relaxed from A to increase frequency).
MIN_TRADE_GRADE: str = os.getenv("MIN_TRADE_GRADE", "B").strip()
if MIN_TRADE_GRADE not in ("A+", "A", "B", "C"):
    MIN_TRADE_GRADE = "B"

# Adaptive filters: tighten grade after drawdowns, relax after stable performance
ENABLE_ADAPTIVE_FILTERS: bool = _bool("ENABLE_ADAPTIVE_FILTERS", False)

# ── Adaptive intelligence layer ───────────────────────────────────────────────
# Engine ranking: prioritise strong engines, downweight weak ones
ENABLE_ADAPTIVE_ENGINE_WEIGHTING: bool = _bool("ENABLE_ADAPTIVE_ENGINE_WEIGHTING", False)

# Equity curve protection: tighten grades on drawdown, relax on recovery
ENABLE_EQUITY_PROTECTION: bool = _bool("ENABLE_EQUITY_PROTECTION", False)

# Adaptive grades: tighten after losing streaks, relax after stable periods
ENABLE_ADAPTIVE_GRADES: bool = _bool("ENABLE_ADAPTIVE_GRADES", False)

# Auto-disable engines with strongly negative expectancy (re-enabled after cooldown)
ENABLE_AUTO_DISABLE_ENGINES: bool = _bool("ENABLE_AUTO_DISABLE_ENGINES", False)

# Correlation guard: prevent multiple highly correlated simultaneous positions
ENABLE_CORRELATION_GUARD: bool = _bool("ENABLE_CORRELATION_GUARD", True)
MAX_CORRELATED_POSITIONS: int = _int("MAX_CORRELATED_POSITIONS", 2)

# ── Portfolio-aware adaptive layer ────────────────────────────────────────────
# Engine governor: tier system (TRUSTED/NEUTRAL/PROBATION) with promotion/demotion
ENABLE_ENGINE_GOVERNOR: bool = _bool("ENABLE_ENGINE_GOVERNOR", False)

# Sentiment filter: CoinGecko-based sentiment modifier on rank_score (filter-only)
ENABLE_SENTIMENT_FILTER: bool = _bool("ENABLE_SENTIMENT_FILTER", False)

# Portfolio brain: sector exposure tracking and portfolio health scoring
ENABLE_PORTFOLIO_BRAIN: bool = _bool("ENABLE_PORTFOLIO_BRAIN", False)
MAX_SECTOR_EXPOSURE: float = _float("MAX_SECTOR_EXPOSURE", 0.40)

# Market avoidance: dangerous environment detection with grade floor adjustments
ENABLE_MARKET_AVOIDANCE: bool = _bool("ENABLE_MARKET_AVOIDANCE", False)

# Learning memory: engine × regime × session performance modifiers
ENABLE_LEARNING_MEMORY: bool = _bool("ENABLE_LEARNING_MEMORY", False)

# Shadow engines: paper-trading simulation (NEVER places live trades)
ENABLE_SHADOW_ENGINES: bool = _bool("ENABLE_SHADOW_ENGINES", False)

# Weekly intelligence report: sent Sunday UTC at 08:00+
ENABLE_WEEKLY_INTELLIGENCE: bool = _bool("ENABLE_WEEKLY_INTELLIGENCE", False)

# ── Operational intelligence layer ───────────────────────────────────────────
# Anomaly detection: auto-reduce aggressiveness or pause on market/system events
ENABLE_ANOMALY_DETECTION: bool = _bool("ENABLE_ANOMALY_DETECTION", False)

# Confidence scoring: 0-100 daily score adjusts risk scale and grade floor
ENABLE_CONFIDENCE_SCORE: bool = _bool("ENABLE_CONFIDENCE_SCORE", False)

# Live vs shadow comparative analytics: persist shadow trades to DB
ENABLE_SHADOW_ANALYTICS: bool = _bool("ENABLE_SHADOW_ANALYTICS", False)

# ── Telegram command bot (operations console) ─────────────────────────────────
# Set ENABLE_TELEGRAM_BOT=true in .env to activate the background polling daemon.
ENABLE_TELEGRAM_BOT: bool = _bool("ENABLE_TELEGRAM_BOT", False)

# Trade approval mode — bot sends APPROVE/REJECT before every entry.
# Requires ENABLE_TELEGRAM_BOT=true.  Times out (auto-reject) after N seconds.
MANUAL_APPROVAL_MODE: bool = _bool("MANUAL_APPROVAL_MODE", False)
MANUAL_APPROVAL_TIMEOUT: int = _int("MANUAL_APPROVAL_TIMEOUT", 300)

# /panic command: if true, also writes TESTNET=true to .env (requires bot restart)
PANIC_SWITCH_ENABLE_TESTNET: bool = _bool("PANIC_SWITCH_ENABLE_TESTNET", False)

# Voice alerts via gTTS (requires: pip install gtts)
ENABLE_TELEGRAM_VOICE_ALERTS: bool = _bool("ENABLE_TELEGRAM_VOICE_ALERTS", False)

# Daily PDF report — sent at REPORT_HOUR_UTC if ENABLE_TELEGRAM_PDF_REPORT=true
ENABLE_TELEGRAM_PDF_REPORT: bool = _bool("ENABLE_TELEGRAM_PDF_REPORT", False)

# Hourly market summary interval (hours, 0 = disabled)
TELEGRAM_SUMMARY_INTERVAL_HOURS: int = _int("TELEGRAM_SUMMARY_INTERVAL_HOURS", 1)

# ── Multi-pair scanner ────────────────────────────────────────────────────────
_SYMBOLS_RAW: str = os.getenv("SYMBOLS", "")
SYMBOLS: list[str] = (
    [s.strip() for s in _SYMBOLS_RAW.split(",") if s.strip()]
    if _SYMBOLS_RAW.strip()
    else [SYMBOL]
)

# ── Expanded symbol set ───────────────────────────────────────────────────────
# When true, appends liquid alt-coins to the scan list (deduped).
ENABLE_EXPANDED_SYMBOLS: bool = _bool("ENABLE_EXPANDED_SYMBOLS", False)
_EXPANDED_SYMBOLS: list[str] = [
    "XRPUSDT", "DOGEUSDT", "ADAUSDT", "LINKUSDT",
    "AVAXUSDT", "SUIUSDT", "TONUSDT",
]
if ENABLE_EXPANDED_SYMBOLS:
    SYMBOLS = list(dict.fromkeys(SYMBOLS + _EXPANDED_SYMBOLS))

# ── Live setup engines (all disabled by default) ─────────────────────────────
# Each engine is independent and appends to the existing scan pipeline.
# Enable individually via .env once ready to validate live.
ENABLE_PULLBACK_SETUP: bool      = _bool("ENABLE_PULLBACK_SETUP",      False)
ENABLE_BREAKOUT_SETUP: bool      = _bool("ENABLE_BREAKOUT_SETUP",      False)
ENABLE_NY_MOMENTUM_SETUP: bool   = _bool("ENABLE_NY_MOMENTUM_SETUP",   False)
ENABLE_MEAN_REVERSION_SETUP: bool = _bool("ENABLE_MEAN_REVERSION_SETUP", False)

# Soft 15-minute confirmation check — logs caution and reduces rank_score by 15
# but NEVER blocks a trade (hard_fail=False by design).
ENABLE_15M_CONFIRMATION: bool    = _bool("ENABLE_15M_CONFIRMATION",    False)

MAX_OPEN_TRADES: int = _int("MAX_OPEN_TRADES", 1)
MAX_TOTAL_RISK: float = _float("MAX_TOTAL_RISK", 0.01)


# ── Dashboard (read-only web UI) ─────────────────────────────────────────────
ENABLE_DASHBOARD: bool = _bool("ENABLE_DASHBOARD", False)
DASHBOARD_HOST: str = os.getenv("DASHBOARD_HOST", "127.0.0.1")
DASHBOARD_PORT: int = _int("DASHBOARD_PORT", 8080)
# Required — dashboard refuses to start if empty
DASHBOARD_PASSWORD: str = os.getenv("DASHBOARD_PASSWORD", "")

# ── Arbitrage scanner (watch-only) ───────────────────────────────────────────
ENABLE_ARBITRAGE_SCANNER: bool = _bool("ENABLE_ARBITRAGE_SCANNER", False)
# ARB_AUTO_TRADE is hard-locked to False regardless of any env var setting.
# If set to True in .env the scanner logs a safety warning and ignores it.
ARB_AUTO_TRADE: bool = False
ARB_MIN_NET_PROFIT_PCT: float = _float("ARB_MIN_NET_PROFIT_PCT", 0.35)
ARB_FEE_PCT: float = _float("ARB_FEE_PCT", 0.10)           # per-leg fee %
ARB_SLIPPAGE_BUFFER_PCT: float = _float("ARB_SLIPPAGE_BUFFER_PCT", 0.15)  # total slippage %
ARB_MAX_SPREAD_PCT: float = _float("ARB_MAX_SPREAD_PCT", 0.20)  # reject illiquid routes
ARB_TOP_N: int = _int("ARB_TOP_N", 5)                       # max alerts per scan

# ── Trend scanner ─────────────────────────────────────────────────────────────
ENABLE_TREND_SCANNER: bool = _bool("ENABLE_TREND_SCANNER", False)
TREND_SCANNER_TOP_N: int = _int("TREND_SCANNER_TOP_N", 5)
TREND_MIN_QUOTE_VOLUME: float = _float("TREND_MIN_QUOTE_VOLUME", 10_000_000)
TREND_MAX_SPREAD_PCT: float = _float("TREND_MAX_SPREAD_PCT", 0.20)
TREND_MIN_VOLUME_SPIKE: float = _float("TREND_MIN_VOLUME_SPIKE", 1.8)
TREND_ALERT_SCORE_THRESHOLD: float = _float("TREND_ALERT_SCORE_THRESHOLD", 75)
ENABLE_TREND_AUTO_TRADE: bool = _bool("ENABLE_TREND_AUTO_TRADE", False)
TREND_MAX_24H_MOVE_PCT: float = _float("TREND_MAX_24H_MOVE_PCT", 25.0)
TREND_MAX_WICK_RATIO: float = _float("TREND_MAX_WICK_RATIO", 3.0)
TREND_MIN_15M_CONFIRMATION: float = _float("TREND_MIN_15M_CONFIRMATION", 0.5)
TREND_MIN_1H_CONFIRMATION: float = _float("TREND_MIN_1H_CONFIRMATION", 1.0)
TREND_MIN_4H_CONFIRMATION: float = _float("TREND_MIN_4H_CONFIRMATION", 2.0)
TREND_REQUIRE_MULTI_TIMEFRAME: bool = _bool("TREND_REQUIRE_MULTI_TIMEFRAME", True)
TREND_REQUIRE_VOLUME_CONFIRMATION: bool = _bool("TREND_REQUIRE_VOLUME_CONFIRMATION", True)


def validate_live_credentials() -> None:
    """Call this in live trading mode to ensure API keys are present."""
    if not BINANCE_API_KEY or not BINANCE_SECRET_KEY:
        sys.exit(
            "[config] BINANCE_API_KEY and BINANCE_SECRET_KEY must be set "
            "for live trading. See .env.example."
        )
