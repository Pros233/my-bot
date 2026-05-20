"""
main.py — Entry point.

Usage:
  python main.py                    # live trading (Binance Testnet)
  python main.py --mode backtest    # full pipeline (current behaviour)
  python main.py --mode research    # only profiles listed in RESEARCH_TARGETS
  python main.py --mode regime      # RegimeClassifier only → regime_log.csv
  python main.py --mode validate    # robustness gate checks on existing CSVs

Legacy flag preserved for backwards compatibility:
  python main.py --backtest         # same as --mode backtest
"""
from __future__ import annotations

import argparse
import signal
import socket
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional

import pandas as pd
import pandas_ta as ta  # noqa: F401 — register .ta accessor globally
from binance.client import Client

import alerts
import config
import pause_manager
import performance
import trade_journal
import arbitrage_scanner
import trend_scanner
import trend_outcome_tracker
import regime as reg
import consensus as con
import risk
import logger
import dashboard
import backtest as bt
import telegram_bot
import trade_filters
import trade_grader
from executor import ExecutionEngine
from strategies.range_mr import get_signal_2h, resample_1h_to_2h
import rejection_analytics
import setup_engines
import engine_ranker
import market_state as mstate
import equity_protection
import correlation_guard
import engine_governor
import shadow_engine
import sentiment_engine
import portfolio_brain
import market_avoidance
import learning_memory
import weekly_intelligence_report as wir
import anomaly_detector
import confidence_score
import shadow_analytics
import social_sentiment
import funding_arb
import grid_engine
import defi_signals


# ── Data helpers ──────────────────────────────────────────────────────────────

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


def fetch_candles(client: Client, data_client: Client | None = None) -> pd.DataFrame:
    klines = (data_client or client).get_klines(
        symbol=config.SYMBOL,
        interval=config.INTERVAL,
        limit=config.LOOKBACK_CANDLES,
    )
    df = _klines_to_df(klines)
    # Drop the still-forming (unclosed) candle — last row
    return df.iloc[:-1]


def get_usdt_balance(client: Client) -> float:
    account = client.get_account()
    for asset in account["balances"]:
        if asset["asset"] == "USDT":
            return float(asset["free"])
    return 0.0


# ── Candle timing ─────────────────────────────────────────────────────────────

def _seconds_until_next_hour() -> float:
    """Seconds until the next UTC hourly candle close (with 5 s buffer)."""
    now = datetime.now(timezone.utc)
    next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    return max((next_hour - now).total_seconds() + 5, 0.0)


# ── Scanner types ─────────────────────────────────────────────────────────────

@dataclass
class ScanResult:
    symbol: str
    df: pd.DataFrame
    trend: str = ""
    vol: str = ""
    adx: float = 0.0
    atr_pct: float = 0.0
    consensus: object = None   # con.ConsensusResult | None
    rmr: object = None         # RangeMRSignal | None
    engine_signal: object = None  # EngineSignal | None
    market_state: object = None   # market_state.MarketState | None
    decision: str = "NO_DATA"  # BUY | RMR_LONG | ENGINE_LONG | HOLD | SKIP | NO_DATA
    rank_score: float = 0.0
    rank_reason: str = ""
    reject_reason: str = ""


def _fetch_candles_for(
    client: Client,
    data_client: Optional[Client],
    symbol: str,
) -> pd.DataFrame:
    klines = (data_client or client).get_klines(
        symbol=symbol,
        interval=config.INTERVAL,
        limit=config.LOOKBACK_CANDLES,
    )
    return _klines_to_df(klines).iloc[:-1]  # drop still-forming candle


def scan_symbol(
    client: Client,
    data_client: Optional[Client],
    symbol: str,
    now_utc: datetime,
) -> ScanResult:
    """Fetch candles and evaluate regime, consensus, and RMR for one symbol."""
    try:
        df = _fetch_candles_for(client, data_client, symbol)
    except Exception as exc:
        logger.log_warning(f"SCAN | {symbol} | fetch failed: {exc}")
        return ScanResult(symbol=symbol, df=pd.DataFrame(),
                          decision="NO_DATA", reject_reason=f"fetch error: {exc}")

    if df.empty or len(df) < 50:
        return ScanResult(symbol=symbol, df=df,
                          decision="NO_DATA", reject_reason="insufficient candle data")

    trend, vol = reg.classify(df)
    adx_df = df.ta.adx(length=config.ADX_PERIOD)
    atr_series = df.ta.atr(length=config.ATR_PERIOD)
    adx_val = float(adx_df[f"ADX_{config.ADX_PERIOD}"].iloc[-1])
    atr_val = float(atr_series.iloc[-1])
    atr_pct = (atr_val / float(df["close"].iloc[-1])) * 100.0

    consensus = con.compute(df, trend, vol)

    # Classify market state (non-raising — returns UNKNOWN on error)
    try:
        ms = mstate.classify(df, adx_val, atr_pct, trend, vol)
    except Exception:
        ms = None

    result = ScanResult(
        symbol=symbol, df=df, trend=trend, vol=vol,
        adx=adx_val, atr_pct=atr_pct, consensus=consensus,
        market_state=ms,
    )

    if not reg.regime_allows_trade(trend, vol):
        result.decision = "SKIP"
        result.reject_reason = f"regime {reg.regime_label(trend, vol)}"
        return result

    # ── RMR check (2H, LONG-biased, even hours only) ──────────────────────────
    if config.ENABLE_RANGE_MR and now_utc.hour % 2 == 0:
        df_2h = resample_1h_to_2h(df)
        rmr = get_signal_2h(df_2h)
        result.rmr = rmr
        if rmr.direction == "LONG":
            result.decision = "RMR_LONG"
            result.rank_score = 200.0 + consensus.ratio * 100.0
            result.rank_reason = (
                f"RMR LONG [{rmr.signal_type}] ADX={adx_val:.1f} "
                f"ATR={rmr.atr_bucket}({atr_pct:.2f}%) VOL={rmr.volume_bucket} "
                f"VWAP_dist={rmr.vwap_distance_r:.2f}R"
            )
            return result
        else:
            result.reject_reason = rmr.reject_reason

    # ── Consensus BUY check ───────────────────────────────────────────────────
    if consensus.decision == con.BUY:
        result.decision = "BUY"
        result.rank_score = 100.0 + consensus.ratio * 100.0
        result.rank_reason = (
            f"score={consensus.score:.2f}/{consensus.max_possible:.2f} "
            f"({consensus.ratio * 100:.1f}%)"
        )
    else:
        result.decision = "HOLD"

    # ── Setup engines (all optional, never override existing signal) ──────────
    if result.decision not in ("RMR_LONG", "BUY"):
        _engine_checks = [
            (config.ENABLE_PULLBACK_SETUP,        lambda: setup_engines.get_pullback_signal(df)),
            (config.ENABLE_BREAKOUT_SETUP,        lambda: setup_engines.get_breakout_signal(df)),
            (config.ENABLE_NY_MOMENTUM_SETUP,     lambda: setup_engines.get_ny_momentum_signal(df, now_utc)),
            (config.ENABLE_MEAN_REVERSION_SETUP,  lambda: setup_engines.get_mean_reversion_signal(df)),
            (getattr(config, "ENABLE_INTRADAY_SCALP", False),
                                                  lambda: setup_engines.get_intraday_scalp_signal(df, now_utc)),
        ]
        for _enabled, _engine_fn in _engine_checks:
            if not _enabled:
                continue
            try:
                _esig = _engine_fn()
                if _esig.direction == "LONG":
                    result.decision     = "ENGINE_LONG"
                    result.engine_signal = _esig
                    result.rank_score   = 150.0 + _esig.rank_boost
                    result.rank_reason  = f"[{_esig.engine}] {_esig.reason}"
                    break
            except Exception as _eng_exc:
                logger.log_warning(f"SCAN | {symbol} | engine error (non-critical): {_eng_exc}")

    # ── Soft 15m confirmation check ───────────────────────────────────────────
    if (getattr(config, "ENABLE_15M_CONFIRMATION", False)
            and result.decision in ("RMR_LONG", "BUY", "ENGINE_LONG")):
        try:
            _df15 = _fetch_candles_for(client, data_client, symbol)  # re-uses 1H; 15m not separately fetched here
            _rsi15 = float(_df15["close"].ewm(span=14, adjust=False).mean().iloc[-1])
            _ema21_15 = float(_df15["close"].ewm(span=21, adjust=False).mean().iloc[-1])
            _close15  = float(_df15["close"].iloc[-1])
            _caution = ""
            if _rsi15 > 75:
                _caution = f"15m-proxy RSI={_rsi15:.0f} > 75"
            elif _close15 < _ema21_15:
                _caution = f"15m-proxy close={_close15:.2f} < EMA21={_ema21_15:.2f}"
            if _caution:
                logger.log_info(
                    f"15M_CAUTION | {symbol} | {_caution} — rank_score reduced by 15"
                )
                result.rank_score = max(0.0, result.rank_score - 15.0)
        except Exception as _15m_exc:
            logger.log_warning(f"15m check failed (non-critical): {_15m_exc}")

    return result


# ── Performance report helper ─────────────────────────────────────────────────

def _send_performance_report(
    open_count: int,
    balance: float,
    symbols: list[str],
    report_type: str = "Daily",
) -> None:
    """Pull stats from trades.db and send a Telegram performance report."""
    try:
        now_utc = datetime.now(timezone.utc)
        today = now_utc.strftime("%Y-%m-%d")
        iso = now_utc.isocalendar()
        this_week = f"{iso[0]}-W{iso[1]:02d}"

        alerts.alert_daily_report(
            total_pnl=performance.total_pnl(),
            daily_pnl=performance.daily_pnl(today),
            weekly_pnl=performance.weekly_pnl(this_week),
            total_trades=performance.total_trades(),
            win_rate=performance.win_rate(),
            max_dd_pct=performance.max_drawdown_pct(),
            best_sym=performance.best_symbol() or "—",
            worst_sym=performance.worst_symbol() or "—",
            open_positions=open_count,
            is_paused=pause_manager.is_paused(),
            symbols=symbols,
            report_type=report_type,
        )
    except Exception as exc:
        logger.log_warning(f"Performance report failed (non-critical): {exc}")


# ── Live trading loop ─────────────────────────────────────────────────────────

def live(client: Client, data_client: Client | None = None) -> None:
    symbols = config.SYMBOLS
    engines: dict[str, ExecutionEngine] = {
        sym: ExecutionEngine(client, sym) for sym in symbols
    }

    logger.log_info(
        f"Live trading started — {config.INTERVAL} "
        f"({'TESTNET' if config.TESTNET else 'LIVE'}) "
        f"| symbols: {', '.join(symbols)} | MAX_OPEN_TRADES={config.MAX_OPEN_TRADES}"
    )
    if config.ENABLE_RANGE_MR:
        logger.log_info(
            "RMR strategy active — promoted setup RMR_2H_NO_FAR_VWAP_LONG_BIASED_BLOCK_LONG_HIGHATR_HIGHVOL "
            "(48/168 variants pass 1460-day walk-forward gate, May 2026)."
        )
    else:
        logger.log_warning(
            "MACD+VWAP research setup failed walk-forward validation. "
            "Set ENABLE_RANGE_MR=true to activate the validated RMR strategy."
        )

    # Reconcile orphaned orders on all symbols
    for engine in engines.values():
        engine.sync_open_orders()

    alerts.alert_startup(symbols)

    # ── Start Telegram command bot daemon ────────────────────────────────────
    telegram_bot.start(client, engines)

    # ── Initialise trade journal DB and print historical stats ────────────────
    trade_journal._ensure_db()
    try:
        _n = performance.total_trades()
        if _n > 0:
            logger.log_info(
                f"HISTORICAL STATS | trades={_n} | "
                f"win_rate={performance.win_rate():.1%} | "
                f"total_pnl={performance.total_pnl():+.4f} USDT | "
                f"max_dd={performance.max_drawdown_pct():.2f}% | "
                f"consec_losses={performance.consecutive_losses()}"
            )
        else:
            logger.log_info("HISTORICAL STATS | No trades recorded yet.")
    except Exception as _exc:
        logger.log_warning(f"Could not read historical stats: {_exc}")

    # ── Report tracking (reset per bot run) ───────────────────────────────────
    _last_daily_report: str = ""
    _last_weekly_report: str = ""
    _last_summary_hour: int = -1          # Telegram hourly summary
    _last_pinned_update: float = 0.0      # Telegram pinned dashboard (monotonic)
    _last_rejection_summary_ts: float = 0.0  # 12h rejection analytics Telegram summary
    _last_learning_summary_ts: float = 0.0   # 24h engine learning summary
    _last_governor_check_ts: float = 0.0     # governor tier check (every 4h)
    _last_weekly_intel_ts: float = 0.0       # weekly intelligence report
    _cycle_confidence: float = 50.0          # cached confidence score for this cycle
    _cycle_start_ts: float = 0.0             # for exec delay measurement
    _last_confidence_state: str = ""         # for state-change Telegram alerts
    _last_intel_summary_ts: float = 0.0     # hourly intel: funding arb + social trending
    _last_funding_alert_ts: float = 0.0     # dedup funding arb extreme alerts

    def _cycle_timeout(_signum, _frame):
        raise TimeoutError("Candle cycle timed out (network stall)")

    signal.signal(signal.SIGALRM, _cycle_timeout)
    signal.siginterrupt(signal.SIGALRM, True)

    # Single shutdown handler covers all symbols
    def _shutdown_all(signum, _frame):
        logger.log_error(f"EMERGENCY SHUTDOWN: Signal {signum} received")
        for eng in engines.values():
            eng._cancel_all_orders()
        sys.exit(1)

    signal.signal(signal.SIGINT, _shutdown_all)
    signal.signal(signal.SIGTERM, _shutdown_all)

    while True:
        try:
            # ── Wait for next candle close ────────────────────────────────────
            wait_secs = _seconds_until_next_hour()
            logger.log_info(f"Sleeping {wait_secs:.0f}s until next candle close…")
            wake_at = datetime.now(timezone.utc) + timedelta(seconds=wait_secs)
            while True:
                remaining = (wake_at - datetime.now(timezone.utc)).total_seconds()
                if remaining <= 0:
                    break
                time.sleep(min(30, remaining))

            # Scale hard timeout with symbol count (30 s per symbol, min 90 s)
            signal.alarm(max(90, 30 * len(symbols)))
            try:
                client.session.close()
                if data_client is not None:
                    data_client.session.close()

                # Clock sync once per cycle on the primary engine
                engines[symbols[0]].check_clock_sync()

                now_utc = datetime.now(timezone.utc)
                _cycle_start_ts = time.monotonic()

                # ── Scan all symbols ──────────────────────────────────────────
                results: dict[str, ScanResult] = {
                    sym: scan_symbol(client, data_client, sym, now_utc)
                    for sym in symbols
                }

                # ── Anomaly detection (per symbol, non-raising) ───────────────
                if getattr(config, "ENABLE_ANOMALY_DETECTION", False):
                    _new_anomalies: list[str] = []
                    try:
                        for _sym, _r in results.items():
                            if _r.df.empty:
                                continue
                            _events = anomaly_detector.check_market_anomalies(
                                _r.df, _sym, _r.adx, _r.atr_pct
                            )
                            for _ev in _events:
                                _is_new = anomaly_detector.record_anomaly(_ev)
                                if _is_new:
                                    _new_anomalies.append(
                                        f"*Anomaly* [{_ev.severity}] `{_ev.anomaly_type}` "
                                        f"`{_sym}` — {_ev.message}"
                                    )
                        # Exec delay check
                        _exec_delay = time.monotonic() - _cycle_start_ts
                        _sys_events = anomaly_detector.check_system_anomalies(_exec_delay)
                        for _ev in _sys_events:
                            _is_new = anomaly_detector.record_anomaly(_ev)
                            if _is_new:
                                _new_anomalies.append(
                                    f"*Anomaly* [{_ev.severity}] `{_ev.anomaly_type}` "
                                    f"— {_ev.message}"
                                )
                        # Send Telegram alerts for new anomalies
                        if _new_anomalies and getattr(config, "ENABLE_TELEGRAM_BOT", False):
                            for _amsg in _new_anomalies:
                                try:
                                    telegram_bot.send_rejection_summary(_amsg)
                                except Exception:
                                    pass
                    except Exception as _an_exc:
                        logger.log_warning(f"anomaly_detector cycle error: {_an_exc}")

                # ── Confidence score (computed once per cycle) ─────────────────
                if getattr(config, "ENABLE_CONFIDENCE_SCORE", False):
                    try:
                        _ms_list = [r.market_state for r in results.values() if r.market_state]
                        _open_syms_conf = [s for s, e in engines.items() if e.has_open_position()]
                        _cycle_confidence = confidence_score.compute_confidence(
                            market_states=_ms_list,
                            balance=0.0,   # balance not yet fetched; will re-use below
                            open_positions=_open_syms_conf,
                        )
                        _conf_state = confidence_score.get_confidence_state(_cycle_confidence)
                        if _conf_state == confidence_score.DEFENSIVE:
                            logger.log_info(
                                f"CONFIDENCE | score={_cycle_confidence:.0f} [{_conf_state}] "
                                f"— pausing new entries"
                            )
                        elif _conf_state == confidence_score.CAUTIOUS:
                            logger.log_info(
                                f"CONFIDENCE | score={_cycle_confidence:.0f} [{_conf_state}] "
                                f"— reducing aggressiveness"
                            )
                        # Alert on state change
                        if _conf_state != _last_confidence_state and _last_confidence_state != "":
                            _state_icon = {
                                confidence_score.NORMAL:    "✅",
                                confidence_score.CAUTIOUS:  "⚠",
                                confidence_score.DEFENSIVE: "⛔",
                            }
                            _conf_msg = (
                                f"*Confidence Score* {_state_icon.get(_conf_state,'⚪')}\n"
                                f"`{_last_confidence_state}` → *{_conf_state}*\n"
                                f"Score: `{_cycle_confidence:.0f}/100`"
                            )
                            if getattr(config, "ENABLE_TELEGRAM_BOT", False):
                                try:
                                    telegram_bot.send_rejection_summary(_conf_msg)
                                except Exception:
                                    pass
                        _last_confidence_state = _conf_state
                    except Exception as _cs_exc:
                        logger.log_warning(f"confidence_score cycle error: {_cs_exc}")

                # ── Check open positions with fresh candle data ───────────────
                for sym, engine in engines.items():
                    if engine.has_open_position():
                        r = results[sym]
                        df_sym = r.df if not r.df.empty else None
                        _pos_before_close = engine.position   # save before check
                        exit_price = engine.check_position(df_sym)
                        if exit_price is not None:
                            logger.log_info(f"{sym} position closed at {exit_price:.2f}")
                            # Record stop-loss for cooldown filter
                            if (_pos_before_close and
                                    exit_price <= _pos_before_close.stop_price * 1.002):
                                trade_filters.record_stop_loss(sym)

                # ── Shadow engine evaluation (paper only — NEVER live) ────────
                if getattr(config, "ENABLE_SHADOW_ENGINES", False):
                    try:
                        _prices = {
                            sym: float(results[sym].df["close"].iloc[-1])
                            for sym in symbols
                            if sym in results and not results[sym].df.empty
                        }
                        _shadow_closed = shadow_engine.evaluate_shadows(_prices)
                        for _sc in _shadow_closed:
                            logger.log_info(
                                f"SHADOW_CLOSE | {_sc['engine']} {_sc['symbol']} "
                                f"{_sc['direction']} → {_sc['outcome']} "
                                f"pnl={_sc['pnl_pct']:+.2f}%"
                            )
                    except Exception as _sh_exc:
                        logger.log_warning(f"shadow_engine.evaluate_shadows failed: {_sh_exc}")

                balance = get_usdt_balance(client)
                pause_manager.update_balance(balance)
                open_count = sum(1 for e in engines.values() if e.has_open_position())

                # ── Log per-symbol CYCLE + SCAN lines ────────────────────────
                for sym, r in results.items():
                    if r.consensus is not None:
                        logger.log_cycle(
                            now_utc, r.trend, r.vol, r.adx, r.atr_pct,
                            r.consensus, r.decision,
                        )
                        if r.rmr is not None and r.decision != "RMR_LONG":
                            logger.log_info(
                                f"RMR skip [{r.rmr.signal_type}] ADX={r.adx:.1f}: "
                                f"{r.rmr.reject_reason}"
                            )
                        if r.engine_signal is not None and r.decision == "ENGINE_LONG":
                            logger.log_info(
                                f"ENGINE | {sym} | [{r.engine_signal.engine}] "
                                f"{r.engine_signal.reason}"
                            )
                    score_str = (
                        f"{r.consensus.score:.2f}/{r.consensus.max_possible:.2f} "
                        f"({r.consensus.ratio * 100:.1f}%)"
                        if r.consensus else "N/A"
                    )
                    _ms_str = mstate.describe(r.market_state) if r.market_state else "market_state=N/A"
                    logger.log_info(
                        f"SCAN | {sym} | regime={r.trend}+{r.vol} | ADX={r.adx:.1f} | "
                        f"score={score_str} | decision={r.decision} | "
                        f"{r.rank_reason or r.reject_reason} | {_ms_str}"
                    )

                # ── Engine ranker: auto-disable check (once per cycle) ───────
                try:
                    engine_ranker.check_auto_disable()
                except Exception:
                    pass

                # ── Engine governor: tier check (every 4h) ────────────────────
                if getattr(config, "ENABLE_ENGINE_GOVERNOR", False):
                    _GOV_INTERVAL = 4 * 3600
                    if time.monotonic() - _last_governor_check_ts >= _GOV_INTERVAL:
                        try:
                            _gov_notifs = engine_governor.check_all_tiers()
                            for _gn in _gov_notifs:
                                if getattr(config, "ENABLE_TELEGRAM_BOT", False):
                                    telegram_bot.send_rejection_summary(_gn)
                            _last_governor_check_ts = time.monotonic()
                        except Exception as _gov_exc:
                            logger.log_warning(f"engine_governor check failed: {_gov_exc}")

                # ── Rank candidates ───────────────────────────────────────────
                # Apply market-state affinity and engine ranker multiplier
                # to rank_score before sorting (non-raising).
                _raw_candidates = [
                    r for r in results.values()
                    if r.decision in ("BUY", "RMR_LONG", "ENGINE_LONG")
                ]
                for _rc in _raw_candidates:
                    try:
                        _eng_name = (
                            _rc.engine_signal.engine if _rc.engine_signal else
                            "RMR" if _rc.decision == "RMR_LONG" else "CONSENSUS"
                        )
                        # Market-state affinity boost
                        if _rc.market_state is not None:
                            _rc.rank_score += mstate.affinity_rank_boost(
                                _rc.market_state, _eng_name
                            )
                        # Adaptive engine weighting multiplier
                        _rc.rank_score *= engine_ranker.rank_score_multiplier(_eng_name)
                        # Block auto-disabled engines
                        if not engine_ranker.is_engine_allowed(_eng_name):
                            _rc.decision = "HOLD"
                            _rc.reject_reason = f"{_eng_name} auto-disabled (negative expectancy)"
                        # Engine governor tier multiplier
                        if getattr(config, "ENABLE_ENGINE_GOVERNOR", False):
                            _rc.rank_score *= engine_governor.rank_multiplier_for_tier(_eng_name)
                        # Sentiment modifier (filter-only, never drives entries)
                        if getattr(config, "ENABLE_SENTIMENT_FILTER", False):
                            _rc.rank_score += sentiment_engine.sentiment_modifier(_rc.symbol)
                        # Social sentiment modifier (trending coins + fear/greed)
                        if getattr(config, "ENABLE_SOCIAL_SENTIMENT", False):
                            _rc.rank_score += social_sentiment.social_rank_modifier(_rc.symbol)
                        # DeFi ecosystem signal modifier
                        if getattr(config, "ENABLE_DEFI_SIGNALS", False):
                            _rc.rank_score += defi_signals.defi_signal_modifier(_rc.symbol)
                        # Learning memory modifier
                        if getattr(config, "ENABLE_LEARNING_MEMORY", False):
                            _regime_key = f"{_rc.trend}+{_rc.vol}"
                            _sess_key   = trade_filters.hour_to_session(now_utc.hour)
                            _rc.rank_score += learning_memory.memory_modifier(
                                _eng_name, _regime_key, _sess_key
                            )
                    except Exception:
                        pass

                candidates = sorted(
                    [r for r in _raw_candidates if r.decision in ("BUY", "RMR_LONG", "ENGINE_LONG")],
                    key=lambda r: r.rank_score,
                    reverse=True,
                )

                # Rejection analytics tracking — reset each cycle
                _ra_grade: str = ""
                _ra_skip: bool = False
                _ra_filter_hits: list = []
                _ra_executed_sym: str = ""

                if open_count >= config.MAX_OPEN_TRADES:
                    for r in candidates:
                        logger.log_info(
                            f"SKIP | {r.symbol} | reason=position already open "
                            f"({open_count}/{config.MAX_OPEN_TRADES} slots used)"
                        )
                elif pause_manager.is_paused():
                    if candidates:
                        reason = pause_manager.pause_reason()
                        logger.log_info(
                            f"PAUSED | reason={reason} | no new entries "
                            f"({len(candidates)} candidate(s) suppressed)"
                        )
                elif candidates:
                    best = candidates[0]
                    for r in candidates[1:]:
                        logger.log_info(
                            f"SKIP | {r.symbol} | reason=lower rank "
                            f"({r.rank_score:.1f}) than {best.symbol} ({best.rank_score:.1f})"
                        )
                    logger.log_info(
                        f"BEST | symbol={best.symbol} | decision={best.decision} | "
                        f"score={best.rank_score:.1f} | reason={best.rank_reason}"
                    )

                    engine = engines[best.symbol]

                    # ── Trade quality filters + grader ────────────────────────
                    try:
                        _filter_results = trade_filters.run_all(
                            best.symbol, best.df, now_utc, results, client
                        )
                        _grade, _grade_score, _grade_reasons = trade_grader.grade_trade(
                            symbol=best.symbol,
                            trend=best.trend,
                            vol=best.vol,
                            adx=best.adx,
                            atr_pct=best.atr_pct,
                            score_pct=best.consensus.ratio * 100.0 if best.consensus else 0.0,
                            now_utc=now_utc,
                            filter_results=_filter_results,
                        )
                        _session_str = trade_filters.session_from_results(_filter_results)
                        _eff_min_grade = trade_grader.adaptive_min_grade(
                            consecutive_losses=performance.consecutive_losses(),
                            daily_loss_pct=(performance.daily_pnl(
                                now_utc.strftime("%Y-%m-%d")) / balance * 100
                            ) if balance > 0 else 0.0,
                            consecutive_wins=performance.consecutive_wins(),
                            weekly_pnl=performance.weekly_pnl(
                                f"{now_utc.isocalendar()[0]}-W{now_utc.isocalendar()[1]:02d}"
                            ),
                        )
                        _skip_trade = not trade_grader.grade_passes_minimum(_grade)
                        if _skip_trade:
                            logger.log_info(
                                f"SKIP | {best.symbol} | grade={_grade} "
                                f"(score={_grade_score}) below min={_eff_min_grade} | "
                                + " | ".join(_grade_reasons[:3])
                            )
                    except Exception as _flt_exc:
                        # Fail-safe: filter/grade crash must never block trading
                        logger.log_warning(f"filter/grade pipeline error (non-critical): {_flt_exc}")
                        _grade, _grade_score, _session_str = "A", 0, "Unknown"
                        _skip_trade = False
                        _filter_results = []

                    # Capture for rejection analytics (after try/except so always set)
                    _ra_grade = _grade
                    _ra_skip = _skip_trade
                    _ra_filter_hits = [f.name for f in _filter_results if not f.passed]

                    # ── Equity protection grade override ──────────────────────
                    try:
                        _ep_grade = equity_protection.effective_min_grade(
                            getattr(config, "MIN_TRADE_GRADE", "B"), balance
                        )
                        _ep_state = equity_protection.get_state(balance)
                        if _ep_state != "normal":
                            _ep_rank = {"A+": 0, "A": 1, "B": 2, "C": 3}
                            _grade_current_rank = _ep_rank.get(_grade, 4)
                            _ep_required_rank   = _ep_rank.get(_ep_grade, 0)
                            if _grade_current_rank > _ep_required_rank:
                                _skip_trade = True
                                logger.log_info(
                                    f"EQUITY_PROTECT | {best.symbol} | "
                                    f"state={_ep_state} requires grade≥{_ep_grade} "
                                    f"but got {_grade} — skipping"
                                )
                    except Exception as _ep_exc:
                        logger.log_warning(f"equity_protection check error (non-critical): {_ep_exc}")

                    # ── Adaptive engine grade override ────────────────────────
                    try:
                        _ae_eng = (
                            best.engine_signal.engine if best.engine_signal else
                            "RMR" if best.decision == "RMR_LONG" else "CONSENSUS"
                        )
                        _ae_min = engine_ranker.effective_min_grade_for_engine(_ae_eng)
                        _ep_rank = {"A+": 0, "A": 1, "B": 2, "C": 3}
                        if _ep_rank.get(_grade, 4) > _ep_rank.get(_ae_min, 2):
                            _skip_trade = True
                            logger.log_info(
                                f"ADAPTIVE_WEIGHT | {best.symbol} | "
                                f"engine {_ae_eng} requires grade≥{_ae_min} "
                                f"but got {_grade} — skipping"
                            )
                    except Exception as _ae_exc:
                        logger.log_warning(f"adaptive engine grade check error (non-critical): {_ae_exc}")

                    # ── Correlation guard ─────────────────────────────────────
                    try:
                        _open_syms = [s for s, e in engines.items() if e.has_open_position()]
                        _corr_ok, _corr_reason = correlation_guard.check_new_position(
                            best.symbol, _open_syms
                        )
                        if not _corr_ok:
                            _skip_trade = True
                            logger.log_info(
                                f"CORR_GUARD | {best.symbol} | blocked: {_corr_reason}"
                            )
                    except Exception as _cg_exc:
                        logger.log_warning(f"correlation_guard check error (non-critical): {_cg_exc}")

                    # ── Portfolio brain: sector exposure ──────────────────────
                    if not _skip_trade and getattr(config, "ENABLE_PORTFOLIO_BRAIN", False):
                        try:
                            _open_syms = [s for s, e in engines.items() if e.has_open_position()]
                            _pb_ok, _pb_reason = portfolio_brain.check_sector_exposure(
                                best.symbol, _open_syms, balance
                            )
                            if not _pb_ok:
                                _skip_trade = True
                                logger.log_info(
                                    f"PORTFOLIO_BRAIN | {best.symbol} | blocked: {_pb_reason}"
                                )
                                if getattr(config, "ENABLE_TELEGRAM_BOT", False):
                                    try:
                                        _pb_summary = portfolio_brain.get_portfolio_summary(
                                            _open_syms, balance
                                        )
                                        _pb_msg = (
                                            f"*Portfolio Brain* ⚠\n"
                                            f"`{best.symbol}` blocked — {_pb_reason}\n"
                                            f"Health: `{_pb_summary.get('health_score', 0):.0f}/100` "
                                            f"[{_pb_summary.get('health_label', '?')}]"
                                        )
                                        telegram_bot.send_rejection_summary(_pb_msg)
                                    except Exception:
                                        pass
                        except Exception as _pb_exc:
                            logger.log_warning(f"portfolio_brain check error (non-critical): {_pb_exc}")

                    # ── Anomaly detector: pause or reduce aggressiveness ──────
                    if not _skip_trade and getattr(config, "ENABLE_ANOMALY_DETECTION", False):
                        try:
                            if anomaly_detector.should_pause_entries():
                                _skip_trade = True
                                _active_anom = anomaly_detector.get_active_anomalies()
                                _crit = [a for a in _active_anom if a.get("severity") == "CRITICAL"]
                                logger.log_info(
                                    f"ANOMALY_PAUSE | {best.symbol} | "
                                    f"CRITICAL anomaly active: "
                                    + "; ".join(a.get("message","") for a in _crit[:2])
                                )
                        except Exception as _ap_exc:
                            logger.log_warning(f"anomaly pause check error: {_ap_exc}")

                    # ── Confidence score: DEFENSIVE → pause; CAUTIOUS → tighten ─
                    if not _skip_trade and getattr(config, "ENABLE_CONFIDENCE_SCORE", False):
                        try:
                            _conf_state_now = confidence_score.get_confidence_state(_cycle_confidence)
                            if _conf_state_now == confidence_score.DEFENSIVE:
                                _skip_trade = True
                                logger.log_info(
                                    f"CONFIDENCE_PAUSE | {best.symbol} | "
                                    f"score={_cycle_confidence:.0f} DEFENSIVE — no new entries"
                                )
                            elif _conf_state_now == confidence_score.CAUTIOUS:
                                _conf_min_grade = confidence_score.effective_min_grade(
                                    _grade, _cycle_confidence
                                )
                                _ep_rank = {"A+": 0, "A": 1, "B": 2, "C": 3}
                                if _ep_rank.get(_grade, 4) > _ep_rank.get(_conf_min_grade, 0):
                                    _skip_trade = True
                                    logger.log_info(
                                        f"CONFIDENCE_GRADE | {best.symbol} | "
                                        f"score={_cycle_confidence:.0f} CAUTIOUS requires "
                                        f"grade≥{_conf_min_grade} but got {_grade}"
                                    )
                        except Exception as _cse_exc:
                            logger.log_warning(f"confidence grade check error: {_cse_exc}")

                    # ── Market avoidance: grade floor tightening ──────────────
                    if not _skip_trade and getattr(config, "ENABLE_MARKET_AVOIDANCE", False):
                        try:
                            _av_result = market_avoidance.check_avoidance(
                                best.df, best.symbol, now_utc
                            )
                            if _av_result.severity != market_avoidance.SEVERITY_NONE:
                                _av_grade_floor = market_avoidance.grade_floor_from_avoidance(
                                    _av_result, _grade
                                )
                                _ep_rank = {"A+": 0, "A": 1, "B": 2, "C": 3}
                                if _ep_rank.get(_grade, 4) > _ep_rank.get(_av_grade_floor, 0):
                                    _skip_trade = True
                                    _av_log = (
                                        f"AVOIDANCE | {best.symbol} | "
                                        f"{_av_result.severity} — grade={_grade} below "
                                        f"floor={_av_grade_floor}: "
                                        + "; ".join(_av_result.reasons)
                                    )
                                    logger.log_info(_av_log)
                                    if getattr(config, "ENABLE_TELEGRAM_BOT", False):
                                        try:
                                            _sev_icon = {
                                                market_avoidance.SEVERITY_WARNING:  "⚠",
                                                market_avoidance.SEVERITY_CRITICAL: "🔴",
                                                market_avoidance.SEVERITY_CAUTION:  "💛",
                                            }
                                            _av_msg = (
                                                f"*Market Avoidance* "
                                                f"{_sev_icon.get(_av_result.severity, '⚠')}\n"
                                                f"`{best.symbol}` blocked — "
                                                f"grade `{_grade}` below floor `{_av_grade_floor}`\n"
                                                f"_{'; '.join(_av_result.reasons)}_"
                                            )
                                            telegram_bot.send_rejection_summary(_av_msg)
                                        except Exception:
                                            pass
                        except Exception as _av_exc:
                            logger.log_warning(f"market_avoidance check error (non-critical): {_av_exc}")

                    if best.decision == "RMR_LONG":
                        rmr = best.rmr
                        rmr_params = risk.TradeParams(
                            entry_price=rmr.entry_price,
                            effective_entry=rmr.entry_price,
                            stop_price=rmr.stop_price,
                            tp_price=rmr.tp_price,
                            stop_distance=rmr.stop_distance,
                            position_size=round(
                                min(
                                    balance * config.RISK_PER_TRADE / rmr.stop_distance,
                                    balance * config.MAX_POSITION_PCT / rmr.entry_price,
                                ) * (confidence_score.risk_scale_factor(_cycle_confidence)
                                     if getattr(config, "ENABLE_CONFIDENCE_SCORE", False) else 1.0)
                                * (0.75 if getattr(config, "ENABLE_ANOMALY_DETECTION", False)
                                   and anomaly_detector.should_reduce_aggressiveness() else 1.0),
                                5,
                            ),
                            risk_amount=round(balance * config.RISK_PER_TRADE, 4),
                            fee_estimate=0.0,
                            halved=False,
                        )
                        logger.log_info(
                            f"RMR LONG [{rmr.signal_type}] | ADX={best.adx:.1f} "
                            f"ATR={rmr.atr_bucket}({best.atr_pct:.2f}%) VOL={rmr.volume_bucket} "
                            f"VWAP={rmr.vwap:.2f} VWAP_dist={rmr.vwap_distance_r:.2f}R | "
                            f"entry={rmr.entry_price:.2f} SL={rmr.stop_price:.2f} "
                            f"TP={rmr.tp_price:.2f} RR={config.RMR_TP_RR_RATIO:.1f}R"
                        )
                        _tg_approved = True
                        if getattr(config, "MANUAL_APPROVAL_MODE", False):
                            signal.alarm(0)
                            _tg_detail = (
                                f"RMR LONG [{rmr.signal_type}] | `{best.symbol}`\n"
                                f"Entry `${rmr.entry_price:,.2f}` | SL `${rmr.stop_price:,.2f}` "
                                f"| TP `${rmr.tp_price:,.2f}` | {config.RMR_TP_RR_RATIO:.1f}R"
                            )
                            _tg_approved = telegram_bot.request_approval(
                                best.symbol, _tg_detail,
                                timeout=getattr(config, "MANUAL_APPROVAL_TIMEOUT", 300),
                            )
                            signal.alarm(max(90, 30 * len(symbols)))

                        if _skip_trade or not _tg_approved:
                            if not _tg_approved:
                                logger.log_info(f"RMR trade rejected via Telegram: {best.symbol}")
                            success = False
                        else:
                            success = engine.execute_buy(
                                rmr_params,
                                strategy="RMR",
                                regime=f"{best.trend}+{best.vol}",
                                adx=best.adx,
                                atr_pct=best.atr_pct,
                                score_pct=best.consensus.ratio * 100.0 if best.consensus else 0.0,
                                balance=balance,
                            )
                            if success and engine.position:
                                engine.position.session     = _session_str
                                engine.position.trade_grade = _grade
                            if success:
                                _ra_executed_sym = best.symbol
                            if success and getattr(config, "ENABLE_TELEGRAM_BOT", False):
                                try:
                                    _pos = engine.position
                                    if _pos:
                                        from telegram_charts import generate_trade_chart
                                        _chart = generate_trade_chart(
                                            best.symbol, client,
                                            _pos.fill_price, _pos.stop_price, _pos.tp_price,
                                        )
                                        if _chart:
                                            telegram_bot.send_photo(_chart,
                                                caption=f"*{best.symbol}* RMR LONG [{_grade}] @ `${_pos.fill_price:,.2f}`")
                                        if getattr(config, "ENABLE_TELEGRAM_VOICE_ALERTS", False):
                                            from telegram_voice import trade_opened_voice
                                            _voice = trade_opened_voice(
                                                best.symbol, "BUY",
                                                _pos.fill_price, _pos.stop_price, _pos.tp_price,
                                            )
                                            if _voice:
                                                telegram_bot.send_voice(_voice)
                                except Exception as _tg_exc:
                                    logger.log_warning(f"TG trade chart send failed: {_tg_exc}")
                        if not success:
                            logger.log_warning(f"RMR order failed for {best.symbol} — staying flat.")

                    elif best.decision == "BUY":
                        halve = reg.should_halve_position(best.trend, best.vol)
                        entry_price = float(best.df["close"].iloc[-1])
                        params = risk.calculate(best.df, entry_price, balance, halve)

                        _tg_approved = True
                        if getattr(config, "MANUAL_APPROVAL_MODE", False):
                            signal.alarm(0)
                            _tg_detail = (
                                f"BUY | `{best.symbol}`\n"
                                f"Entry `${entry_price:,.2f}` | SL `${params.stop_price:,.2f}` "
                                f"| TP `${params.tp_price:,.2f}`\n"
                                f"Score `{best.rank_reason}`"
                            )
                            _tg_approved = telegram_bot.request_approval(
                                best.symbol, _tg_detail,
                                timeout=getattr(config, "MANUAL_APPROVAL_TIMEOUT", 300),
                            )
                            signal.alarm(max(90, 30 * len(symbols)))

                        if _skip_trade or not _tg_approved:
                            if not _tg_approved:
                                logger.log_info(f"BUY trade rejected via Telegram: {best.symbol}")
                            success = False
                        else:
                            success = engine.execute_buy(
                                params,
                                regime=f"{best.trend}+{best.vol}",
                                adx=best.adx,
                                atr_pct=best.atr_pct,
                                score_pct=best.consensus.ratio * 100.0 if best.consensus else 0.0,
                                balance=balance,
                            )
                            if success and engine.position:
                                engine.position.session     = _session_str
                                engine.position.trade_grade = _grade
                            if success:
                                _ra_executed_sym = best.symbol
                            if success and getattr(config, "ENABLE_TELEGRAM_BOT", False):
                                try:
                                    _pos = engine.position
                                    if _pos:
                                        from telegram_charts import generate_trade_chart
                                        _chart = generate_trade_chart(
                                            best.symbol, client,
                                            _pos.fill_price, _pos.stop_price, _pos.tp_price,
                                        )
                                        if _chart:
                                            telegram_bot.send_photo(_chart,
                                                caption=f"*{best.symbol}* BUY opened @ `${_pos.fill_price:,.2f}`")
                                        if getattr(config, "ENABLE_TELEGRAM_VOICE_ALERTS", False):
                                            from telegram_voice import trade_opened_voice
                                            _voice = trade_opened_voice(
                                                best.symbol, "BUY",
                                                _pos.fill_price, _pos.stop_price, _pos.tp_price,
                                            )
                                            if _voice:
                                                telegram_bot.send_voice(_voice)
                                except Exception as _tg_exc:
                                    logger.log_warning(f"TG trade chart send failed: {_tg_exc}")
                        if not success:
                            logger.log_warning(f"Order failed for {best.symbol} — staying flat.")

                    elif best.decision == "ENGINE_LONG":
                        esig = best.engine_signal
                        engine_params = risk.TradeParams(
                            entry_price=esig.entry_price,
                            effective_entry=esig.entry_price,
                            stop_price=esig.stop_price,
                            tp_price=esig.tp_price,
                            stop_distance=esig.stop_distance,
                            position_size=round(
                                min(
                                    balance * config.RISK_PER_TRADE / esig.stop_distance,
                                    balance * config.MAX_POSITION_PCT / esig.entry_price,
                                ) * (confidence_score.risk_scale_factor(_cycle_confidence)
                                     if getattr(config, "ENABLE_CONFIDENCE_SCORE", False) else 1.0)
                                * (0.75 if getattr(config, "ENABLE_ANOMALY_DETECTION", False)
                                   and anomaly_detector.should_reduce_aggressiveness() else 1.0),
                                5,
                            ),
                            risk_amount=round(balance * config.RISK_PER_TRADE, 4),
                            fee_estimate=0.0,
                            halved=False,
                        )
                        logger.log_info(
                            f"ENGINE LONG [{esig.engine}] | {best.symbol} | "
                            f"entry={esig.entry_price:.2f} SL={esig.stop_price:.2f} "
                            f"TP={esig.tp_price:.2f} | {esig.reason}"
                        )
                        _tg_approved = True
                        if getattr(config, "MANUAL_APPROVAL_MODE", False):
                            signal.alarm(0)
                            _tg_detail = (
                                f"ENGINE LONG [{esig.engine}] | `{best.symbol}`\n"
                                f"Entry `${esig.entry_price:,.2f}` | SL `${esig.stop_price:,.2f}` "
                                f"| TP `${esig.tp_price:,.2f}`\n{esig.reason}"
                            )
                            _tg_approved = telegram_bot.request_approval(
                                best.symbol, _tg_detail,
                                timeout=getattr(config, "MANUAL_APPROVAL_TIMEOUT", 300),
                            )
                            signal.alarm(max(90, 30 * len(symbols)))

                        if _skip_trade or not _tg_approved:
                            if not _tg_approved:
                                logger.log_info(
                                    f"ENGINE_LONG trade rejected via Telegram: {best.symbol}"
                                )
                            success = False
                        else:
                            success = engine.execute_buy(
                                engine_params,
                                strategy=esig.engine,
                                regime=f"{best.trend}+{best.vol}",
                                adx=best.adx,
                                atr_pct=best.atr_pct,
                                score_pct=best.consensus.ratio * 100.0 if best.consensus else 0.0,
                                balance=balance,
                            )
                            if success and engine.position:
                                engine.position.session     = _session_str
                                engine.position.trade_grade = _grade
                            if success:
                                _ra_executed_sym = best.symbol
                            if success and getattr(config, "ENABLE_TELEGRAM_BOT", False):
                                try:
                                    _pos = engine.position
                                    if _pos:
                                        from telegram_charts import generate_trade_chart
                                        _chart = generate_trade_chart(
                                            best.symbol, client,
                                            _pos.fill_price, _pos.stop_price, _pos.tp_price,
                                        )
                                        if _chart:
                                            telegram_bot.send_photo(
                                                _chart,
                                                caption=(
                                                    f"*{best.symbol}* ENGINE LONG "
                                                    f"[{esig.engine}] [{_grade}] "
                                                    f"@ `${_pos.fill_price:,.2f}`"
                                                ),
                                            )
                                except Exception as _tg_exc:
                                    logger.log_warning(f"TG trade chart send failed: {_tg_exc}")
                        if not success:
                            logger.log_warning(
                                f"ENGINE_LONG order failed for {best.symbol} — staying flat."
                            )

                # ── Dashboard for each symbol ─────────────────────────────────
                for sym, r in results.items():
                    if r.consensus is None:
                        continue
                    eng = engines[sym]
                    pos = eng.position
                    dashboard.print_status(
                        symbol=sym,
                        interval=config.INTERVAL,
                        trend=r.trend, vol=r.vol,
                        adx=r.adx, atr_pct=r.atr_pct,
                        consensus=r.consensus,
                        decision=r.decision,
                        balance=balance,
                        open_trades=1 if eng.has_open_position() else 0,
                        position_size=pos.size if pos else None,
                        stop_price=pos.stop_price if pos else None,
                        tp_price=pos.tp_price if pos else None,
                    )

                # ── Rejection analytics recording ─────────────────────────────
                try:
                    _ra_session_now = trade_filters.hour_to_session(now_utc.hour)
                    for _ra_sym, _ra_r in results.items():
                        _ra_is_candidate = _ra_r.decision in ("BUY", "RMR_LONG", "ENGINE_LONG")
                        _ra_is_best = bool(candidates) and candidates[0].symbol == _ra_sym
                        _ra_strategy = (
                            _ra_r.engine_signal.engine if _ra_r.engine_signal else
                            "RMR" if _ra_r.decision == "RMR_LONG" else
                            "CONSENSUS" if _ra_r.decision == "BUY" else
                            "NONE"
                        )

                        if not _ra_is_candidate:
                            # Rejected before reaching the candidate stage
                            rejection_analytics.record_scan(
                                symbol=_ra_sym,
                                session=_ra_session_now,
                                rejected=True,
                                reject_reason=_ra_r.reject_reason or _ra_r.decision,
                                strategy=_ra_strategy,
                            )
                        elif _ra_is_best and not pause_manager.is_paused() and open_count < config.MAX_OPEN_TRADES:
                            # Best candidate — went through filter/grade pipeline
                            _ra_did_exec = _ra_executed_sym == _ra_sym
                            rejection_analytics.record_scan(
                                symbol=_ra_sym,
                                session=_ra_session_now,
                                rejected=_ra_skip or not _ra_did_exec,
                                reject_reason=(f"grade={_ra_grade}" if _ra_skip else ""),
                                grade=_ra_grade,
                                filter_hits=_ra_filter_hits,
                                executed=_ra_did_exec,
                                strategy=_ra_strategy,
                            )
                        else:
                            # Candidate but blocked: paused, position full, or lower rank
                            _ra_reason = (
                                "paused" if pause_manager.is_paused()
                                else "position_full" if open_count >= config.MAX_OPEN_TRADES
                                else "lower_rank"
                            )
                            rejection_analytics.record_scan(
                                symbol=_ra_sym,
                                session=_ra_session_now,
                                rejected=True,
                                reject_reason=_ra_reason,
                                strategy=_ra_strategy,
                            )

                    # Funnel log line (cumulative totals)
                    _funnel = rejection_analytics.get_funnel()
                    logger.log_info(
                        f"FUNNEL | scanned={_funnel['scanned']} "
                        f"rejected={_funnel['rejected']} "
                        f"executed={_funnel['executed']} | "
                        f"A+={_funnel['grade_Aplus']} "
                        f"A={_funnel['grade_A']} "
                        f"B={_funnel['grade_B']} "
                        f"C={_funnel['grade_C']}"
                    )
                except Exception as _ra_exc:
                    logger.log_warning(
                        f"rejection_analytics record failed (non-critical): {_ra_exc}"
                    )

                # ── Daily / weekly performance reports ───────────────────────
                if config.ENABLE_DAILY_REPORT and now_utc.hour == config.REPORT_HOUR_UTC:
                    today_str = now_utc.strftime("%Y-%m-%d")
                    if _last_daily_report != today_str:
                        _send_performance_report(open_count, balance, symbols, "Daily")
                        _last_daily_report = today_str

                if config.ENABLE_WEEKLY_REPORT and now_utc.hour == config.REPORT_HOUR_UTC:
                    iso_wk = now_utc.isocalendar()
                    week_str = f"{iso_wk[0]}-W{iso_wk[1]:02d}"
                    if _last_weekly_report != week_str and now_utc.weekday() == 6:
                        _send_performance_report(open_count, balance, symbols, "Weekly")
                        _last_weekly_report = week_str

                # ── Telegram periodic tasks (all non-raising) ─────────────────
                if getattr(config, "ENABLE_TELEGRAM_BOT", False):
                    # Hourly market summary
                    _summary_interval = getattr(config, "TELEGRAM_SUMMARY_INTERVAL_HOURS", 1)
                    if _summary_interval > 0 and now_utc.hour != _last_summary_hour:
                        if now_utc.hour % _summary_interval == 0:
                            telegram_bot.send_market_summary(results)
                            _last_summary_hour = now_utc.hour

                    # Risk alerts (1-hour cooldown internally)
                    telegram_bot.check_risk_alerts(client)

                    # Whale alerts from scan results
                    telegram_bot.check_whale_alerts(results)

                    # Pinned mini-dashboard (every 5 minutes by wall clock)
                    _now_mono = time.monotonic()
                    if _now_mono - _last_pinned_update >= 300:
                        try:
                            _paused_str = ("PAUSED" if pause_manager.is_paused()
                                           else "ACTIVE")
                            _open_syms = [s for s, e in engines.items()
                                          if e.has_open_position()]
                            _pinned_text = (
                                f"*BTC Bot* | {now_utc.strftime('%H:%M UTC')}\n"
                                f"Mode: `{'TESTNET' if config.TESTNET else 'LIVE'}`\n"
                                f"Status: `{_paused_str}`\n"
                                f"Balance: `${balance:,.2f}`\n"
                                f"Open: `{', '.join(_open_syms) or 'none'}`\n"
                                f"Today PnL: `${performance.daily_pnl(now_utc.strftime('%Y-%m-%d')):+.4f}`"
                            )
                            telegram_bot.update_pinned(_pinned_text)
                        except Exception as _pin_exc:
                            logger.log_warning(f"TG pinned update failed: {_pin_exc}")
                        _last_pinned_update = _now_mono

                    # 24h engine learning summary
                    _LEARN_INTERVAL = 24 * 3600
                    if time.monotonic() - _last_learning_summary_ts >= _LEARN_INTERVAL:
                        try:
                            import telegram_commands as _tc_ls
                            telegram_bot.send_rejection_summary(_tc_ls.cmd_leaderboard())
                            _last_learning_summary_ts = time.monotonic()
                        except Exception as _ls_exc:
                            logger.log_warning(
                                f"24h learning summary failed (non-critical): {_ls_exc}"
                            )

                    # Weekly intelligence report (Sunday UTC, 08:00+)
                    if getattr(config, "ENABLE_WEEKLY_INTELLIGENCE", False):
                        if wir.should_send_weekly(_last_weekly_intel_ts):
                            try:
                                _weekly_text = wir.generate_weekly_report()
                                telegram_bot.send_rejection_summary(_weekly_text)
                                _last_weekly_intel_ts = time.monotonic()
                            except Exception as _wi_exc:
                                logger.log_warning(f"weekly_intelligence_report failed: {_wi_exc}")

                    # 12h rejection analytics summary
                    _RA_INTERVAL = 12 * 3600
                    if time.monotonic() - _last_rejection_summary_ts >= _RA_INTERVAL:
                        try:
                            import telegram_commands as _tc_ra
                            telegram_bot.send_rejection_summary(_tc_ra.cmd_rejections())
                            _last_rejection_summary_ts = time.monotonic()
                        except Exception as _rej_exc:
                            logger.log_warning(
                                f"12h rejection summary failed (non-critical): {_rej_exc}"
                            )

                    # Daily PDF report
                    if (getattr(config, "ENABLE_TELEGRAM_PDF_REPORT", False)
                            and now_utc.hour == config.REPORT_HOUR_UTC
                            and _last_daily_report == now_utc.strftime("%Y-%m-%d")):
                        try:
                            from telegram_reports import generate_daily_pdf
                            _pdf = generate_daily_pdf()
                            if _pdf:
                                _pdf_date = now_utc.strftime("%Y-%m-%d")
                                telegram_bot.send_document(
                                    _pdf,
                                    filename=f"btcbot_report_{_pdf_date}.pdf",
                                    caption=f"*Daily Report* — {_pdf_date}",
                                )
                        except Exception as _pdf_exc:
                            logger.log_warning(f"TG PDF report failed: {_pdf_exc}")

            finally:
                signal.alarm(0)

            # ── Arbitrage scanner (outside SIGALRM window — watch only) ────
            if config.ENABLE_ARBITRAGE_SCANNER:
                try:
                    arbitrage_scanner.run_scan(data_client or client)
                except Exception as _arb_exc:
                    logger.log_warning(
                        f"arbitrage_scanner: scan failed (non-critical): {_arb_exc}"
                    )

            # ── Grid engine update (virtual grid — never live orders) ────────
            if getattr(config, "ENABLE_GRID_ENGINE", False):
                try:
                    for _gsym in ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]:
                        if _gsym not in results:
                            continue
                        _gr = results[_gsym]
                        if _gr.df.empty:
                            continue
                        _gprice = float(_gr.df["close"].iloc[-1])
                        _gatr   = float(_gr.atr_pct * _gprice / 100.0) if _gr.atr_pct else 0.0
                        _gevts  = grid_engine.update_grid(_gsym, _gprice, _gatr)
                        if _gevts.get("filled") or _gevts.get("closed"):
                            for _gfill in _gevts.get("filled", []):
                                logger.log_info(
                                    f"GRID_FILL [{_gsym}] virtual buy @ {_gfill['price']:.2f}"
                                    f" TP={_gfill['tp']:.2f}"
                                )
                            for _gclose in _gevts.get("closed", []):
                                logger.log_info(
                                    f"GRID_CLOSE [{_gsym}] virtual exit @ {_gclose['tp']:.2f}"
                                    f" pnl={_gclose['pnl']:+.6f}"
                                )
                except Exception as _ge_exc:
                    logger.log_warning(f"grid_engine update failed (non-critical): {_ge_exc}")

            # ── Funding arb: extreme rate alert ──────────────────────────────
            if (getattr(config, "ENABLE_FUNDING_ARB", False)
                    and getattr(config, "ENABLE_TELEGRAM_BOT", False)):
                try:
                    _ARB_ALERT_COOLDOWN = 3600  # 1h between extreme alerts
                    if time.monotonic() - _last_funding_alert_ts >= _ARB_ALERT_COOLDOWN:
                        _extreme_arb = [
                            s for s in funding_arb.get_arb_signals(
                                getattr(config, "FUNDING_ARB_ALERT_PCT", 0.03)
                            )
                            if s["strength"] == "EXTREME"
                        ]
                        if _extreme_arb:
                            _arb_lines = ["*Funding Arb* Extreme Rates Detected"]
                            for _as in _extreme_arb[:4]:
                                _arb_lines.append(
                                    f"`{_as['symbol']}` {_as['direction'].replace('_',' ')}: "
                                    f"`{_as['rate_pct']:+.4f}%` "
                                    f"({_as['annualized_pct']:+.1f}% ann)"
                                )
                            try:
                                telegram_bot.send_rejection_summary("\n".join(_arb_lines))
                            except Exception:
                                pass
                            _last_funding_alert_ts = time.monotonic()
                except Exception as _fa_exc:
                    logger.log_warning(f"funding_arb alert failed (non-critical): {_fa_exc}")

            # ── Hourly intel summary (trending + funding arb) ─────────────────
            if (getattr(config, "ENABLE_HOURLY_INTEL_SUMMARY", False)
                    and getattr(config, "ENABLE_TELEGRAM_BOT", False)):
                try:
                    _INTEL_INTERVAL_H = getattr(config, "HOURLY_INTEL_INTERVAL_HOURS", 1)
                    _intel_elapsed_h  = (time.monotonic() - _last_intel_summary_ts) / 3600
                    if _intel_elapsed_h >= _INTEL_INTERVAL_H:
                        import telegram_commands as _tc_intel
                        _intel_msg = _tc_intel.cmd_intel_summary()
                        if _intel_msg:
                            telegram_bot.send_rejection_summary(_intel_msg)
                        _last_intel_summary_ts = time.monotonic()
                except Exception as _is_exc:
                    logger.log_warning(f"hourly intel summary failed (non-critical): {_is_exc}")

            # ── Trend scanner (outside SIGALRM window — non-blocking) ──────
            if config.ENABLE_TREND_SCANNER:
                try:
                    trend_scanner.run_scan(client)
                except Exception as _ts_exc:
                    logger.log_warning(f"trend_scanner: scan failed (non-critical): {_ts_exc}")
                try:
                    _n_outcomes = trend_outcome_tracker.track(client)
                    if _n_outcomes:
                        logger.log_info(f"trend_outcome_tracker: {_n_outcomes} outcome(s) recorded")
                except Exception as _to_exc:
                    logger.log_warning(f"trend_outcome_tracker: failed (non-critical): {_to_exc}")

        except (TimeoutError, socket.timeout, ConnectionError, OSError) as exc:
            logger.log_warning(f"Network error in cycle (will retry next candle): {exc}")
            continue

        except Exception as exc:
            logger.log_error("Unhandled exception in main loop", exc)
            alerts.alert_exception(exc, "main loop")
            logger.log_error(f"EMERGENCY SHUTDOWN: {exc}")
            alerts.alert_emergency_shutdown(str(exc))
            for eng in engines.values():
                eng._cancel_all_orders()
            sys.exit(1)


# ── Entry ─────────────────────────────────────────────────────────────────────

def _make_client(testnet: bool = False) -> "Client":
    """Create a Binance client for data fetching (public endpoints)."""
    try:
        return Client(
            api_key=config.BINANCE_API_KEY or "",
            api_secret=config.BINANCE_SECRET_KEY or "",
            testnet=testnet,
            requests_params={"timeout": 10},
        )
    except Exception as exc:
        logger.log_error("Failed to initialise Binance client", exc)
        sys.exit(1)


def _run_backtest_mode(balance: float) -> None:
    """--mode backtest: full existing pipeline."""
    # Historical klines are a public endpoint on the MAIN Binance cluster.
    client = _make_client(testnet=False)
    try:
        bt.run(client, initial_balance=balance)
    except Exception as exc:
        logger.log_error("Backtest failed", exc)
        raise


def _run_research_mode(balance: float) -> None:
    """--mode research: run regime-stratified walk-forward on RESEARCH_TARGETS."""
    from walk_forward_by_regime import run_regime_walk_forward
    from tiered_exit import ExitConfig

    client = _make_client(testnet=False)
    df = bt.fetch_historical(client, config.WF_BACKTEST_DAYS)

    targets = config.RESEARCH_TARGETS
    if targets:
        logger.log_info(f"Research mode — targets: {targets}")
    else:
        logger.log_info("Research mode — running regime walk-forward (all)")

    exit_cfg = ExitConfig(
        enable_partial_tp=config.ENABLE_PARTIAL_TP,
        enable_time_stop=config.ENABLE_TIME_STOP,
        enable_momentum_exit=config.ENABLE_MOMENTUM_EXIT,
    )
    passed = run_regime_walk_forward(df, initial_balance=balance, exit_config=exit_cfg)
    sys.exit(0 if passed else 1)


def _run_regime_mode() -> None:
    """--mode regime: classify regimes and write regime_log.csv only."""
    from regime_classifier import RegimeClassifier

    client = _make_client(testnet=False)
    df = bt.fetch_historical(client, config.WF_BACKTEST_DAYS)

    clf = RegimeClassifier(log_to_csv=True)
    series = clf.classify_series(df)

    from collections import Counter
    counts = Counter(series)
    print(f"\n  Regime classification complete ({len(df)} bars):")
    for label, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"    {label:<20} {n:>6}  ({n / len(df):.1%})")
    from regime_classifier import REGIME_LOG_PATH
    print(f"\n  regime_log.csv → {REGIME_LOG_PATH}")


def _run_validate_mode() -> None:
    """--mode validate: check robustness gate against existing summary CSV."""
    from pathlib import Path
    import csv as csv_mod
    from sample_guard import SampleGuard, WindowResult

    summary_csv = Path("outputs/research/regime_performance_summary.csv")
    if not summary_csv.exists():
        print(f"[validate] {summary_csv} not found. Run --mode research first.")
        sys.exit(1)

    with open(summary_csv, newline="") as f:
        rows = list(csv_mod.DictReader(f))

    windows = [
        WindowResult(
            window_start=r["window_start"],
            window_end=r["window_end"],
            dominant_regime=r["dominant_regime"],
            trade_count=int(r["trade_count"]),
            pf=float(r["pf"]),
            win_rate=float(r["win_rate"]),
            median_r=float(r["median_r"]),
            worst_window_pf=float(r["worst_window_pf"]),
            tp_hit_rate=float(r["tp_hit_rate"]),
        )
        for r in rows
    ]

    guard = SampleGuard()
    guard.validate(windows)
    agg = guard.aggregate_valid(windows)

    print(f"\n  Validate mode — {len(windows)} windows from {summary_csv}")
    print(f"    Valid windows  : {agg['valid_count']} / {agg['total_count']}  ({agg['valid_window_pct']:.1%})")
    print(f"    Avg PF         : {agg['avg_pf']:.3f}")
    print(f"    Worst PF       : {agg['worst_pf']:.3f}")
    print(f"    Avg win rate   : {agg['avg_win_rate']:.1%}")

    gate_pass = guard.passes_gate(windows) and agg["valid_window_pct"] >= config.VALID_WINDOW_PCT
    print(f"    Robustness gate: {'PASS' if gate_pass else 'FAIL'}")
    sys.exit(0 if gate_pass else 1)


def main() -> None:
    # Bound all socket operations — prevents CLOSE_WAIT hangs on Binance testnet
    socket.setdefaulttimeout(15)

    parser = argparse.ArgumentParser(description="BTC/USDT trading bot")
    parser.add_argument(
        "--mode",
        choices=["backtest", "research", "regime", "validate"],
        default=None,
        help=(
            "backtest = full pipeline (default when --backtest used); "
            "research = regime walk-forward on RESEARCH_TARGETS; "
            "regime = classify regimes, write regime_log.csv; "
            "validate = check robustness gate on existing CSVs"
        ),
    )
    # Legacy flag — treated as --mode backtest
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="[legacy] Run historical backtest (same as --mode backtest)",
    )
    parser.add_argument(
        "--balance",
        type=float,
        default=10_000.0,
        help="Starting balance for backtest simulation (default: 10000 USDT)",
    )
    args = parser.parse_args()

    # Resolve effective mode
    mode = args.mode
    if mode is None and args.backtest:
        mode = "backtest"

    if mode == "backtest":
        _run_backtest_mode(args.balance)

    elif mode == "research":
        _run_research_mode(args.balance)

    elif mode == "regime":
        _run_regime_mode()

    elif mode == "validate":
        _run_validate_mode()

    else:
        # No mode flag → live trading
        config.validate_live_credentials()
        client = _make_client(testnet=config.TESTNET)
        # Fetch candles from mainnet — testnet has severely limited history (~80 bars)
        # Orders are still executed on testnet via the primary client
        data_client = _make_client(testnet=False) if config.TESTNET else None
        live(client, data_client)


if __name__ == "__main__":
    main()
