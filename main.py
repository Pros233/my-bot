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
from datetime import datetime, timezone, timedelta

import pandas as pd
import pandas_ta as ta  # noqa: F401 — register .ta accessor globally
from binance.client import Client

import config
import regime as reg
import consensus as con
import risk
import logger
import dashboard
import backtest as bt
from executor import ExecutionEngine
from strategies.range_mr import get_signal_2h, resample_1h_to_2h


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


# ── Live trading loop ─────────────────────────────────────────────────────────

def live(client: Client, data_client: Client | None = None) -> None:
    engine = ExecutionEngine(client)

    logger.log_info(f"Live trading started — {config.SYMBOL} {config.INTERVAL} "
                    f"({'TESTNET' if config.TESTNET else 'LIVE'})")
    if config.ENABLE_RANGE_MR:
        logger.log_info(
            "RMR strategy active — promoted setup RMR_2H_NO_FAR_VWAP_LONG_BIASED_BLOCK_LONG_HIGHATR_HIGHVOL "
            "(48/168 variants pass 1460-day walk-forward gate, May 2026)."
        )
    else:
        # MACD+VWAP consensus has not passed walk-forward validation.
        logger.log_warning(
            "MACD+VWAP research setup failed walk-forward validation. "
            "Set ENABLE_RANGE_MR=true to activate the validated RMR strategy."
        )

    # Initial reconnect safety — reconcile any orphaned orders
    engine.sync_open_orders()

    def _cycle_timeout(_signum, _frame):
        raise TimeoutError("Candle cycle timed out (network stall)")

    signal.signal(signal.SIGALRM, _cycle_timeout)
    signal.siginterrupt(signal.SIGALRM, True)  # don't restart syscalls on SIGALRM

    while True:
        try:
            # ── Wait for next candle close ────────────────────────────────────
            wait_secs = _seconds_until_next_hour()
            logger.log_info(f"Sleeping {wait_secs:.0f}s until next candle close…")
            time.sleep(wait_secs)

            # ── Hard 90 s OS-level timeout covers all network calls in cycle ──
            # finally block guarantees alarm is cancelled on every exit path,
            # including inner continue statements.
            signal.alarm(90)
            try:
                # ── Refresh HTTP session — drop any stale CLOSE_WAIT socket ──
                client.session.close()
                if data_client is not None:
                    data_client.session.close()

                # ── Clock sync check ──────────────────────────────────────────
                engine.check_clock_sync()

                # ── Fetch candles ─────────────────────────────────────────────
                df = fetch_candles(client, data_client)
                if df.empty or len(df) < 50:
                    logger.log_warning("Insufficient candle data, skipping cycle.")
                    continue

                # ── Regime classification ─────────────────────────────────────
                trend, vol = reg.classify(df)

                adx_df = df.ta.adx(length=config.ADX_PERIOD)
                atr_series = df.ta.atr(length=config.ATR_PERIOD)
                adx_val = float(adx_df[f"ADX_{config.ADX_PERIOD}"].iloc[-1])
                atr_val = float(atr_series.iloc[-1])
                close_val = float(df["close"].iloc[-1])
                atr_pct = (atr_val / close_val) * 100.0

                now_utc = datetime.now(timezone.utc)

                if not reg.regime_allows_trade(trend, vol):
                    logger.log_skip(now_utc, f"Regime {reg.regime_label(trend, vol)} — no trade.")
                    balance = get_usdt_balance(client)
                    result = con.compute(df, trend, vol)
                    dashboard.print_status(
                        symbol=config.SYMBOL,
                        interval=config.INTERVAL,
                        trend=trend, vol=vol,
                        adx=adx_val, atr_pct=atr_pct,
                        consensus=result,
                        decision="SKIP",
                        balance=balance,
                        open_trades=1 if engine.has_open_position() else 0,
                    )
                    continue

                # ── Check existing position ───────────────────────────────────
                if engine.has_open_position():
                    exit_price = engine.check_position(df)
                    if exit_price is not None:
                        logger.log_info(f"Position closed at {exit_price:.2f}")

                # ── Compute consensus ─────────────────────────────────────────
                result = con.compute(df, trend, vol)
                balance = get_usdt_balance(client)

                logger.log_cycle(now_utc, trend, vol, adx_val, atr_pct, result, result.decision)

                # ── Execute if signal and no open position ────────────────────
                entry_size = None
                stop_p = None
                tp_p = None

                if result.decision == con.BUY and not engine.has_open_position():
                    halve = reg.should_halve_position(trend, vol)
                    entry_price = float(df["close"].iloc[-1])
                    params = risk.calculate(df, entry_price, balance, halve)

                    entry_size = params.position_size
                    stop_p = params.stop_price
                    tp_p = params.tp_price

                    success = engine.execute_buy(params)
                    if not success:
                        logger.log_warning("Order execution failed — staying flat.")
                        entry_size = None

                # ── Range MR signal (2H, LONG-biased) ────────────────────────
                # Evaluated only on even hours when a new 2H bar has just closed.
                if (
                    config.ENABLE_RANGE_MR
                    and not engine.has_open_position()
                    and now_utc.hour % 2 == 0
                ):
                    df_2h = resample_1h_to_2h(df)
                    rmr = get_signal_2h(df_2h)
                    if rmr.direction == "LONG":
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
                                ),
                                5,
                            ),
                            risk_amount=round(balance * config.RISK_PER_TRADE, 4),
                            fee_estimate=0.0,
                            halved=False,
                        )
                        logger.log_info(
                            f"RMR LONG [{rmr.signal_type}] | ADX={adx_val:.1f} "
                            f"ATR={rmr.atr_bucket}({atr_pct:.2f}%) VOL={rmr.volume_bucket} "
                            f"VWAP={rmr.vwap:.2f} VWAP_dist={rmr.vwap_distance_r:.2f}R | "
                            f"entry={rmr.entry_price:.2f} SL={rmr.stop_price:.2f} "
                            f"TP={rmr.tp_price:.2f} RR={config.RMR_TP_RR_RATIO:.1f}R"
                        )
                        success = engine.execute_buy(rmr_params)
                        if success:
                            entry_size = rmr_params.position_size
                            stop_p = rmr_params.stop_price
                            tp_p = rmr_params.tp_price
                        else:
                            logger.log_warning("RMR order execution failed — staying flat.")
                    else:
                        logger.log_info(
                            f"RMR skip [{rmr.signal_type}] ADX={adx_val:.1f}: {rmr.reject_reason}"
                        )

                # ── Dashboard ─────────────────────────────────────────────────
                pos = engine.position
                dashboard.print_status(
                    symbol=config.SYMBOL,
                    interval=config.INTERVAL,
                    trend=trend, vol=vol,
                    adx=adx_val, atr_pct=atr_pct,
                    consensus=result,
                    decision=result.decision,
                    balance=balance,
                    open_trades=1 if engine.has_open_position() else 0,
                    position_size=pos.size if pos else entry_size,
                    stop_price=pos.stop_price if pos else stop_p,
                    tp_price=pos.tp_price if pos else tp_p,
                )

            finally:
                signal.alarm(0)  # always cancel — covers continue, return, exception

        except KeyboardInterrupt:
            engine.emergency_shutdown("KeyboardInterrupt")

        except (TimeoutError, socket.timeout, ConnectionError, OSError) as exc:
            logger.log_warning(f"Network error in cycle (will retry next candle): {exc}")
            continue

        except Exception as exc:
            logger.log_error("Unhandled exception in main loop", exc)
            engine.emergency_shutdown(str(exc))


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
