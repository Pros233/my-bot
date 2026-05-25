"""
trade_block_diagnosis.py — Diagnose why trades are being blocked.

Run on VPS:
    cd /opt/btcbot && .venv/bin/python trade_block_diagnosis.py

Shows:
  - Adaptive grade effective floor
  - Exchange filter validation for key symbols
  - Recent TP/SL placement failures
  - Recent min-notional / lot-size skips
  - Recent adaptive grade tightenings
"""
from __future__ import annotations

import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, "/opt/btcbot")
from dotenv import load_dotenv
load_dotenv("/opt/btcbot/.env")


def _section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def _check_adaptive_grade() -> None:
    _section("1. ADAPTIVE GRADE FLOOR")
    try:
        import config, trade_grader, performance
        base = getattr(config, "MIN_TRADE_GRADE", "A")
        enabled = getattr(config, "ENABLE_ADAPTIVE_GRADES", False)
        print(f"  ENABLE_ADAPTIVE_GRADES : {enabled}")
        print(f"  Base MIN_TRADE_GRADE   : {base}")

        if not enabled:
            print("  Adaptive grades are DISABLED — effective floor = base")
            return

        consec_losses = performance.consecutive_losses()
        consec_wins   = performance.consecutive_wins()
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        iso   = datetime.now(timezone.utc).isocalendar()
        week  = f"{iso[0]}-W{iso[1]:02d}"
        bal   = 10_000.0  # placeholder; actual balance from update_balance

        try:
            import pause_manager
            state = pause_manager._load_state()
            bal = state.get("balance", 10_000.0) or 10_000.0
        except Exception:
            pass

        daily_pnl  = performance.daily_pnl(today)
        weekly_pnl = performance.weekly_pnl(week)
        daily_pct  = (daily_pnl / bal * 100) if bal > 0 else 0.0

        eff = trade_grader.adaptive_min_grade(
            consecutive_losses=consec_losses,
            daily_loss_pct=daily_pct,
            consecutive_wins=consec_wins,
            weekly_pnl=weekly_pnl,
        )

        print(f"  Consecutive losses     : {consec_losses}")
        print(f"  Consecutive wins       : {consec_wins}")
        print(f"  Daily PnL              : ${daily_pnl:.4f} ({daily_pct:.2f}%)")
        print(f"  Weekly PnL             : ${weekly_pnl:.4f}")
        print(f"  Effective grade floor  : {eff}", end="")
        if eff != base:
            print(f"  ← TIGHTENED from {base}")
        else:
            print("  (at base)")
    except Exception as exc:
        print(f"  ERROR: {exc}")


def _check_exchange_filters() -> None:
    _section("2. EXCHANGE FILTER VALIDATION (sample symbols)")
    try:
        from binance.client import Client
        import config, exchange_filters

        client = Client(config.BINANCE_API_KEY, config.BINANCE_SECRET_KEY)

        # Use realistic balance and risk params
        try:
            account = client.get_account()
            bal = next(
                (float(a["free"]) for a in account["balances"] if a["asset"] == "USDT"),
                100.0,
            )
        except Exception:
            bal = 100.0

        risk_amt = bal * config.RISK_PER_TRADE
        print(f"  Balance (USDT)         : ${bal:.2f}")
        print(f"  Risk per trade         : ${risk_amt:.4f} ({config.RISK_PER_TRADE*100:.2f}%)")
        print()

        samples = [
            ("BTCUSDT",  100_000.0, 1500.0),  # BTC price, stop distance
            ("BNBUSDT",     700.0,    7.0),
            ("DOGEUSDT",      0.18,   0.003),
            ("XRPUSDT",       2.50,   0.04),
            ("SOLUSDT",     180.0,    3.0),
        ]

        print(f"  {'Symbol':<12} {'step_size':>10} {'min_qty':>10} {'min_notl':>10} "
              f"{'calc_qty':>10} {'notional':>10} {'valid':>6} {'reason'}")
        print(f"  {'-'*100}")

        for sym, price, stop_dist in samples:
            try:
                f = exchange_filters.get_filters(client, sym)
                raw_qty = risk_amt / stop_dist
                result  = exchange_filters.validate_order(client, sym, raw_qty, price)
                ok = "OK" if result.valid else "FAIL"
                reason = "" if result.valid else result.reason
                print(
                    f"  {sym:<12} {f.step_size:>10} {f.min_qty:>10} "
                    f"{f.min_notional:>10.2f} {result.adjusted_qty:>10} "
                    f"{result.notional:>10.2f} {ok:>6}  {reason}"
                )
            except Exception as exc:
                print(f"  {sym:<12} ERROR: {exc}")
    except Exception as exc:
        print(f"  ERROR: {exc}")


def _scan_log(label: str, patterns: list[str], lines_back: int = 5000) -> None:
    """Scan the last N lines of bot.log for pattern matches."""
    log_path = Path("/opt/btcbot/bot.log")
    if not log_path.exists():
        print("  bot.log not found")
        return
    try:
        all_lines = log_path.read_text(errors="replace").splitlines()
        recent = all_lines[-lines_back:]
        combined = re.compile("|".join(patterns), re.IGNORECASE)
        hits = [l for l in recent if combined.search(l)]
        if hits:
            for h in hits[-10:]:
                print(f"  {h.strip()}")
        else:
            print(f"  (no {label} found in last {lines_back} lines)")
    except Exception as exc:
        print(f"  ERROR scanning log: {exc}")


def _check_tp_sl_failures() -> None:
    _section("3. TP / STOP ORDER FAILURES (recent log)")
    _scan_log("TP/SL failures", [
        "TP_ORDER_FAILED", "STOP_ORDER_FAILED",
        "EMERGENCY_EXIT_PROTECTION_FAILED", "EMERGENCY_EXIT",
        "stop-only protection",
        "stop placement failed",
        "TP placement failed",
    ])


def _check_notional_skips() -> None:
    _section("4. MIN NOTIONAL / LOT SIZE SKIPS (recent log)")
    _scan_log("notional/lot skips", [
        "ORDER SKIP", "PRE-FLIGHT", "min_notional", "lot_size", "min_qty",
    ])


def _check_adaptive_tightenings() -> None:
    _section("5. ADAPTIVE GRADE TIGHTENINGS (recent log)")
    _scan_log("adaptive tightenings", [
        "ADAPTIVE_GRADE", "ADAPTIVE |", "grade floor adjusted",
        "ADAPTIVE_WEIGHT",
    ])


def main() -> None:
    print()
    print("trade_block_diagnosis.py")
    print(f"Run at: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

    _check_adaptive_grade()
    _check_exchange_filters()
    _check_tp_sl_failures()
    _check_notional_skips()
    _check_adaptive_tightenings()

    print()
    print("=" * 60)
    print("  Diagnosis complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
