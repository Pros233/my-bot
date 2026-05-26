"""
trade_block_diagnosis.py — Diagnose why trades are being blocked.

Run on VPS:
    cd /opt/btcbot && .venv/bin/python trade_block_diagnosis.py
    cd /opt/btcbot && .venv/bin/python trade_block_diagnosis.py --scan15-preview

Shows:
  - Adaptive grade effective floor
  - Exchange filter validation for key symbols
  - Recent TP/SL placement failures
  - Recent min-notional / lot-size skips
  - Recent adaptive grade tightenings

With --scan15-preview:
  - 15m scanner config and stats
  - Live 15m candidate scan results (read-only, no execution)
  - Confirmation gate preview for each candidate
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


def _check_scan15_preview() -> None:
    _section("6. 15M CANDIDATE SCANNER PREVIEW (read-only)")
    try:
        import config as _cfg
        import candidate_scanner_15m as _c15
        import candidate_confirmation as _cc
        from binance.client import Client

        enabled = getattr(_cfg, "ENABLE_15M_CANDIDATE_SCAN", False)
        print(f"  ENABLE_15M_CANDIDATE_SCAN : {enabled}")
        print(f"  Min rank score            : {getattr(_cfg, 'SCAN_15M_MIN_RANK_SCORE', 55.0):.0f}")
        print(f"  Min confirm score         : {getattr(_cfg, 'SCAN_15M_MIN_CONFIRMATION_SCORE', 55.0):.0f}")
        print(f"  Cooldown (minutes)        : {getattr(_cfg, 'SCAN_15M_COOLDOWN_MINUTES', 14)}")

        stats = _c15.get_stats()
        print(f"\n  Stats — total_scans={stats.get('total_scans',0)} "
              f"candidates={stats.get('total_candidates',0)} "
              f"executed={stats.get('total_executed',0)}")

        if not enabled:
            print("\n  Scanner is disabled — set ENABLE_15M_CANDIDATE_SCAN=true to use.")
            return

        raw_syms = getattr(_cfg, "SCAN_15M_SYMBOLS", "")
        syms = (
            [s.strip() for s in raw_syms.split(",") if s.strip()]
            if raw_syms.strip()
            else list(_cfg.SYMBOLS)
        )
        print(f"\n  Symbols to scan: {', '.join(syms)}")

        try:
            client = Client(_cfg.BINANCE_API_KEY, _cfg.BINANCE_SECRET_KEY)
        except Exception as e:
            print(f"  Cannot connect to Binance: {e}")
            return

        now_utc = datetime.now(timezone.utc)
        print(f"  Running scan at {now_utc.strftime('%H:%M UTC')} …")

        candidates = _c15.scan_15m_candidates(client, None, syms, now_utc)
        if not candidates:
            print("  No candidates found.")
            return

        print(f"\n  {'Symbol':<12} {'Setup':<30} {'Rank':>5} {'Grade':>6} "
              f"{'RSI':>5} {'ADX':>5} {'Vol':>5}")
        print(f"  {'-'*80}")
        for c in candidates:
            print(f"  {c.symbol:<12} {c.setup_name:<30} {c.rank_score:>5.0f} "
                  f"{c.grade_estimate:>6} {c.rsi:>5.0f} {c.adx:>5.0f} "
                  f"{c.volume_ratio:>5.1f}x")

        min_rank = float(getattr(_cfg, "SCAN_15M_MIN_RANK_SCORE", 55.0))
        qualified = [c for c in candidates if c.rank_score >= min_rank]
        if not qualified:
            print(f"\n  No candidates above rank threshold {min_rank:.0f}.")
            return

        # Get balance for confirmation preview
        try:
            account = client.get_account()
            bal = next(
                (float(a["free"]) for a in account["balances"] if a["asset"] == "USDT"),
                100.0,
            )
        except Exception:
            bal = 100.0

        print(f"\n  Confirmation gate preview (balance=${bal:.2f}, open_count=0):")
        print(f"  {'Symbol':<12} {'Gate':<25} {'Score':>5} {'Result'}")
        print(f"  {'-'*70}")
        for c in qualified:
            res = _cc.confirm_15m_candidate(
                candidate=c,
                market_states={},
                balance=bal,
                open_count=0,
                confidence_score=65.0,   # assume CAUTIOUS for conservative preview
                client=client,
                now_utc=now_utc,
            )
            flag   = "PASS" if res.confirmed else "BLOCK"
            detail = res.blocking_gate if not res.confirmed else "all gates ok"
            print(f"  {c.symbol:<12} {detail:<25} {res.confirmation_score:>5.0f}  {flag}")

    except Exception as exc:
        print(f"  ERROR: {exc}")


def main() -> None:
    import argparse as _ap
    parser = _ap.ArgumentParser(description="Trade block diagnosis")
    parser.add_argument(
        "--scan15-preview", action="store_true",
        help="Add 15m scanner config, live scan results, and confirmation preview",
    )
    args = parser.parse_args()

    print()
    print("trade_block_diagnosis.py")
    print(f"Run at: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

    _check_adaptive_grade()
    _check_exchange_filters()
    _check_tp_sl_failures()
    _check_notional_skips()
    _check_adaptive_tightenings()

    if args.scan15_preview:
        _check_scan15_preview()

    print()
    print("=" * 60)
    print("  Diagnosis complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
