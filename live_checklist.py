"""
live_checklist.py — Pre-live safety checklist.

Run this on the VPS BEFORE switching from TESTNET to live trading.
Every [FAIL] item must be resolved before switching.

Usage:
    python live_checklist.py
    ssh root@134.209.197.173 "cd /opt/btcbot && .venv/bin/python live_checklist.py"

Exit codes:
    0 — all checks passed (safe to switch)
    1 — one or more checks failed (do NOT switch)
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

_vps_env = Path("/opt/btcbot/.env")
load_dotenv(dotenv_path=_vps_env if _vps_env.exists() else Path(".env"))

import config         # noqa: E402
import pause_manager  # noqa: E402
import trade_journal  # noqa: E402

# ── Result tracking ───────────────────────────────────────────────────────────

PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"

_results: list[tuple[str, str, str]] = []
_has_failure = False


def _record(status: str, label: str, detail: str = "") -> None:
    global _has_failure
    _results.append((status, label, detail))
    if status == FAIL:
        _has_failure = True


# ── Individual checks ─────────────────────────────────────────────────────────

def _check_testnet() -> None:
    if config.TESTNET:
        _record(PASS, "TESTNET=true  (currently testnet — switch script will set false)")
    else:
        _record(WARN, "TESTNET=false — bot is already in LIVE mode (re-running checklist?)")


def _check_api_keys() -> None:
    missing = [k for k in ("BINANCE_API_KEY", "BINANCE_SECRET_KEY")
               if not getattr(config, k)]
    if missing:
        _record(FAIL, "Binance API keys missing",
                f"Not set: {', '.join(missing)}")
    else:
        _record(PASS, "Binance API keys present")


def _check_symbols() -> None:
    syms = config.SYMBOLS
    if syms:
        _record(PASS, f"SYMBOLS loaded: {', '.join(syms)}")
    else:
        _record(FAIL, "SYMBOLS is empty", "Set SYMBOLS= in .env")


def _check_risk_per_trade() -> None:
    v = config.RISK_PER_TRADE
    if v <= 0.0025:
        _record(PASS, f"RISK_PER_TRADE={v} (≤ 0.0025 limit)")
    else:
        _record(FAIL, f"RISK_PER_TRADE={v} exceeds 0.0025",
                "Set RISK_PER_TRADE=0.001 or lower before going live.")


def _check_max_open_trades() -> None:
    v = config.MAX_OPEN_TRADES
    if v == 1:
        _record(PASS, "MAX_OPEN_TRADES=1")
    else:
        _record(FAIL, f"MAX_OPEN_TRADES={v}", "Must be 1 for live mode.")


def _check_max_total_risk() -> None:
    v = config.MAX_TOTAL_RISK
    if v <= 0.01:
        _record(PASS, f"MAX_TOTAL_RISK={v} (≤ 0.01 limit)")
    else:
        _record(FAIL, f"MAX_TOTAL_RISK={v} exceeds 0.01",
                "Set MAX_TOTAL_RISK=0.005 before going live.")


def _check_telegram() -> None:
    if not config.ENABLE_TELEGRAM_ALERTS:
        _record(FAIL, "Telegram alerts DISABLED",
                "Set ENABLE_TELEGRAM_ALERTS=true in .env")
        return
    if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
        _record(FAIL, "Telegram token or chat_id missing",
                "Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env")
        return
    try:
        import requests
        url = (f"https://api.telegram.org/bot"
               f"{config.TELEGRAM_BOT_TOKEN}/sendMessage")
        r = requests.post(
            url,
            json={
                "chat_id": config.TELEGRAM_CHAT_ID,
                "text": "*Live checklist* — Telegram connection verified ✓",
                "parse_mode": "Markdown",
            },
            timeout=10,
        )
        if r.ok:
            _record(PASS, "Telegram alert sent and confirmed OK")
        else:
            _record(FAIL, f"Telegram API error {r.status_code}", r.text[:200])
    except Exception as exc:
        _record(FAIL, "Telegram send failed", str(exc)[:200])


def _check_trades_db() -> None:
    try:
        trade_journal._ensure_db()
        _record(PASS, f"trades.db accessible ({trade_journal.DB_PATH.resolve()})")
    except Exception as exc:
        _record(FAIL, "trades.db cannot be initialised", str(exc))


def _check_pause_status() -> None:
    try:
        if pause_manager.is_paused():
            reason = pause_manager.pause_reason()
            _record(FAIL, "Bot is PAUSED",
                    f"reason={reason}  →  run: bash unpause_bot.sh")
        else:
            _record(PASS, "Bot is not paused")
    except Exception as exc:
        _record(WARN, "Could not determine pause status", str(exc))


def _check_open_orders(client) -> None:
    any_orders = False
    for sym in config.SYMBOLS:
        try:
            orders = client.get_open_orders(symbol=sym)
            if orders:
                any_orders = True
                _record(FAIL, f"Open orders on {sym}: {len(orders)}",
                        "Cancel all open orders before switching to live.")
        except Exception as exc:
            _record(WARN, f"Could not check open orders for {sym}", str(exc))
    if not any_orders:
        _record(PASS, f"No open orders on any symbol")


def _check_open_positions(client) -> None:
    """Check for above-dust coin balances that may indicate stuck positions."""
    dust = {"BTC": 0.0001, "ETH": 0.001, "SOL": 0.01, "BNB": 0.01}
    try:
        account = client.get_account()
        held = []
        for a in account["balances"]:
            name = a["asset"]
            if name in dust:
                total = float(a["free"]) + float(a["locked"])
                if total > dust[name]:
                    held.append(f"{name}={total:.6f}")
        if held:
            _record(WARN, f"Non-dust coin balances detected: {', '.join(held)}",
                    "Verify these are intentional, not stuck positions.")
        else:
            _record(PASS, "No open positions detected (all coin balances at dust level)")
    except Exception as exc:
        _record(WARN, "Could not check account balances", str(exc))


def _check_service() -> None:
    try:
        result = subprocess.run(
            ["systemctl", "is-active", "btcbot"],
            capture_output=True, text=True, timeout=5,
        )
        status = result.stdout.strip()
        if status == "active":
            _record(PASS, "btcbot systemd service: active")
        else:
            _record(FAIL, f"btcbot service status: {status}",
                    "Run: systemctl start btcbot")
    except FileNotFoundError:
        _record(WARN, "systemctl not available (run this check on the VPS)",
                "ssh root@134.209.197.173 \"cd /opt/btcbot && .venv/bin/python live_checklist.py\"")
    except Exception as exc:
        _record(WARN, "Could not check btcbot service", str(exc))


def _check_log_errors() -> None:
    log_candidates = [Path("/opt/btcbot/bot.log"), Path("bot.log")]
    log_file = next((p for p in log_candidates if p.exists()), None)
    if log_file is None:
        _record(WARN, "bot.log not found", "Cannot verify recent log health.")
        return
    try:
        lines = log_file.read_text(errors="replace").splitlines()[-100:]
        # Signal 15 = normal systemctl restart — not an error to block launch
        errors = [
            ln for ln in lines
            if "[ERROR]" in ln and "Signal 15" not in ln
        ]
        if errors:
            _record(FAIL, f"{len(errors)} unexpected ERROR(s) in last 100 log lines",
                    errors[-1].strip()[:120])
        else:
            _record(PASS, "No unexpected ERRORs in last 100 bot log lines")
    except Exception as exc:
        _record(WARN, "Could not read bot.log", str(exc))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    SEP = "═" * 64

    print(f"\n{SEP}")
    print(f"  Live Mode Safety Checklist")
    print(SEP)
    print()

    # ── Non-API checks ────────────────────────────────────────────────────────
    _check_testnet()
    _check_api_keys()
    _check_symbols()
    _check_risk_per_trade()
    _check_max_open_trades()
    _check_max_total_risk()
    _check_telegram()
    _check_trades_db()
    _check_pause_status()

    # ── Binance API checks ────────────────────────────────────────────────────
    client = None
    if config.BINANCE_API_KEY and config.BINANCE_SECRET_KEY:
        try:
            from binance.client import Client
            client = Client(
                api_key=config.BINANCE_API_KEY,
                api_secret=config.BINANCE_SECRET_KEY,
                testnet=config.TESTNET,
                requests_params={"timeout": 10},
            )
        except Exception as exc:
            _record(FAIL, "Cannot connect to Binance API", str(exc)[:150])

    if client:
        _check_open_orders(client)
        _check_open_positions(client)
    else:
        _record(FAIL, "Skipping Binance checks — API connection unavailable")

    # ── System checks ─────────────────────────────────────────────────────────
    _check_service()
    _check_log_errors()

    # ── Print results ─────────────────────────────────────────────────────────
    icons = {PASS: "✓", FAIL: "✗", WARN: "⚠"}
    for status, label, detail in _results:
        tag = f"[{status}]"
        print(f"  {tag:<6}  {icons[status]} {label}")
        if detail:
            print(f"            → {detail}")

    print()
    print(SEP)

    if _has_failure:
        print(f"  ✗  CHECKLIST FAILED — DO NOT SWITCH TO LIVE")
        print(f"     Resolve all [FAIL] items above, then re-run.")
    else:
        print(f"  ✓  ALL CHECKS PASSED — SAFE TO SWITCH TO SMALL LIVE CAPITAL")
        print(f"     Run:  bash /opt/btcbot/switch_to_live_small.sh")

    print(SEP)
    print()

    sys.exit(1 if _has_failure else 0)


if __name__ == "__main__":
    main()
