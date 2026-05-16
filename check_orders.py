"""
check_orders.py — Binance order and balance snapshot.

Loads /opt/btcbot/.env when running on the VPS, otherwise local .env.
Connects to Binance Testnet or mainnet based on TESTNET env var.
Never prints API keys.

Usage:
    python check_orders.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# ── Load .env ─────────────────────────────────────────────────────────────────
_vps_env = Path("/opt/btcbot/.env")
if _vps_env.exists():
    load_dotenv(dotenv_path=_vps_env)
    print(f"[config] Loaded {_vps_env}")
else:
    load_dotenv()
    print("[config] Loaded local .env")

# ── Parse settings ────────────────────────────────────────────────────────────
TESTNET = os.getenv("TESTNET", "true").strip().lower() in ("true", "1", "yes")
API_KEY = os.getenv("BINANCE_API_KEY", "")
API_SECRET = os.getenv("BINANCE_SECRET_KEY", "")

_SYMBOL = os.getenv("SYMBOL", "BTCUSDT")
_SYMBOLS_RAW = os.getenv("SYMBOLS", "")
SYMBOLS = (
    [s.strip() for s in _SYMBOLS_RAW.split(",") if s.strip()]
    if _SYMBOLS_RAW.strip()
    else [_SYMBOL]
)

BALANCE_ASSETS = ["BTC", "ETH", "SOL", "BNB", "USDT"]

# ── Connect ───────────────────────────────────────────────────────────────────
try:
    from binance.client import Client
except ImportError:
    print("[ERROR] python-binance not installed. Run: pip install python-binance")
    sys.exit(1)

MODE = "TESTNET" if TESTNET else "LIVE"
print(f"\n{'=' * 58}")
print(f"  Binance Order Check — {MODE}")
print(f"  Symbols : {', '.join(SYMBOLS)}")
print(f"  API key : {API_KEY[:8]}...{API_KEY[-4:] if len(API_KEY) > 12 else '****'}")
print(f"{'=' * 58}\n")

try:
    client = Client(
        api_key=API_KEY,
        api_secret=API_SECRET,
        testnet=TESTNET,
        requests_params={"timeout": 10},
    )
except Exception as exc:
    print(f"[ERROR] Could not connect to Binance: {exc}")
    sys.exit(1)

# ── Balances ──────────────────────────────────────────────────────────────────
print("── Balances ─────────────────────────────────────────────")
try:
    account = client.get_account()
    found = False
    for asset in account["balances"]:
        if asset["asset"] in BALANCE_ASSETS:
            free = float(asset["free"])
            locked = float(asset["locked"])
            if free > 0 or locked > 0:
                print(f"  {asset['asset']:<6}  free={free:>16.6f}  locked={locked:>16.6f}")
                found = True
    if not found:
        print("  (no non-zero balances found for tracked assets)")
except Exception as exc:
    print(f"  [ERROR] Could not fetch balances: {exc}")

# ── Per-symbol: ticker + open orders ──────────────────────────────────────────
for sym in SYMBOLS:
    print(f"\n── {sym} {'─' * (52 - len(sym))}")

    # Ticker price
    try:
        ticker = client.get_symbol_ticker(symbol=sym)
        print(f"  Current price : ${float(ticker['price']):>14,.4f}")
    except Exception as exc:
        print(f"  [ERROR] Ticker: {exc}")

    # Open orders
    try:
        orders = client.get_open_orders(symbol=sym)
        if not orders:
            print("  Open orders   : none")
        else:
            print(f"  Open orders   : {len(orders)}")
            print(
                f"  {'ID':<12} {'SIDE':<5} {'TYPE':<22} "
                f"{'PRICE':>12}  {'STOP PRICE':>12}  {'QTY':>10}  STATUS"
            )
            print(f"  {'-'*90}")
            for o in orders:
                print(
                    f"  {o['orderId']:<12} {o['side']:<5} {o['type']:<22} "
                    f"${float(o['price']):>11,.4f}  "
                    f"${float(o['stopPrice']):>11,.4f}  "
                    f"{float(o['origQty']):>10.5f}  {o['status']}"
                )
    except Exception as exc:
        print(f"  [ERROR] Open orders: {exc}")

print(f"\n{'=' * 58}\n")
