"""
export_trades_csv.py — Export trades.db to trades_export.csv.

Usage:
    python export_trades_csv.py
    ssh root@134.209.197.173 "cd /opt/btcbot && .venv/bin/python export_trades_csv.py"
"""
from __future__ import annotations

import csv
import sqlite3
from pathlib import Path
from dotenv import load_dotenv

_vps_env = Path("/opt/btcbot/.env")
load_dotenv(dotenv_path=_vps_env if _vps_env.exists() else Path(".env"))

from trade_journal import DB_PATH  # noqa: E402

CSV_PATH = Path("trades_export.csv")

if not DB_PATH.exists():
    print(f"[ERROR] {DB_PATH} not found. No trades have been recorded yet.")
    raise SystemExit(1)

with sqlite3.connect(DB_PATH) as conn:
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM trades ORDER BY id ASC").fetchall()

if not rows:
    print("No trades in database — nothing to export.")
    raise SystemExit(0)

with open(CSV_PATH, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows([dict(r) for r in rows])

print(f"Exported {len(rows)} trade(s) → {CSV_PATH.resolve()}")
