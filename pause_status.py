"""
pause_status.py — Print current drawdown-pause status.

Usage:
    python pause_status.py
    ssh root@134.209.197.173 "cd /opt/btcbot && .venv/bin/python pause_status.py"
"""
from __future__ import annotations

from pathlib import Path
from dotenv import load_dotenv

_vps_env = Path("/opt/btcbot/.env")
if _vps_env.exists():
    load_dotenv(dotenv_path=_vps_env)
else:
    load_dotenv()

import pause_manager  # noqa: E402 — must load .env first

s = pause_manager.get_status()

print(f"\n{'=' * 52}")
print(f"  Pause Status")
print(f"{'=' * 52}")
print(f"  Auto-pause enabled : {__import__('config').AUTO_PAUSE_ON_DRAWDOWN}")
print(f"  Status             : {'PAUSED ⛔' if s['paused'] else 'ACTIVE ✅'}")
if s['paused']:
    print(f"  Reason             : {s['reason']}")
    print(f"  Paused at          : {s['paused_at_utc']}")
    if s['pause_until_utc']:
        print(f"  Auto-resumes at    : {s['pause_until_utc']}")
    else:
        print(f"  Auto-resumes at    : manual unpause required")
print(f"{'─' * 52}")
print(f"  Balance            : ${s['balance']:,.2f} USDT")
print(f"  Daily PnL          : {s['daily_pnl_pct']:+.3f}%  (limit: -{s['max_daily_loss_pct']:.1f}%)")
print(f"  Weekly PnL         : {s['weekly_pnl_pct']:+.3f}%  (limit: -{s['max_weekly_loss_pct']:.1f}%)")
print(f"  Consecutive losses : {s['consecutive_losses']}  (limit: {s['max_consecutive_losses']})")
print(f"{'=' * 52}\n")
