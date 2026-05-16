#!/usr/bin/env bash
# switch_to_live_small.sh — Switch from TESTNET to live mode (small capital).
#
# Safety settings applied:
#   TESTNET=false
#   RISK_PER_TRADE=0.001    (0.1% per trade — half the testnet limit)
#   MAX_OPEN_TRADES=1
#   MAX_TOTAL_RISK=0.005    (0.5% total account exposure cap)
#
# Run on VPS: bash /opt/btcbot/switch_to_live_small.sh
#
# Prerequisites: run live_checklist.py first and confirm ALL CHECKS PASSED.

set -euo pipefail
cd /opt/btcbot

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  SWITCHING TO LIVE MODE — REAL MONEY WILL BE TRADED          ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo "  $(date -u '+%Y-%m-%d %H:%M UTC')"
echo ""
echo "  Settings that will be applied:"
echo "    TESTNET=false"
echo "    RISK_PER_TRADE=0.001   (0.1% per trade)"
echo "    MAX_OPEN_TRADES=1"
echo "    MAX_TOTAL_RISK=0.005   (0.5% total cap)"
echo ""
echo "  Press Ctrl+C within 10 seconds to CANCEL..."
sleep 10

echo ""
echo "Proceeding..."

# ── Backup current .env ───────────────────────────────────────────────────────
BACKUP=".env.backup.$(date +%Y%m%d_%H%M%S)"
cp .env "$BACKUP"
echo "  .env backed up → $BACKUP"

# ── Apply live safety settings ────────────────────────────────────────────────
sed -i 's/^TESTNET=.*/TESTNET=false/' .env
sed -i 's/^RISK_PER_TRADE=.*/RISK_PER_TRADE=0.001/' .env
sed -i 's/^MAX_OPEN_TRADES=.*/MAX_OPEN_TRADES=1/' .env
sed -i 's/^MAX_TOTAL_RISK=.*/MAX_TOTAL_RISK=0.005/' .env

echo "  Settings written to .env:"
grep -E 'TESTNET|RISK_PER_TRADE|MAX_OPEN_TRADES|MAX_TOTAL_RISK' .env | sed 's/^/    /'

# ── Restart bot ───────────────────────────────────────────────────────────────
echo ""
echo "Restarting btcbot..."
systemctl restart btcbot
sleep 5

# ── Telegram alert ────────────────────────────────────────────────────────────
.venv/bin/python - <<'PYEOF' 2>/dev/null || echo "  (Telegram alert — non-critical failure)"
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path("/opt/btcbot/.env"))
import requests
token = os.getenv("TELEGRAM_BOT_TOKEN", "")
chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
if token and chat_id:
    import datetime
    ts = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    r = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        json={
            "chat_id": chat_id,
            "text": (
                "*LIVE MODE ENABLED — SMALL CAPITAL*\n"
                "RISK\\_PER\\_TRADE=0.001 | MAX\\_OPEN\\_TRADES=1 | MAX\\_TOTAL\\_RISK=0.005\n"
                f"Time: {ts}"
            ),
            "parse_mode": "Markdown",
        },
        timeout=10,
    )
    print(f"  Telegram alert sent: {r.status_code}")
PYEOF

# ── Verify startup ────────────────────────────────────────────────────────────
echo ""
echo "=== Last 6 log lines ==="
tail -6 /opt/btcbot/bot.log

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  LIVE MODE ACTIVE                                            ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo "  Monitor : tail -f /opt/btcbot/bot.log"
echo "  Report  : cd /opt/btcbot && .venv/bin/python report_performance.py"
echo "  Revert  : bash /opt/btcbot/switch_back_to_testnet.sh"
echo "  Backup  : $BACKUP"
echo ""
