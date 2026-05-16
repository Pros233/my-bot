#!/usr/bin/env bash
# switch_back_to_testnet.sh — Revert to TESTNET mode immediately.
#
# Use this any time you want to stop live trading and return to paper mode.
# Run on VPS: bash /opt/btcbot/switch_back_to_testnet.sh

set -euo pipefail
cd /opt/btcbot

echo ""
echo "=== Switch Back to TESTNET ==="
echo "  $(date -u '+%Y-%m-%d %H:%M UTC')"
echo ""

# ── Set TESTNET=true ──────────────────────────────────────────────────────────
sed -i 's/^TESTNET=.*/TESTNET=true/' .env

# Verify the change landed (sed is a no-op if TESTNET was already true)
if grep -q '^TESTNET=true' .env; then
    echo "  TESTNET=true set."
else
    echo "  WARNING: could not set TESTNET=true — check .env manually."
fi

echo ""
echo "Restarting btcbot..."
systemctl restart btcbot
sleep 4

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
                "*TESTNET MODE ENABLED*\n"
                "Switched back from live trading. No real money at risk.\n"
                f"Time: {ts}"
            ),
            "parse_mode": "Markdown",
        },
        timeout=10,
    )
    print(f"  Telegram alert sent: {r.status_code}")
PYEOF

# ── Status ────────────────────────────────────────────────────────────────────
echo ""
echo "=== Current config ==="
grep -E 'TESTNET|RISK_PER_TRADE|MAX_OPEN_TRADES|MAX_TOTAL_RISK' .env | sed 's/^/  /'

echo ""
echo "=== Last 5 log lines ==="
tail -5 /opt/btcbot/bot.log

echo ""
echo "=== TESTNET mode active — no real money at risk ==="
echo ""
