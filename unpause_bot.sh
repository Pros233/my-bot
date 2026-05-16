#!/usr/bin/env bash
# unpause_bot.sh — Delete PAUSED file, send resume alert, restart btcbot.
# Run on the VPS: bash /opt/btcbot/unpause_bot.sh

set -euo pipefail
cd /opt/btcbot

echo "=== Unpause Bot ==="

if [ ! -f PAUSED ]; then
    echo "Bot is not paused — nothing to do."
else
    # Send resume alert and delete PAUSED file via pause_manager
    .venv/bin/python -c "import pause_manager; pause_manager.manual_unpause()"
fi

echo "Restarting btcbot service..."
systemctl restart btcbot
sleep 3

echo ""
echo "=== Current status ==="
.venv/bin/python pause_status.py

echo "=== Last 5 log lines ==="
tail -5 /opt/btcbot/bot.log
