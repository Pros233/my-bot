#!/usr/bin/env bash
# stop_dashboard.sh — Stop the read-only bot dashboard.
# Run on VPS: bash /opt/btcbot/stop_dashboard.sh
set -euo pipefail

echo ""
echo "=== Stopping BTC Bot Dashboard ==="
systemctl stop btcbot-dashboard
sleep 1
echo "  Status: $(systemctl is-active btcbot-dashboard 2>/dev/null || echo 'inactive')"
echo "  Dashboard stopped."
echo ""
