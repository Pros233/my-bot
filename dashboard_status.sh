#!/usr/bin/env bash
# dashboard_status.sh — Show dashboard service status and recent log.
# Run on VPS: bash /opt/btcbot/dashboard_status.sh
set -euo pipefail

echo ""
echo "=== BTC Bot Dashboard Status ==="
echo "  $(date -u '+%Y-%m-%d %H:%M UTC')"
echo ""
systemctl status btcbot-dashboard --no-pager -l || true
echo ""
echo "=== Last 10 dashboard log lines ==="
tail -10 /opt/btcbot/dashboard.log 2>/dev/null || echo "  (no log yet)"
echo ""
