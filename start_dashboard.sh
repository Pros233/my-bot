#!/usr/bin/env bash
# start_dashboard.sh — Start the read-only bot dashboard.
# Run on VPS: bash /opt/btcbot/start_dashboard.sh
set -euo pipefail

echo ""
echo "=== Starting BTC Bot Dashboard ==="
echo "  $(date -u '+%Y-%m-%d %H:%M UTC')"
echo ""

# Confirm password is set before starting
if ! grep -q '^DASHBOARD_PASSWORD=.\+' /opt/btcbot/.env 2>/dev/null; then
    echo "ERROR: DASHBOARD_PASSWORD is not set in .env."
    echo "  Add: DASHBOARD_PASSWORD=yourpassword"
    exit 1
fi

systemctl start btcbot-dashboard
sleep 2
STATUS=$(systemctl is-active btcbot-dashboard)
echo "  Status: $STATUS"

if [ "$STATUS" = "active" ]; then
    PORT=$(grep '^DASHBOARD_PORT=' /opt/btcbot/.env 2>/dev/null | cut -d= -f2 || echo 8080)
    echo ""
    echo "  Dashboard is running on 127.0.0.1:${PORT} (localhost only)"
    echo ""
    echo "  To access from your Mac:"
    echo "    ssh -L ${PORT}:127.0.0.1:${PORT} root@134.209.197.173"
    echo "    Then open: http://127.0.0.1:${PORT}"
    echo ""
    echo "  Logs: tail -f /opt/btcbot/dashboard.log"
else
    echo "ERROR: Dashboard failed to start."
    echo "Last 10 log lines:"
    tail -10 /opt/btcbot/dashboard.log 2>/dev/null || journalctl -u btcbot-dashboard -n 10 --no-pager
    exit 1
fi
