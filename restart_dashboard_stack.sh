#!/usr/bin/env bash
# restart_dashboard_stack.sh — Restart nginx + btcbot-dashboard, verify both healthy.
# Run on VPS: bash /opt/btcbot/restart_dashboard_stack.sh
set -euo pipefail

echo ""
echo "=== Restarting BTC Bot Dashboard Stack ==="
echo "  $(date -u '+%Y-%m-%d %H:%M UTC')"
echo ""

# ── 1. Test nginx config before touching anything ────────────────────────────
echo "[1/4] Testing nginx config..."
if ! nginx -t 2>&1; then
    echo "ERROR: nginx config invalid — aborting. Fix the config first."
    exit 1
fi
echo "  nginx config OK."

# ── 2. Restart nginx ─────────────────────────────────────────────────────────
echo ""
echo "[2/4] Restarting nginx..."
systemctl restart nginx
sleep 1
NGINX=$(systemctl is-active nginx)
echo "  nginx: $NGINX"
if [ "$NGINX" != "active" ]; then
    echo "ERROR: nginx failed to start."
    journalctl -u nginx -n 20 --no-pager
    exit 1
fi

# ── 3. Restart btcbot-dashboard ──────────────────────────────────────────────
echo ""
echo "[3/4] Restarting btcbot-dashboard..."
systemctl restart btcbot-dashboard
sleep 2
DASH=$(systemctl is-active btcbot-dashboard)
echo "  btcbot-dashboard: $DASH"
if [ "$DASH" != "active" ]; then
    echo "ERROR: btcbot-dashboard failed to start."
    tail -20 /opt/btcbot/dashboard.log 2>/dev/null || journalctl -u btcbot-dashboard -n 20 --no-pager
    exit 1
fi

# ── 4. Quick reachability check ──────────────────────────────────────────────
echo ""
echo "[4/4] Checking internal reachability..."
sleep 1
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8080/login 2>/dev/null || echo "000")
if [ "$HTTP_CODE" = "200" ] || [ "$HTTP_CODE" = "302" ]; then
    echo "  Flask dashboard responding (HTTP $HTTP_CODE) — OK"
else
    echo "  WARNING: Flask dashboard returned HTTP $HTTP_CODE (expected 200 or 302)"
fi

echo ""
echo "=== Stack healthy. ==="
echo "  Dashboard: https://mybot233.duckdns.org"
echo "  Logs:      tail -f /var/log/nginx/btcbot-access.log"
echo "             tail -f /opt/btcbot/dashboard.log"
echo ""
