#!/usr/bin/env bash
# dashboard_status.sh — Show full dashboard + nginx + SSL + firewall status.
# Run on VPS: bash /opt/btcbot/dashboard_status.sh
set -euo pipefail

DOMAIN="mybot233.duckdns.org"
CERT_PATH="/etc/letsencrypt/live/${DOMAIN}/fullchain.pem"

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║        BTC Bot Dashboard — Stack Status              ║"
echo "║  $(date -u '+%Y-%m-%d %H:%M UTC')                          ║"
echo "╚══════════════════════════════════════════════════════╝"

# ── Bot service ──────────────────────────────────────────────────────────────
echo ""
echo "── Bot Service ──────────────────────────────────────────"
BOT_STATUS=$(systemctl is-active btcbot 2>/dev/null || echo "inactive")
echo "  btcbot:            $BOT_STATUS"
TESTNET=$(grep -E '^TESTNET=' /opt/btcbot/.env 2>/dev/null | cut -d= -f2 || echo "?")
echo "  Mode:              $([ "$TESTNET" = "false" ] && echo "LIVE" || echo "TESTNET ($TESTNET)")"

# ── Dashboard service ────────────────────────────────────────────────────────
echo ""
echo "── Dashboard Service ────────────────────────────────────"
DASH_STATUS=$(systemctl is-active btcbot-dashboard 2>/dev/null || echo "inactive")
echo "  btcbot-dashboard:  $DASH_STATUS"
echo "  Internal bind:     127.0.0.1:8080"
LISTENING=$(ss -tlnp 2>/dev/null | grep ':8080' | awk '{print $4}' || echo "not listening")
echo "  Socket:            ${LISTENING:-not listening}"

# ── Nginx ────────────────────────────────────────────────────────────────────
echo ""
echo "── Nginx ────────────────────────────────────────────────"
NGINX_STATUS=$(systemctl is-active nginx 2>/dev/null || echo "not installed")
echo "  nginx:             $NGINX_STATUS"
if systemctl is-active --quiet nginx 2>/dev/null; then
    nginx -t 2>&1 | sed 's/^/  config: /'
    PORTS=$(ss -tlnp 2>/dev/null | grep nginx | awk '{print $4}' | tr '\n' ' ')
    echo "  listening on:      ${PORTS:-none}"
fi

# ── SSL certificate ──────────────────────────────────────────────────────────
echo ""
echo "── SSL Certificate ──────────────────────────────────────"
echo "  Domain:            $DOMAIN"
if [ -f "$CERT_PATH" ]; then
    EXPIRY=$(openssl x509 -enddate -noout -in "$CERT_PATH" 2>/dev/null | cut -d= -f2 || echo "?")
    DAYS=$(( ( $(date -d "$EXPIRY" +%s 2>/dev/null || date -jf "%b %d %T %Y %Z" "$EXPIRY" +%s 2>/dev/null || echo 0) - $(date +%s) ) / 86400 ))
    echo "  Certificate:       valid (expires: $EXPIRY)"
    echo "  Days remaining:    $DAYS"
    if [ "$DAYS" -lt 14 ]; then
        echo "  WARNING: Certificate expires soon — run: certbot renew"
    fi
    # Check auto-renewal timer
    TIMER=$(systemctl is-active certbot.timer 2>/dev/null || systemctl is-active snap.certbot.renew.timer 2>/dev/null || echo "not found")
    echo "  Auto-renew timer:  $TIMER"
else
    echo "  Certificate:       NOT FOUND (run setup_https.sh)"
fi

# ── Fail2ban ─────────────────────────────────────────────────────────────────
echo ""
echo "── Fail2ban ─────────────────────────────────────────────"
F2B_STATUS=$(systemctl is-active fail2ban 2>/dev/null || echo "not installed")
echo "  fail2ban:          $F2B_STATUS"
if systemctl is-active --quiet fail2ban 2>/dev/null; then
    fail2ban-client status nginx-http-auth 2>/dev/null | grep -E 'Banned|Currently' | sed 's/^/  /' || true
    fail2ban-client status nginx-limit-req 2>/dev/null | grep -E 'Banned|Currently' | sed 's/^/  /' || true
fi

# ── Firewall ─────────────────────────────────────────────────────────────────
echo ""
echo "── Firewall (ufw) ───────────────────────────────────────"
ufw status 2>/dev/null | sed 's/^/  /' || echo "  ufw: not available"

# ── Public access ────────────────────────────────────────────────────────────
echo ""
echo "── Public Access ────────────────────────────────────────"
echo "  Dashboard URL:     https://${DOMAIN}"
echo "  Auth:              HTTP Basic Auth + Flask session"
echo "  Read-only:         YES (no trade endpoints)"

echo ""
echo "── Recent Nginx Log (last 10 lines) ─────────────────────"
tail -10 /var/log/nginx/btcbot-access.log 2>/dev/null || echo "  (no log yet)"

echo ""
