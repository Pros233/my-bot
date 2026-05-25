#!/usr/bin/env bash
# deploy_safe.sh — safe local-to-droplet deploy
#
# USAGE (from local machine):
#   rsync -av --delete \
#     --exclude='.env' \
#     --exclude='trades.db' \
#     --exclude='PAUSED' \
#     --exclude='PAUSE_STATE.json' \
#     --exclude='engine_governor.json' \
#     --exclude='anomaly_detector.json' \
#     --exclude='rejection_analytics.json' \
#     --exclude='trend_watchlist.db' \
#     --exclude='*.db' \
#     --exclude='*.csv' \
#     --exclude='*.log' \
#     --exclude='**pycache**' \
#     --exclude='.venv' \
#     ./ root@134.209.197.173:/opt/btcbot/
#   ssh root@134.209.197.173 "bash /opt/btcbot/deploy_safe.sh"
#
# .env, trades.db, and all runtime state are NEVER touched by rsync.
set -euo pipefail

REMOTE_DIR="/opt/btcbot"

echo "======================================================"
echo "  deploy_safe.sh — BTC bot safe deploy"
echo "  $(date -u '+%Y-%m-%d %H:%M UTC')"
echo "======================================================"

# ── 1. py_compile check ──────────────────────────────────
echo ""
echo "[1/4] Running py_compile checks..."
FAIL=0
for f in "$REMOTE_DIR"/*.py; do
    if python3 -m py_compile "$f" 2>&1; then
        echo "  OK   $f"
    else
        echo "  FAIL $f"
        FAIL=1
    fi
done
if [ "$FAIL" -eq 1 ]; then
    echo ""
    echo "ERROR: py_compile failed — aborting restart."
    exit 1
fi
echo "  All .py files passed syntax check."

# ── 2. Confirm .env is intact (never touched by rsync) ───
echo ""
echo "[2/4] Verifying .env is present and untouched..."
if [ ! -f "$REMOTE_DIR/.env" ]; then
    echo "ERROR: .env missing — refusing to restart."
    exit 1
fi
echo "  .env present."

# ── 3. Restart service ───────────────────────────────────
echo ""
echo "[3/4] Restarting btcbot..."
systemctl restart btcbot
sleep 3
STATUS=$(systemctl is-active btcbot)
echo "  systemctl is-active btcbot -> $STATUS"
if [ "$STATUS" != "active" ]; then
    echo "ERROR: btcbot failed to start. Last 20 log lines:"
    journalctl -u btcbot -n 20 --no-pager
    exit 1
fi

# ── 4. Print live config from .env ───────────────────────
echo ""
echo "[4/4] Current .env config:"
for key in TESTNET RISK_PER_TRADE MAX_TOTAL_RISK; do
    val=$(grep -E "^${key}=" "$REMOTE_DIR/.env" 2>/dev/null | cut -d= -f2 || echo "(not set)")
    printf "  %-20s = %s\n" "$key" "$val"
done

echo ""
echo "======================================================"
echo "  Deploy complete. Bot is ACTIVE."
echo "  .env was NOT touched by this deploy."
echo "======================================================"
