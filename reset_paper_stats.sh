#!/usr/bin/env bash
# reset_paper_stats.sh — Backup trades.db, clear all stats, restart btcbot.
# Run on VPS: bash /opt/btcbot/reset_paper_stats.sh
#
# What this does:
#   1. Backs up trades.db with a timestamp suffix
#   2. Removes trades.db  (auto-recreates on first trade)
#   3. Clears PAUSE_STATE.json and PAUSED file if present
#   4. Restarts btcbot

set -euo pipefail
cd /opt/btcbot

TS=$(date +%Y%m%d_%H%M%S)
echo "=== Reset Paper Stats === ($(date -u '+%Y-%m-%d %H:%M UTC'))"

# ── Backup + clear trades.db ──────────────────────────────────────────────────
if [ -f trades.db ]; then
    BACKUP="trades_backup_${TS}.db"
    cp trades.db "$BACKUP"
    echo "  Backed up → $BACKUP"
    rm trades.db
    echo "  trades.db removed"
else
    echo "  trades.db not found — nothing to backup"
fi

# ── Clear pause state ─────────────────────────────────────────────────────────
if [ -f PAUSE_STATE.json ]; then
    cp PAUSE_STATE.json "pause_state_backup_${TS}.json"
    rm PAUSE_STATE.json
    echo "  PAUSE_STATE.json cleared"
fi

if [ -f PAUSED ]; then
    rm PAUSED
    echo "  PAUSED file removed"
fi

# ── Restart bot ───────────────────────────────────────────────────────────────
echo ""
echo "Restarting btcbot..."
systemctl restart btcbot
sleep 4
tail -5 /opt/btcbot/bot.log

echo ""
echo "=== Reset complete ==="
[ -f "$BACKUP" ] && echo "  Backup file: /opt/btcbot/$BACKUP"
echo "  trades.db will auto-create on the next completed trade."
