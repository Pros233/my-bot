#!/usr/bin/env bash
# ─────────────────────────────
# BTC/USDT Live Bot Dashboard
# Shows RMR, trend, partial TP in real-time
# ─────────────────────────────

BOT_LOG="./trades.log"

if [ ! -f "$BOT_LOG" ]; then
    echo "Error: trades.log not found in current directory."
    exit 1
fi

echo "Monitoring bot log: $BOT_LOG"
echo "Press Ctrl+C to stop monitoring"

# Clear previous temp file
TMP_LOG="/tmp/bot_monitor_tmp.log"
> "$TMP_LOG"

tail -n0 -f "$BOT_LOG" | while read line; do
    # Filter for key entries
    if [[ "$line" == *"CYCLE"* ]] || \
       [[ "$line" == *"RMR LONG"* ]] || \
       [[ "$line" == *"RMR skip"* ]] || \
       [[ "$line" == *"Partial TP"* ]] || \
       [[ "$line" == *"Position closed"* ]]; then

        echo -e "\n===== $(date -u +'%Y-%m-%d %H:%M:%S UTC') ====="
        echo "$line"

        # Extract key values (RMR type, ADX, ATR%, VWAP distance)
        if [[ "$line" == *"RMR LONG"* ]]; then
            RMR_TYPE=$(echo "$line" | grep -oP '(?<=RMR LONG \[).+?(?=\])')
            ADX_VAL=$(echo "$line" | grep -oP 'ADX=\K[0-9.]+')
            ATR_BUCKET=$(echo "$line" | grep -oP 'ATR=\K[^ ]+')
            VOL_BUCKET=$(echo "$line" | grep -oP 'VOL=\K[^ ]+')
            VWAP_DIST=$(echo "$line" | grep -oP 'VWAP_dist=\K[0-9.]+')
            ENTRY=$(echo "$line" | grep -oP 'entry=\K[0-9.]+')
            SL=$(echo "$line" | grep -oP 'SL=\K[0-9.]+')
            TP=$(echo "$line" | grep -oP 'TP=\K[0-9.]+')
            echo "Type: $RMR_TYPE | ADX: $ADX_VAL | ATR: $ATR_BUCKET | VOL: $VOL_BUCKET | VWAP_dist: $VWAP_DIST"
            echo "Entry: $ENTRY | SL: $SL | TP: $TP"
        fi

        # Show skips
        if [[ "$line" == *"RMR skip"* ]]; then
            echo "RMR skipped — reason: $(echo "$line" | cut -d':' -f2-)"
        fi

        # Show partial TP fills
        if [[ "$line" == *"Partial TP"* ]]; then
            echo "Partial TP executed: $line"
        fi

        # Show closed positions
        if [[ "$line" == *"Position closed"* ]]; then
            echo "Position fully closed: $line"
        fi
    fi
done
