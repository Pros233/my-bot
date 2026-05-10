#!/usr/bin/env bash

# ────────────────────────────────
# BTC/USDT Live Bot Supervisor
# Ensures persistent TESTNET run with RMR + dynamic exits
# ────────────────────────────────

# CONFIG: adjust if needed
BOT_CMD=".venv/bin/python main.py"
ENV_VARS="ENABLE_RANGE_MR=true ENABLE_PARTIAL_TP=true RMR_TREND_ENTRY=true ENABLE_MOMENTUM_EXIT=true TESTNET=true"
PID_FILE="./bot.pid"
LOG_FILE="./bot.log"
CHECK_INTERVAL=60          # seconds between process checks

# Function to start the bot
start_bot() {
    echo "$(date -u) | Starting bot..."
    # caffeinate -is: -i prevents idle sleep, -s prevents full system sleep (lid close/power mgmt)
    nohup caffeinate -is bash -c "$ENV_VARS $BOT_CMD" >> "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
    echo "$(date -u) | Bot PID: $(cat $PID_FILE)"
}

# Function to check if bot is running
check_bot() {
    if [ -f "$PID_FILE" ]; then
        BOT_PID=$(cat "$PID_FILE")
        if ps -p $BOT_PID > /dev/null 2>&1; then
            return 0  # running
        else
            return 1  # dead
        fi
    else
        return 1  # pid file missing
    fi
}

# Function to kill bot gracefully (SIGTERM first, then SIGKILL)
kill_bot() {
    if [ -f "$PID_FILE" ]; then
        BOT_PID=$(cat "$PID_FILE")
        if ps -p $BOT_PID > /dev/null 2>&1; then
            echo "$(date -u) | Stopping bot PID $BOT_PID (SIGTERM)..."
            kill -15 $BOT_PID
            # Wait up to 10s for graceful shutdown
            for i in $(seq 1 10); do
                sleep 1
                ps -p $BOT_PID > /dev/null 2>&1 || break
            done
            # Force kill if still alive
            if ps -p $BOT_PID > /dev/null 2>&1; then
                echo "$(date -u) | Force-killing bot PID $BOT_PID (SIGKILL)..."
                kill -9 $BOT_PID
                sleep 1
            fi
        fi
        rm -f "$PID_FILE"
    fi
}

# Guard: only one supervisor at a time
SUPERVISOR_PID_FILE="./supervisor.pid"
if [ -f "$SUPERVISOR_PID_FILE" ]; then
    EXISTING=$(cat "$SUPERVISOR_PID_FILE")
    if ps -p $EXISTING > /dev/null 2>&1; then
        echo "Supervisor already running (PID $EXISTING). Exiting."
        exit 1
    fi
fi
echo $$ > "$SUPERVISOR_PID_FILE"
trap "rm -f $SUPERVISOR_PID_FILE $PID_FILE; exit" INT TERM EXIT

# Ensure clean start
kill_bot
start_bot

# ────────────────────────────────
# Main loop: watch and auto-restart
# ────────────────────────────────
while true; do
    sleep $CHECK_INTERVAL

    if ! check_bot; then
        echo "$(date -u) | Bot not running. Restarting..."
        start_bot
    fi

    # Optionally: check log growth for hangs
    tail -n 5 "$LOG_FILE"
done
