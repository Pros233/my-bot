"""
telegram_bot.py — Background Telegram command-bot with long-polling.

Runs as a daemon thread so it never blocks the trading loop.
All public functions are non-raising — failures are logged as warnings.

Security:
  - Every update is verified against TELEGRAM_CHAT_ID before processing.
  - Unknown chat IDs receive no response (silent drop).
  - No API keys or secrets are ever included in any message.
  - /panic is gated by TELEGRAM_CHAT_ID and requires ENABLE_TELEGRAM_BOT=true.
"""
from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from typing import Optional

import requests

import config
import logger

# ── Module-level state (set by start()) ───────────────────────────────────────
_client = None           # Binance Client (for command handlers)
_engines: dict = {}      # symbol → ExecutionEngine
_stop_event = threading.Event()
_thread: Optional[threading.Thread] = None
_offset = 0              # Telegram getUpdates offset

# ── Approval gate (MANUAL_APPROVAL_MODE) ─────────────────────────────────────
_approval_lock = threading.Lock()
_pending: dict[str, threading.Event] = {}   # symbol → Event
_results: dict[str, bool] = {}              # symbol → approved?

# ── Pinned mini-dashboard ─────────────────────────────────────────────────────
_pinned_id: int = 0
_pinned_lock = threading.Lock()

# ── Risk alert dedup (avoid spamming) ─────────────────────────────────────────
_risk_alerted: dict[str, float] = {}   # key → monotonic timestamp of last alert
_RISK_COOLDOWN = 3600.0                # seconds between same-type risk alerts


# ── Low-level API helpers ─────────────────────────────────────────────────────

def _url(method: str) -> str:
    return f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/{method}"


def _post(method: str, timeout: int = 15, **kwargs) -> Optional[dict]:
    """POST to Telegram API. Returns parsed result dict or None."""
    try:
        r = requests.post(_url(method), json=kwargs, timeout=timeout)
        data = r.json()
        if not data.get("ok"):
            logger.log_warning(f"TG {method}: {data.get('description','?')}")
            return None
        return data.get("result")
    except Exception as exc:
        logger.log_warning(f"TG {method} failed: {exc}")
        return None


def _get_updates() -> list:
    """Long-poll for updates. Returns list of update dicts."""
    try:
        r = requests.get(
            _url("getUpdates"),
            params={"offset": _offset, "timeout": 20, "limit": 100,
                    "allowed_updates": ["message", "callback_query"]},
            timeout=30,
        )
        data = r.json()
        return data.get("result", []) if data.get("ok") else []
    except Exception:
        return []


def _send_text(text: str, reply_markup=None) -> Optional[int]:
    """Send a plain Markdown message. Returns message_id or None."""
    kwargs: dict = {
        "chat_id": config.TELEGRAM_CHAT_ID,
        "text":    text[:4096],
        "parse_mode": "Markdown",
    }
    if reply_markup is not None:
        kwargs["reply_markup"] = reply_markup
    result = _post("sendMessage", **kwargs)
    return result["message_id"] if result else None


def _edit_text(message_id: int, text: str) -> bool:
    result = _post("editMessageText",
                   chat_id=config.TELEGRAM_CHAT_ID,
                   message_id=message_id,
                   text=text[:4096],
                   parse_mode="Markdown")
    return result is not None


def _answer_cb(cb_id: str, text: str = "") -> None:
    _post("answerCallbackQuery", callback_query_id=cb_id, text=text[:200])


def _chat_ok(chat_id) -> bool:
    return str(chat_id).strip() == str(config.TELEGRAM_CHAT_ID).strip()


# ── Photo / document / voice senders ─────────────────────────────────────────

def send_photo(photo_bytes: bytes, caption: str = "") -> None:
    """Send PNG photo. Never raises."""
    try:
        requests.post(
            _url("sendPhoto"),
            data={"chat_id": config.TELEGRAM_CHAT_ID,
                  "caption": caption[:1024], "parse_mode": "Markdown"},
            files={"photo": ("chart.png", photo_bytes, "image/png")},
            timeout=30,
        )
    except Exception as exc:
        logger.log_warning(f"TG sendPhoto failed: {exc}")


def send_document(doc_bytes: bytes, filename: str, caption: str = "") -> None:
    """Send a document (e.g. PDF). Never raises."""
    try:
        requests.post(
            _url("sendDocument"),
            data={"chat_id": config.TELEGRAM_CHAT_ID,
                  "caption": caption[:1024], "parse_mode": "Markdown"},
            files={"document": (filename, doc_bytes, "application/octet-stream")},
            timeout=60,
        )
    except Exception as exc:
        logger.log_warning(f"TG sendDocument failed: {exc}")


def send_voice(voice_bytes: bytes) -> None:
    """Send a voice message (OGG/MP3). Never raises."""
    try:
        requests.post(
            _url("sendVoice"),
            data={"chat_id": config.TELEGRAM_CHAT_ID},
            files={"voice": ("alert.mp3", voice_bytes, "audio/mpeg")},
            timeout=30,
        )
    except Exception as exc:
        logger.log_warning(f"TG sendVoice failed: {exc}")


# ── Public interface ──────────────────────────────────────────────────────────

def set_context(client, engines: dict) -> None:
    global _client, _engines
    _client = client
    _engines = dict(engines)


def is_running() -> bool:
    return _thread is not None and _thread.is_alive()


def start(client=None, engines: Optional[dict] = None) -> None:
    global _thread
    if client:
        set_context(client, engines or {})
    if not getattr(config, "ENABLE_TELEGRAM_BOT", False):
        return
    if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
        logger.log_warning("ENABLE_TELEGRAM_BOT=true but BOT_TOKEN or CHAT_ID missing")
        return
    _stop_event.clear()
    _thread = threading.Thread(target=_poll_loop, name="tg-bot", daemon=True)
    _thread.start()
    logger.log_info("Telegram command bot started")


def stop() -> None:
    _stop_event.set()


def alert(text: str) -> None:
    """Send an alert message from the trading bot (non-command path). Never raises."""
    if not config.ENABLE_TELEGRAM_ALERTS:
        return
    if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
        return
    try:
        _send_text(text)
    except Exception:
        pass


# ── Trade approval gate ───────────────────────────────────────────────────────

def request_approval(symbol: str, detail: str,
                     timeout: int = 300) -> bool:
    """
    Send APPROVE/REJECT inline buttons and block until user responds or timeout.
    Returns True if approved. If Telegram is disabled, always returns True.
    """
    if not getattr(config, "ENABLE_TELEGRAM_BOT", False):
        return True
    if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
        return True

    event = threading.Event()
    with _approval_lock:
        _pending[symbol] = event
        _results.pop(symbol, None)

    text = (
        f"*APPROVAL REQUIRED* | `{symbol}`\n"
        f"{detail}\n\n"
        f"_Timeout: {timeout // 60}m — no response = auto-reject_"
    )
    buttons = {"inline_keyboard": [[
        {"text": "✅ APPROVE", "callback_data": f"approve:{symbol}"},
        {"text": "❌ REJECT",  "callback_data": f"reject:{symbol}"},
    ]]}
    _send_text(text, reply_markup=buttons)

    fired = event.wait(timeout=float(timeout))

    with _approval_lock:
        approved = _results.pop(symbol, False)
        _pending.pop(symbol, None)

    if not fired:
        _send_text(f"*TRADE TIMEOUT* | `{symbol}` auto-rejected after {timeout // 60}m.")
        return False

    _send_text(f"*TRADE {'APPROVED ✅' if approved else 'REJECTED ❌'}* | `{symbol}`")
    return approved


# ── Pinned mini-dashboard ─────────────────────────────────────────────────────

def update_pinned(text: str) -> None:
    """Edit the pinned message. Create + pin if none yet."""
    global _pinned_id
    if not getattr(config, "ENABLE_TELEGRAM_BOT", False):
        return
    with _pinned_lock:
        if _pinned_id and _edit_text(_pinned_id, text):
            return
        msg_id = _send_text(text)
        if msg_id:
            _pinned_id = msg_id
            _post("pinChatMessage",
                  chat_id=config.TELEGRAM_CHAT_ID,
                  message_id=msg_id,
                  disable_notification=True)


# ── Risk alerts (with dedup) ──────────────────────────────────────────────────

def check_risk_alerts(client) -> None:
    """Check risk thresholds and send alerts if needed. Never raises."""
    if not getattr(config, "ENABLE_TELEGRAM_BOT", False):
        return
    try:
        import performance
        import pause_manager
        now = time.monotonic()

        def _cooldown_ok(key: str) -> bool:
            last = _risk_alerted.get(key, 0.0)
            if now - last > _RISK_COOLDOWN:
                _risk_alerted[key] = now
                return True
            return False

        # Get balance and PnL
        try:
            usdt = 0.0
            if client:
                acct = client.get_account()
                for b in acct.get("balances", []):
                    if b["asset"] == "USDT":
                        usdt = float(b["free"])
                        break
        except Exception:
            usdt = 0.0

        perf = {
            "daily":  performance.daily_pnl(datetime.now(timezone.utc).strftime("%Y-%m-%d")),
            "weekly": performance.weekly_pnl(
                f"{datetime.now(timezone.utc).isocalendar()[0]}"
                f"-W{datetime.now(timezone.utc).isocalendar()[1]:02d}"
            ),
            "consec": performance.consecutive_losses(),
        }

        daily_limit  = usdt * getattr(config, "MAX_DAILY_LOSS", 0.02)
        weekly_limit = usdt * getattr(config, "MAX_WEEKLY_LOSS", 0.05)
        daily_used   = max(0.0, -perf["daily"])
        weekly_used  = max(0.0, -perf["weekly"])
        consec       = perf["consec"]

        if daily_limit > 0 and daily_used / daily_limit >= 0.5 and _cooldown_ok("daily_loss_50"):
            _send_text(
                f"⚠ *RISK ALERT — Daily Loss 50%*\n"
                f"Used: `${daily_used:.4f}` / `${daily_limit:.4f}`\n"
                f"Mode: {'TESTNET' if config.TESTNET else 'LIVE'}"
            )
        if weekly_limit > 0 and weekly_used / weekly_limit >= 0.5 and _cooldown_ok("weekly_loss_50"):
            _send_text(
                f"⚠ *RISK ALERT — Weekly Loss 50%*\n"
                f"Used: `${weekly_used:.4f}` / `${weekly_limit:.4f}`\n"
                f"Mode: {'TESTNET' if config.TESTNET else 'LIVE'}"
            )
        if consec >= 2 and _cooldown_ok(f"consec_{consec}"):
            _send_text(
                f"⚠ *RISK ALERT — {consec} Consecutive Losses*\n"
                f"{'Consider pausing. Use /pause' if consec >= 3 else 'Monitor closely.'}\n"
                f"Mode: {'TESTNET' if config.TESTNET else 'LIVE'}"
            )
        if pause_manager.is_paused() and _cooldown_ok("paused"):
            _send_text(
                f"⚠ *RISK ALERT — Trading PAUSED*\n"
                f"Reason: {pause_manager.pause_reason()}\n"
                f"Use /unpause to resume."
            )
    except Exception as exc:
        logger.log_warning(f"TG risk check failed (non-critical): {exc}")


# ── Market summary ────────────────────────────────────────────────────────────

def send_market_summary(results: dict) -> None:
    """Send deterministic hourly market summary. Never raises."""
    try:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        lines = [f"*Market Summary* | {ts}\n"]
        signals = []

        for sym, r in results.items():
            base    = sym.replace("USDT", "")
            trend   = getattr(r, "trend", "?")
            vol     = getattr(r, "vol", "?")
            adx     = getattr(r, "adx", 0.0)
            atr_pct = getattr(r, "atr_pct", 0.0)
            dec     = getattr(r, "decision", "HOLD")
            signals.append(dec)

            if dec in ("RMR_LONG", "BUY"):
                icon, desc = "🟢", f"active setup ({dec}), ADX={adx:.0f}"
            elif vol == "HIGH_VOLATILITY":
                icon, desc = "⚡", f"high vol ATR={atr_pct:.2f}% — caution"
            elif trend == "TRENDING":
                icon, desc = "📈", f"trending ADX={adx:.0f} — watching"
            elif trend == "RANGING":
                icon, desc = "↔️", f"ranging ADX={adx:.0f}"
            else:
                icon, desc = "⬜", f"no setup"
            lines.append(f"{icon} *{base}*: {desc}")

        active = sum(1 for s in signals if s in ("BUY", "RMR_LONG"))
        if active >= 2:
            lines.append("\n⚡ Multiple setups — risk management critical")
        elif active == 1:
            lines.append("\n✅ One active setup")
        else:
            lines.append("\n⬜ No valid setups — standing aside")

        # Append latest arb if any
        try:
            import sqlite3
            from pathlib import Path
            for db in (Path("/opt/btcbot/arbitrage_watchlist.db"),
                       Path("arbitrage_watchlist.db")):
                if db.exists():
                    with sqlite3.connect(str(db)) as conn:
                        row = conn.execute(
                            "SELECT route, net_profit_pct FROM arbitrage_signals "
                            "WHERE net_profit_pct > 0 ORDER BY id DESC LIMIT 1"
                        ).fetchone()
                    if row:
                        lines.append(f"\n💹 *ARB*: `{row[0]}` net `+{row[1]:.3f}%` WATCH ONLY")
                    break
        except Exception:
            pass

        _send_text("\n".join(lines))
    except Exception as exc:
        logger.log_warning(f"TG market summary failed (non-critical): {exc}")


# ── Rejection analytics summary ──────────────────────────────────────────────

def send_rejection_summary(text: str) -> None:
    """
    Send a rejection analytics summary message via the command bot.
    Requires ENABLE_TELEGRAM_BOT=true and valid TOKEN/CHAT_ID. Never raises.
    """
    if not getattr(config, "ENABLE_TELEGRAM_BOT", False):
        return
    if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
        return
    try:
        _send_text(text)
    except Exception as exc:
        logger.log_warning(f"TG rejection summary failed (non-critical): {exc}")


# ── Whale alerts ──────────────────────────────────────────────────────────────

def check_whale_alerts(scan_results: dict) -> None:
    """Alert on unusual volume spikes from scan results. Never raises."""
    if not getattr(config, "ENABLE_TELEGRAM_BOT", False):
        return
    try:
        now = time.monotonic()
        for sym, r in scan_results.items():
            df = getattr(r, "df", None)
            if df is None or df.empty or len(df) < 21:
                continue
            try:
                vol_now = float(df["volume"].iloc[-1])
                vol_ma  = float(df["volume"].iloc[-21:-1].mean())
                if vol_ma <= 0:
                    continue
                spike = (vol_now / vol_ma - 1) * 100
                key = f"whale_{sym}"
                if spike >= 200 and (now - _risk_alerted.get(key, 0)) > 3600:
                    _risk_alerted[key] = now
                    price_move = (
                        float(df["close"].iloc[-1]) / float(df["close"].iloc[-2]) - 1
                    ) * 100
                    _send_text(
                        f"🐋 *WHALE ALERT* | `{sym}`\n"
                        f"Volume spike: `+{spike:.0f}%` vs 20-bar MA\n"
                        f"Price move:   `{price_move:+.2f}%`\n"
                        f"WATCH ONLY — no trade placed."
                    )
            except Exception:
                continue
    except Exception as exc:
        logger.log_warning(f"TG whale check failed (non-critical): {exc}")


# ── Polling loop ──────────────────────────────────────────────────────────────

def _poll_loop() -> None:
    global _offset
    logger.log_info("TG poll loop started")
    while not _stop_event.is_set():
        try:
            updates = _get_updates()
            for upd in updates:
                _offset = upd["update_id"] + 1
                try:
                    _dispatch(upd)
                except Exception as exc:
                    logger.log_warning(f"TG dispatch error: {exc}")
        except Exception as exc:
            logger.log_warning(f"TG poll error: {exc}")
            time.sleep(5)
        else:
            if not updates:
                time.sleep(1)


def _dispatch(upd: dict) -> None:
    # ── Callback query (inline button press) ─────────────────────────────────
    if "callback_query" in upd:
        cq = upd["callback_query"]
        if not _chat_ok(cq.get("from", {}).get("id")):
            _answer_cb(cq["id"], "Unauthorized")
            return
        _answer_cb(cq["id"])
        _handle_callback(cq.get("data", ""))
        return

    # ── Text message ──────────────────────────────────────────────────────────
    msg = upd.get("message", {})
    if not msg:
        return
    if not _chat_ok(msg.get("chat", {}).get("id")):
        return
    text = msg.get("text", "").strip()
    if not text.startswith("/"):
        return

    parts = text.split()
    cmd   = parts[0].lower().lstrip("/").split("@")[0]
    args  = parts[1:]
    try:
        _handle_command(cmd, args)
    except Exception as exc:
        logger.log_warning(f"TG /{cmd} error: {exc}")
        _send_text(f"⚠ Command error: `{type(exc).__name__}`")


def _handle_callback(data: str) -> None:
    if ":" not in data:
        return
    action, symbol = data.split(":", 1)
    with _approval_lock:
        if symbol in _pending:
            _results[symbol] = (action == "approve")
            _pending[symbol].set()


def _handle_command(cmd: str, args: list) -> None:
    import telegram_commands as tc

    if cmd == "status":
        _send_text(tc.cmd_status(_client, _engines))

    elif cmd == "balance":
        _send_text(tc.cmd_balance(_client))

    elif cmd == "pnl":
        _send_text(tc.cmd_pnl())

    elif cmd == "open":
        _send_text(tc.cmd_open(_engines))

    elif cmd == "orders":
        _send_text(tc.cmd_orders(_client))

    elif cmd == "trades":
        _send_text(tc.cmd_trades())

    elif cmd == "risk":
        _send_text(tc.cmd_risk(_client))

    elif cmd == "pause":
        _send_text(tc.cmd_pause())

    elif cmd == "unpause":
        _send_text(tc.cmd_unpause())

    elif cmd == "chart":
        raw = (args[0].upper() if args else "BTC")
        sym = raw if raw.endswith("USDT") else raw + "USDT"
        if sym not in config.SYMBOLS:
            _send_text(
                f"Unknown symbol: `{sym}`\n"
                f"Available: `{', '.join(config.SYMBOLS)}`"
            )
            return
        try:
            from telegram_charts import generate_price_chart
            chart = generate_price_chart(sym, _client)
            if chart:
                send_photo(chart, caption=f"*{sym}* — 48H price chart")
            else:
                _send_text(f"Chart unavailable for `{sym}`")
        except Exception as exc:
            _send_text(f"Chart error: `{exc}`")

    elif cmd == "panic":
        _send_text(tc.cmd_panic(_client, _engines))

    elif cmd == "best":
        _send_text(tc.cmd_best())

    elif cmd == "worst":
        _send_text(tc.cmd_worst())

    elif cmd == "conditions":
        _send_text(tc.cmd_conditions())

    elif cmd == "sessions":
        _send_text(tc.cmd_sessions())

    elif cmd == "regimes":
        _send_text(tc.cmd_regimes())

    elif cmd == "grades":
        _send_text(tc.cmd_grades())

    elif cmd == "rejections":
        _send_text(tc.cmd_rejections())

    elif cmd == "enginerank":
        _send_text(tc.cmd_enginerank())

    elif cmd == "marketstate":
        _send_text(tc.cmd_marketstate())

    elif cmd == "expectancy":
        _send_text(tc.cmd_expectancy())

    elif cmd == "leaderboard":
        _send_text(tc.cmd_leaderboard())

    elif cmd == "correlation":
        _send_text(tc.cmd_correlation(_engines))

    elif cmd == "equity":
        _send_text(tc.cmd_equity())

    elif cmd == "health":
        _send_text(tc.cmd_health())

    elif cmd == "funnel":
        _send_text(tc.cmd_funnel())

    elif cmd == "frequency":
        _send_text(tc.cmd_frequency())

    elif cmd == "strategies":
        _send_text(tc.cmd_strategies())

    elif cmd == "engines":
        _send_text(tc.cmd_engines())

    elif cmd == "setups":
        _send_text(tc.cmd_setups())

    elif cmd == "anomalies":
        _send_text(tc.cmd_anomalies())

    elif cmd == "confidence":
        _send_text(tc.cmd_confidence())

    elif cmd == "livevshadow":
        days = int(args[0]) if args and args[0].isdigit() else 30
        _send_text(tc.cmd_livevshadow(days))

    elif cmd == "governor":
        _send_text(tc.cmd_governor())

    elif cmd == "shadow":
        _send_text(tc.cmd_shadow())

    elif cmd == "shadowreport":
        _send_text(tc.cmd_shadowreport())

    elif cmd == "sentiment":
        _send_text(tc.cmd_sentiment())

    elif cmd == "trending":
        _send_text(tc.cmd_trending())

    elif cmd == "portfolio":
        _send_text(tc.cmd_portfolio(_engines))

    elif cmd == "avoidance":
        _send_text(tc.cmd_avoidance())

    elif cmd == "weekly":
        _send_text(tc.cmd_weekly())

    elif cmd == "memory":
        _send_text(tc.cmd_memory())

    elif cmd == "help":
        _send_text(tc.cmd_help())

    else:
        _send_text(f"Unknown command: `/{cmd}`\nSee /help")
