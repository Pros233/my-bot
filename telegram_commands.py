"""
telegram_commands.py — Handler functions for each Telegram /command.

Called by telegram_bot._handle_command(). Each function returns a
Markdown-formatted string for the bot to send. Never raises — all
exceptions are caught and returned as error strings.

Security:
  - No API keys, secrets, or raw credentials are included in any reply.
  - /panic is intentionally destructive — gated by TELEGRAM_CHAT_ID only.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import config
import logger


# ── Formatting helpers ────────────────────────────────────────────────────────

def _pct(v: float) -> str:
    return f"{v:+.2f}%"


def _usdt(v: float) -> str:
    return f"${v:+.4f}" if abs(v) < 1 else f"${v:+.2f}"


def _mode() -> str:
    return "TESTNET" if config.TESTNET else "LIVE"


# ── /status ───────────────────────────────────────────────────────────────────

def cmd_status(client, engines: dict) -> str:
    try:
        import pause_manager
        import performance

        lines = [f"*Bot Status* | {datetime.now(timezone.utc).strftime('%H:%M UTC')}"]
        lines.append(f"Mode:  `{_mode()}`")
        lines.append(f"Paused: `{'YES — ' + pause_manager.pause_reason() if pause_manager.is_paused() else 'No'}`")

        # Balance
        try:
            usdt = 0.0
            if client:
                for b in client.get_account().get("balances", []):
                    if b["asset"] == "USDT":
                        usdt = float(b["free"])
                        break
            lines.append(f"Balance: `${usdt:,.2f} USDT`")
        except Exception:
            lines.append("Balance: `unavailable`")

        # Open positions
        open_count = sum(1 for e in engines.values() if e.has_open_position())
        lines.append(f"Open positions: `{open_count}/{len(engines)}`")

        for sym, eng in engines.items():
            if eng.has_open_position():
                p = eng.position
                unrealized = ""
                try:
                    ticker = client.get_symbol_ticker(symbol=sym)
                    price = float(ticker["price"])
                    pnl_pct = (price - p.fill_price) / p.fill_price * 100
                    unrealized = f"  unrealized `{_pct(pnl_pct)}`"
                except Exception:
                    pass
                lines.append(
                    f"  `{sym}` {p.side} @ `${p.fill_price:,.2f}`"
                    f"  SL `${p.stop_price:,.2f}` TP `${p.tp_price:,.2f}`"
                    f"{unrealized}"
                )

        # Quick PnL snapshot
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        daily = performance.daily_pnl(today)
        total = performance.total_pnl()
        lines.append(f"Today PnL:  `{_usdt(daily)}`")
        lines.append(f"All-time:   `{_usdt(total)}` ({performance.total_trades()} trades)")

        return "\n".join(lines)
    except Exception as exc:
        logger.log_warning(f"cmd_status error: {exc}")
        return f"⚠ /status error: `{exc}`"


# ── /balance ──────────────────────────────────────────────────────────────────

def cmd_balance(client) -> str:
    try:
        if not client:
            return "⚠ Client not available"
        acct = client.get_account()
        lines = [f"*Balances* | {_mode()}"]
        for b in acct.get("balances", []):
            free  = float(b["free"])
            locked = float(b["locked"])
            if free > 0 or locked > 0:
                asset = b["asset"]
                if locked > 0:
                    lines.append(f"`{asset}`: {free:.6f} (locked: {locked:.6f})")
                else:
                    lines.append(f"`{asset}`: {free:.6f}")
        return "\n".join(lines) if len(lines) > 1 else f"*Balances*: all zero | {_mode()}"
    except Exception as exc:
        logger.log_warning(f"cmd_balance error: {exc}")
        return f"⚠ /balance error: `{exc}`"


# ── /pnl ──────────────────────────────────────────────────────────────────────

def cmd_pnl() -> str:
    try:
        import performance
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        iso   = datetime.now(timezone.utc).isocalendar()
        week  = f"{iso[0]}-W{iso[1]:02d}"

        daily  = performance.daily_pnl(today)
        weekly = performance.weekly_pnl(week)
        total  = performance.total_pnl()
        n      = performance.total_trades()
        wr     = performance.win_rate()
        pf     = performance.profit_factor()
        exp    = performance.expectancy()
        dd     = performance.max_drawdown_pct()
        avg_w  = performance.average_win()
        avg_l  = performance.average_loss()
        cl     = performance.consecutive_losses()

        pf_str = f"{pf:.2f}" if pf != float("inf") else "∞"

        return (
            f"*PnL Report* | {_mode()}\n"
            f"Today:   `{_usdt(daily)}`\n"
            f"Week:    `{_usdt(weekly)}`\n"
            f"All-time:`{_usdt(total)}`\n"
            f"\n"
            f"Trades:  `{n}` | WR `{wr*100:.1f}%`\n"
            f"PF:      `{pf_str}` | Exp `{_usdt(exp)}`\n"
            f"MaxDD:   `{dd:.2f}%`\n"
            f"Avg Win: `{_usdt(avg_w)}` | Avg Loss `{_usdt(avg_l)}`\n"
            f"Consec losses: `{cl}`"
        )
    except Exception as exc:
        logger.log_warning(f"cmd_pnl error: {exc}")
        return f"⚠ /pnl error: `{exc}`"


# ── /open ─────────────────────────────────────────────────────────────────────

def cmd_open(engines: dict) -> str:
    try:
        positions = [(sym, e.position) for sym, e in engines.items() if e.has_open_position()]
        if not positions:
            return "*Open Positions*: none"

        lines = [f"*Open Positions* ({len(positions)})"]
        for sym, p in positions:
            opened = ""
            try:
                from datetime import datetime
                dt = datetime.fromtimestamp(p.entry_time, tz=timezone.utc)
                opened = f"\n  Opened: `{dt.strftime('%Y-%m-%d %H:%M UTC')}`"
            except Exception:
                pass
            lines.append(
                f"\n`{sym}` {p.side} `{p.size:.6f}`\n"
                f"  Entry `${p.fill_price:,.2f}` | SL `${p.stop_price:,.2f}` | TP `${p.tp_price:,.2f}`\n"
                f"  Strategy: `{p.strategy}` | Regime: `{p.regime}`"
                f"{opened}"
            )
        return "\n".join(lines)
    except Exception as exc:
        logger.log_warning(f"cmd_open error: {exc}")
        return f"⚠ /open error: `{exc}`"


# ── /orders ───────────────────────────────────────────────────────────────────

def cmd_orders(client) -> str:
    try:
        if not client:
            return "⚠ Client not available"
        all_orders = []
        for sym in config.SYMBOLS:
            try:
                orders = client.get_open_orders(symbol=sym)
                all_orders.extend(orders)
            except Exception:
                pass

        if not all_orders:
            return "*Open Orders*: none"

        lines = [f"*Open Orders* ({len(all_orders)})"]
        for o in all_orders:
            lines.append(
                f"`{o['symbol']}` {o['side']} {o['type']}\n"
                f"  Qty `{o['origQty']}` @ `${float(o['price']):,.2f}`\n"
                f"  ID `{o['orderId']}`"
            )
        return "\n".join(lines)
    except Exception as exc:
        logger.log_warning(f"cmd_orders error: {exc}")
        return f"⚠ /orders error: `{exc}`"


# ── /trades ───────────────────────────────────────────────────────────────────

def cmd_trades() -> str:
    try:
        import performance
        trades = performance.recent_trades(limit=10)
        if not trades:
            return "*Recent Trades*: none recorded"

        lines = [f"*Recent Trades* (last {len(trades)})"]
        for t in trades:
            icon  = "✅" if t["pnl_usdt"] > 0 else "❌"
            lines.append(
                f"{icon} `{t['symbol']}` {t['side']}\n"
                f"  PnL `{_usdt(t['pnl_usdt'])}` (`{t['pnl_pct']:+.2f}%`)\n"
                f"  {t.get('close_reason','?')} | {int(t.get('duration_minutes',0))}m"
            )
        return "\n".join(lines)
    except Exception as exc:
        logger.log_warning(f"cmd_trades error: {exc}")
        return f"⚠ /trades error: `{exc}`"


# ── /risk ─────────────────────────────────────────────────────────────────────

def cmd_risk(client) -> str:
    try:
        import performance
        import pause_manager

        usdt = 0.0
        try:
            if client:
                for b in client.get_account().get("balances", []):
                    if b["asset"] == "USDT":
                        usdt = float(b["free"])
                        break
        except Exception:
            pass

        today  = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        iso    = datetime.now(timezone.utc).isocalendar()
        week   = f"{iso[0]}-W{iso[1]:02d}"
        daily  = performance.daily_pnl(today)
        weekly = performance.weekly_pnl(week)
        consec = performance.consecutive_losses()

        daily_limit  = usdt * config.MAX_DAILY_LOSS  if usdt > 0 else 0
        weekly_limit = usdt * config.MAX_WEEKLY_LOSS if usdt > 0 else 0
        daily_pct    = abs(daily) / daily_limit  * 100 if daily_limit > 0 else 0
        weekly_pct   = abs(weekly) / weekly_limit * 100 if weekly_limit > 0 else 0

        paused = pause_manager.is_paused()
        reason = pause_manager.pause_reason()

        return (
            f"*Risk Exposure* | {_mode()}\n"
            f"Balance: `${usdt:,.2f}`\n"
            f"\n"
            f"Daily loss:  `{_usdt(daily)}` / `${daily_limit:.2f}` limit  (`{daily_pct:.0f}%`)\n"
            f"Weekly loss: `{_usdt(weekly)}` / `${weekly_limit:.2f}` limit (`{weekly_pct:.0f}%`)\n"
            f"Consec losses: `{consec}` / `{config.MAX_CONSECUTIVE_LOSSES}`\n"
            f"\n"
            f"Status: `{'PAUSED — ' + reason if paused else 'ACTIVE'}`"
        )
    except Exception as exc:
        logger.log_warning(f"cmd_risk error: {exc}")
        return f"⚠ /risk error: `{exc}`"


# ── /pause ────────────────────────────────────────────────────────────────────

def cmd_pause() -> str:
    try:
        import pause_manager
        if pause_manager.is_paused():
            return (
                f"*Already paused* | reason: `{pause_manager.pause_reason()}`\n"
                "Use /unpause to resume."
            )
        # Write PAUSED file directly (manual pause)
        pause_manager._init()
        pause_manager._write_paused_file("manual_telegram")
        logger.log_info("MANUAL PAUSE via Telegram /pause command")
        return (
            "*Trading PAUSED* via Telegram.\n"
            "New entries blocked. Existing positions continue.\n"
            "Use /unpause to resume."
        )
    except Exception as exc:
        logger.log_warning(f"cmd_pause error: {exc}")
        return f"⚠ /pause error: `{exc}`"


# ── /unpause ──────────────────────────────────────────────────────────────────

def cmd_unpause() -> str:
    try:
        import pause_manager
        if not pause_manager.is_paused():
            return "*Not paused* — trading is already active."
        pause_manager.manual_unpause()
        return "*Trading RESUMED* via Telegram. New entries now allowed."
    except Exception as exc:
        logger.log_warning(f"cmd_unpause error: {exc}")
        return f"⚠ /unpause error: `{exc}`"


# ── /panic ────────────────────────────────────────────────────────────────────

def cmd_panic(client, engines: dict) -> str:
    """
    Emergency panic switch:
      1. Cancel all open orders on every symbol.
      2. Pause trading (write PAUSED file).
      3. Optionally set TESTNET=true in .env (if PANIC_SWITCH_ENABLE_TESTNET).
    """
    try:
        report_lines = ["*PANIC SWITCH ACTIVATED*"]

        # 1. Cancel all orders
        cancelled: list[str] = []
        errors: list[str] = []
        for sym, eng in engines.items():
            try:
                eng._cancel_all_orders()
                cancelled.append(sym)
            except Exception as exc:
                errors.append(f"{sym}: {exc}")

        if cancelled:
            report_lines.append(f"Orders cancelled: `{', '.join(cancelled)}`")
        if errors:
            report_lines.append(f"Cancel errors: `{'; '.join(errors)}`")

        # Also attempt direct Binance cancel for any symbols not in engines
        if client:
            for sym in config.SYMBOLS:
                if sym not in engines:
                    try:
                        client.cancel_all_open_orders(symbol=sym)
                    except Exception:
                        pass

        # 2. Pause trading
        import pause_manager
        pause_manager._init()
        pause_manager._write_paused_file("panic_telegram")
        logger.log_warning("PANIC SWITCH triggered via Telegram")
        report_lines.append("Trading: *PAUSED*")

        # 3. Optionally switch to testnet
        enable_testnet_switch = getattr(config, "PANIC_SWITCH_ENABLE_TESTNET", False)
        if enable_testnet_switch and not config.TESTNET:
            env_paths = [Path("/opt/btcbot/.env"), Path(".env")]
            switched = False
            for env_path in env_paths:
                if env_path.exists():
                    try:
                        text = env_path.read_text()
                        import re
                        text = re.sub(r"^TESTNET=.*", "TESTNET=true", text, flags=re.MULTILINE)
                        env_path.write_text(text)
                        switched = True
                        logger.log_warning(f"PANIC: set TESTNET=true in {env_path}")
                        break
                    except Exception as exc:
                        report_lines.append(f"⚠ .env update failed: `{exc}`")
            if switched:
                report_lines.append("Mode: switched to `TESTNET=true` (restart required)")
            else:
                report_lines.append("⚠ Could not find .env to switch TESTNET")
        elif config.TESTNET:
            report_lines.append("Mode: already `TESTNET` — no switch needed")
        else:
            report_lines.append("Mode: `LIVE` — TESTNET switch disabled in config")

        report_lines.append("\n_Use /unpause to resume when ready._")
        return "\n".join(report_lines)

    except Exception as exc:
        logger.log_warning(f"cmd_panic error: {exc}")
        return f"⚠ /panic error: `{exc}`"


# ── /help ─────────────────────────────────────────────────────────────────────

def cmd_help() -> str:
    symbols_str = " | ".join(s.replace("USDT", "").lower() for s in config.SYMBOLS)
    return (
        "*BTC Bot — Command Reference*\n"
        "\n"
        "*Status & Info*\n"
        "/status — Bot status, mode, open positions, quick PnL\n"
        "/balance — All non-zero Binance balances\n"
        "/pnl — Full performance report (WR, PF, drawdown...)\n"
        "/open — Open positions with entry, SL, TP\n"
        "/orders — Open orders on Binance\n"
        "/trades — Last 10 closed trades\n"
        "/risk — Daily/weekly loss usage vs limits\n"
        "\n"
        "*Control*\n"
        "/pause — Pause new entries (positions continue)\n"
        "/unpause — Resume trading\n"
        f"/chart <sym> — 48H price chart (`{symbols_str}`)\n"
        "/panic — Emergency: cancel orders + pause + optional TESTNET\n"
        "\n"
        "_All commands restricted to authorised chat ID only._"
    )
