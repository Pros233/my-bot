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


# ── Analytics helpers ─────────────────────────────────────────────────────────

def _fmt_analytics_table(rows: list, label: str, top_n: int = 5) -> str:
    """Format a list of analytics dicts into a compact Markdown table."""
    if not rows:
        return f"*{label}*: no data yet"
    lines = [f"*{label}*"]
    for r in rows[:top_n]:
        cat = r.get("category", "?")
        n   = r.get("trades", 0)
        wr  = r.get("win_rate", 0)
        avg = r.get("avg_pnl", 0)
        tot = r.get("total_pnl", 0)
        lines.append(
            f"  `{cat}` — {n}T | WR {wr:.0f}% | avg `{avg:+.4f}` | tot `{tot:+.4f}`"
        )
    return "\n".join(lines)


# ── /best ──────────────────────────────────────────────────────────────────────

def cmd_best() -> str:
    try:
        import performance_advanced as pa
        best = pa.best_market_conditions(top_n=2)
        lines = [f"*Best Market Conditions* | {_mode()}"]
        for dim, rows in best.items():
            if rows:
                r = rows[0]
                lines.append(
                    f"  Best {dim}: `{r['category']}` "
                    f"(WR={r['win_rate']:.0f}%, avg={r['avg_pnl']:+.4f})"
                )
        if len(lines) == 1:
            lines.append("_No trade data yet_")
        return "\n".join(lines)
    except Exception as exc:
        return f"⚠ /best error: `{exc}`"


# ── /worst ─────────────────────────────────────────────────────────────────────

def cmd_worst() -> str:
    try:
        import performance_advanced as pa
        worst = pa.worst_market_conditions(top_n=2)
        lines = [f"*Worst Market Conditions* | {_mode()}"]
        for dim, rows in worst.items():
            if rows:
                r = rows[0]
                lines.append(
                    f"  Worst {dim}: `{r['category']}` "
                    f"(WR={r['win_rate']:.0f}%, avg={r['avg_pnl']:+.4f})"
                )
        if len(lines) == 1:
            lines.append("_No trade data yet_")
        return "\n".join(lines)
    except Exception as exc:
        return f"⚠ /worst error: `{exc}`"


# ── /conditions ────────────────────────────────────────────────────────────────

def cmd_conditions() -> str:
    try:
        import performance_advanced as pa
        s = pa.summary_report()
        if not s:
            return "*Conditions*: no trade data yet"
        lines = [f"*Market Condition Analytics* | {_mode()}", ""]
        lines.append(f"Best session:  `{s.get('best_session', 'N/A')}`")
        lines.append(f"Best regime:   `{s.get('best_regime', 'N/A')}`")
        lines.append(f"Best hour:     `{s.get('best_hour', 'N/A')}`")
        lines.append(f"Best weekday:  `{s.get('best_weekday', 'N/A')}`")
        lines.append(f"Best grade:    `{s.get('best_grade', 'N/A')}`")
        lines.append("")
        lines.append(f"Worst session: `{s.get('worst_session', 'N/A')}`")
        lines.append(f"Worst regime:  `{s.get('worst_regime', 'N/A')}`")
        lines.append(f"Worst hour:    `{s.get('worst_hour', 'N/A')}`")
        lines.append(f"Worst weekday: `{s.get('worst_weekday', 'N/A')}`")
        return "\n".join(lines)
    except Exception as exc:
        return f"⚠ /conditions error: `{exc}`"


# ── /sessions ──────────────────────────────────────────────────────────────────

def cmd_sessions() -> str:
    try:
        import performance_advanced as pa
        rows = pa.pnl_by_session()
        return _fmt_analytics_table(rows, "PnL by Session")
    except Exception as exc:
        return f"⚠ /sessions error: `{exc}`"


# ── /regimes ───────────────────────────────────────────────────────────────────

def cmd_regimes() -> str:
    try:
        import performance_advanced as pa
        rows = pa.pnl_by_regime()
        return _fmt_analytics_table(rows, "PnL by Regime")
    except Exception as exc:
        return f"⚠ /regimes error: `{exc}`"


# ── /grades ────────────────────────────────────────────────────────────────────

def cmd_grades() -> str:
    try:
        import performance_advanced as pa
        rows = pa.pnl_by_grade()
        dist = pa.grade_distribution()
        lines = [f"*Trade Grade Distribution* | {_mode()}"]
        for grade in ("A+", "A", "B", "C", "ungraded"):
            count = dist.get(grade, 0)
            if count:
                lines.append(f"  `{grade}`: {count} trades")
        lines.append("")
        lines.append(_fmt_analytics_table(rows, "PnL by Grade"))
        return "\n".join(lines)
    except Exception as exc:
        return f"⚠ /grades error: `{exc}`"


# ── /rejections ───────────────────────────────────────────────────────────────

def cmd_rejections() -> str:
    """Return a formatted rejection analytics summary."""
    try:
        import rejection_analytics as ra

        summary = ra.get_summary()
        funnel  = ra.get_funnel()

        total    = summary["total_scanned"]
        rejected = summary["total_rejected"]
        executed = summary["total_executed"]
        rate_pct = summary["rejection_rate_pct"]

        lines = [
            f"*Why Trades Are Being Rejected* | {_mode()}",
            "",
            f"Setup funnel (all-time):",
            f"  Scanned:  `{total}`",
            f"  Rejected: `{rejected}`  ({rate_pct:.0f}%)",
            f"  Executed: `{executed}`",
            "",
        ]

        # Top rejection reasons
        top_reasons = summary["top_reasons"]
        if top_reasons:
            total_rej = rejected or 1
            lines.append("*Top Rejection Reasons*")
            for reason, count in top_reasons[:5]:
                pct = count / total_rej * 100
                bar = "█" * int(pct / 5)
                lines.append(f"  `{reason[:38]}`")
                lines.append(f"    {bar} {pct:.0f}% ({count})")
            lines.append("")

        # Grade distribution
        grade_dist = summary["grade_distribution"]
        if grade_dist:
            lines.append("*Grade Distribution*")
            for g in ("A+", "A", "B", "C", "REJECT"):
                n = grade_dist.get(g, 0)
                if n:
                    lines.append(f"  `{g}`: {n}")
            lines.append("")

        # Top filter hits
        top_filters = summary["top_filters_hit"]
        if top_filters:
            lines.append("*Filter Hit Rates*")
            for fname, count in top_filters[:4]:
                lines.append(f"  `{fname}`: {count} hits")
            lines.append("")

        # Top sessions rejected
        top_sess = summary["top_sessions_rejected"]
        if top_sess:
            lines.append("*Most Rejected Sessions*")
            for sess, count in top_sess[:3]:
                lines.append(f"  `{sess}`: {count}")

        if total == 0:
            return "*Rejection Analytics*: no data yet — bot needs to run at least one cycle."

        return "\n".join(lines)

    except Exception as exc:
        logger.log_warning(f"cmd_rejections error: {exc}")
        return f"⚠ /rejections error: `{exc}`"


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
        "*Analytics*\n"
        "/best — Best performing conditions (session/regime/hour)\n"
        "/worst — Worst performing conditions\n"
        "/conditions — Full best/worst summary\n"
        "/sessions — PnL breakdown by session\n"
        "/regimes — PnL breakdown by regime\n"
        "/grades — Trade grade distribution\n"
        "/rejections — Why trades are being rejected (filter/grade funnel)\n"
        "\n"
        "_All commands restricted to authorised chat ID only._"
    )
