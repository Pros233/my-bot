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


# ── /enginerank ───────────────────────────────────────────────────────────────

def cmd_enginerank() -> str:
    """Engine leaderboard ranked by health score."""
    try:
        import engine_ranker as er
        ranked = er.rank_engines(days=60)
        if not ranked:
            return "*Engine Ranking*: no trade data yet."

        lines = [f"*Engine Leaderboard* | {_mode()} | last 60d"]
        for i, r in enumerate(ranked, 1):
            n = r["trades"]
            if n == 0:
                lines.append(f"  {i}. `{r['engine']}` — no trades yet")
                continue
            disabled = " [DISABLED]" if r["disabled"] else ""
            lines.append(
                f"  {i}. `{r['engine']}`{disabled} | score `{r['score']:.0f}`\n"
                f"     trades={n} | exp=`{r['expectancy']:+.4f}` | "
                f"WR={r['win_rate']*100:.0f}% | PF={r['profit_factor']:.2f} | "
                f"DD={r['max_drawdown_pct']:.1f}%"
            )
        return "\n".join(lines)
    except Exception as exc:
        return f"⚠ /enginerank error: `{exc}`"


# ── /marketstate ──────────────────────────────────────────────────────────────

def cmd_marketstate() -> str:
    """Current market state for all scanned symbols."""
    try:
        import market_state as ms

        # Pull latest scan results from rejection_analytics (best proxy without live scan)
        lines = [f"*Market State* | {_mode()}"]
        lines.append(
            "_Market state is computed per-symbol each scan cycle "
            "and logged to the bot log. Use /status or check the dashboard "
            "for the live panel._"
        )

        # Show engine affinity for common states
        lines.append("")
        lines.append("*Engine Affinity Reference*")
        for state_name in [
            ms.STRONG_TREND, ms.RANGING, ms.MEAN_REV_FAVORABLE,
            ms.VOL_EXPANSION, ms.MOMENTUM_EXPANSION,
        ]:
            import dataclasses
            dummy = ms.MarketState(
                state=state_name, trend_quality="moderate",
                vol_state="normal", liquidity="good", momentum="neutral",
                adx=25.0, atr_pct=0.5, bb_width_pctile=0.5, vol_ratio=1.0,
            )
            best = ms.best_engines_for_state(dummy)[:3]
            lines.append(f"  `{state_name}` → best: {', '.join(f'`{e}`' for e in best)}")
        return "\n".join(lines)
    except Exception as exc:
        return f"⚠ /marketstate error: `{exc}`"


# ── /expectancy ───────────────────────────────────────────────────────────────

def cmd_expectancy() -> str:
    """Per-engine expectancy breakdown."""
    try:
        import engine_performance as ep
        all_stats = ep.get_all_stats(days=60)

        lines = [f"*Engine Expectancy* | {_mode()} | last 60d"]
        for engine in ep.ENGINE_NAMES:
            s = all_stats[engine]
            n = s["trades"]
            if n == 0:
                lines.append(f"  `{engine}` — no trades")
                continue
            lines.append(
                f"  `{engine}` ({n}T)\n"
                f"    exp=`{s['expectancy']:+.4f}` | PF=`{s['profit_factor']:.2f}` "
                f"| WR=`{s['win_rate']*100:.0f}%` | Sharpe=`{s['sharpe_ratio']:.2f}`"
            )
        return "\n".join(lines)
    except Exception as exc:
        return f"⚠ /expectancy error: `{exc}`"


# ── /leaderboard ──────────────────────────────────────────────────────────────

def cmd_leaderboard() -> str:
    """Full engine leaderboard with learning summary."""
    try:
        import engine_performance as ep
        import engine_ranker as er
        from datetime import datetime, timezone

        ranked = er.rank_engines(days=30)
        all_stats_90 = ep.get_all_stats(days=90)

        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        lines = [f"*Engine Learning Summary* | {now}"]

        if not any(r["trades"] > 0 for r in ranked):
            lines.append("_No trade data yet — engines are learning…_")
            return "\n".join(lines)

        best = er.best_engine(days=30)
        worst = er.worst_engine(days=30)
        if best:
            bs = all_stats_90[best]
            lines.append(f"Best engine: `{best}` | exp `{bs['expectancy']:+.4f}` | WR {bs['win_rate']*100:.0f}%")
        if worst and worst != best:
            ws = all_stats_90[worst]
            lines.append(f"Worst engine: `{worst}` | exp `{ws['expectancy']:+.4f}` | WR {ws['win_rate']*100:.0f}%")
        lines.append("")

        for r in ranked:
            if r["trades"] == 0:
                continue
            flag = " [disabled]" if r["disabled"] else ""
            lines.append(
                f"  `{r['engine']}`{flag} score=`{r['score']:.0f}` "
                f"| {r['trades']}T | exp=`{r['expectancy']:+.4f}` "
                f"| PF=`{r['profit_factor']:.2f}`"
            )

        # Adaptive state
        try:
            import equity_protection as ep2
            ep_sum = ep2.get_summary()
            lines.append("")
            lines.append(
                f"Equity protection: `{ep_sum['state'].upper()}`"
                + (f" → grade≥`{ep_sum['effective_grade']}`" if ep_sum['tightening_active'] else "")
            )
        except Exception:
            pass

        return "\n".join(lines)
    except Exception as exc:
        return f"⚠ /leaderboard error: `{exc}`"


# ── /correlation ──────────────────────────────────────────────────────────────

def cmd_correlation(engines: dict = None) -> str:
    """Current symbol correlations and open position exposure."""
    try:
        import correlation_guard as cg

        open_syms = []
        if engines:
            open_syms = [s for s, e in engines.items() if e.has_open_position()]

        exposure = cg.get_exposure_summary(open_syms)
        max_corr  = getattr(config, "MAX_CORRELATED_POSITIONS", 2)
        guard_on  = getattr(config, "ENABLE_CORRELATION_GUARD", True)

        lines = [
            f"*Correlation Exposure* | {_mode()}",
            f"Guard: `{'ON' if guard_on else 'OFF'}` | max_correlated=`{max_corr}`",
            "",
        ]

        if not open_syms:
            lines.append("Open positions: `none`")
        else:
            lines.append(f"Open: `{', '.join(open_syms)}`")
            lines.append("")
            lines.append("*Cluster Exposure*")
            for cluster, syms in sorted(exposure.items()):
                risk_flag = " ⚠" if len(syms) >= max_corr else ""
                lines.append(f"  `{cluster}`{risk_flag}: {', '.join(f'`{s}`' for s in syms)}")

        lines.append("")
        lines.append("*Cluster Map*")
        for cluster, syms in sorted(cg._CLUSTERS.items()):
            lines.append(f"  `{cluster}`: {', '.join(s.replace('USDT','') for s in syms)}")

        return "\n".join(lines)
    except Exception as exc:
        return f"⚠ /correlation error: `{exc}`"


# ── /equity ───────────────────────────────────────────────────────────────────

def cmd_equity() -> str:
    """Equity protection state and drawdown summary."""
    try:
        import equity_protection as ep
        s = ep.get_summary()

        state_icon = {"normal": "✅", "selective": "⚠", "defensive": "🛑"}.get(s["state"], "?")
        lines = [
            f"*Equity Protection* | {_mode()}",
            f"Enabled: `{'YES' if s['enabled'] else 'No'}`",
            f"State: {state_icon} `{s['state'].upper()}`",
            f"Base grade: `{s.get('base_grade', '?')}`",
            f"Effective grade: `{s.get('effective_grade', '?')}`",
            f"Tightening: `{'YES' if s['tightening_active'] else 'No'}`",
            "",
            f"Consec losses: `{s.get('consecutive_losses', 0)}`",
            f"Consec wins:   `{s.get('consecutive_wins', 0)}`",
            f"Max drawdown:  `{s.get('max_drawdown_pct', 0.0):.2f}%`",
            f"Daily PnL:     `{_usdt(s.get('daily_pnl', 0.0))}`",
            f"Weekly PnL:    `{_usdt(s.get('weekly_pnl', 0.0))}`",
        ]
        return "\n".join(lines)
    except Exception as exc:
        return f"⚠ /equity error: `{exc}`"


# ── /health ───────────────────────────────────────────────────────────────────

def cmd_health() -> str:
    """Full system health: engines, equity, correlation, adaptive state."""
    try:
        import engine_ranker as er
        import equity_protection as ep
        import correlation_guard as cg

        lines = [f"*System Health* | {_mode()}"]

        # Adaptive weighting
        adapt_on = getattr(config, "ENABLE_ADAPTIVE_ENGINE_WEIGHTING", False)
        auto_dis  = getattr(config, "ENABLE_AUTO_DISABLE_ENGINES", False)
        eq_prot   = getattr(config, "ENABLE_EQUITY_PROTECTION", False)
        lines.append(
            f"\nAdaptive weighting: `{'ON' if adapt_on else 'off'}` | "
            f"Auto-disable: `{'ON' if auto_dis else 'off'}` | "
            f"Eq protection: `{'ON' if eq_prot else 'off'}`"
        )

        # Engine scores
        ranked = er.rank_engines(days=60)
        lines.append("\n*Engine Health Scores*")
        for r in ranked:
            n = r["trades"]
            bar = "█" * int(r["score"] / 10)
            dis = " [off]" if r["disabled"] else ""
            lines.append(
                f"  `{r['engine']:12s}`{dis} {bar} `{r['score']:.0f}`"
                + (f" ({n}T)" if n > 0 else " (no data)")
            )

        # Equity protection
        ep_sum = ep.get_summary()
        lines.append(
            f"\n*Equity*: `{ep_sum['state'].upper()}` | "
            f"DD=`{ep_sum.get('max_drawdown_pct', 0.0):.2f}%` | "
            f"grade≥`{ep_sum.get('effective_grade', '?')}`"
        )

        # Correlation
        guard_on = getattr(config, "ENABLE_CORRELATION_GUARD", True)
        max_corr = getattr(config, "MAX_CORRELATED_POSITIONS", 2)
        lines.append(
            f"*Correlation*: guard=`{'ON' if guard_on else 'off'}` | "
            f"max_correlated=`{max_corr}`"
        )

        # Min grade
        min_grade = getattr(config, "MIN_TRADE_GRADE", "A")
        lines.append(f"*Min grade*: `{min_grade}` | effective=`{ep_sum.get('effective_grade', min_grade)}`")

        return "\n".join(lines)
    except Exception as exc:
        return f"⚠ /health error: `{exc}`"


# ── /funnel ───────────────────────────────────────────────────────────────────

def cmd_funnel() -> str:
    """Setup funnel: scanned → rejected → executed with grade breakdown."""
    try:
        import rejection_analytics as ra
        f = ra.get_funnel()
        s = ra.get_summary()
        total = f["scanned"]
        if total == 0:
            return "*Setup Funnel*: no data yet."

        rej_pct  = round(f["rejected"] / total * 100) if total else 0
        exec_pct = round(f["executed"] / total * 100) if total else 0
        lines = [
            f"*Setup Funnel* (all-time)",
            f"",
            f"  Scanned:  `{total}`",
            f"  Rejected: `{f['rejected']}` ({rej_pct}%)",
            f"  Passed:   `{f['passed']}`",
            f"  Executed: `{f['executed']}` ({exec_pct}%)",
            f"",
            f"*Grade Breakdown*",
            f"  `A+` {f['grade_Aplus']}  `A` {f['grade_A']}  "
            f"`B` {f['grade_B']}  `C` {f['grade_C']}  `REJECT` {f['grade_REJECT']}",
        ]
        top = s["top_reasons"]
        if top:
            lines += ["", "*Top Rejection Reasons*"]
            for reason, count in top[:4]:
                pct = round(count / max(f["rejected"], 1) * 100)
                lines.append(f"  `{reason[:40]}` — {pct}%")
        return "\n".join(lines)
    except Exception as exc:
        return f"⚠ /funnel error: `{exc}`"


# ── /frequency ────────────────────────────────────────────────────────────────

def cmd_frequency() -> str:
    """Trade frequency analytics: setups/day, executions/day."""
    try:
        import rejection_analytics as ra
        freq = ra.get_frequency_stats()
        series = ra.get_daily_series(days=7)

        lines = [
            f"*Trade Frequency* (7-day avg)",
            f"",
            f"  Setups/day:  `{freq['avg_scanned_per_day_7d']:.1f}`",
            f"  Trades/day:  `{freq['avg_executed_per_day_7d']:.1f}`",
            f"",
            f"*Daily Breakdown* (last 7 days)",
        ]
        dates   = series["dates"][-7:]
        scanned = series["scanned"][-7:]
        executed = series["executed"][-7:]
        for i, d in enumerate(dates):
            lines.append(
                f"  `{d[-5:]}` — setups: {scanned[i]}  executed: {executed[i]}"
            )
        return "\n".join(lines)
    except Exception as exc:
        return f"⚠ /frequency error: `{exc}`"


# ── /strategies ───────────────────────────────────────────────────────────────

def cmd_strategies() -> str:
    """Per-strategy scan/execute breakdown."""
    try:
        import rejection_analytics as ra
        freq = ra.get_frequency_stats()
        by_strat = freq.get("by_strategy", {})

        if not by_strat:
            return "*Strategies*: no per-strategy data yet."

        lines = [f"*Strategy Breakdown* (all-time)"]
        for strat, counts in sorted(by_strat.items(), key=lambda x: -x[1]["scanned"]):
            sc = counts["scanned"]
            ex = counts["executed"]
            rj = counts["rejected"]
            exec_pct = round(ex / sc * 100) if sc > 0 else 0
            lines.append(
                f"  `{strat}` — scanned {sc} | executed {ex} ({exec_pct}%) | rejected {rj}"
            )
        return "\n".join(lines)
    except Exception as exc:
        return f"⚠ /strategies error: `{exc}`"


# ── /engines ──────────────────────────────────────────────────────────────────

def cmd_engines() -> str:
    """Show which setup engines are currently enabled/disabled."""
    try:
        flags = {
            "RMR (range mean-reversion)":  getattr(config, "ENABLE_RANGE_MR", False),
            "Pullback continuation":        getattr(config, "ENABLE_PULLBACK_SETUP", False),
            "Volatility breakout":          getattr(config, "ENABLE_BREAKOUT_SETUP", False),
            "NY open momentum":             getattr(config, "ENABLE_NY_MOMENTUM_SETUP", False),
            "Mean-reversion micro":         getattr(config, "ENABLE_MEAN_REVERSION_SETUP", False),
        }
        min_grade = getattr(config, "MIN_TRADE_GRADE", "A")
        expanded  = getattr(config, "ENABLE_EXPANDED_SYMBOLS", False)
        conf_15m  = getattr(config, "ENABLE_15M_CONFIRMATION", False)

        lines = [f"*Setup Engines* | min_grade=`{min_grade}`"]
        lines.append(
            f"Symbols: `{', '.join(config.SYMBOLS[:5])}{'…' if len(config.SYMBOLS) > 5 else ''}`"
        )
        lines.append(f"Expanded symbols: `{'ON' if expanded else 'off'}`")
        lines.append(f"15m confirmation: `{'ON' if conf_15m else 'off'}`")
        lines.append("")
        for name, enabled in flags.items():
            icon = "✅" if enabled else "⬜"
            lines.append(f"  {icon} `{name}`")
        active = sum(1 for v in flags.values() if v)
        lines.append(f"\n_Active engines: {active}/{len(flags)}_")
        return "\n".join(lines)
    except Exception as exc:
        return f"⚠ /engines error: `{exc}`"


# ── /setups ───────────────────────────────────────────────────────────────────

def cmd_setups() -> str:
    """Combined funnel + frequency + engine status in one compact view."""
    try:
        import rejection_analytics as ra
        freq   = ra.get_frequency_stats()
        funnel = ra.get_funnel()

        total  = funnel["scanned"]
        exec_n = funnel["executed"]
        rej_pct = round(funnel["rejected"] / max(total, 1) * 100)

        flags = {
            "RMR":       getattr(config, "ENABLE_RANGE_MR", False),
            "PULLBACK":  getattr(config, "ENABLE_PULLBACK_SETUP", False),
            "BREAKOUT":  getattr(config, "ENABLE_BREAKOUT_SETUP", False),
            "NY_MOM":    getattr(config, "ENABLE_NY_MOMENTUM_SETUP", False),
            "MICRO_MR":  getattr(config, "ENABLE_MEAN_REVERSION_SETUP", False),
        }
        active_engines = [k for k, v in flags.items() if v]
        min_grade = getattr(config, "MIN_TRADE_GRADE", "A")

        lines = [
            f"*Setup Summary* | {_mode()} | grade≥`{min_grade}`",
            f"",
            f"Engines: `{', '.join(active_engines) or 'RMR only'}`",
            f"Symbols: `{len(config.SYMBOLS)}`"
            + (f" (+expanded)" if getattr(config, "ENABLE_EXPANDED_SYMBOLS", False) else ""),
            f"",
            f"Funnel (all-time)",
            f"  Scanned `{total}` → Rejected `{funnel['rejected']}` ({rej_pct}%)"
            f" → Executed `{exec_n}`",
            f"",
            f"7d avg  — setups/day `{freq['avg_scanned_per_day_7d']:.1f}`"
            f"  trades/day `{freq['avg_executed_per_day_7d']:.1f}`",
        ]
        by_strat = freq.get("by_strategy", {})
        if by_strat:
            lines.append("")
            lines.append("*By Strategy*")
            for strat, counts in sorted(by_strat.items(), key=lambda x: -x[1]["executed"]):
                ep = round(counts["executed"] / max(counts["scanned"], 1) * 100)
                lines.append(
                    f"  `{strat}` — {counts['scanned']} scanned | {counts['executed']} executed ({ep}%)"
                )
        return "\n".join(lines)
    except Exception as exc:
        return f"⚠ /setups error: `{exc}`"


# ── /help ─────────────────────────────────────────────────────────────────────

def cmd_anomalies() -> str:
    """Active anomalies and system auto-response status."""
    try:
        import anomaly_detector as ad
        summary = ad.get_anomaly_summary()
        active  = summary.get("active", [])

        lines = [
            f"*Anomaly Detector* | total=`{summary.get('total',0)}` "
            f"warn=`{summary.get('warnings',0)}` crit=`{summary.get('criticals',0)}`",
            "",
        ]
        if summary.get("pause_entries"):
            lines.append("⛔ *ENTRIES PAUSED* — CRITICAL anomaly active")
        elif summary.get("reduce_aggressiveness"):
            lines.append("⚠ *Aggressiveness reduced* — WARNING anomaly active (risk ×0.75)")
        else:
            lines.append("✅ No auto-response active")
        lines.append("")

        if not active:
            lines.append("_No active anomalies._")
        else:
            sev_icon = {"WARNING": "⚠", "CRITICAL": "🔴"}
            for a in active:
                icon = sev_icon.get(a.get("severity",""), "⚪")
                lines.append(
                    f"{icon} `{a.get('anomaly_type','?')}` | `{a.get('symbol','?')}` | "
                    f"×{a.get('count',1)} | {a.get('message','')}"
                )
        return "\n".join(lines)
    except Exception as exc:
        return f"⚠ /anomalies error: `{exc}`"


def cmd_confidence() -> str:
    """System confidence score and trading behavior state."""
    try:
        import confidence_score as cs
        summary = cs.get_confidence_summary()
        score  = summary.get("score", 0.0)
        state  = summary.get("state", "?")
        scale  = summary.get("risk_scale", 1.0)
        comps  = summary.get("components", {})
        thresh = summary.get("thresholds", {})

        state_icon = {
            cs.NORMAL:    "✅",
            cs.CAUTIOUS:  "⚠",
            cs.DEFENSIVE: "⛔",
        }
        icon = state_icon.get(state, "⚪")

        lines = [
            f"*System Confidence* {icon}",
            f"Score: `{score:.0f}/100` [{state}]  risk_scale=`{scale:.2f}`",
            f"Thresholds: normal≥`{thresh.get('normal',70)}` cautious≥`{thresh.get('cautious',40)}`",
            "",
            "*Components*",
        ]
        for k, v in comps.items():
            bar_filled = "█" * int(v / 10) + "░" * (10 - int(v / 10))
            lines.append(f"  `{k:<24}` `{v:5.1f}` [{bar_filled}]")

        lines.append("")
        if state == cs.DEFENSIVE:
            lines.append("_⛔ No new entries until confidence recovers above 40._")
        elif state == cs.CAUTIOUS:
            lines.append("_⚠ Risk scaled down, grade floor tightened by 1 level._")
        else:
            lines.append("_✅ Full operation — no restrictions._")

        return "\n".join(lines)
    except Exception as exc:
        return f"⚠ /confidence error: `{exc}`"


def cmd_livevshadow(days: int = 30) -> str:
    """Live vs shadow comparative analytics."""
    try:
        import shadow_analytics as sa
        summary      = sa.get_comparison_summary(days)
        engines      = summary.get("engines", {})
        outperformers = summary.get("outperforming", [])

        lines = [
            f"*Live vs Shadow* | last `{days}d`",
            f"Live trades: `{summary.get('total_live_trades',0)}` | "
            f"Shadow trades: `{summary.get('total_shadow_trades',0)}`",
            "",
        ]

        if not engines:
            lines.append("_No comparison data yet._")
            return "\n".join(lines)

        lines.append("*Per-Engine* (live exp vs shadow exp%)")
        for eng, data in sorted(engines.items()):
            live   = data.get("live",   {})
            shadow = data.get("shadow", {})
            delta  = data.get("delta_expectancy")
            flag   = " ★" if data.get("outperforms") else ""
            l_t = live.get("trades", 0)
            s_t = shadow.get("trades", 0)
            l_e = live.get("expectancy")
            s_e = shadow.get("expectancy_pct")
            l_str = f"{l_e:+.4f}" if l_e is not None else "—"
            s_str = f"{s_e:+.4f}%" if s_e is not None else "—"
            d_str = f"{delta:+.4f}" if delta is not None else "—"
            lines.append(
                f"  `{eng}` L:{l_t}T `{l_str}` | S:{s_t}T `{s_str}` | Δ`{d_str}`{flag}"
            )

        if outperformers:
            lines.append("")
            lines.append(f"*★ Outperformers ({len(outperformers)})*")
            for item in outperformers:
                eng   = item.get("engine","?")
                delta = item.get("delta_expectancy", 0.0)
                lines.append(f"  `{eng}` shadow outperforms live by `{delta:+.4f}`")
            lines.append("_→ Consider promoting to live testing._")

        return "\n".join(lines)
    except Exception as exc:
        return f"⚠ /livevshadow error: `{exc}`"


def cmd_governor() -> str:
    """Engine governor — tier summary (TRUSTED/NEUTRAL/PROBATION)."""
    try:
        import engine_governor as eg
        summary = eg.get_tier_summary()
        engines = summary.get("engines", {})
        counts  = summary.get("tier_counts", {})

        lines = ["*Engine Governor*"]
        lines.append(
            f"TRUSTED: `{counts.get('TRUSTED',0)}` | "
            f"NEUTRAL: `{counts.get('NEUTRAL',0)}` | "
            f"PROBATION: `{counts.get('PROBATION',0)}`"
        )
        lines.append("")

        tier_icon = {eg.TRUSTED: "✅", eg.NEUTRAL: "⚪", eg.PROBATION: "🔴"}
        for eng, data in sorted(engines.items()):
            tier  = data.get("tier", "NEUTRAL")
            score = data.get("score", 0.0)
            since = str(data.get("since", ""))[:10]
            icon  = tier_icon.get(tier, "⚪")
            lines.append(f"{icon} `{eng}` [{tier}] score=`{score}` since=`{since}`")
            for ev in data.get("history", [])[-2:]:
                lines.append(f"    _{ev.get('tier','?')} — {ev.get('reason','')[:60]}_")

        return "\n".join(lines)
    except Exception as exc:
        return f"⚠ /governor error: `{exc}`"


def cmd_shadow() -> str:
    """Shadow lab — paper-trade simulation stats."""
    try:
        import shadow_engine as se
        summary = se.get_shadow_summary()
        engines_data = summary.get("engines", {})

        lines = [
            f"*Shadow Lab* | open=`{summary.get('total_open',0)}` "
            f"closed=`{summary.get('total_closed',0)}`",
            "",
        ]
        if not engines_data:
            lines.append("_No shadow data yet._")
            return "\n".join(lines)

        for eng, s in sorted(engines_data.items()):
            n  = s.get("trades", 0)
            wr = s.get("win_rate", 0.0) * 100
            exp = s.get("expectancy", 0.0)
            pnl = s.get("total_pnl_pct", 0.0)
            op  = s.get("open", 0)
            lines.append(
                f"`{eng}` — {n}T WR`{wr:.0f}%` "
                f"exp=`{exp:+.3f}%` pnl=`{pnl:+.2f}%` open=`{op}`"
            )
        return "\n".join(lines)
    except Exception as exc:
        return f"⚠ /shadow error: `{exc}`"


def cmd_shadowreport() -> str:
    """Shadow lab — open shadow positions."""
    try:
        import shadow_engine as se
        open_trades = se.get_open_shadows()
        if not open_trades:
            return "*Shadow Lab* — _No open shadow trades._"

        lines = [f"*Shadow Lab* | {len(open_trades)} open position(s)", ""]
        for t in open_trades[:15]:
            lines.append(
                f"`{t['engine']}` {t['symbol']} {t['direction']} "
                f"@ `{t['entry']:.4f}` SL`{t['stop']:.4f}` TP`{t['tp']:.4f}`"
            )
        return "\n".join(lines)
    except Exception as exc:
        return f"⚠ /shadowreport error: `{exc}`"


def cmd_sentiment() -> str:
    """Sentiment heatmap — CoinGecko sentiment per symbol."""
    try:
        import sentiment_engine as se
        summary = se.get_sentiment_summary()
        symbols_data = summary.get("symbols", {})

        lines = [
            f"*Sentiment Heatmap* | avg=`{summary.get('avg_score',0.0):+.2f}` "
            f"({summary.get('avg_label','neutral')}) "
            f"[cache age `{summary.get('cache_age_s',0):.0f}s`]",
            "",
        ]
        if not symbols_data:
            lines.append("_No sentiment data (CoinGecko unavailable)._")
            return "\n".join(lines)

        label_icon = {
            "greed": "🟢", "mild_bullish": "🔼",
            "neutral": "⚪", "mild_bearish": "🔽", "fear": "🔴",
        }
        for sym, d in sorted(symbols_data.items()):
            icon = label_icon.get(d.get("label", "neutral"), "⚪")
            lines.append(
                f"{icon} `{sym.replace('USDT','')}` "
                f"score=`{d['score']:+.2f}` "
                f"modifier=`{d['modifier']:+.1f}` "
                f"24h=`{d['change_24h']:+.1f}%`"
            )
        return "\n".join(lines)
    except Exception as exc:
        return f"⚠ /sentiment error: `{exc}`"


def cmd_trending() -> str:
    """Top bullish and bearish symbols by sentiment."""
    try:
        import sentiment_engine as se
        summary = se.get_sentiment_summary()
        bullish = summary.get("top_bullish", [])
        bearish = summary.get("top_bearish", [])

        lines = ["*Sentiment Trending*", ""]
        if bullish:
            lines.append("*Bullish ↑*")
            for sym in bullish:
                d = summary.get("symbols", {}).get(sym, {})
                lines.append(f"  `{sym}` score=`{d.get('score',0):+.2f}` 24h=`{d.get('change_24h',0):+.1f}%`")
            lines.append("")
        if bearish:
            lines.append("*Bearish ↓*")
            for sym in bearish:
                d = summary.get("symbols", {}).get(sym, {})
                lines.append(f"  `{sym}` score=`{d.get('score',0):+.2f}` 24h=`{d.get('change_24h',0):+.1f}%`")

        return "\n".join(lines) if len(lines) > 2 else "*Sentiment Trending* — _no data_"
    except Exception as exc:
        return f"⚠ /trending error: `{exc}`"


def cmd_portfolio(engines: dict = None) -> str:
    """Portfolio brain — sector exposure and health score."""
    try:
        import portfolio_brain as pb
        open_syms = []
        if engines:
            open_syms = [s for s, e in engines.items() if e.has_open_position()]

        summary = pb.get_portfolio_summary(open_syms)
        health  = summary.get("health_score", 0.0)
        label   = summary.get("health_label", "?")
        sector_exp = summary.get("sector_exposure", {})
        comps   = summary.get("components", {})

        lines = [
            f"*Portfolio Brain* | health=`{health:.0f}/100` [{label}]",
            f"Open positions: `{summary.get('open_positions',0)}`",
            "",
        ]
        if sector_exp:
            lines.append("*Sector Exposure*")
            for sec, d in sorted(sector_exp.items()):
                over = " ⚠ OVER LIMIT" if d.get("over_limit") else ""
                lines.append(
                    f"  `{sec}` {d['count']} pos ({d['exposure']*100:.0f}%){over}"
                )
            lines.append("")

        lines.append("*Health Components*")
        for k, v in comps.items():
            lines.append(f"  {k}: `{v:.0f}`")

        return "\n".join(lines)
    except Exception as exc:
        return f"⚠ /portfolio error: `{exc}`"


def cmd_avoidance() -> str:
    """Market avoidance — current environment conditions (no live data without df)."""
    try:
        lines = [
            "*Market Avoidance*",
            "_Checks run per-symbol during scan. Conditions detected:_",
            "",
            "  `DEAD_LIQUIDITY`  — vol < 20% of avg → CRITICAL (A+ required)",
            "  `VOLATILITY_CHAOS` — ATR > 3.5× avg → CRITICAL",
            "  `POST_NEWS_CHOP`  — sharp move + indecisive bars → WARNING (A)",
            "  `EXHAUSTED_TREND` — 5+ bar trend + declining vol → CAUTION",
            "  `WEEKEND_THIN`    — Sat/Sun 22:00-08:00 UTC → CAUTION",
            "  `SPREAD_EXPLOSION` — avg wick > 75% of bar range → WARNING",
            "",
            "_Active results appear in bot logs per scan cycle._",
        ]
        return "\n".join(lines)
    except Exception as exc:
        return f"⚠ /avoidance error: `{exc}`"


def cmd_weekly() -> str:
    """Weekly intelligence report — last 7 days."""
    try:
        import weekly_intelligence_report as wir
        return wir.generate_weekly_report()
    except Exception as exc:
        return f"⚠ /weekly error: `{exc}`"


def cmd_memory() -> str:
    """Learning memory — strongest/weakest engine × regime × session pairs."""
    try:
        import learning_memory as lm
        summary = lm.get_memory_summary()

        lines = [
            f"*Learning Memory* | patterns=`{summary.get('total_patterns',0)}` "
            f"[min_sample={summary.get('min_sample',8)}]",
            "",
        ]

        strongest = summary.get("strongest_pairs", [])
        if strongest:
            lines.append("*Strongest Pairs ↑*")
            for p in strongest:
                lines.append(
                    f"  `{p['engine']}` × `{p['regime']}` × `{p['session']}` "
                    f"exp=`{p['expectancy']:+.4f}` ({p['trades']}T) "
                    f"mod=`{p['modifier']:+.1f}`"
                )
            lines.append("")

        weakest = summary.get("weakest_pairs", [])
        if weakest:
            lines.append("*Weakest Pairs ↓*")
            for p in weakest:
                lines.append(
                    f"  `{p['engine']}` × `{p['regime']}` × `{p['session']}` "
                    f"exp=`{p['expectancy']:+.4f}` ({p['trades']}T) "
                    f"mod=`{p['modifier']:+.1f}`"
                )
            lines.append("")

        eng_avg = summary.get("engine_avg_expectancy", {})
        if eng_avg:
            lines.append("*Engine Avg Expectancy*")
            for eng, avg in sorted(eng_avg.items(), key=lambda x: -x[1]):
                icon = "↑" if avg >= 0 else "↓"
                lines.append(f"  {icon} `{eng}` `{avg:+.4f}`")

        return "\n".join(lines) if len(lines) > 2 else "*Learning Memory* — _insufficient data_"
    except Exception as exc:
        return f"⚠ /memory error: `{exc}`"


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
        "/enginerank — Engine leaderboard ranked by health score\n"
        "/expectancy — Per-engine expectancy breakdown\n"
        "/leaderboard — Full engine learning summary\n"
        "/marketstate — Market state reference and engine affinity\n"
        "/correlation — Correlated position exposure\n"
        "/equity — Equity protection state and drawdown\n"
        "/health — Full system health (engines, equity, correlation)\n"
        "/funnel — Setup funnel with grade breakdown\n"
        "/frequency — Setups/day and trades/day (7d)\n"
        "/strategies — Per-strategy scan/execute breakdown\n"
        "/engines — Which setup engines are active\n"
        "/setups — Combined setup summary (engines + funnel + frequency)\n"
        "\n"
        "*Operational Intelligence*\n"
        "/anomalies — Active anomalies and auto-response status\n"
        "/confidence — System confidence score (0-100) and trading state\n"
        "/livevshadow — Live vs shadow comparative analytics\n"
        "\n"
        "*Portfolio Intelligence*\n"
        "/governor — Engine tier system (TRUSTED/NEUTRAL/PROBATION)\n"
        "/shadow — Shadow lab paper-trade simulation stats\n"
        "/shadowreport — Open shadow positions\n"
        "/sentiment — CoinGecko sentiment heatmap per symbol\n"
        "/trending — Top bullish/bearish symbols by sentiment\n"
        "/portfolio — Sector exposure and portfolio health score\n"
        "/avoidance — Market avoidance conditions reference\n"
        "/weekly — Weekly intelligence report (last 7 days)\n"
        "/memory — Learning memory: strongest/weakest engine×regime×session\n"
        "\n"
        "*Session 6 — Advanced Strategies*\n"
        "/scalp — Intraday scalp engine status and recent signals\n"
        "/grid — Virtual grid tracker: levels, hits, virtual PnL\n"
        "/funding — Funding rate arb: extreme rates + market bias\n"
        "/defi — DeFi ecosystem: TVL momentum + top yield pools\n"
        "/social — Social sentiment: trending coins + Fear/Greed index\n"
        "/intel — Combined hourly intel: trending coins + arb opportunities\n"
        "\n"
        "_All commands restricted to authorised chat ID only._"
    )


# ── /scalp ────────────────────────────────────────────────────────────────────

def cmd_scalp() -> str:
    try:
        import config as _cfg
        lines = ["*Intraday Scalp Engine*"]
        enabled = getattr(_cfg, "ENABLE_INTRADAY_SCALP", False)
        lines.append(f"Status: `{'ENABLED' if enabled else 'DISABLED'}`")
        if not enabled:
            lines.append("_Set ENABLE\\_INTRADAY\\_SCALP=true in .env to activate_")
            lines.append("\nEngine: `INTRADAY_SCALP`")
            lines.append("Session: `07:00-20:00 UTC`")
            lines.append("TP/SL ratio: `1.2R` (scalp-tight)")
            lines.append("Conditions: EMA9>EMA21 | RSI 45-62 | 3-bar close accel | ADX<35")
        else:
            lines.append("\nEngine: `INTRADAY_SCALP`")
            lines.append("Session gate: `07:00-20:00 UTC`")
            lines.append("TP ratio: `1.2R` | Stop: `EMA9 - 0.3×ATR`")
            lines.append("Rank boost: `+45` when fired")
            lines.append("\n_Signals appear in /status and /engines when active_")
        return "\n".join(lines)
    except Exception as exc:
        return f"*Scalp*: error — `{exc}`"


# ── /grid ─────────────────────────────────────────────────────────────────────

def cmd_grid() -> str:
    try:
        import grid_engine as _ge
        import config as _cfg
        enabled = getattr(_cfg, "ENABLE_GRID_ENGINE", False)
        lines = ["*Virtual Grid Tracker*"]
        lines.append(f"Status: `{'ENABLED' if enabled else 'DISABLED'}`")
        lines.append(f"Mode: `SHADOW (no live orders)`")
        if not enabled:
            lines.append("_Set ENABLE\\_GRID\\_ENGINE=true in .env to activate_")
            return "\n".join(lines)

        all_grids = _ge.get_all_grid_status()
        for sym, gs in all_grids.items():
            if gs.get("status") == "inactive":
                lines.append(f"\n`{sym}`: not initialised yet")
                continue
            lines.append(f"\n`{sym}`")
            lines.append(f"  Center: `${gs.get('center_price',0):,.2f}` | ATR: `{gs.get('atr',0):.2f}`")
            lines.append(f"  Levels: `{gs.get('open_levels',0)} open` / `{gs.get('total_levels',0)} total`")
            lines.append(f"  Open positions: `{gs.get('open_positions',0)}`")
            lines.append(f"  Hits: `{gs.get('hits',0)}` | Virtual PnL: `{gs.get('total_virtual_pnl',0):+.4f}`")
            next_buy = gs.get("next_buy_level", 0)
            if next_buy:
                lines.append(f"  Next buy level: `${next_buy:,.2f}`")
        return "\n".join(lines)
    except Exception as exc:
        return f"*Grid*: error — `{exc}`"


# ── /funding ──────────────────────────────────────────────────────────────────

def cmd_funding() -> str:
    try:
        import funding_arb as _fa
        import config as _cfg
        enabled = getattr(_cfg, "ENABLE_FUNDING_ARB", False)
        lines = ["*Funding Rate Arbitrage*"]
        lines.append(f"Status: `{'ENABLED (watch-only)' if enabled else 'DISABLED'}`")
        lines.append("`AUTO_TRADE: locked OFF`")

        summary = _fa.get_funding_summary()
        bias    = summary.get("market_funding_bias", "neutral")
        avg     = summary.get("avg_rate_pct", 0.0)
        extreme = summary.get("extreme_count", 0)
        age     = summary.get("cache_age_s", 0)

        lines.append(f"\nMarket bias: `{bias.replace('_',' ').upper()}`")
        lines.append(f"Avg rate: `{avg:+.5f}%` | Extreme signals: `{extreme}`")
        lines.append(f"Data age: `{int(age)}s ago`")

        signals = summary.get("arb_signals", [])
        if signals:
            lines.append("\n*Top Arb Opportunities*")
            for s in signals[:5]:
                lines.append(
                    f"`{s['symbol']}` [{s['strength']}] {s['direction'].replace('_',' ')}\n"
                    f"  Rate `{s['rate_pct']:+.4f}%` | Ann `{s['annualized_pct']:+.1f}%`\n"
                    f"  Next funding: `{s['next_funding']}`"
                )
        else:
            lines.append("\n_No notable arb opportunities right now_")

        lines.append(f"\n*All Rates*")
        rates = sorted(summary.get("rates", {}).values(),
                       key=lambda x: abs(x.get("rate_pct", 0)), reverse=True)
        for r in rates[:8]:
            lines.append(f"`{r['symbol']}`: `{r['rate_pct']:+.4f}%`")

        return "\n".join(lines)
    except Exception as exc:
        return f"*Funding*: error — `{exc}`"


# ── /defi ─────────────────────────────────────────────────────────────────────

def cmd_defi() -> str:
    try:
        import defi_signals as _ds
        import config as _cfg
        enabled = getattr(_cfg, "ENABLE_DEFI_SIGNALS", False)
        lines = ["*DeFi Ecosystem Signals*"]
        lines.append(f"Status: `{'ENABLED' if enabled else 'DISABLED'}`")
        lines.append("Source: `DeFiLlama (free API)`")

        summary = _ds.get_defi_summary()
        age     = summary.get("cache_age_s", 0)
        env     = summary.get("yield_environment", "unknown")
        avg_apy = summary.get("avg_top10_apy", 0.0)

        lines.append(f"Yield env: `{env.replace('_',' ').upper()}`")
        lines.append(f"Avg top-10 APY: `{avg_apy:.1f}%`")
        lines.append(f"Data age: `{int(age)}s ago`")

        chains = summary.get("chains", {})
        if chains:
            lines.append("\n*Chain TVL Momentum (24h)*")
            for name, data in sorted(chains.items(), key=lambda x: x[1].get("change_1d", 0), reverse=True):
                chg   = data.get("change_1d", 0.0)
                tvl   = data.get("tvl_usd", 0.0)
                health= data.get("health", "stable")
                icon  = "+" if chg > 0 else ""
                lines.append(f"`{name}` [{health}]: `{icon}{chg:.2f}%` TVL `${tvl/1e9:.2f}B`")

        pools = summary.get("top_yield_pools", [])
        if pools:
            lines.append("\n*Top Yield Pools (by TVL)*")
            for p in pools[:5]:
                lines.append(
                    f"`{p.get('project','')}` [{p.get('chain','')}]: "
                    f"`{p.get('apy',0):.1f}% APY` | TVL `${p.get('tvl_usd',0)/1e6:.0f}M`"
                )
        return "\n".join(lines)
    except Exception as exc:
        return f"*DeFi*: error — `{exc}`"


# ── /social ───────────────────────────────────────────────────────────────────

def cmd_social() -> str:
    try:
        import social_sentiment as _ss
        import config as _cfg
        enabled = getattr(_cfg, "ENABLE_SOCIAL_SENTIMENT", False)
        lines = ["*Social Sentiment*"]
        lines.append(f"Status: `{'ENABLED' if enabled else 'DISABLED'}`")
        lines.append("Sources: `CoinGecko Trending + Fear/Greed Index`")

        summary = _ss.get_social_summary()

        fng   = summary.get("fear_greed", {})
        fng_v = fng.get("value", 50)
        fng_l = fng.get("label", "Neutral")
        lines.append(f"\n*Fear & Greed Index*: `{fng_v}/100` — `{fng_l}`")

        # Progress bar
        bars = int(fng_v / 10)
        bar  = "█" * bars + "░" * (10 - bars)
        lines.append(f"`[{bar}]`")

        glb = summary.get("global_market", {})
        if glb:
            mkt_chg = glb.get("market_cap_change_24h", 0.0)
            btc_dom = glb.get("btc_dominance", 0.0)
            lines.append(f"Market cap 24h: `{mkt_chg:+.2f}%` | BTC dom: `{btc_dom:.1f}%`")

        coins = summary.get("trending_coins", [])
        if coins:
            lines.append("\n*Trending Coins (CoinGecko)*")
            for c in coins:
                sym  = c.get("symbol", "")
                name = c.get("name", "")
                rank = c.get("rank", 9999)
                usdt = c.get("usdt_sym", "")
                rank_str = f" (MCap #{rank})" if rank < 500 else ""
                lines.append(f"• `{sym}` — {name}{rank_str} (`{usdt}`)")
        else:
            lines.append("\n_Trending data not yet loaded_")

        signals = summary.get("buy_signals", [])
        if signals:
            lines.append(f"\n*Buy Signals*: `{len(signals)}` trending coins meeting criteria")
            for s in signals[:3]:
                strength = s.get("strength", "")
                lines.append(f"• [{strength}] `{s.get('symbol','')}` — {s.get('name','')}")
        else:
            lines.append("\n_No buy signals from trending coins right now_")

        return "\n".join(lines)
    except Exception as exc:
        return f"*Social*: error — `{exc}`"


# ── /intel (combined hourly summary) ──────────────────────────────────────────

def cmd_intel_summary() -> str:
    """
    Compact hourly intel summary for auto-posting and /intel command.
    Combines: trending coins, fear/greed, funding arb highlights.
    """
    try:
        import social_sentiment as _ss
        import funding_arb as _fa
        from datetime import datetime, timezone

        now_str = datetime.now(timezone.utc).strftime("%H:%M UTC")
        lines   = [f"*Hourly Intel* | {now_str}"]

        # Fear & Greed
        fng   = _ss.get_fear_greed()
        fng_v = fng.get("value", 50)
        fng_l = fng.get("label", "Neutral")
        lines.append(f"Fear/Greed: `{fng_v}/100` {fng_l}")

        # Top 3 trending
        coins = _ss.get_trending_coins()[:3]
        if coins:
            trend_str = " · ".join(c.get("symbol", "") for c in coins)
            lines.append(f"Trending: `{trend_str}`")

        # Funding arb top signal
        arb = _fa.get_arb_signals(min_rate_pct=0.02)
        if arb:
            top = arb[0]
            lines.append(
                f"Top funding: `{top['symbol']}` `{top['rate_pct']:+.4f}%` "
                f"({top['direction'].replace('_',' ')})"
            )

        # Buy signals
        signals = _ss.trending_buy_signals()
        if signals:
            sig_str = " · ".join(s.get("symbol", "") for s in signals[:3])
            lines.append(f"Social signals: `{sig_str}`")

        return "\n".join(lines)
    except Exception as exc:
        return f"*Intel*: error — `{exc}`"
