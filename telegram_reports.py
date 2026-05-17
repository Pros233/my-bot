"""
telegram_reports.py — Daily PDF performance report via matplotlib PdfPages.

Returns PDF bytes suitable for telegram_bot.send_document().
Never raises — returns None on any failure.

Requires: matplotlib (pip install matplotlib)
"""
from __future__ import annotations

import io
from datetime import datetime, timezone
from typing import Optional

import config
import logger


def generate_daily_pdf() -> Optional[bytes]:
    """
    Generate a multi-page PDF report covering:
      Page 1 — Headline stats table + equity curve
      Page 2 — Daily PnL bar chart (last 30 days)
      Page 3 — Per-symbol breakdown table

    Returns PDF bytes or None.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        import performance

        today   = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        iso     = datetime.now(timezone.utc).isocalendar()
        week    = f"{iso[0]}-W{iso[1]:02d}"
        mode    = "TESTNET" if config.TESTNET else "LIVE"

        daily   = performance.daily_pnl(today)
        weekly  = performance.weekly_pnl(week)
        total   = performance.total_pnl()
        n       = performance.total_trades()
        wr      = performance.win_rate()
        pf      = performance.profit_factor()
        exp     = performance.expectancy()
        dd      = performance.max_drawdown_pct()
        avg_w   = performance.average_win()
        avg_l   = performance.average_loss()
        sharpe  = performance.sharpe_like_ratio()
        cl      = performance.consecutive_losses()
        cw      = performance.consecutive_wins()

        pf_str  = f"{pf:.2f}" if pf != float("inf") else "∞"

        buf = io.BytesIO()
        with PdfPages(buf) as pdf:

            # ── Page 1: Headline stats + equity curve ─────────────────────────
            fig, (ax_tbl, ax_eq) = plt.subplots(
                2, 1, figsize=(8.27, 11.69),
                facecolor="#0d1117",
                gridspec_kw={"height_ratios": [1, 2]},
            )
            fig.suptitle(
                f"BTC Bot Daily Report — {today}  [{mode}]",
                color="#e6edf3", fontsize=14, fontweight="bold", y=0.97,
            )

            # Stats table
            ax_tbl.set_facecolor("#0d1117")
            ax_tbl.axis("off")
            stats = [
                ["Today PnL",     f"${daily:+.4f}"],
                ["Week PnL",      f"${weekly:+.4f}"],
                ["All-time PnL",  f"${total:+.2f}"],
                ["Total trades",  str(n)],
                ["Win rate",      f"{wr*100:.1f}%"],
                ["Profit factor", pf_str],
                ["Expectancy",    f"${exp:+.4f}"],
                ["Max drawdown",  f"{dd:.2f}%"],
                ["Avg win",       f"${avg_w:+.4f}"],
                ["Avg loss",      f"${avg_l:+.4f}"],
                ["Sharpe proxy",  f"{sharpe:.3f}"],
                ["Consec wins",   str(cw)],
                ["Consec losses", str(cl)],
            ]
            tbl = ax_tbl.table(
                cellText=stats,
                colLabels=["Metric", "Value"],
                cellLoc="left",
                loc="center",
                colWidths=[0.55, 0.35],
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(9)
            for (row, col), cell in tbl.get_celld().items():
                cell.set_facecolor("#161b22")
                cell.set_edgecolor("#30363d")
                cell.set_text_props(color="#e6edf3")

            # Equity curve
            all_trades = performance.recent_trades(limit=500)
            if all_trades:
                pnls   = [t["pnl_usdt"] for t in reversed(all_trades)]
                equity = []
                cumsum = 0.0
                for p in pnls:
                    cumsum += p
                    equity.append(cumsum)
                ax_eq.set_facecolor("#161b22")
                for spine in ax_eq.spines.values():
                    spine.set_edgecolor("#30363d")
                color = "#3fb950" if equity[-1] >= 0 else "#f85149"
                ax_eq.plot(equity, color=color, linewidth=1.5)
                ax_eq.axhline(0, color="#8b949e", linewidth=0.5, linestyle="--")
                ax_eq.fill_between(range(len(equity)), equity, 0,
                                   alpha=0.15, color=color)
                ax_eq.set_title("Equity Curve (all trades)", color="#e6edf3", fontsize=10)
                ax_eq.tick_params(colors="#8b949e")
                ax_eq.set_ylabel("PnL (USDT)", color="#8b949e", fontsize=8)
                ax_eq.yaxis.tick_right()
            else:
                ax_eq.set_facecolor("#161b22")
                ax_eq.text(0.5, 0.5, "No trades yet", ha="center", va="center",
                           color="#8b949e", transform=ax_eq.transAxes)
                ax_eq.axis("off")

            fig.tight_layout(rect=[0, 0, 1, 0.95])
            pdf.savefig(fig, facecolor="#0d1117")
            plt.close(fig)

            # ── Page 2: Daily PnL bar chart (last 30 days) ───────────────────
            daily_rows = performance.daily_summary()
            if daily_rows:
                fig2, ax2 = plt.subplots(figsize=(8.27, 5), facecolor="#0d1117")
                ax2.set_facecolor("#161b22")
                for spine in ax2.spines.values():
                    spine.set_edgecolor("#30363d")
                days  = [r["day"] for r in reversed(daily_rows)]
                pnls2 = [r["pnl"] for r in reversed(daily_rows)]
                colors = ["#3fb950" if p >= 0 else "#f85149" for p in pnls2]
                ax2.bar(range(len(days)), pnls2, color=colors, width=0.7)
                ax2.axhline(0, color="#8b949e", linewidth=0.5)
                ax2.set_xticks(range(len(days)))
                ax2.set_xticklabels(
                    [d[-5:] for d in days],
                    rotation=45, ha="right", fontsize=7, color="#8b949e",
                )
                ax2.set_title("Daily PnL — Last 30 Days", color="#e6edf3", fontsize=11)
                ax2.tick_params(colors="#8b949e")
                ax2.set_ylabel("PnL (USDT)", color="#8b949e", fontsize=8)
                ax2.yaxis.tick_right()
                fig2.tight_layout()
                pdf.savefig(fig2, facecolor="#0d1117")
                plt.close(fig2)

            # ── Page 3: Per-symbol table ──────────────────────────────────────
            by_sym = performance.pnl_by_symbol()
            if by_sym:
                fig3, ax3 = plt.subplots(figsize=(8.27, 4), facecolor="#0d1117")
                ax3.set_facecolor("#0d1117")
                ax3.axis("off")
                sym_data = [
                    [sym,
                     str(d["trades"]),
                     f"{d['win_rate']*100:.1f}%",
                     f"${d['total_pnl']:+.4f}",
                     f"${d['avg_pnl']:+.4f}"]
                    for sym, d in by_sym.items()
                ]
                tbl3 = ax3.table(
                    cellText=sym_data,
                    colLabels=["Symbol", "Trades", "WR", "Total PnL", "Avg PnL"],
                    cellLoc="center",
                    loc="center",
                    colWidths=[0.25, 0.12, 0.12, 0.22, 0.22],
                )
                tbl3.auto_set_font_size(False)
                tbl3.set_fontsize(9)
                for (row, col), cell in tbl3.get_celld().items():
                    cell.set_facecolor("#161b22")
                    cell.set_edgecolor("#30363d")
                    cell.set_text_props(color="#e6edf3")
                ax3.set_title("Per-Symbol Breakdown", color="#e6edf3", fontsize=11, pad=20)
                fig3.tight_layout()
                pdf.savefig(fig3, facecolor="#0d1117")
                plt.close(fig3)

        buf.seek(0)
        return buf.read()

    except ImportError:
        logger.log_warning("matplotlib not installed — daily PDF unavailable")
        return None
    except Exception as exc:
        logger.log_warning(f"telegram_reports.generate_daily_pdf failed: {exc}")
        return None
