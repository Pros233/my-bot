"""
telegram_charts.py — matplotlib price chart generation for /chart command.

Returns PNG bytes suitable for telegram_bot.send_photo().
Never raises — returns None on any failure.

Requires: matplotlib (pip install matplotlib)
"""
from __future__ import annotations

import io
from typing import Optional

import config
import logger


def generate_price_chart(symbol: str, client) -> Optional[bytes]:
    """
    Fetch 48H of 1H OHLC data for *symbol* and render a price chart with:
      - Candlestick bars (green/red)
      - EMA 9 and EMA 21 overlays
      - Volume bars at bottom
      - Entry/SL/TP horizontal lines if a position is open

    Returns PNG bytes or None.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import pandas as pd
        import pandas_ta as ta  # noqa: F401

        # ── Fetch candles ────────────────────────────────────────────────────
        klines = client.get_klines(symbol=symbol, interval="1h", limit=49)
        if not klines or len(klines) < 2:
            return None

        df = pd.DataFrame(klines, columns=[
            "ts", "open", "high", "low", "close", "volume",
            "close_time", "quote_vol", "trades", "tbbase", "tbquote", "ignore",
        ])
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = df[col].astype(float)
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df = df.iloc[:-1]  # drop forming candle

        # ── Indicators ───────────────────────────────────────────────────────
        df["ema9"]  = df["close"].ewm(span=9,  adjust=False).mean()
        df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()

        # ── Entry/SL/TP from live position ───────────────────────────────────
        entry_price: Optional[float] = None
        sl_price:    Optional[float] = None
        tp_price:    Optional[float] = None
        try:
            import telegram_bot
            eng = telegram_bot._engines.get(symbol)
            if eng and eng.has_open_position():
                p = eng.position
                entry_price = p.fill_price
                sl_price    = p.stop_price
                tp_price    = p.tp_price
        except Exception:
            pass

        # ── Layout ───────────────────────────────────────────────────────────
        fig, (ax_price, ax_vol) = plt.subplots(
            2, 1,
            figsize=(10, 6),
            gridspec_kw={"height_ratios": [3, 1]},
            facecolor="#0d1117",
        )
        for ax in (ax_price, ax_vol):
            ax.set_facecolor("#161b22")
            ax.tick_params(colors="#8b949e", labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor("#30363d")

        xs = range(len(df))

        # Candlestick bars
        for i, (_, row) in enumerate(df.iterrows()):
            color = "#3fb950" if row["close"] >= row["open"] else "#f85149"
            ax_price.plot([i, i], [row["low"], row["high"]], color=color, linewidth=0.8)
            ax_price.bar(i, abs(row["close"] - row["open"]),
                         bottom=min(row["open"], row["close"]),
                         color=color, width=0.6)

        # EMAs
        ax_price.plot(list(xs), df["ema9"].tolist(),  color="#58a6ff", linewidth=1,   label="EMA9")
        ax_price.plot(list(xs), df["ema21"].tolist(), color="#f78166", linewidth=1,   label="EMA21")

        # Position lines
        if entry_price:
            ax_price.axhline(entry_price, color="#e3b341", linewidth=1,   linestyle="--", label=f"Entry ${entry_price:,.2f}")
        if sl_price:
            ax_price.axhline(sl_price,    color="#f85149", linewidth=1,   linestyle=":",  label=f"SL ${sl_price:,.2f}")
        if tp_price:
            ax_price.axhline(tp_price,    color="#3fb950", linewidth=1,   linestyle=":",  label=f"TP ${tp_price:,.2f}")

        # X-tick labels (every 8 bars = 8H)
        tick_positions = list(range(0, len(df), 8))
        tick_labels    = [df["ts"].iloc[i].strftime("%m/%d %H:%M") for i in tick_positions]
        ax_price.set_xticks(tick_positions)
        ax_price.set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=7)
        ax_price.set_xlim(-0.5, len(df) - 0.5)
        ax_price.set_title(
            f"{symbol}  48H  —  ${df['close'].iloc[-1]:,.2f}",
            color="#e6edf3", fontsize=10, pad=6,
        )
        ax_price.legend(
            fontsize=7, loc="upper left",
            facecolor="#161b22", edgecolor="#30363d", labelcolor="#8b949e",
        )
        ax_price.yaxis.tick_right()

        # Volume bars
        for i, (_, row) in enumerate(df.iterrows()):
            color = "#3fb950" if row["close"] >= row["open"] else "#f85149"
            ax_vol.bar(i, row["volume"], color=color, alpha=0.6, width=0.6)
        ax_vol.set_xlim(-0.5, len(df) - 0.5)
        ax_vol.set_xticks([])
        ax_vol.set_ylabel("Vol", color="#8b949e", fontsize=7)
        ax_vol.yaxis.tick_right()

        fig.tight_layout(pad=0.5)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                    facecolor="#0d1117")
        plt.close(fig)
        buf.seek(0)
        return buf.read()

    except ImportError:
        logger.log_warning("matplotlib not installed — /chart unavailable")
        return None
    except Exception as exc:
        logger.log_warning(f"telegram_charts.generate_price_chart failed: {exc}")
        return None


def generate_trade_chart(
    symbol: str,
    client,
    entry_price: float,
    sl_price: float,
    tp_price: float,
    side: str = "BUY",
    close_price: Optional[float] = None,
) -> Optional[bytes]:
    """
    Generate a chart at trade open/close. Like generate_price_chart but
    always includes entry/SL/TP lines and optionally a close marker.
    Returns PNG bytes or None.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd

        klines = client.get_klines(symbol=symbol, interval="1h", limit=49)
        if not klines or len(klines) < 2:
            return None

        df = pd.DataFrame(klines, columns=[
            "ts", "open", "high", "low", "close", "volume",
            "close_time", "quote_vol", "trades", "tbbase", "tbquote", "ignore",
        ])
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = df[col].astype(float)
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df = df.iloc[:-1]

        df["ema9"]  = df["close"].ewm(span=9,  adjust=False).mean()
        df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()

        fig, (ax_price, ax_vol) = plt.subplots(
            2, 1,
            figsize=(10, 5),
            gridspec_kw={"height_ratios": [3, 1]},
            facecolor="#0d1117",
        )
        for ax in (ax_price, ax_vol):
            ax.set_facecolor("#161b22")
            ax.tick_params(colors="#8b949e", labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor("#30363d")

        xs = range(len(df))
        for i, (_, row) in enumerate(df.iterrows()):
            color = "#3fb950" if row["close"] >= row["open"] else "#f85149"
            ax_price.plot([i, i], [row["low"], row["high"]], color=color, linewidth=0.8)
            ax_price.bar(i, abs(row["close"] - row["open"]),
                         bottom=min(row["open"], row["close"]),
                         color=color, width=0.6)

        ax_price.plot(list(xs), df["ema9"].tolist(),  color="#58a6ff", linewidth=1, label="EMA9")
        ax_price.plot(list(xs), df["ema21"].tolist(), color="#f78166", linewidth=1, label="EMA21")
        ax_price.axhline(entry_price, color="#e3b341", linewidth=1.2, linestyle="--", label=f"Entry ${entry_price:,.2f}")
        ax_price.axhline(sl_price,    color="#f85149", linewidth=1,   linestyle=":",  label=f"SL ${sl_price:,.2f}")
        ax_price.axhline(tp_price,    color="#3fb950", linewidth=1,   linestyle=":",  label=f"TP ${tp_price:,.2f}")
        if close_price is not None:
            ax_price.axhline(close_price, color="#bc8cff", linewidth=1.2, linestyle="-", label=f"Close ${close_price:,.2f}")

        tick_positions = list(range(0, len(df), 8))
        tick_labels    = [df["ts"].iloc[i].strftime("%m/%d %H:%M") for i in tick_positions]
        ax_price.set_xticks(tick_positions)
        ax_price.set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=7)
        ax_price.set_xlim(-0.5, len(df) - 0.5)

        status = "OPEN" if close_price is None else "CLOSED"
        ax_price.set_title(
            f"{symbol}  {side}  [{status}]  —  ${df['close'].iloc[-1]:,.2f}",
            color="#e6edf3", fontsize=10, pad=6,
        )
        ax_price.legend(
            fontsize=7, loc="upper left",
            facecolor="#161b22", edgecolor="#30363d", labelcolor="#8b949e",
        )
        ax_price.yaxis.tick_right()

        for i, (_, row) in enumerate(df.iterrows()):
            color = "#3fb950" if row["close"] >= row["open"] else "#f85149"
            ax_vol.bar(i, row["volume"], color=color, alpha=0.6, width=0.6)
        ax_vol.set_xlim(-0.5, len(df) - 0.5)
        ax_vol.set_xticks([])
        ax_vol.set_ylabel("Vol", color="#8b949e", fontsize=7)
        ax_vol.yaxis.tick_right()

        fig.tight_layout(pad=0.5)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="#0d1117")
        plt.close(fig)
        buf.seek(0)
        return buf.read()

    except ImportError:
        return None
    except Exception as exc:
        logger.log_warning(f"telegram_charts.generate_trade_chart failed: {exc}")
        return None
