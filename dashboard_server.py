#!/usr/bin/env python3
"""
dashboard_server.py — Advanced read-only web dashboard for the BTC trading bot.

Binds to 127.0.0.1 only.  Access via SSH tunnel:
  ssh -L 8080:127.0.0.1:8080 root@134.209.197.173
  Then open: http://127.0.0.1:8080

SAFETY:  Zero buy / sell / order / cancel / execute endpoints exist.
         API keys and secrets are never sent to the browser.
         100 % read-only.
"""
from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone, timedelta
from functools import wraps
from pathlib import Path

try:
    from flask import (
        Flask, Response, jsonify, redirect,
        render_template_string, request, session,
        stream_with_context, url_for,
    )
except ImportError:
    sys.exit("[dashboard] Flask not installed. Run: .venv/bin/pip install flask")

# ── Bot module imports ────────────────────────────────────────────────────────
try:
    import config
    import performance
    import pause_manager
except ImportError as exc:
    sys.exit(f"[dashboard] Cannot import bot modules: {exc}")

# ── Optional: pandas + regime (for heatmap) ───────────────────────────────────
try:
    import pandas as pd
    import pandas_ta as _pta  # noqa: F401 — registers .ta accessor
    import regime as _reg
    _PANDAS_OK = True
except ImportError:
    _PANDAS_OK = False

# ── Startup guard ─────────────────────────────────────────────────────────────
if not config.DASHBOARD_PASSWORD:
    sys.exit(
        "[dashboard] DASHBOARD_PASSWORD is not set in .env.\n"
        "Add DASHBOARD_PASSWORD=<password> to .env then restart."
    )

# ── Flask app ──────────────────────────────────────────────────────────────────
_SECRET = hashlib.sha256(
    (config.DASHBOARD_PASSWORD + "dash-v2").encode()
).hexdigest()
app = Flask(__name__)
app.secret_key = _SECRET
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=False,
)

# ── Cache ──────────────────────────────────────────────────────────────────────
_cache: dict = {}
_cache_ts: dict[str, float] = {}
_cache_lock = threading.Lock()


def _cached(key: str, fn, ttl: float = 15.0):
    with _cache_lock:
        if key in _cache and time.monotonic() - _cache_ts.get(key, 0.0) < ttl:
            return _cache[key]
    try:
        val = fn()
    except Exception:
        with _cache_lock:
            return _cache.get(key)
    with _cache_lock:
        _cache[key] = val
        _cache_ts[key] = time.monotonic()
    return val


# ── Binance client (lazy) ─────────────────────────────────────────────────────
_bn = None
_bn_lock = threading.Lock()


def _client():
    global _bn
    with _bn_lock:
        if _bn is None:
            try:
                from binance.client import Client
                _bn = Client(
                    config.BINANCE_API_KEY,
                    config.BINANCE_SECRET_KEY,
                    testnet=config.TESTNET,
                    requests_params={"timeout": 8},
                )
            except Exception:
                pass
    return _bn


# ── Kline → dict helper ───────────────────────────────────────────────────────
def _kline_row(k) -> dict:
    return {
        "t": int(k[0]), "o": float(k[1]), "h": float(k[2]),
        "l": float(k[3]), "c": float(k[4]), "v": float(k[5]),
    }


def _klines_to_df(klines):
    df = pd.DataFrame(klines, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)
    return df[["open", "high", "low", "close", "volume"]]


# ── Data fetchers ─────────────────────────────────────────────────────────────

def _fetch_balances() -> dict:
    c = _client()
    if not c:
        return {}
    out = {}
    for b in c.get_account().get("balances", []):
        asset, free, locked = b["asset"], float(b["free"]), float(b["locked"])
        if asset in ("USDT", "BTC", "ETH", "SOL", "BNB") or free + locked > 0:
            out[asset] = {"free": free, "locked": locked}
    return out


def _fetch_open_orders() -> list:
    c = _client()
    if not c:
        return []
    orders = []
    for sym in config.SYMBOLS:
        try:
            for o in c.get_open_orders(symbol=sym):
                ts = o.get("time", 0)
                o["time_fmt"] = (
                    datetime.utcfromtimestamp(ts / 1000).strftime("%m-%d %H:%M")
                    if ts else "—"
                )
                orders.append(o)
        except Exception:
            pass
    return orders


def _fetch_bot_status() -> str:
    try:
        r = subprocess.run(
            ["systemctl", "is-active", "btcbot"],
            capture_output=True, text=True, timeout=5,
        )
        return r.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _fetch_perf() -> dict:
    try:
        now = datetime.now(timezone.utc)
        today = now.strftime("%Y-%m-%d")
        iso = now.isocalendar()
        week = f"{iso[0]}-W{iso[1]:02d}"
        return {
            "total_pnl": performance.total_pnl(),
            "daily_pnl": performance.daily_pnl(today),
            "weekly_pnl": performance.weekly_pnl(week),
            "total_trades": performance.total_trades(),
            "win_rate": performance.win_rate(),
            "max_dd_pct": performance.max_drawdown_pct(),
            "consecutive_losses": performance.consecutive_losses(),
        }
    except Exception:
        return {
            "total_pnl": 0.0, "daily_pnl": 0.0, "weekly_pnl": 0.0,
            "total_trades": 0, "win_rate": 0.0,
            "max_dd_pct": 0.0, "consecutive_losses": 0,
        }


def _fetch_logs(n: int = 20) -> list[str]:
    for p in (Path("/opt/btcbot/bot.log"), Path("bot.log")):
        if p.exists():
            try:
                lines = p.read_text(errors="replace").splitlines()
                important = [
                    l for l in lines
                    if "[INFO]" in l or "[WARNING]" in l or "[ERROR]" in l
                ]
                return list(reversed(important[-n:]))
            except Exception:
                pass
    return []


def _fetch_trades(n: int = 500) -> list[dict]:
    for db in (Path("/opt/btcbot/trades.db"), Path("trades.db")):
        if db.exists():
            try:
                with sqlite3.connect(str(db)) as conn:
                    conn.row_factory = sqlite3.Row
                    rows = conn.execute(
                        "SELECT * FROM trades ORDER BY id DESC LIMIT ?", (n,)
                    ).fetchall()
                return [dict(r) for r in rows]
            except Exception:
                pass
    return []


def _fetch_trade_by_id(trade_id: int) -> dict:
    for db in (Path("/opt/btcbot/trades.db"), Path("trades.db")):
        if db.exists():
            try:
                with sqlite3.connect(str(db)) as conn:
                    conn.row_factory = sqlite3.Row
                    row = conn.execute(
                        "SELECT * FROM trades WHERE id=?", (trade_id,)
                    ).fetchone()
                    if row:
                        return dict(row)
            except Exception:
                pass
    return {}


def _fetch_arb_alerts(n: int = 10) -> list[dict]:
    for db in (Path("/opt/btcbot/arbitrage_watchlist.db"), Path("arbitrage_watchlist.db")):
        if db.exists():
            try:
                with sqlite3.connect(str(db)) as conn:
                    conn.row_factory = sqlite3.Row
                    rows = conn.execute(
                        """SELECT detected_at_utc, arb_type, route,
                                  gross_profit_pct, net_profit_pct, liquidity_score
                           FROM arbitrage_signals WHERE alert_sent=1
                           ORDER BY id DESC LIMIT ?""",
                        (n,),
                    ).fetchall()
                return [dict(r) for r in rows]
            except Exception:
                pass
    return []


def _fetch_trend_alerts(n: int = 10) -> list[dict]:
    for db in (Path("/opt/btcbot/trend_watchlist.db"), Path("trend_watchlist.db")):
        if db.exists():
            try:
                with sqlite3.connect(str(db)) as conn:
                    conn.row_factory = sqlite3.Row
                    tables = conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    ).fetchall()
                    if not tables:
                        return []
                    tbl = tables[0][0]
                    rows = conn.execute(
                        f"SELECT * FROM [{tbl}] ORDER BY id DESC LIMIT ?", (n,)
                    ).fetchall()
                return [dict(r) for r in rows]
            except Exception:
                pass
    return []


# ── Chart candles ─────────────────────────────────────────────────────────────

def _fetch_all_charts() -> dict:
    c = _client()
    if not c:
        return {}
    result = {}
    for sym in config.SYMBOLS:
        try:
            klines = c.get_klines(symbol=sym, interval="1h", limit=52)
            if len(klines) < 5:
                continue
            candles = [_kline_row(k) for k in klines[:-1]]  # drop open candle
            cur = candles[-1]["c"]
            result[sym] = {
                "candles": candles[-48:],
                "current_price": cur,
                "change_1h":  round((cur / candles[-2]["c"] - 1) * 100, 2) if len(candles) >= 2 else 0,
                "change_4h":  round((cur / candles[-5]["c"] - 1) * 100, 2) if len(candles) >= 5 else 0,
                "change_24h": round((cur / candles[-25]["c"] - 1) * 100, 2) if len(candles) >= 25 else 0,
            }
        except Exception:
            pass
    return result


def _fetch_trade_candles(trade: dict) -> list:
    c = _client()
    if not c or not trade:
        return []
    try:
        sym = trade.get("symbol", config.SYMBOL)
        opened = trade.get("opened_at_utc", "")
        closed = trade.get("closed_at_utc", opened) or opened
        if not opened:
            return []

        def _parse(s: str) -> datetime:
            s = s[:26].replace("Z", "+00:00")
            for fmt in ("%Y-%m-%dT%H:%M:%S.%f+00:00", "%Y-%m-%dT%H:%M:%S+00:00",
                        "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
                try:
                    return datetime.strptime(s[:len(fmt)], fmt)
                except ValueError:
                    pass
            return datetime.utcnow()

        t_open = int(_parse(opened).timestamp() * 1000)
        t_close = int(_parse(closed).timestamp() * 1000)
        start_ms = t_open - 24 * 3_600_000
        end_ms = t_close + 12 * 3_600_000

        klines = c.get_klines(
            symbol=sym, interval="1h",
            startTime=start_ms, endTime=end_ms, limit=200,
        )
        return [_kline_row(k) for k in klines]
    except Exception:
        return []


# ── Signal heatmap ────────────────────────────────────────────────────────────

def _fetch_heatmap() -> dict:
    c = _client()
    if not c:
        return {}

    if _PANDAS_OK:
        return _fetch_heatmap_full(c)

    # Fallback: use 24h ticker stats
    result = {}
    try:
        tickers = {t["symbol"]: t for t in c.get_ticker()}
    except Exception:
        return {}
    for sym in config.SYMBOLS:
        t = tickers.get(sym, {})
        try:
            pct = float(t.get("priceChangePercent", 0))
            vol_ratio = float(t.get("count", 0)) / 10000
            if pct > 3:
                signal, color = "BULLISH", "#3fb950"
            elif pct < -3:
                signal, color = "BEARISH", "#f85149"
            elif abs(pct) < 1:
                signal, color = "RANGING", "#d29922"
            else:
                signal, color = "NEUTRAL", "#8b949e"
            result[sym] = {
                "regime": signal, "vol": "—", "adx": 0,
                "atr_pct": 0, "signal": signal, "color": color,
                "price": float(t.get("lastPrice", 0)),
                "change_24h": pct,
            }
        except Exception:
            pass
    return result


def _fetch_heatmap_full(c) -> dict:
    result = {}
    _SIG_COLORS = {
        "BULLISH":    "#3fb950",
        "RANGING":    "#d29922",
        "HIGH_VOL":   "#f0883e",
        "BEARISH":    "#f85149",
        "NO_SETUP":   "#8b949e",
    }
    for sym in config.SYMBOLS:
        try:
            klines = c.get_klines(symbol=sym, interval="1h", limit=210)
            df = _klines_to_df(klines)
            trend, vol = _reg.classify(df)

            adx_df = df.ta.adx(length=14)
            atr_s = df.ta.atr(length=14)
            adx = float(adx_df[f"ADX_14"].iloc[-1]) if adx_df is not None else 0.0
            atr = float(atr_s.iloc[-1]) if atr_s is not None else 0.0
            close = float(df["close"].iloc[-1])
            atr_pct = (atr / close * 100) if close else 0.0

            if vol == "HIGH_VOLATILITY":
                signal = "HIGH_VOL"
            elif trend == "TRENDING":
                signal = "BULLISH"
            elif trend == "RANGING":
                signal = "RANGING"
            else:
                signal = "NO_SETUP"

            result[sym] = {
                "regime": trend, "vol": vol,
                "adx": round(adx, 1), "atr_pct": round(atr_pct, 2),
                "signal": signal, "color": _SIG_COLORS.get(signal, "#8b949e"),
                "price": close, "change_24h": 0,
            }
        except Exception:
            result[sym] = {
                "regime": "?", "vol": "?", "adx": 0, "atr_pct": 0,
                "signal": "NO_SETUP", "color": "#8b949e",
                "price": 0, "change_24h": 0,
            }
    return result


# ── Equity curve ──────────────────────────────────────────────────────────────

def _fetch_equity() -> dict:
    trades = _fetch_trades(2000)
    if not trades:
        return {"dates": [], "cumulative": [], "drawdown": [],
                "daily_dates": [], "daily_values": []}

    sorted_trades = sorted(
        trades,
        key=lambda t: t.get("closed_at_utc") or t.get("opened_at_utc") or "",
    )
    cumul = 0.0
    peak = 0.0
    dates, cumulative, drawdown = [], [], []
    daily: dict[str, float] = {}

    for t in sorted_trades:
        pnl = float(t.get("realized_pnl") or 0)
        ts = t.get("closed_at_utc") or t.get("opened_at_utc") or ""
        date = ts[:10] if ts else "?"

        cumul += pnl
        peak = max(peak, cumul)
        dd = ((cumul - peak) / abs(peak) * 100) if peak != 0 else 0.0

        dates.append(ts[:16] if ts else "?")
        cumulative.append(round(cumul, 4))
        drawdown.append(round(dd, 4))
        daily[date] = round(daily.get(date, 0.0) + pnl, 4)

    daily_dates = sorted(daily)
    return {
        "dates": dates,
        "cumulative": cumulative,
        "drawdown": drawdown,
        "daily_dates": daily_dates,
        "daily_values": [daily[d] for d in daily_dates],
    }


# ── Risk exposure ─────────────────────────────────────────────────────────────

def _fetch_risk() -> dict:
    try:
        balances = _cached("balances", _fetch_balances) or {}
        open_orders = _cached("open_orders", _fetch_open_orders) or []
        perf = _cached("perf", _fetch_perf, 30.0) or {}

        usdt = balances.get("USDT", {}).get("free", 0.0)
        total_est = sum(
            b.get("free", 0) + b.get("locked", 0)
            for sym, b in balances.items() if sym == "USDT"
        )

        open_val = sum(
            float(o.get("origQty", 0)) * float(o.get("price") or 0)
            for o in open_orders if o.get("side") == "BUY"
        )
        exposure_pct = (open_val / (usdt + open_val) * 100) if (usdt + open_val) > 0 else 0.0
        risk_usdt = usdt * config.RISK_PER_TRADE

        daily_limit = usdt * getattr(config, "MAX_DAILY_LOSS", 0.02)
        weekly_limit = usdt * getattr(config, "MAX_WEEKLY_LOSS", 0.05)
        daily_used = max(0.0, -float(perf.get("daily_pnl", 0)))
        weekly_used = max(0.0, -float(perf.get("weekly_pnl", 0)))

        return {
            "usdt_balance": round(usdt, 2),
            "exposure_pct": round(exposure_pct, 2),
            "open_position_value": round(open_val, 2),
            "risk_usdt_per_trade": round(risk_usdt, 4),
            "daily_loss_limit": round(daily_limit, 4),
            "daily_loss_used": round(daily_used, 4),
            "daily_remaining": round(max(0, daily_limit - daily_used), 4),
            "weekly_loss_limit": round(weekly_limit, 4),
            "weekly_loss_used": round(weekly_used, 4),
            "weekly_remaining": round(max(0, weekly_limit - weekly_used), 4),
            "max_open_trades": config.MAX_OPEN_TRADES,
            "open_trades_count": sum(1 for o in open_orders if o.get("side") == "BUY"),
            "risk_per_trade_pct": config.RISK_PER_TRADE * 100,
        }
    except Exception:
        return {}


# ── AI market summary ─────────────────────────────────────────────────────────

def _generate_summary() -> dict:
    try:
        heatmap = _cached("heatmap", _fetch_heatmap, 300.0) or {}
        perf = _cached("perf", _fetch_perf, 30.0) or {}
        logs = _fetch_logs(5)

        lines: list[str] = []
        signals: list[str] = []

        for sym, d in heatmap.items():
            base = sym.replace("USDT", "")
            sig = d.get("signal", "NO_SETUP")
            adx = d.get("adx", 0)
            atr_pct = d.get("atr_pct", 0)
            signals.append(sig)

            if sig == "BULLISH":
                desc = f"trending ({d.get('regime','?')}), ADX={adx:.0f} — watching for entry"
            elif sig == "BEARISH":
                desc = f"downtrend, ADX={adx:.0f} — no long setup"
            elif sig == "HIGH_VOL":
                desc = f"high volatility (ATR={atr_pct:.2f}%) — caution, reduced size"
            elif sig == "RANGING":
                desc = f"ranging, ADX={adx:.0f} — RMR setup possible"
            else:
                desc = f"no clear setup, ADX={adx:.0f}"
            lines.append(f"{base}: {desc}.")

        wins = signals.count("BULLISH")
        hvols = signals.count("HIGH_VOL")
        ranges = signals.count("RANGING")
        if wins >= 3:
            overall = "broad bullish trend — prioritise RMR longs."
        elif hvols >= 2:
            overall = "elevated volatility across board — smaller size or stand aside."
        elif ranges >= 3:
            overall = "market in consolidation — RMR setups most likely."
        else:
            overall = "mixed signals — be selective, wait for high-quality setups."

        # Inject last error if any
        errors = [l for l in logs if "[ERROR]" in l]
        if errors:
            lines.append(f"⚠ Last error: {errors[0][-80:]}")

        lines.append(f"Overall: {overall}")
        return {
            "text": "\n".join(lines),
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        }
    except Exception:
        return {"text": "Summary unavailable.", "generated_at": "—"}


# ── Auth ──────────────────────────────────────────────────────────────────────

def login_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not session.get("logged_in"):
            if request.path.startswith("/api/"):
                return jsonify({"error": "unauthorized"}), 401
            return redirect(url_for("login_page"))
        return fn(*args, **kwargs)
    return wrapper


# ── HTML Template ─────────────────────────────────────────────────────────────

_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>BTC Bot{% if page=='trade' %} — Trade #{{ trade.get('id','') }}{% endif %}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{--bg:#0d1117;--card:#161b22;--border:#21262d;--text:#c9d1d9;--dim:#8b949e;--g:#3fb950;--r:#f85149;--y:#d29922;--o:#f0883e;--b:#58a6ff}
body{background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,sans-serif;font-size:14px;line-height:1.5}
a{color:var(--b);text-decoration:none}
/* Header */
.hdr{background:var(--card);border-bottom:1px solid var(--border);padding:10px 20px;display:flex;align-items:center;gap:12px;flex-wrap:wrap;position:sticky;top:0;z-index:100}
.hdr h1{font-size:16px;font-weight:700;color:#f0f6fc}
.badge{padding:2px 9px;border-radius:10px;font-size:10px;font-weight:700;letter-spacing:.5px}
.bl{background:#0f2d1a;color:var(--g);border:1px solid var(--g)}
.bt{background:#2d1f0f;color:var(--y);border:1px solid var(--y)}
.bro{background:#1a1a2e;color:var(--dim);border:1px solid var(--border);font-size:9px}
.dot{display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:4px;vertical-align:middle}
.da{background:var(--g);box-shadow:0 0 5px var(--g)}.di{background:var(--r)}.du{background:var(--dim)}
.hdr-r{margin-left:auto;display:flex;gap:10px;align-items:center}
.dim{color:var(--dim)}
.btn-sm{background:var(--card);border:1px solid var(--border);color:var(--text);padding:4px 11px;border-radius:6px;cursor:pointer;font-size:12px}
.btn-sm:hover{background:var(--border)}
/* Layout */
.wrap{max-width:1380px;margin:0 auto;padding:16px 20px}
.stitle{font-size:10px;font-weight:700;color:var(--dim);text-transform:uppercase;letter-spacing:.8px;margin:20px 0 8px}
/* Cards */
.g4{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}
.g3{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}
.g2{display:grid;grid-template-columns:repeat(2,1fr);gap:10px}
.g1{display:grid;grid-template-columns:1fr;gap:10px}
.card{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:14px 16px}
.card-sm{padding:10px 14px}
.lbl{font-size:9px;color:var(--dim);text-transform:uppercase;letter-spacing:.5px;margin-bottom:4px}
.val{font-size:19px;font-weight:700;color:#f0f6fc}
.sub{font-size:10px;color:var(--dim);margin-top:3px}
/* Color helpers */
.cg{color:var(--g)}.cr{color:var(--r)}.cy{color:var(--y)}.co{color:var(--o)}.cb{color:var(--b)}.cd{color:var(--dim)}
.bg{border-color:rgba(63,185,80,.4)}.br{border-color:rgba(248,81,73,.4)}.by{border-color:rgba(210,153,34,.4)}
/* Tables */
.tbl-wrap{overflow-x:auto;-webkit-overflow-scrolling:touch}
table{width:100%;border-collapse:collapse;font-size:12.5px}
th{background:var(--bg);color:var(--dim);font-size:9px;text-transform:uppercase;letter-spacing:.5px;font-weight:700;padding:7px 10px;border-bottom:1px solid var(--border);white-space:nowrap;text-align:left}
td{padding:7px 10px;border-bottom:1px solid var(--card);color:var(--text);vertical-align:middle}
tr:last-child td{border-bottom:none}
tr.clickable:hover td{background:#1c2128;cursor:pointer}
.mono{font-family:monospace}
.nodata{color:var(--dim);font-style:italic;padding:10px 0;font-size:12px}
/* Charts */
.chart-hdr{display:flex;align-items:center;gap:8px;margin-bottom:8px;flex-wrap:wrap}
.chart-price{font-size:17px;font-weight:700;color:#f0f6fc}
.chg{font-size:10px;padding:2px 6px;border-radius:4px;font-weight:600}
.chg-p{background:#0f2d1a;color:var(--g)}.chg-n{background:#2d0f0f;color:var(--r)}.chg-0{background:var(--border);color:var(--dim)}
/* Heatmap */
.heatmap-card{text-align:center;padding:12px}
.heatmap-sig{font-size:14px;font-weight:700;margin:4px 0}
/* Equity canvas */
.equity-wrap{padding:10px;min-height:200px}
/* Log */
.log-box{background:#010409;border:1px solid var(--border);border-radius:8px;padding:10px;font-family:monospace;font-size:11px;max-height:320px;overflow-y:auto}
.ll{padding:2px 0;border-bottom:1px solid #0d111750;white-space:pre-wrap;word-break:break-all}
.ll:last-child{border-bottom:none}
.li{color:#8b949e}.lw{color:var(--y)}.le{color:var(--r)}
/* Notifications */
.notif-box{background:#010409;border:1px solid var(--border);border-radius:8px;padding:10px;font-family:monospace;font-size:11px;max-height:260px;overflow-y:auto}
.ni{color:#58a6ff;padding:2px 0;display:block;border-bottom:1px solid #0d111750}
.nw{color:var(--y);padding:2px 0;display:block;border-bottom:1px solid #0d111750}
.ne{color:var(--r);padding:2px 0;display:block;border-bottom:1px solid #0d111750}
/* Summary */
.summary-text{font-family:monospace;font-size:13px;line-height:1.9;white-space:pre-line;color:var(--text)}
/* Risk bars */
.risk-bar-bg{background:var(--border);border-radius:3px;height:6px;margin-top:5px}
.risk-bar-fill{height:6px;border-radius:3px;transition:width .3s}
/* Login */
.login-wrap{min-height:100vh;display:flex;align-items:center;justify-content:center}
.login-box{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:30px 36px;width:320px}
.login-box h2{font-size:18px;color:#f0f6fc;margin-bottom:4px}
.login-box p{color:var(--dim);font-size:12px;margin-bottom:20px}
.login-box input{width:100%;background:var(--bg);border:1px solid var(--border);border-radius:6px;color:#f0f6fc;padding:9px 11px;font-size:13px;margin-bottom:10px}
.login-box input:focus{outline:none;border-color:var(--b)}
.login-box button{width:100%;background:#238636;border:1px solid #2ea043;border-radius:6px;color:#fff;padding:9px;font-size:13px;font-weight:600;cursor:pointer}
.login-box button:hover{background:#2ea043}
.login-err{color:var(--r);font-size:12px;margin-top:8px}
.ro-note{background:#0f1f0f;border:1px solid rgba(63,185,80,.3);border-radius:6px;color:var(--g);font-size:11px;padding:7px 10px;margin-top:14px;text-align:center}
/* Mobile */
@media(max-width:900px){.g4{grid-template-columns:repeat(2,1fr)}.g3{grid-template-columns:repeat(2,1fr)}}
@media(max-width:560px){.g4,.g3,.g2{grid-template-columns:1fr}.hdr{padding:8px 12px;gap:8px}.wrap{padding:10px 12px}.val{font-size:16px}}
/* Trade detail */
.trade-detail-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.td-row{display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid var(--border)}
.td-row:last-child{border-bottom:none}
.td-lbl{color:var(--dim);font-size:12px}
.td-val{font-weight:600;font-size:12px;font-family:monospace}
@media(max-width:600px){.trade-detail-grid{grid-template-columns:1fr}}
/* Footer */
.footer{text-align:center;color:#30363d;font-size:10px;padding:16px 0 4px;border-top:1px solid var(--border);margin-top:20px}
</style>
</head>
<body>

{% if page == 'login' %}
<div class="login-wrap">
  <div class="login-box">
    <h2>BTC Bot Dashboard</h2>
    <p>Read-only monitoring. No trading actions available.</p>
    <form method="post" action="/login">
      <input type="password" name="password" placeholder="Dashboard password" autofocus required>
      <button type="submit">Sign in</button>
      {% if error %}<div class="login-err">{{ error }}</div>{% endif %}
    </form>
    <div class="ro-note">READ-ONLY &mdash; No buy / sell / switch controls</div>
  </div>
</div>

{% elif page == 'trade' %}
{% set t = trade %}
{% set pnl = t.get('realized_pnl') or 0 %}
<div class="hdr">
  <a href="/" class="btn-sm">← Dashboard</a>
  <h1>Trade #{{ t.get('id','?') }} — {{ t.get('symbol','') }}</h1>
  <span class="badge {{ 'bl' if not testnet else 'bt' }}">{{ 'LIVE' if not testnet else 'TESTNET' }}</span>
  <span class="badge bro">READ-ONLY</span>
</div>
<div class="wrap">

  <div class="stitle">Trade Detail</div>
  <div class="trade-detail-grid">
    <div class="card">
      {% set fields = [
        ('Symbol',      t.get('symbol','')),
        ('Strategy',    t.get('strategy','')),
        ('Regime',      t.get('regime','')),
        ('Side',        'LONG'),
        ('Entry',       ('%.4f' % (t.get('fill_price') or t.get('entry_price') or 0))),
        ('Stop',        ('%.4f' % (t.get('stop_price') or 0))),
        ('TP',          ('%.4f' % (t.get('tp_price') or 0))),
        ('Exit',        ('%.4f' % (t.get('exit_price') or 0))),
        ('Size',        ('%.5f' % (t.get('size') or 0))),
        ('PnL',         ('%+.4f USDT' % pnl)),
        ('Close',       t.get('close_reason','')),
        ('ADX',         ('%.1f' % (t.get('adx') or 0))),
        ('ATR%',        ('%.2f%%' % (t.get('atr_pct') or 0))),
        ('Score',       ('%.1f%%' % (t.get('score_pct') or 0))),
        ('Opened',      (t.get('opened_at_utc') or '')[:16]),
        ('Closed',      (t.get('closed_at_utc') or '')[:16]),
      ] %}
      {% for lbl, val in fields %}
      <div class="td-row">
        <span class="td-lbl">{{ lbl }}</span>
        <span class="td-val {{ 'cg' if (lbl=='PnL' and pnl>=0) else 'cr' if (lbl=='PnL' and pnl<0) else '' }}">{{ val or '—' }}</span>
      </div>
      {% endfor %}
    </div>

    <div class="card">
      <div class="lbl">Price Chart Around Trade</div>
      <canvas id="trade-chart" style="max-height:340px"></canvas>
      <div class="sub" id="trade-chart-msg" style="margin-top:6px">Loading candles…</div>
    </div>
  </div>

</div>
<script>
(async () => {
  const tradeId = {{ t.get('id',0) }};
  const entryPrice = {{ t.get('fill_price') or t.get('entry_price') or 0 }};
  const exitPrice  = {{ t.get('exit_price') or 0 }};
  const pnl        = {{ pnl }};
  try {
    const r = await fetch('/api/trade/' + tradeId);
    const data = await r.json();
    const candles = data.candles || [];
    document.getElementById('trade-chart-msg').textContent = candles.length ? '' : 'No candle data.';
    if (!candles.length) return;
    const labels = candles.map(c => {
      const d = new Date(c.t);
      return d.getUTCMonth()+1+'/'+d.getUTCDate()+' '+String(d.getUTCHours()).padStart(2,'0')+':00';
    });
    const closes = candles.map(c => c.c);
    const ctx = document.getElementById('trade-chart');
    new Chart(ctx, {
      type: 'line',
      data: {
        labels,
        datasets: [
          { label:'Close', data: closes, borderColor:'#58a6ff', borderWidth:1.5, pointRadius:0, fill:false },
          entryPrice ? { label:'Entry', data: candles.map(c => null).map((_, i) => null),
            type:'scatter', pointRadius:0 } : null,
        ].filter(Boolean),
      },
      options: {
        responsive:true,
        plugins:{
          legend:{labels:{color:'#8b949e',font:{size:10}}},
          annotation: {},
          tooltip: {
            callbacks:{
              afterBody: (items) => {
                const idx = items[0]?.dataIndex;
                const c = candles[idx];
                return c ? [`O:${c.o} H:${c.h} L:${c.l} C:${c.c}`] : [];
              }
            }
          }
        },
        scales:{
          x:{ticks:{color:'#8b949e',maxTicksLimit:8,font:{size:10}},grid:{color:'#21262d'}},
          y:{ticks:{color:'#8b949e',font:{size:10}},grid:{color:'#21262d'}},
        },
        animation:false,
      }
    });
    // Draw entry/exit as vertical reference lines via plugin
    // (manual canvas annotations since annotation plugin not loaded)
  } catch(e) {
    document.getElementById('trade-chart-msg').textContent = 'Chart unavailable.';
  }
})();
</script>
<div class="footer">BTC Bot Dashboard &mdash; Read-Only &mdash; No buy/sell/order controls.</div>

{% else %}
<!-- ═══════════════ MAIN DASHBOARD ═══════════════ -->

<div class="hdr">
  <h1>BTC Bot</h1>
  <span class="badge {{ 'bl' if not testnet else 'bt' }}">{{ 'LIVE' if not testnet else 'TESTNET' }}</span>
  <span class="badge bro">READ-ONLY</span>
  <span><span class="dot {{ 'da' if bot_status=='active' else 'di' if bot_status=='inactive' else 'du' }}"></span>
    <span class="{{ 'cg' if bot_status=='active' else 'cr' }}">{{ bot_status }}</span>
  </span>
  <span class="dim" style="font-size:11px">{{ symbols|join(' · ') }}</span>
  <div class="hdr-r">
    <span class="dim" style="font-size:11px">↻ <span id="cd">10</span>s</span>
    <a href="/logout"><button class="btn-sm">Sign out</button></a>
  </div>
</div>

<div class="wrap">

<!-- ── AI Market Summary ─────────────────────────────────────────────────── -->
<div class="stitle">AI Market Summary</div>
<div class="card">
  <div class="summary-text" id="summary-text">{{ summary.get('text','Loading…') }}</div>
  <div class="sub" style="margin-top:6px">Generated: {{ summary.get('generated_at','—') }}
    &nbsp;<a href="#" onclick="reloadSummary();return false;" style="font-size:10px">↺ refresh</a>
  </div>
</div>

<!-- ── Status ─────────────────────────────────────────────────────────────── -->
<div class="stitle">Status</div>
<div class="g4" style="margin-bottom:10px">
  <div class="card {{ 'bg' if bot_status=='active' else 'br' }}">
    <div class="lbl">Bot Service</div>
    <div class="val {{ 'cg' if bot_status=='active' else 'cr' }}">{{ bot_status.upper() }}</div>
    <div class="sub">{{ 'LIVE' if not testnet else 'TESTNET' }} mode</div>
  </div>
  <div class="card {{ 'br' if paused else 'bg' }}">
    <div class="lbl">Trading</div>
    <div class="val {{ 'cr' if paused else 'cg' }}">{{ 'PAUSED' if paused else 'ACTIVE' }}</div>
    <div class="sub">{{ pause_reason or 'No restrictions' }}</div>
  </div>
  <div class="card">
    <div class="lbl">Open Orders</div>
    <div class="val">{{ open_orders|length }}</div>
    <div class="sub">Across {{ symbols|length }} symbol(s)</div>
  </div>
  <div class="card">
    <div class="lbl">Total Trades</div>
    <div class="val">{{ perf.get('total_trades',0) }}</div>
    <div class="sub">Win rate {{ '%.1f' % (perf.get('win_rate',0)*100) }}%</div>
  </div>
</div>

<!-- ── PnL ────────────────────────────────────────────────────────────────── -->
<div class="stitle">Performance</div>
{% set dp=perf.get('daily_pnl',0) %}{% set wp=perf.get('weekly_pnl',0) %}{% set tp=perf.get('total_pnl',0) %}{% set dd=perf.get('max_dd_pct',0) %}
<div class="g4" style="margin-bottom:10px">
  <div class="card {{ 'bg' if dp>=0 else 'br' }}">
    <div class="lbl">Today PnL</div>
    <div class="val {{ 'cg' if dp>=0 else 'cr' }}">{{ '%+.4f' % dp }}</div>
    <div class="sub">USDT</div>
  </div>
  <div class="card {{ 'bg' if wp>=0 else 'br' }}">
    <div class="lbl">This Week PnL</div>
    <div class="val {{ 'cg' if wp>=0 else 'cr' }}">{{ '%+.4f' % wp }}</div>
    <div class="sub">USDT</div>
  </div>
  <div class="card {{ 'bg' if tp>=0 else 'br' }}">
    <div class="lbl">Total PnL</div>
    <div class="val {{ 'cg' if tp>=0 else 'cr' }}">{{ '%+.4f' % tp }}</div>
    <div class="sub">USDT</div>
  </div>
  <div class="card {{ 'br' if dd>5 else 'by' if dd>2 else '' }}">
    <div class="lbl">Max Drawdown</div>
    <div class="val {{ 'cr' if dd>5 else 'cy' if dd>2 else 'cg' }}">{{ '%.2f' % dd }}%</div>
    <div class="sub">Consec. losses: {{ perf.get('consecutive_losses',0) }}</div>
  </div>
</div>

<!-- ── Signal Heatmap ─────────────────────────────────────────────────────── -->
<div class="stitle">Signal Heatmap</div>
<div class="g4" id="heatmap-grid">
  <div class="card cd">Loading heatmap…</div>
</div>

<!-- ── Price Charts ───────────────────────────────────────────────────────── -->
<div class="stitle">Price Charts (1H candles)</div>
<div class="g2" id="chart-grid">
  {% for sym in symbols %}
  <div class="card" id="chartcard-{{ sym }}">
    <div class="chart-hdr">
      <strong>{{ sym }}</strong>
      <span class="chart-price" id="price-{{ sym }}">—</span>
      <span class="chg chg-0" id="ch1h-{{ sym }}">1h —</span>
      <span class="chg chg-0" id="ch4h-{{ sym }}">4h —</span>
      <span class="chg chg-0" id="ch24h-{{ sym }}">24h —</span>
    </div>
    <canvas id="chart-{{ sym }}" style="max-height:80px"></canvas>
    <canvas id="vol-{{ sym }}" style="max-height:32px;margin-top:2px"></canvas>
  </div>
  {% endfor %}
</div>

<!-- ── Risk Exposure ──────────────────────────────────────────────────────── -->
<div class="stitle">Risk Exposure</div>
<div class="g3" id="risk-grid">
  <div class="card cd">Loading…</div>
</div>

<!-- ── Equity Curve ───────────────────────────────────────────────────────── -->
<div class="stitle">PnL Equity Curve</div>
<div class="card">
  <div class="equity-wrap">
    <canvas id="equity-chart"></canvas>
    <div class="sub" id="equity-msg" style="margin-top:6px">Loading equity data…</div>
  </div>
</div>

<!-- ── Asset Balances ─────────────────────────────────────────────────────── -->
<div class="stitle">Asset Balances</div>
<div class="card">
  <div class="tbl-wrap">
    <table>
      <tr><th>Asset</th><th>Free</th><th>Locked</th><th>Total</th></tr>
      {% for asset in ['USDT','BTC','ETH','SOL','BNB'] %}
      {% set b=balances.get(asset,{}) %}{% set fr=b.get('free',0) %}{% set lk=b.get('locked',0) %}
      <tr>
        <td><strong>{{ asset }}</strong></td>
        <td class="mono {{ 'cg' if fr>0 else 'cd' }}">{{ '%.6f'%fr }}</td>
        <td class="mono cd">{{ '%.6f'%lk }}</td>
        <td class="mono {{ 'cg' if (fr+lk)>0 else 'cd' }}">{{ '%.6f'%(fr+lk) }}</td>
      </tr>
      {% endfor %}
    </table>
  </div>
</div>

<!-- ── Open Orders ────────────────────────────────────────────────────────── -->
<div class="stitle">Open Orders</div>
<div class="card">
  {% if open_orders %}
  <div class="tbl-wrap"><table>
    <tr><th>Symbol</th><th>Side</th><th>Type</th><th>Qty</th><th>Price</th><th>Stop</th><th>Status</th><th>Time</th></tr>
    {% for o in open_orders %}
    <tr>
      <td><strong>{{ o.get('symbol','') }}</strong></td>
      <td class="{{ 'cg' if o.get('side')=='BUY' else 'cr' }}"><strong>{{ o.get('side','') }}</strong></td>
      <td class="mono cd">{{ o.get('type','') }}</td>
      <td class="mono">{{ o.get('origQty','') }}</td>
      <td class="mono">{{ o.get('price','') }}</td>
      <td class="mono cd">{{ o.get('stopPrice','—') }}</td>
      <td class="cd">{{ o.get('status','') }}</td>
      <td class="cd">{{ o.get('time_fmt','—') }}</td>
    </tr>
    {% endfor %}
  </table></div>
  {% else %}<div class="nodata">No open orders.</div>{% endif %}
</div>

<!-- ── Recent Trades ──────────────────────────────────────────────────────── -->
<div class="stitle">Recent Trades (click for detail)</div>
<div class="card">
  {% if trades %}
  <div class="tbl-wrap"><table>
    <tr><th>Closed</th><th>Symbol</th><th>Strategy</th><th>Entry</th><th>Exit</th><th>PnL</th><th>Reason</th><th>Regime</th></tr>
    {% for t in trades[:10] %}
    {% set pnl=t.get('realized_pnl',0) or 0 %}
    <tr class="clickable" onclick="window.location='/trade/{{ t.get('id','') }}'">
      <td class="cd mono">{{ (t.get('closed_at_utc') or t.get('opened_at_utc') or '')[:16] }}</td>
      <td><strong>{{ t.get('symbol','') }}</strong></td>
      <td class="cd">{{ t.get('strategy','') }}</td>
      <td class="mono">{{ '%.2f'%(t.get('fill_price') or t.get('entry_price') or 0) }}</td>
      <td class="mono">{{ '%.2f'%(t.get('exit_price') or 0) }}</td>
      <td class="mono {{ 'cg' if pnl>=0 else 'cr' }}"><strong>{{ '%+.4f'%pnl }}</strong></td>
      <td class="cd">{{ t.get('close_reason','') }}</td>
      <td class="cd" style="font-size:10px">{{ t.get('regime','') }}</td>
    </tr>
    {% endfor %}
  </table></div>
  {% else %}<div class="nodata">No trades recorded yet.</div>{% endif %}
</div>

{% if arb_alerts %}
<!-- ── Arb Alerts ──────────────────────────────────────────────────────────── -->
<div class="stitle">Arbitrage Alerts — Watch Only</div>
<div class="card">
  <div class="tbl-wrap"><table>
    <tr><th>Time</th><th>Type</th><th>Route</th><th>Gross</th><th>Net</th><th>Liq</th></tr>
    {% for a in arb_alerts %}
    {% set net=a.get('net_profit_pct',0) or 0 %}
    <tr>
      <td class="cd mono">{{ (a.get('detected_at_utc') or '')[:16] }}</td>
      <td class="cd">{{ a.get('arb_type','') }}</td>
      <td class="mono" style="font-size:10px">{{ a.get('route','') }}</td>
      <td class="{{ 'cg' if (a.get('gross_profit_pct') or 0)>=0 else 'cr' }}">{{ '%+.3f'%(a.get('gross_profit_pct') or 0) }}%</td>
      <td class="{{ 'cg' if net>=0 else 'cr' }}"><strong>{{ '%+.3f'%net }}%</strong></td>
      <td class="cd">{{ a.get('liquidity_score','') }}</td>
    </tr>
    {% endfor %}
  </table></div>
</div>
{% endif %}

{% if trend_alerts %}
<div class="stitle">Trend Alerts</div>
<div class="card">
  <div class="tbl-wrap"><table>
    <tr>{% for c in trend_cols %}<th>{{ c }}</th>{% endfor %}</tr>
    {% for t in trend_alerts %}
    <tr>{% for c in trend_cols %}<td class="mono cd" style="font-size:10px">{{ t.get(c,'') }}</td>{% endfor %}</tr>
    {% endfor %}
  </table></div>
</div>
{% endif %}

<!-- ── Bot Log ────────────────────────────────────────────────────────────── -->
<div class="stitle">Bot Log (latest 20 events)</div>
<div class="log-box">
  {% if logs %}{% for line in logs %}
    {% if '[ERROR]' in line %}<div class="ll le">{{ line }}</div>
    {% elif '[WARNING]' in line %}<div class="ll lw">{{ line }}</div>
    {% else %}<div class="ll li">{{ line }}</div>{% endif %}
  {% endfor %}{% else %}<span class="cd">No log data.</span>{% endif %}
</div>

<!-- ── Live Notifications ─────────────────────────────────────────────────── -->
<div class="stitle">Live Notifications</div>
<div class="notif-box" id="notif-box">
  <span class="cd" id="notif-status">Connecting…</span>
</div>

</div><!-- /wrap -->
<div class="footer">BTC Bot Dashboard &mdash; Read-Only &mdash; {{ now_utc }} UTC &mdash; No buy/sell/order controls.</div>

<!-- ═══ JAVASCRIPT ═══ -->
<script>
const _SYMBOLS = {{ symbols | tojson }};

/* ── Auto-refresh countdown ── */
let _cd = 10;
const _cdEl = document.getElementById('cd');
setInterval(() => { _cd--; if(_cdEl) _cdEl.textContent=_cd; if(_cd<=0) location.reload(); }, 1000);

/* ── Chart.js defaults ── */
Chart.defaults.color = '#8b949e';
Chart.defaults.font.family = 'monospace';
Chart.defaults.font.size = 10;

/* ── Mini price charts ── */
async function loadCharts() {
  try {
    const r = await fetch('/api/charts');
    if (!r.ok) return;
    const data = await r.json();
    for (const [sym, info] of Object.entries(data)) {
      const priceEl = document.getElementById('price-' + sym);
      if (priceEl) priceEl.textContent = info.current_price.toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:4});

      const mkChg = (id, val, label) => {
        const el = document.getElementById(id);
        if (!el) return;
        el.textContent = label + (val >= 0 ? '+' : '') + val.toFixed(2) + '%';
        el.className = 'chg ' + (val > 0 ? 'chg-p' : val < 0 ? 'chg-n' : 'chg-0');
      };
      mkChg('ch1h-'+sym, info.change_1h, '1h ');
      mkChg('ch4h-'+sym, info.change_4h, '4h ');
      mkChg('ch24h-'+sym, info.change_24h, '24h ');

      const closes = info.candles.map(c => c.c);
      const vols   = info.candles.map(c => c.v);
      const labels = info.candles.map((c,i) => i % 8 === 0 ? new Date(c.t).getUTCHours() + 'h' : '');
      const color  = closes[closes.length-1] >= closes[0] ? '#3fb950' : '#f85149';

      const cCtx = document.getElementById('chart-' + sym);
      if (cCtx) new Chart(cCtx, {
        type:'line',
        data:{ labels, datasets:[{ data:closes, borderColor:color, borderWidth:1.5, pointRadius:0,
          fill:true, backgroundColor:color+'15' }]},
        options:{ responsive:true, animation:false,
          plugins:{ legend:{display:false}, tooltip:{enabled:false} },
          scales:{ x:{display:false}, y:{display:false} }
        }
      });

      const vCtx = document.getElementById('vol-' + sym);
      if (vCtx) new Chart(vCtx, {
        type:'bar',
        data:{ labels, datasets:[{ data:vols, backgroundColor:'#8b949e33', borderWidth:0 }]},
        options:{ responsive:true, animation:false,
          plugins:{ legend:{display:false}, tooltip:{enabled:false} },
          scales:{ x:{display:false}, y:{display:false} }
        }
      });
    }
  } catch(e) {}
}

/* ── Signal heatmap ── */
async function loadHeatmap() {
  try {
    const r = await fetch('/api/heatmap');
    if (!r.ok) return;
    const data = await r.json();
    const grid = document.getElementById('heatmap-grid');
    if (!grid) return;
    grid.innerHTML = '';
    for (const [sym, d] of Object.entries(data)) {
      const card = document.createElement('div');
      card.className = 'card heatmap-card';
      card.style.borderColor = d.color;
      card.innerHTML = `
        <div class="lbl">${sym}</div>
        <div class="heatmap-sig" style="color:${d.color}">${d.signal}</div>
        <div class="sub">${d.regime} + ${d.vol}</div>
        <div class="sub">ADX ${d.adx} &middot; ATR ${d.atr_pct}%</div>
        <div class="sub" style="margin-top:2px">$${d.price.toLocaleString(undefined,{maximumFractionDigits:2})}</div>`;
      grid.appendChild(card);
    }
  } catch(e) {}
}

/* ── Risk widget ── */
async function loadRisk() {
  try {
    const r = await fetch('/api/risk');
    if (!r.ok) return;
    const d = await r.json();
    const grid = document.getElementById('risk-grid');
    if (!grid) return;

    const barPct = (used, limit) => {
      if (!limit) return 0;
      return Math.min(100, used / limit * 100).toFixed(1);
    };
    const dailyPct  = barPct(d.daily_loss_used, d.daily_loss_limit);
    const weeklyPct = barPct(d.weekly_loss_used, d.weekly_loss_limit);
    const tradePct  = d.max_open_trades ? (d.open_trades_count / d.max_open_trades * 100).toFixed(0) : 0;

    grid.innerHTML = `
      <div class="card"><div class="lbl">Account Exposure</div>
        <div class="val ${d.exposure_pct>10?'cr':d.exposure_pct>5?'cy':'cg'}">${d.exposure_pct}%</div>
        <div class="sub">Open value: $${d.open_position_value}</div>
      </div>
      <div class="card"><div class="lbl">Risk / Trade</div>
        <div class="val">${d.risk_per_trade_pct}%</div>
        <div class="sub">~$${d.risk_usdt_per_trade} USDT</div>
      </div>
      <div class="card"><div class="lbl">Open Trades</div>
        <div class="val">${d.open_trades_count} / ${d.max_open_trades}</div>
        <div class="risk-bar-bg"><div class="risk-bar-fill" style="width:${tradePct}%;background:#58a6ff"></div></div>
      </div>
      <div class="card"><div class="lbl">Daily Loss Budget</div>
        <div class="val ${parseFloat(dailyPct)>80?'cr':parseFloat(dailyPct)>50?'cy':'cg'}">${d.daily_remaining >= 0 ? '$'+d.daily_remaining : 'EXCEEDED'}</div>
        <div class="sub">Used $${d.daily_loss_used} / $${d.daily_loss_limit}</div>
        <div class="risk-bar-bg"><div class="risk-bar-fill" style="width:${dailyPct}%;background:${parseFloat(dailyPct)>80?'#f85149':'#d29922'}"></div></div>
      </div>
      <div class="card"><div class="lbl">Weekly Loss Budget</div>
        <div class="val ${parseFloat(weeklyPct)>80?'cr':parseFloat(weeklyPct)>50?'cy':'cg'}">${d.weekly_remaining >= 0 ? '$'+d.weekly_remaining : 'EXCEEDED'}</div>
        <div class="sub">Used $${d.weekly_loss_used} / $${d.weekly_loss_limit}</div>
        <div class="risk-bar-bg"><div class="risk-bar-fill" style="width:${weeklyPct}%;background:${parseFloat(weeklyPct)>80?'#f85149':'#d29922'}"></div></div>
      </div>
      <div class="card"><div class="lbl">USDT Balance</div>
        <div class="val">$${d.usdt_balance}</div>
        <div class="sub">Free in account</div>
      </div>`;
  } catch(e) {}
}

/* ── Equity curve ── */
async function loadEquity() {
  try {
    const r = await fetch('/api/equity');
    if (!r.ok) return;
    const data = await r.json();
    const msg = document.getElementById('equity-msg');
    if (!data.dates || !data.dates.length) {
      if (msg) msg.textContent = 'No trade history yet.';
      return;
    }
    if (msg) msg.textContent = '';
    const ctx = document.getElementById('equity-chart');
    if (!ctx) return;
    new Chart(ctx, {
      type:'line',
      data:{
        labels: data.dates,
        datasets:[
          { label:'Cumulative PnL (USDT)', data:data.cumulative,
            borderColor:'#3fb950', borderWidth:2, pointRadius:0, fill:false, yAxisID:'y' },
          { label:'Drawdown %', data:data.drawdown,
            borderColor:'#f85149', borderWidth:1, pointRadius:0, fill:true,
            backgroundColor:'rgba(248,81,73,0.08)', yAxisID:'y1' },
        ]
      },
      options:{
        responsive:true, animation:false,
        interaction:{ mode:'index', intersect:false },
        plugins:{ legend:{ labels:{ color:'#8b949e', font:{size:11} } } },
        scales:{
          x:{ ticks:{color:'#8b949e', maxTicksLimit:8, font:{size:10}}, grid:{color:'#21262d'} },
          y:{ ticks:{color:'#8b949e', font:{size:10}}, grid:{color:'#21262d'}, title:{display:true,text:'USDT',color:'#8b949e'} },
          y1:{ position:'right', ticks:{color:'#f85149', font:{size:10}},
               grid:{drawOnChartArea:false}, title:{display:true,text:'DD%',color:'#f85149'} },
        }
      }
    });
  } catch(e) {}
}

/* ── Live notifications (SSE) ── */
function connectSSE() {
  const box = document.getElementById('notif-box');
  const status = document.getElementById('notif-status');
  try {
    const es = new EventSource('/api/events');
    if (status) { status.textContent = ''; }

    es.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        const msg = data.msg || '';
        if (!msg || !box) return;
        const div = document.createElement('div');
        div.className = msg.includes('[ERROR]') ? 'ne' : msg.includes('[WARNING]') ? 'nw' : 'ni';
        div.textContent = msg;
        if (status) status.remove();
        box.insertBefore(div, box.firstChild);
        while (box.children.length > 60) box.removeChild(box.lastChild);
      } catch(ex) {}
    };

    es.onerror = () => {
      if (status) { status.textContent = 'SSE disconnected — reconnecting…'; box.insertBefore(status, box.firstChild); }
      es.close();
      setTimeout(connectSSE, 5000);
    };
  } catch(e) {
    if (status) status.textContent = 'SSE unavailable.';
  }
}

/* ── AI summary refresh ── */
async function reloadSummary() {
  try {
    const r = await fetch('/api/summary?refresh=1');
    const d = await r.json();
    const el = document.getElementById('summary-text');
    if (el) el.textContent = d.text || '';
  } catch(e) {}
}

/* ── Boot ── */
document.addEventListener('DOMContentLoaded', () => {
  loadCharts();
  loadHeatmap();
  loadRisk();
  loadEquity();
  connectSSE();
});
</script>

{% endif %}
</body>
</html>"""


# ══════════════════════════════════════════════════════════════════════════════
# ROUTES — ALL READ-ONLY.  NO BUY / SELL / ORDER / EXECUTE ENDPOINTS.
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/login", methods=["GET", "POST"])
def login_page():
    error = ""
    if request.method == "POST":
        pw = request.form.get("password", "")
        if secrets.compare_digest(pw, config.DASHBOARD_PASSWORD):
            session["logged_in"] = True
            return redirect(url_for("dashboard"))
        error = "Incorrect password."
    return render_template_string(_TEMPLATE, page="login", error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login_page"))


@app.route("/")
@login_required
def dashboard():
    balances    = _cached("balances",    _fetch_balances,    15.0) or {}
    open_orders = _cached("open_orders", _fetch_open_orders, 15.0) or []
    bot_status  = _cached("bot_status",  _fetch_bot_status,  10.0) or "unknown"
    perf        = _cached("perf",        _fetch_perf,        30.0) or {}
    logs        = _fetch_logs(20)
    trades      = _fetch_trades(10)
    arb_alerts  = _fetch_arb_alerts(10)
    trend_raw   = _fetch_trend_alerts(10)
    summary     = _cached("summary",     _generate_summary,  3600.0) or {}

    try:
        paused       = pause_manager.is_paused()
        pause_reason = pause_manager.pause_reason() if paused else ""
    except Exception:
        paused, pause_reason = False, ""

    _skip = {"id", "raw_data", "json", "data"}
    trend_cols = (
        [c for c in trend_raw[0].keys() if c not in _skip][:8] if trend_raw else []
    )

    return render_template_string(
        _TEMPLATE,
        page="dashboard",
        testnet=config.TESTNET,
        bot_status=bot_status,
        symbols=config.SYMBOLS,
        balances=balances,
        open_orders=open_orders,
        paused=paused,
        pause_reason=pause_reason,
        perf=perf,
        logs=logs,
        trades=trades,
        arb_alerts=arb_alerts,
        trend_alerts=trend_raw,
        trend_cols=trend_cols,
        summary=summary,
        now_utc=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        error="",
    )


@app.route("/trade/<int:trade_id>")
@login_required
def trade_detail(trade_id: int):
    trade = _fetch_trade_by_id(trade_id)
    return render_template_string(
        _TEMPLATE,
        page="trade",
        trade=trade,
        trade_id=trade_id,
        testnet=config.TESTNET,
        error="",
    )


# ── JSON API endpoints (read-only) ────────────────────────────────────────────

@app.route("/api/status")
@login_required
def api_status():
    return jsonify({
        "mode":       "TESTNET" if config.TESTNET else "LIVE",
        "bot_status": _cached("bot_status", _fetch_bot_status, 10.0),
        "symbols":    config.SYMBOLS,
        "paused":     pause_manager.is_paused() if hasattr(pause_manager, "is_paused") else False,
        "perf":       _cached("perf", _fetch_perf, 30.0) or {},
        "balances":   _cached("balances", _fetch_balances, 15.0) or {},
    })


@app.route("/api/charts")
@login_required
def api_charts():
    return jsonify(_cached("charts", _fetch_all_charts, 60.0) or {})


@app.route("/api/heatmap")
@login_required
def api_heatmap():
    return jsonify(_cached("heatmap", _fetch_heatmap, 300.0) or {})


@app.route("/api/equity")
@login_required
def api_equity():
    return jsonify(_cached("equity", _fetch_equity, 120.0) or {})


@app.route("/api/risk")
@login_required
def api_risk():
    return jsonify(_cached("risk", _fetch_risk, 20.0) or {})


@app.route("/api/trades")
@login_required
def api_trades():
    trades = _fetch_trades(50)
    safe = [
        {k: v for k, v in t.items()
         if k not in ("api_key", "secret", "password")}
        for t in trades
    ]
    return jsonify(safe)


@app.route("/api/trade/<int:trade_id>")
@login_required
def api_trade_detail(trade_id: int):
    trade = _fetch_trade_by_id(trade_id)
    if not trade:
        return jsonify({"error": "not found"}), 404
    candles = _fetch_trade_candles(trade)
    return jsonify({"trade": trade, "candles": candles})


@app.route("/api/summary")
@login_required
def api_summary():
    if request.args.get("refresh"):
        with _cache_lock:
            _cache.pop("summary", None)
    return jsonify(_cached("summary", _generate_summary, 3600.0) or {})


@app.route("/api/events")
@login_required
def api_events():
    """Server-Sent Events: stream new bot log lines to the browser."""
    def generate():
        log_path = Path("/opt/btcbot/bot.log")
        if not log_path.exists():
            log_path = Path("bot.log")

        pos = 0
        if log_path.exists():
            pos = max(0, log_path.stat().st_size - 8192)

        while True:
            try:
                if log_path.exists():
                    with open(log_path, "r", errors="replace") as f:
                        f.seek(pos)
                        chunk = f.read()
                        pos = f.tell()
                    for line in chunk.splitlines():
                        line = line.strip()
                        if line and (
                            "[INFO]" in line or "[WARNING]" in line or "[ERROR]" in line
                        ):
                            yield f"data: {json.dumps({'msg': line})}\n\n"
            except Exception:
                pass
            time.sleep(3)
            yield ": keepalive\n\n"

    return Response(
        stream_with_context(generate()),
        content_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ══════════════════════════════════════════════════════════════════════════════
# NO TRADING ENDPOINTS BELOW.  Zero order_market_buy / order_market_sell /
# create_order / cancel_order calls exist anywhere in this file.
# ══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    host = config.DASHBOARD_HOST
    port = config.DASHBOARD_PORT
    if host not in ("127.0.0.1", "localhost"):
        print(f"[dashboard] WARNING: DASHBOARD_HOST={host!r} — must be 127.0.0.1")
    mode = "TESTNET" if config.TESTNET else "LIVE"
    print(f"[dashboard] Starting — http://{host}:{port}  ({mode})")
    print(f"[dashboard] SSH tunnel: ssh -L {port}:127.0.0.1:{port} root@134.209.197.173")
    print(f"[dashboard] Then open:  http://127.0.0.1:{port}")
    app.run(host=host, port=port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
