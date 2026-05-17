#!/usr/bin/env python3
"""
dashboard_server.py — Advanced read-only web dashboard for the BTC trading bot.

Binds to 127.0.0.1 only.  Access via SSH tunnel:
  ssh -L 8080:127.0.0.1:8080 root@134.209.197.173
  Then open: http://127.0.0.1:8080

SAFETY:  Zero buy / sell / order / cancel / execute endpoints exist.
         API keys and secrets are never sent to the browser.
         100 % read-only.  No full-page refresh — AJAX polling only.
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
from datetime import datetime, timezone
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

try:
    import performance_advanced as _perf_adv
    _PERF_ADV_OK = True
except ImportError:
    _PERF_ADV_OK = False

# ── Optional: pandas + regime (for heatmap) ───────────────────────────────────
try:
    import pandas as pd
    import pandas_ta as _pta  # noqa: F401
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


# ── Kline helpers ─────────────────────────────────────────────────────────────
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
            "total_pnl":          performance.total_pnl(),
            "daily_pnl":          performance.daily_pnl(today),
            "weekly_pnl":         performance.weekly_pnl(week),
            "total_trades":       performance.total_trades(),
            "win_rate":           performance.win_rate(),
            "max_dd_pct":         performance.max_drawdown_pct(),
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


def _fetch_arb_alerts(n: int = 20) -> list[dict]:
    for db in (Path("/opt/btcbot/arbitrage_watchlist.db"), Path("arbitrage_watchlist.db")):
        if db.exists():
            try:
                with sqlite3.connect(str(db)) as conn:
                    conn.row_factory = sqlite3.Row
                    rows = conn.execute(
                        """SELECT detected_at_utc, arb_type, route,
                                  gross_profit_pct, net_profit_pct,
                                  spread_pct, liquidity_score
                           FROM arbitrage_signals
                           ORDER BY id DESC LIMIT ?""",
                        (n,),
                    ).fetchall()
                return [dict(r) for r in rows]
            except Exception:
                pass
    return []


def _fetch_trend_alerts(n: int = 20) -> list[dict]:
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
            candles = [_kline_row(k) for k in klines[:-1]]
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

        t_open  = int(_parse(opened).timestamp() * 1000)
        t_close = int(_parse(closed).timestamp() * 1000)
        start_ms = t_open  - 24 * 3_600_000
        end_ms   = t_close + 12 * 3_600_000

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
    # Fallback: 24h ticker
    result = {}
    try:
        tickers = {t["symbol"]: t for t in c.get_ticker()}
    except Exception:
        return {}
    for sym in config.SYMBOLS:
        t = tickers.get(sym, {})
        try:
            pct = float(t.get("priceChangePercent", 0))
            if pct > 3:
                signal, color = "BULLISH", "#3fb950"
            elif pct < -3:
                signal, color = "BEARISH", "#f85149"
            elif abs(pct) < 1:
                signal, color = "RANGING", "#d29922"
            else:
                signal, color = "NEUTRAL", "#8b949e"
            result[sym] = {
                "regime": signal, "vol": "—", "adx": 0, "atr_pct": 0,
                "signal": signal, "color": color,
                "price": float(t.get("lastPrice", 0)), "change_24h": pct,
            }
        except Exception:
            pass
    return result


def _fetch_heatmap_full(c) -> dict:
    _SIG_COLORS = {
        "BULLISH":  "#3fb950",
        "RANGING":  "#d29922",
        "HIGH_VOL": "#f0883e",
        "BEARISH":  "#f85149",
        "NO_SETUP": "#8b949e",
    }
    result = {}
    for sym in config.SYMBOLS:
        try:
            klines = c.get_klines(symbol=sym, interval="1h", limit=210)
            df = _klines_to_df(klines)
            trend, vol = _reg.classify(df)
            adx_df = df.ta.adx(length=14)
            atr_s  = df.ta.atr(length=14)
            adx    = float(adx_df["ADX_14"].iloc[-1]) if adx_df is not None else 0.0
            atr    = float(atr_s.iloc[-1]) if atr_s is not None else 0.0
            close  = float(df["close"].iloc[-1])
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
        pnl  = float(t.get("realized_pnl") or 0)
        ts   = t.get("closed_at_utc") or t.get("opened_at_utc") or ""
        date = ts[:10] if ts else "?"
        cumul += pnl
        peak   = max(peak, cumul)
        dd     = ((cumul - peak) / abs(peak) * 100) if peak != 0 else 0.0
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
        balances    = _cached("balances",    _fetch_balances,    15.0) or {}
        open_orders = _cached("open_orders", _fetch_open_orders, 15.0) or []
        perf        = _cached("perf",        _fetch_perf,        30.0) or {}

        usdt    = balances.get("USDT", {}).get("free", 0.0)
        open_val = sum(
            float(o.get("origQty", 0)) * float(o.get("price") or 0)
            for o in open_orders if o.get("side") == "BUY"
        )
        exposure_pct = (open_val / (usdt + open_val) * 100) if (usdt + open_val) > 0 else 0.0
        risk_usdt    = usdt * config.RISK_PER_TRADE

        daily_limit  = usdt * getattr(config, "MAX_DAILY_LOSS",  0.02)
        weekly_limit = usdt * getattr(config, "MAX_WEEKLY_LOSS", 0.05)
        daily_used   = max(0.0, -float(perf.get("daily_pnl",  0)))
        weekly_used  = max(0.0, -float(perf.get("weekly_pnl", 0)))

        return {
            "usdt_balance":       round(usdt, 2),
            "exposure_pct":       round(exposure_pct, 2),
            "open_position_value": round(open_val, 2),
            "risk_usdt_per_trade": round(risk_usdt, 4),
            "daily_loss_limit":   round(daily_limit,  4),
            "daily_loss_used":    round(daily_used,   4),
            "daily_remaining":    round(max(0, daily_limit  - daily_used),  4),
            "weekly_loss_limit":  round(weekly_limit, 4),
            "weekly_loss_used":   round(weekly_used,  4),
            "weekly_remaining":   round(max(0, weekly_limit - weekly_used), 4),
            "max_open_trades":    config.MAX_OPEN_TRADES,
            "open_trades_count":  sum(1 for o in open_orders if o.get("side") == "BUY"),
            "risk_per_trade_pct": config.RISK_PER_TRADE * 100,
        }
    except Exception:
        return {}


# ── AI market summary ─────────────────────────────────────────────────────────

def _generate_summary() -> dict:
    try:
        heatmap = _cached("heatmap", _fetch_heatmap, 300.0) or {}
        perf    = _cached("perf",    _fetch_perf,    30.0)  or {}
        logs    = _fetch_logs(5)

        lines: list[str] = []
        signals: list[str] = []

        for sym, d in heatmap.items():
            base    = sym.replace("USDT", "")
            sig     = d.get("signal", "NO_SETUP")
            adx     = d.get("adx", 0)
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

        wins   = signals.count("BULLISH")
        hvols  = signals.count("HIGH_VOL")
        ranges = signals.count("RANGING")
        if wins >= 3:
            overall = "broad bullish trend — prioritise RMR longs."
        elif hvols >= 2:
            overall = "elevated volatility across board — smaller size or stand aside."
        elif ranges >= 3:
            overall = "market in consolidation — RMR setups most likely."
        else:
            overall = "mixed signals — be selective, wait for high-quality setups."

        errors = [l for l in logs if "[ERROR]" in l]
        if errors:
            lines.append(f"Last error: {errors[0][-80:]}")

        lines.append(f"Overall: {overall}")
        return {
            "text":         "\n".join(lines),
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
.hdr{background:var(--card);border-bottom:1px solid var(--border);padding:10px 20px;display:flex;align-items:center;gap:12px;flex-wrap:wrap;position:sticky;top:0;z-index:100}
.hdr h1{font-size:16px;font-weight:700;color:#f0f6fc}
.badge{padding:2px 9px;border-radius:10px;font-size:10px;font-weight:700;letter-spacing:.5px}
.bl{background:#0f2d1a;color:var(--g);border:1px solid var(--g)}
.bt{background:#2d1f0f;color:var(--y);border:1px solid var(--y)}
.bro{background:#1a1a2e;color:var(--dim);border:1px solid var(--border);font-size:9px}
.bwatch{background:#1a1708;color:var(--y);border:1px solid var(--y);font-size:9px;letter-spacing:.3px}
.dot{display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:4px;vertical-align:middle}
.da{background:var(--g);box-shadow:0 0 5px var(--g)}.di{background:var(--r)}.du{background:var(--dim)}
.hdr-r{margin-left:auto;display:flex;gap:10px;align-items:center}
.dim{color:var(--dim)}
.btn-sm{background:var(--card);border:1px solid var(--border);color:var(--text);padding:4px 11px;border-radius:6px;cursor:pointer;font-size:12px}
.btn-sm:hover{background:var(--border)}
.wrap{max-width:1380px;margin:0 auto;padding:16px 20px}
.stitle{font-size:10px;font-weight:700;color:var(--dim);text-transform:uppercase;letter-spacing:.8px;margin:20px 0 8px;display:flex;align-items:center;gap:8px}
.stitle-ts{font-size:9px;font-weight:400;color:#30363d;text-transform:none;letter-spacing:0}
.g4{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}
.g3{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}
.g2{display:grid;grid-template-columns:repeat(2,1fr);gap:10px}
.card{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:14px 16px}
.lbl{font-size:9px;color:var(--dim);text-transform:uppercase;letter-spacing:.5px;margin-bottom:4px}
.val{font-size:19px;font-weight:700;color:#f0f6fc}
.sub{font-size:10px;color:var(--dim);margin-top:3px}
.cg{color:var(--g)}.cr{color:var(--r)}.cy{color:var(--y)}.co{color:var(--o)}.cb{color:var(--b)}.cd{color:var(--dim)}
.bg{border-color:rgba(63,185,80,.4)}.br{border-color:rgba(248,81,73,.4)}.by{border-color:rgba(210,153,34,.4)}
.tbl-wrap{overflow-x:auto;-webkit-overflow-scrolling:touch}
table{width:100%;border-collapse:collapse;font-size:12.5px}
th{background:var(--bg);color:var(--dim);font-size:9px;text-transform:uppercase;letter-spacing:.5px;font-weight:700;padding:7px 10px;border-bottom:1px solid var(--border);white-space:nowrap;text-align:left}
td{padding:7px 10px;border-bottom:1px solid var(--card);color:var(--text);vertical-align:middle}
tr:last-child td{border-bottom:none}
tr.clickable:hover td{background:#1c2128;cursor:pointer}
.mono{font-family:monospace}
.nodata{color:var(--dim);font-style:italic;padding:10px 0;font-size:12px}
.chart-hdr{display:flex;align-items:center;gap:8px;margin-bottom:8px;flex-wrap:wrap}
.chart-price{font-size:17px;font-weight:700;color:#f0f6fc}
.chg{font-size:10px;padding:2px 6px;border-radius:4px;font-weight:600}
.chg-p{background:#0f2d1a;color:var(--g)}.chg-n{background:#2d0f0f;color:var(--r)}.chg-0{background:var(--border);color:var(--dim)}
.heatmap-card{text-align:center;padding:12px}
.heatmap-sig{font-size:14px;font-weight:700;margin:4px 0}
.equity-wrap{padding:10px;min-height:200px}
.log-box{background:#010409;border:1px solid var(--border);border-radius:8px;padding:10px;font-family:monospace;font-size:11px;max-height:320px;overflow-y:auto}
.ll{padding:2px 0;border-bottom:1px solid #0d111750;white-space:pre-wrap;word-break:break-all}
.ll:last-child{border-bottom:none}
.li{color:#8b949e}.lw{color:var(--y)}.le{color:var(--r)}
.notif-box{background:#010409;border:1px solid var(--border);border-radius:8px;padding:10px;font-family:monospace;font-size:11px;max-height:260px;overflow-y:auto}
.ni{color:#58a6ff;padding:2px 0;display:block;border-bottom:1px solid #0d111750}
.nw{color:var(--y);padding:2px 0;display:block;border-bottom:1px solid #0d111750}
.ne{color:var(--r);padding:2px 0;display:block;border-bottom:1px solid #0d111750}
.summary-text{font-family:monospace;font-size:13px;line-height:1.9;white-space:pre-line;color:var(--text)}
.risk-bar-bg{background:var(--border);border-radius:3px;height:6px;margin-top:5px}
.risk-bar-fill{height:6px;border-radius:3px;transition:width .4s}
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
@media(max-width:900px){.g4{grid-template-columns:repeat(2,1fr)}.g3{grid-template-columns:repeat(2,1fr)}}
@media(max-width:560px){.g4,.g3,.g2{grid-template-columns:1fr}.hdr{padding:8px 12px;gap:8px}.wrap{padding:10px 12px}.val{font-size:16px}}
.trade-detail-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.td-row{display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid var(--border)}
.td-row:last-child{border-bottom:none}
.td-lbl{color:var(--dim);font-size:12px}
.td-val{font-weight:600;font-size:12px;font-family:monospace}
@media(max-width:600px){.trade-detail-grid{grid-template-columns:1fr}}
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
  <a href="/" class="btn-sm">&#8592; Dashboard</a>
  <h1>Trade #{{ t.get('id','?') }} &mdash; {{ t.get('symbol','') }}</h1>
  <span class="badge {{ 'bl' if not testnet else 'bt' }}">{{ 'LIVE' if not testnet else 'TESTNET' }}</span>
  <span class="badge bro">READ-ONLY</span>
</div>
<div class="wrap">
  <div class="stitle">Trade Detail</div>
  <div class="trade-detail-grid">
    <div class="card">
      {% set fields = [
        ('Symbol',   t.get('symbol','')),
        ('Strategy', t.get('strategy','')),
        ('Regime',   t.get('regime','')),
        ('Side',     'LONG'),
        ('Entry',    ('%.4f' % (t.get('fill_price') or t.get('entry_price') or 0))),
        ('Stop',     ('%.4f' % (t.get('stop_price') or 0))),
        ('TP',       ('%.4f' % (t.get('tp_price') or 0))),
        ('Exit',     ('%.4f' % (t.get('exit_price') or 0))),
        ('Size',     ('%.5f' % (t.get('size') or 0))),
        ('PnL',      ('%+.4f USDT' % pnl)),
        ('Close',    t.get('close_reason','')),
        ('ADX',      ('%.1f' % (t.get('adx') or 0))),
        ('ATR%',     ('%.2f%%' % (t.get('atr_pct') or 0))),
        ('Score',    ('%.1f%%' % (t.get('score_pct') or 0))),
        ('Opened',   (t.get('opened_at_utc') or '')[:16]),
        ('Closed',   (t.get('closed_at_utc') or '')[:16]),
      ] %}
      {% for lbl, val in fields %}
      <div class="td-row">
        <span class="td-lbl">{{ lbl }}</span>
        <span class="td-val {{ 'cg' if (lbl=='PnL' and pnl>=0) else 'cr' if (lbl=='PnL' and pnl<0) else '' }}">{{ val or '&mdash;' }}</span>
      </div>
      {% endfor %}
    </div>
    <div class="card">
      <div class="lbl">Price Chart Around Trade</div>
      <canvas id="trade-chart" style="max-height:340px"></canvas>
      <div class="sub" id="trade-chart-msg" style="margin-top:6px">Loading candles&hellip;</div>
    </div>
  </div>
</div>
<script>
(async () => {
  const tradeId    = {{ t.get('id',0) }};
  const entryPrice = {{ t.get('fill_price') or t.get('entry_price') or 0 }};
  const exitPrice  = {{ t.get('exit_price') or 0 }};
  try {
    const r = await fetch('/api/trade/' + tradeId);
    const data = await r.json();
    const candles = data.candles || [];
    const msg = document.getElementById('trade-chart-msg');
    if (msg) msg.textContent = candles.length ? '' : 'No candle data.';
    if (!candles.length) return;
    const labels = candles.map(c => {
      const d = new Date(c.t);
      return (d.getUTCMonth()+1)+'/'+d.getUTCDate()+' '+String(d.getUTCHours()).padStart(2,'0')+':00';
    });
    const closes = candles.map(c => c.c);
    // Draw horizontal lines for entry/exit using annotation-free approach
    const entryData = candles.map(() => entryPrice || null);
    const exitData  = candles.map(() => exitPrice  || null);
    new Chart(document.getElementById('trade-chart'), {
      type: 'line',
      data: {
        labels,
        datasets: [
          { label:'Close',    data:closes,    borderColor:'#58a6ff', borderWidth:1.5, pointRadius:0, fill:false },
          entryPrice ? { label:'Entry', data:entryData, borderColor:'#3fb950', borderWidth:1, borderDash:[4,4], pointRadius:0, fill:false } : null,
          exitPrice  ? { label:'Exit',  data:exitData,  borderColor:'#f85149', borderWidth:1, borderDash:[4,4], pointRadius:0, fill:false } : null,
        ].filter(Boolean),
      },
      options: {
        responsive:true, animation:false,
        plugins:{
          legend:{labels:{color:'#8b949e',font:{size:10}}},
          tooltip:{callbacks:{afterBody:(items)=>{const c=candles[items[0]?.dataIndex];return c?[`O:${c.o} H:${c.h} L:${c.l} C:${c.c}`]:[];}}}
        },
        scales:{
          x:{ticks:{color:'#8b949e',maxTicksLimit:8,font:{size:10}},grid:{color:'#21262d'}},
          y:{ticks:{color:'#8b949e',font:{size:10}},grid:{color:'#21262d'}},
        }
      }
    });
  } catch(e) {
    const msg = document.getElementById('trade-chart-msg');
    if (msg) msg.textContent = 'Chart unavailable.';
  }
})();
</script>
<div class="footer">BTC Bot Dashboard &mdash; Read-Only &mdash; No buy/sell/order controls.</div>

{% else %}
<!-- ═══════════════════════ MAIN DASHBOARD ═══════════════════════ -->

<div class="hdr">
  <h1>BTC Bot</h1>
  <span class="badge {{ 'bl' if not testnet else 'bt' }}">{{ 'LIVE' if not testnet else 'TESTNET' }}</span>
  <span class="badge bro">READ-ONLY</span>
  <span id="hdr-bot-dot"><span class="dot du"></span><span class="cd">—</span></span>
  <span id="hdr-paused-badge" style="display:none"><span class="badge bt">PAUSED</span></span>
  <span class="dim" style="font-size:11px">{{ symbols|join(' &middot; ') }}</span>
  <div class="hdr-r">
    <span class="dim" style="font-size:10px" id="hdr-updated">—</span>
    <a href="/logout"><button class="btn-sm">Sign out</button></a>
  </div>
</div>

<div class="wrap">

<!-- A. AI Market Summary -->
<div class="stitle">AI Market Summary <span class="stitle-ts" id="summary-ts"></span></div>
<div class="card">
  <div class="summary-text" id="summary-text">{{ summary.get('text','Loading&hellip;') }}</div>
  <div class="sub" style="margin-top:6px">Generated: <span id="summary-gen">{{ summary.get('generated_at','—') }}</span>
    &nbsp;<a href="#" onclick="reloadSummary();return false;" style="font-size:10px">&#8635; refresh</a>
  </div>
</div>

<!-- A. Status -->
<div class="stitle">Status <span class="stitle-ts" id="status-ts"></span></div>
<div class="g4" id="status-grid">
  <div class="card" id="card-bot-svc">
    <div class="lbl">Bot Service</div>
    <div class="val cd" id="bot-svc-val">—</div>
    <div class="sub" id="bot-svc-sub">{{ 'LIVE' if not testnet else 'TESTNET' }} mode</div>
  </div>
  <div class="card" id="card-trading">
    <div class="lbl">Trading</div>
    <div class="val cd" id="trading-val">—</div>
    <div class="sub" id="trading-sub">—</div>
  </div>
  <div class="card" id="card-orders">
    <div class="lbl">Open Orders</div>
    <div class="val" id="orders-val">—</div>
    <div class="sub">Across {{ symbols|length }} symbol(s)</div>
  </div>
  <div class="card" id="card-trades-hdr">
    <div class="lbl">Total Trades</div>
    <div class="val" id="total-trades-val">—</div>
    <div class="sub" id="win-rate-sub">Win rate —</div>
  </div>
</div>

<!-- B. Performance -->
<div class="stitle">Performance <span class="stitle-ts" id="perf-ts"></span></div>
<div class="g4">
  <div class="card" id="card-today-pnl">
    <div class="lbl">Today PnL</div>
    <div class="val cd" id="today-pnl-val">—</div>
    <div class="sub">USDT</div>
  </div>
  <div class="card" id="card-week-pnl">
    <div class="lbl">This Week PnL</div>
    <div class="val cd" id="week-pnl-val">—</div>
    <div class="sub">USDT</div>
  </div>
  <div class="card" id="card-total-pnl">
    <div class="lbl">Total PnL</div>
    <div class="val cd" id="total-pnl-val">—</div>
    <div class="sub">USDT</div>
  </div>
  <div class="card" id="card-dd">
    <div class="lbl">Max Drawdown</div>
    <div class="val cd" id="dd-val">—</div>
    <div class="sub" id="dd-sub">Consec. losses: —</div>
  </div>
</div>

<!-- K. Risk Exposure -->
<div class="stitle">Risk Exposure <span class="stitle-ts" id="risk-ts"></span></div>
<div class="g3" id="risk-grid"><div class="card cd">Loading&hellip;</div></div>

<!-- E. Signal Heatmap -->
<div class="stitle">Signal Heatmap <span class="stitle-ts" id="heatmap-ts"></span></div>
<div class="g4" id="heatmap-grid"><div class="card cd">Loading heatmap&hellip;</div></div>

<!-- D. Price Charts -->
<div class="stitle">Price Charts (1H) <span class="stitle-ts" id="charts-ts"></span></div>
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

<!-- C. PnL Equity Curve -->
<div class="stitle">PnL Equity Curve <span class="stitle-ts" id="equity-ts"></span></div>
<div class="card">
  <div class="equity-wrap">
    <canvas id="equity-chart"></canvas>
    <div class="sub" id="equity-msg" style="margin-top:6px">Loading equity data&hellip;</div>
  </div>
</div>

<!-- Asset Balances -->
<div class="stitle">Asset Balances <span class="stitle-ts" id="bal-ts"></span></div>
<div class="card">
  <div class="tbl-wrap">
    <table>
      <tr><th>Asset</th><th>Free</th><th>Locked</th><th>Total</th></tr>
      <tbody id="balances-tbody">
        <tr><td colspan="4" class="nodata">Loading&hellip;</td></tr>
      </tbody>
    </table>
  </div>
</div>

<!-- Open Orders -->
<div class="stitle">Open Orders <span class="stitle-ts" id="orders-ts"></span></div>
<div class="card" id="orders-container"><div class="nodata">Loading&hellip;</div></div>

<!-- H. Trade Replay -->
<div class="stitle">Recent Trades — click for detail <span class="stitle-ts" id="trades-ts"></span></div>
<div class="card" id="trades-container"><div class="nodata">Loading&hellip;</div></div>

<!-- F. Trending Coins -->
<div class="stitle">Trending Coins <span class="badge bwatch">WATCH ONLY</span> <span class="stitle-ts" id="trends-ts"></span></div>
<div class="card" id="trends-container"><div class="nodata cd">Loading trend data&hellip;</div></div>

<!-- G. Arbitrage Watch -->
<div class="stitle">Arbitrage Watch <span class="badge bwatch">WATCH ONLY</span> <span class="stitle-ts" id="arb-ts"></span></div>
<div class="card" id="arb-container"><div class="nodata cd">Loading arbitrage data&hellip;</div></div>

<!-- Bot Log -->
<div class="stitle">Bot Log <span class="stitle-ts" id="log-ts"></span></div>
<div class="log-box" id="log-box"><span class="cd">Loading&hellip;</span></div>

<!-- J. Live Notifications -->
<div class="stitle">Live Notifications</div>
<div class="notif-box" id="notif-box">
  <span class="cd" id="notif-status">Connecting&hellip;</span>
</div>

</div><!-- /wrap -->
<div class="footer" id="footer-ts">BTC Bot Dashboard &mdash; Read-Only &mdash; No buy/sell/order controls.</div>

<!-- ═══ JAVASCRIPT ═══ -->
<script>
'use strict';
const _SYMBOLS = {{ symbols | tojson }};

/* ── helpers ── */
function _ts() { return new Date().toLocaleTimeString([], {hour:'2-digit',minute:'2-digit',second:'2-digit'}); }
function _setTs(id) { const el=document.getElementById(id); if(el) el.textContent='updated '+_ts(); }
function _txt(id, v) { const el=document.getElementById(id); if(el) el.textContent=v; }
function _html(id, v) { const el=document.getElementById(id); if(el) el.innerHTML=v; }
function _cls(id, ...cls) { const el=document.getElementById(id); if(el){ el.className=''; el.classList.add(...cls.filter(Boolean)); } }
function _pnlColor(v) { return v>=0?'cg':'cr'; }
function _pnlFmt(v) { v=parseFloat(v)||0; return (v>=0?'+':'')+v.toFixed(4); }
function _chgClass(v) { return v>0?'chg chg-p':v<0?'chg chg-n':'chg chg-0'; }
function _cardBorder(id, ok) { const el=document.getElementById(id); if(el){ el.classList.remove('bg','br','by'); el.classList.add(ok?'bg':'br'); } }

/* ── Chart.js defaults ── */
Chart.defaults.color = '#8b949e';
Chart.defaults.font.family = 'monospace';
Chart.defaults.font.size = 10;

/* ── A+B+Balances+Orders: loadStatus ── */
let _statusLoading = false;
async function loadStatus() {
  if (_statusLoading) return;
  _statusLoading = true;
  try {
    const r = await fetch('/api/status');
    if (!r.ok) return;
    const d = await r.json();

    /* header dot */
    const active = d.bot_status === 'active';
    const paused = d.paused;
    _html('hdr-bot-dot', `<span class="dot ${active?'da':'di'}"></span><span class="${active?'cg':'cr'}">${d.bot_status||'?'}</span>`);
    const pbEl = document.getElementById('hdr-paused-badge');
    if (pbEl) pbEl.style.display = paused ? '' : 'none';
    _txt('hdr-updated', 'updated '+_ts());

    /* status cards */
    _txt('bot-svc-val', (d.bot_status||'?').toUpperCase());
    _cls('bot-svc-val', active?'cg':'cr', 'val');
    _cardBorder('card-bot-svc', active);

    _txt('trading-val', paused ? 'PAUSED' : 'ACTIVE');
    _cls('trading-val', paused?'cr':'cg', 'val');
    _txt('trading-sub', d.pause_reason || 'No restrictions');
    _cardBorder('card-trading', !paused);

    const orders = d.open_orders || [];
    _txt('orders-val', orders.length);

    const perf = d.perf || {};
    _txt('total-trades-val', perf.total_trades ?? '—');
    _txt('win-rate-sub', `Win rate ${((perf.win_rate||0)*100).toFixed(1)}%`);

    /* perf cards */
    const dp = parseFloat(perf.daily_pnl  || 0);
    const wp = parseFloat(perf.weekly_pnl || 0);
    const tp = parseFloat(perf.total_pnl  || 0);
    const dd = parseFloat(perf.max_dd_pct || 0);

    _txt('today-pnl-val',  _pnlFmt(dp)); _cls('today-pnl-val',  _pnlColor(dp), 'val'); _cardBorder('card-today-pnl', dp>=0);
    _txt('week-pnl-val',   _pnlFmt(wp)); _cls('week-pnl-val',   _pnlColor(wp), 'val'); _cardBorder('card-week-pnl',  wp>=0);
    _txt('total-pnl-val',  _pnlFmt(tp)); _cls('total-pnl-val',  _pnlColor(tp), 'val'); _cardBorder('card-total-pnl', tp>=0);
    _txt('dd-val', dd.toFixed(2)+'%');
    _cls('dd-val', dd>5?'cr':dd>2?'cy':'cg', 'val');
    _txt('dd-sub', `Consec. losses: ${perf.consecutive_losses ?? 0}`);
    _cardBorder('card-dd', dd<=2);

    /* balances */
    const bals = d.balances || {};
    const assets = ['USDT','BTC','ETH','SOL','BNB'];
    let bRows = '';
    for (const a of assets) {
      const b = bals[a] || {};
      const fr = parseFloat(b.free||0), lk = parseFloat(b.locked||0), tot = fr+lk;
      bRows += `<tr>
        <td><strong>${a}</strong></td>
        <td class="mono ${fr>0?'cg':'cd'}">${fr.toFixed(6)}</td>
        <td class="mono cd">${lk.toFixed(6)}</td>
        <td class="mono ${tot>0?'cg':'cd'}">${tot.toFixed(6)}</td>
      </tr>`;
    }
    _html('balances-tbody', bRows);
    _setTs('bal-ts');

    /* open orders */
    const oCont = document.getElementById('orders-container');
    if (oCont) {
      if (!orders.length) {
        oCont.innerHTML = '<div class="nodata">No open orders.</div>';
      } else {
        let h = '<div class="tbl-wrap"><table><tr><th>Symbol</th><th>Side</th><th>Type</th><th>Qty</th><th>Price</th><th>Stop</th><th>Status</th><th>Time</th></tr>';
        for (const o of orders) {
          h += `<tr>
            <td><strong>${o.symbol||''}</strong></td>
            <td class="${o.side==='BUY'?'cg':'cr'}"><strong>${o.side||''}</strong></td>
            <td class="mono cd">${o.type||''}</td>
            <td class="mono">${o.origQty||''}</td>
            <td class="mono">${o.price||''}</td>
            <td class="mono cd">${o.stopPrice||'—'}</td>
            <td class="cd">${o.status||''}</td>
            <td class="cd">${o.time_fmt||'—'}</td>
          </tr>`;
        }
        oCont.innerHTML = h + '</table></div>';
      }
    }
    _setTs('orders-ts');
    _setTs('status-ts');
    _setTs('perf-ts');
  } catch(e) {}
  finally { _statusLoading = false; }
}

/* ── D. Price charts: loadCharts ── */
let _chartsLoading = false;
async function loadCharts() {
  if (_chartsLoading) return;
  _chartsLoading = true;
  try {
    const r = await fetch('/api/charts');
    if (!r.ok) return;
    const data = await r.json();
    for (const [sym, info] of Object.entries(data)) {
      _txt('price-'+sym, info.current_price.toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:4}));

      const mkChg = (id, val, label) => {
        const el = document.getElementById(id);
        if (!el) return;
        el.textContent = label+(val>=0?'+':'')+val.toFixed(2)+'%';
        el.className = _chgClass(val);
      };
      mkChg('ch1h-'+sym,  info.change_1h,  '1h ');
      mkChg('ch4h-'+sym,  info.change_4h,  '4h ');
      mkChg('ch24h-'+sym, info.change_24h, '24h ');

      const closes = info.candles.map(c => c.c);
      const vols   = info.candles.map(c => c.v);
      const labels = info.candles.map((c,i) => i%8===0 ? new Date(c.t).getUTCHours()+'h' : '');
      const color  = closes[closes.length-1] >= closes[0] ? '#3fb950' : '#f85149';

      const cCtx = document.getElementById('chart-'+sym);
      if (cCtx) { Chart.getChart(cCtx)?.destroy(); new Chart(cCtx, {
        type:'line',
        data:{labels,datasets:[{data:closes,borderColor:color,borderWidth:1.5,pointRadius:0,fill:true,backgroundColor:color+'15'}]},
        options:{responsive:true,animation:false,plugins:{legend:{display:false},tooltip:{enabled:false}},scales:{x:{display:false},y:{display:false}}}
      }); }

      const vCtx = document.getElementById('vol-'+sym);
      if (vCtx) { Chart.getChart(vCtx)?.destroy(); new Chart(vCtx, {
        type:'bar',
        data:{labels,datasets:[{data:vols,backgroundColor:'#8b949e33',borderWidth:0}]},
        options:{responsive:true,animation:false,plugins:{legend:{display:false},tooltip:{enabled:false}},scales:{x:{display:false},y:{display:false}}}
      }); }
    }
    _setTs('charts-ts');
  } catch(e) {}
  finally { _chartsLoading = false; }
}

/* ── E. Signal heatmap: loadHeatmap ── */
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
        <div class="sub">${d.regime} &middot; ${d.vol}</div>
        <div class="sub">ADX ${d.adx} &middot; ATR ${d.atr_pct}%</div>
        <div class="sub" style="margin-top:2px">$${(d.price||0).toLocaleString(undefined,{maximumFractionDigits:2})}</div>`;
      grid.appendChild(card);
    }
    _setTs('heatmap-ts');
  } catch(e) {}
}

/* ── K. Risk: loadRisk ── */
async function loadRisk() {
  try {
    const r = await fetch('/api/risk');
    if (!r.ok) return;
    const d = await r.json();
    const grid = document.getElementById('risk-grid');
    if (!grid) return;
    const barPct = (used, limit) => !limit ? 0 : Math.min(100, used/limit*100).toFixed(1);
    const dp = barPct(d.daily_loss_used,  d.daily_loss_limit);
    const wp = barPct(d.weekly_loss_used, d.weekly_loss_limit);
    const tp = d.max_open_trades ? (d.open_trades_count/d.max_open_trades*100).toFixed(0) : 0;
    grid.innerHTML = `
      <div class="card">
        <div class="lbl">Account Exposure</div>
        <div class="val ${d.exposure_pct>10?'cr':d.exposure_pct>5?'cy':'cg'}">${d.exposure_pct}%</div>
        <div class="sub">Open value: $${d.open_position_value}</div>
      </div>
      <div class="card">
        <div class="lbl">Risk / Trade</div>
        <div class="val">${d.risk_per_trade_pct}%</div>
        <div class="sub">~$${d.risk_usdt_per_trade} USDT</div>
      </div>
      <div class="card">
        <div class="lbl">Open Trades</div>
        <div class="val">${d.open_trades_count} / ${d.max_open_trades}</div>
        <div class="risk-bar-bg"><div class="risk-bar-fill" style="width:${tp}%;background:#58a6ff"></div></div>
      </div>
      <div class="card">
        <div class="lbl">Daily Loss Budget</div>
        <div class="val ${parseFloat(dp)>80?'cr':parseFloat(dp)>50?'cy':'cg'}">${d.daily_remaining>=0?'$'+d.daily_remaining:'EXCEEDED'}</div>
        <div class="sub">Used $${d.daily_loss_used} / $${d.daily_loss_limit}</div>
        <div class="risk-bar-bg"><div class="risk-bar-fill" style="width:${dp}%;background:${parseFloat(dp)>80?'#f85149':'#d29922'}"></div></div>
      </div>
      <div class="card">
        <div class="lbl">Weekly Loss Budget</div>
        <div class="val ${parseFloat(wp)>80?'cr':parseFloat(wp)>50?'cy':'cg'}">${d.weekly_remaining>=0?'$'+d.weekly_remaining:'EXCEEDED'}</div>
        <div class="sub">Used $${d.weekly_loss_used} / $${d.weekly_loss_limit}</div>
        <div class="risk-bar-bg"><div class="risk-bar-fill" style="width:${wp}%;background:${parseFloat(wp)>80?'#f85149':'#d29922'}"></div></div>
      </div>
      <div class="card">
        <div class="lbl">USDT Balance</div>
        <div class="val">$${d.usdt_balance}</div>
        <div class="sub">Free in account</div>
      </div>`;
    _setTs('risk-ts');
  } catch(e) {}
}

/* ── C. Equity curve: loadEquity ── */
let _eqChart = null;
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
    if (_eqChart) { _eqChart.destroy(); _eqChart = null; }
    _eqChart = new Chart(ctx, {
      type:'line',
      data:{
        labels: data.dates,
        datasets:[
          {label:'Cumulative PnL (USDT)',data:data.cumulative,borderColor:'#3fb950',borderWidth:2,pointRadius:0,fill:false,yAxisID:'y'},
          {label:'Drawdown %',data:data.drawdown,borderColor:'#f85149',borderWidth:1,pointRadius:0,fill:true,backgroundColor:'rgba(248,81,73,0.08)',yAxisID:'y1'},
        ]
      },
      options:{
        responsive:true,animation:false,
        interaction:{mode:'index',intersect:false},
        plugins:{legend:{labels:{color:'#8b949e',font:{size:11}}}},
        scales:{
          x:{ticks:{color:'#8b949e',maxTicksLimit:8,font:{size:10}},grid:{color:'#21262d'}},
          y:{ticks:{color:'#8b949e',font:{size:10}},grid:{color:'#21262d'},title:{display:true,text:'USDT',color:'#8b949e'}},
          y1:{position:'right',ticks:{color:'#f85149',font:{size:10}},grid:{drawOnChartArea:false},title:{display:true,text:'DD%',color:'#f85149'}},
        }
      }
    });
    _setTs('equity-ts');
  } catch(e) {}
}

/* ── H. Trade Replay: loadTrades ── */
async function loadTrades() {
  try {
    const r = await fetch('/api/trades');
    if (!r.ok) return;
    const trades = await r.json();
    const cont = document.getElementById('trades-container');
    if (!cont) return;
    if (!trades.length) {
      cont.innerHTML = '<div class="nodata">No trades recorded yet.</div>';
    } else {
      let h = '<div class="tbl-wrap"><table><tr><th>Closed</th><th>Symbol</th><th>Strategy</th><th>Entry</th><th>Exit</th><th>PnL</th><th>PnL%</th><th>Duration</th><th>Reason</th><th>Regime</th></tr>';
      for (const t of trades.slice(0,20)) {
        const pnl = parseFloat(t.realized_pnl || 0);
        const entry = parseFloat(t.fill_price || t.entry_price || 0);
        const exit  = parseFloat(t.exit_price || 0);
        const pnlPct = (entry && exit) ? ((exit-entry)/entry*100).toFixed(2)+'%' : '—';
        const opened = (t.opened_at_utc||'').slice(0,16);
        const closed = (t.closed_at_utc||t.opened_at_utc||'').slice(0,16);
        let dur = '—';
        try {
          if (t.opened_at_utc && t.closed_at_utc) {
            const ms = new Date(t.closed_at_utc) - new Date(t.opened_at_utc);
            const mins = Math.round(ms/60000);
            dur = mins >= 60 ? `${(mins/60).toFixed(1)}h` : `${mins}m`;
          }
        } catch(_) {}
        h += `<tr class="clickable" onclick="window.location='/trade/${t.id}'">
          <td class="cd mono">${closed}</td>
          <td><strong>${t.symbol||''}</strong></td>
          <td class="cd">${t.strategy||''}</td>
          <td class="mono">${entry?entry.toFixed(2):'—'}</td>
          <td class="mono">${exit?exit.toFixed(2):'—'}</td>
          <td class="mono ${pnl>=0?'cg':'cr'}"><strong>${_pnlFmt(pnl)}</strong></td>
          <td class="mono ${pnl>=0?'cg':'cr'}">${pnlPct}</td>
          <td class="cd">${dur}</td>
          <td class="cd">${t.close_reason||''}</td>
          <td class="cd" style="font-size:10px">${t.regime||''}</td>
        </tr>`;
      }
      cont.innerHTML = h + '</table></div>';
    }
    _setTs('trades-ts');
  } catch(e) {}
}

/* ── F. Trending Coins: loadTrends ── */
async function loadTrends() {
  try {
    const r = await fetch('/api/trends');
    if (!r.ok) return;
    const rows = await r.json();
    const cont = document.getElementById('trends-container');
    if (!cont) return;
    if (!rows.length) {
      cont.innerHTML = '<div class="nodata cd">No trend alerts yet.</div>';
      _setTs('trends-ts');
      return;
    }
    // Determine columns — prefer known fields, fall back to all keys
    const WANT = ['symbol','grade','score','sentiment','volume_spike','vol_spike','spread','timeframe_confirmations','tf_confirms','detected_at_utc','detected_at'];
    const keys = Object.keys(rows[0]).filter(k => !['id','raw_data','json','data'].includes(k));
    const cols = [...new Set([...WANT.filter(w => keys.includes(w)), ...keys])].slice(0,10);
    const colLabel = c => c.replace(/_/g,' ').replace(/\b\w/g,l=>l.toUpperCase());
    let h = '<div class="tbl-wrap"><table><tr>';
    for (const c of cols) h += `<th>${colLabel(c)}</th>`;
    h += '<th></th></tr>';
    for (const row of rows) {
      h += '<tr>';
      for (const c of cols) {
        const v = row[c] ?? '—';
        const cls = c==='grade' ? (v==='A'||v==='B'?'cg':v==='C'?'cy':'cr') : '';
        h += `<td class="mono ${cls}" style="font-size:11px">${v}</td>`;
      }
      h += '<td><span class="badge bwatch">WATCH ONLY</span></td></tr>';
    }
    cont.innerHTML = h + '</table></div>';
    _setTs('trends-ts');
  } catch(e) {}
}

/* ── G. Arbitrage Watch: loadArb ── */
async function loadArb() {
  try {
    const r = await fetch('/api/arb');
    if (!r.ok) return;
    const rows = await r.json();
    const cont = document.getElementById('arb-container');
    if (!cont) return;
    if (!rows.length) {
      cont.innerHTML = '<div class="nodata cd">No arbitrage opportunities yet.</div>';
      _setTs('arb-ts');
      return;
    }
    let h = '<div class="tbl-wrap"><table><tr><th>Detected</th><th>Type</th><th>Route</th><th>Gross</th><th>Net</th><th>Spread</th><th>Liquidity</th><th></th></tr>';
    for (const a of rows) {
      const gross = parseFloat(a.gross_profit_pct||0);
      const net   = parseFloat(a.net_profit_pct||0);
      const spread= parseFloat(a.spread_pct||0);
      h += `<tr>
        <td class="cd mono" style="font-size:11px">${(a.detected_at_utc||'').slice(0,16)}</td>
        <td class="cd" style="font-size:11px">${(a.arb_type||'').replace('_',' ')}</td>
        <td class="mono" style="font-size:10px">${a.route||''}</td>
        <td class="${gross>=0?'cg':'cr'}">${gross>=0?'+':''}${gross.toFixed(3)}%</td>
        <td class="${net>=0?'cg':'cr'}"><strong>${net>=0?'+':''}${net.toFixed(3)}%</strong></td>
        <td class="cd">${spread.toFixed(3)}%</td>
        <td class="cd">${a.liquidity_score||'—'}</td>
        <td><span class="badge bwatch">WATCH ONLY</span></td>
      </tr>`;
    }
    cont.innerHTML = h + '</table></div>';
    _setTs('arb-ts');
  } catch(e) {}
}

/* ── Bot Log: loadLogs ── */
async function loadLogs() {
  try {
    const r = await fetch('/api/logs');
    if (!r.ok) return;
    const lines = await r.json();
    const box = document.getElementById('log-box');
    if (!box) return;
    if (!lines.length) { box.innerHTML = '<span class="cd">No log data.</span>'; return; }
    box.innerHTML = lines.map(line => {
      const cls = line.includes('[ERROR]')?'ll le':line.includes('[WARNING]')?'ll lw':'ll li';
      return `<div class="${cls}">${line.replace(/</g,'&lt;')}</div>`;
    }).join('');
    _setTs('log-ts');
  } catch(e) {}
}

/* ── J. Live Notifications (SSE) ── */
function connectSSE() {
  const box = document.getElementById('notif-box');
  const status = document.getElementById('notif-status');
  try {
    const es = new EventSource('/api/events');
    if (status) status.textContent = '';
    es.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        const msg = data.msg || '';
        if (!msg || !box) return;
        const div = document.createElement('div');
        div.className = msg.includes('[ERROR]')?'ne':msg.includes('[WARNING]')?'nw':'ni';
        div.textContent = msg;
        if (status && status.parentNode === box) status.remove();
        box.insertBefore(div, box.firstChild);
        while (box.children.length > 80) box.removeChild(box.lastChild);
      } catch(_) {}
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

/* ── I. AI Summary refresh ── */
async function reloadSummary() {
  try {
    const r = await fetch('/api/summary?refresh=1');
    const d = await r.json();
    _txt('summary-text', d.text || '');
    _txt('summary-gen', d.generated_at || '—');
  } catch(e) {}
}

/* ── Footer timestamp ── */
function updateFooter() {
  const el = document.getElementById('footer-ts');
  if (el) el.textContent = 'BTC Bot Dashboard — Read-Only — '+new Date().toUTCString().slice(0,25)+' UTC — No buy/sell/order controls.';
}

/* ── Boot + polling ── */
document.addEventListener('DOMContentLoaded', () => {
  // Immediate loads
  loadStatus();
  loadCharts();
  loadHeatmap();
  loadRisk();
  loadEquity();
  loadTrades();
  loadArb();
  loadTrends();
  loadLogs();
  connectSSE();
  updateFooter();

  // Polling intervals (no full-page reload — zero flicker)
  setInterval(loadStatus,  15_000);   // status, perf, balances, orders
  setInterval(loadLogs,    10_000);   // bot log
  setInterval(loadRisk,    20_000);   // risk exposure
  setInterval(loadTrades,  60_000);   // trade list
  setInterval(loadArb,     60_000);   // arb watch
  setInterval(loadTrends,  60_000);   // trending coins
  setInterval(loadCharts,  60_000);   // price charts
  setInterval(loadEquity, 120_000);   // equity curve
  setInterval(loadHeatmap,300_000);   // signal heatmap
  setInterval(updateFooter, 60_000);  // footer clock
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
    # Pass minimal data for immediate render; all sections updated via AJAX.
    summary = _cached("summary", _generate_summary, 3600.0) or {}
    try:
        paused = pause_manager.is_paused()
    except Exception:
        paused = False
    return render_template_string(
        _TEMPLATE,
        page="dashboard",
        testnet=config.TESTNET,
        symbols=config.SYMBOLS,
        paused=paused,
        summary=summary,
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


# ── JSON API endpoints (all read-only) ────────────────────────────────────────

@app.route("/api/status")
@login_required
def api_status():
    bot_status  = _cached("bot_status",  _fetch_bot_status,  10.0) or "unknown"
    perf        = _cached("perf",        _fetch_perf,        30.0) or {}
    balances    = _cached("balances",    _fetch_balances,    15.0) or {}
    open_orders = _cached("open_orders", _fetch_open_orders, 15.0) or []
    try:
        paused       = pause_manager.is_paused()
        pause_reason = pause_manager.pause_reason() if paused else ""
    except Exception:
        paused, pause_reason = False, ""
    return jsonify({
        "mode":         "TESTNET" if config.TESTNET else "LIVE",
        "bot_status":   bot_status,
        "symbols":      config.SYMBOLS,
        "paused":       paused,
        "pause_reason": pause_reason,
        "perf":         perf,
        "balances":     balances,
        "open_orders":  open_orders,
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
        {k: v for k, v in t.items() if k not in ("api_key", "secret", "password")}
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


@app.route("/api/arb")
@login_required
def api_arb():
    return jsonify(_cached("arb", _fetch_arb_alerts, 60.0) or [])


@app.route("/api/trends")
@login_required
def api_trends():
    return jsonify(_cached("trends", _fetch_trend_alerts, 60.0) or [])


@app.route("/api/logs")
@login_required
def api_logs():
    return jsonify(_fetch_logs(20))


@app.route("/api/summary")
@login_required
def api_summary():
    if request.args.get("refresh"):
        with _cache_lock:
            _cache.pop("summary", None)
    return jsonify(_cached("summary", _generate_summary, 3600.0) or {})


@app.route("/api/analytics")
@login_required
def api_analytics():
    """Advanced analytics by session/regime/hour/weekday/score/ADX/ATR/grade."""
    if not _PERF_ADV_OK:
        return jsonify({"error": "performance_advanced not available"})

    def _build():
        return {
            "by_session":  _perf_adv.pnl_by_session(),
            "by_regime":   _perf_adv.pnl_by_regime(),
            "by_hour":     _perf_adv.pnl_by_hour(),
            "by_weekday":  _perf_adv.pnl_by_weekday(),
            "by_score":    _perf_adv.pnl_by_score_bucket(),
            "by_adx":      _perf_adv.pnl_by_adx_bucket(),
            "by_atr":      _perf_adv.pnl_by_atr_bucket(),
            "by_grade":    _perf_adv.pnl_by_grade(),
            "by_symbol":   _perf_adv.pnl_by_symbol(),
            "by_strategy": _perf_adv.pnl_by_strategy(),
            "best":        _perf_adv.best_market_conditions(top_n=3),
            "worst":       _perf_adv.worst_market_conditions(top_n=3),
            "grade_dist":  _perf_adv.grade_distribution(),
            "summary":     _perf_adv.summary_report(),
        }

    return jsonify(_cached("analytics", _build, 120.0) or {})


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
                        if line and ("[INFO]" in line or "[WARNING]" in line or "[ERROR]" in line):
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
