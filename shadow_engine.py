"""
shadow_engine.py — Paper-trading simulation layer.

Records shadow signals and tracks their TP/SL outcomes across cycles.
NEVER places live trades. NEVER affects live positions. NEVER affects risk.

Shadow trades are evaluated each cycle: if price crosses TP or SL the
virtual trade is closed and the outcome recorded.

Public API
----------
    record_shadow_signal(engine, symbol, direction, entry, stop, tp, reason)
                                      → None
    evaluate_shadows(prices)          → list[dict]  (closed shadow trades)
    get_shadow_stats(engine)          → dict
    get_shadow_summary()              → dict  (all engines)
    get_open_shadows()                → list[dict]

Never raises.
"""
from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import logger

# ── Persistence ───────────────────────────────────────────────────────────────

_SAVE_PATHS = [
    Path("/opt/btcbot/shadow_engine.json"),
    Path("shadow_engine.json"),
]

_lock = threading.Lock()

# State structure:
# {
#   "open":   [{id, engine, symbol, direction, entry, stop, tp, opened_at, reason}],
#   "closed": {engine: [{...closed trade fields...}]},   # capped at 200 per engine
# }
_state: dict = {"open": [], "closed": {}}

_shadow_id = 0


def _save_path() -> Path:
    for p in _SAVE_PATHS:
        if p.parent.exists():
            return p
    return _SAVE_PATHS[-1]


def _load() -> None:
    p = _save_path()
    if not p.exists():
        return
    try:
        data = json.loads(p.read_text())
        with _lock:
            _state["open"]   = data.get("open",   [])
            _state["closed"] = data.get("closed", {})
    except Exception:
        pass


def _save() -> None:
    p = _save_path()
    try:
        with _lock:
            snap = json.dumps(_state, indent=2)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(snap)
        tmp.replace(p)
    except Exception:
        pass


_load()


# ── Core API ──────────────────────────────────────────────────────────────────

def record_shadow_signal(
    engine:    str,
    symbol:    str,
    direction: str,
    entry:     float,
    stop:      float,
    tp:        float,
    reason:    str = "",
) -> None:
    """
    Record a new shadow signal.  Called when a setup is REJECTED from live
    trading but is interesting to track, OR for dedicated shadow engines.
    """
    global _shadow_id
    try:
        ts = datetime.now(timezone.utc).isoformat()
        with _lock:
            _shadow_id += 1
            trade = {
                "id":        _shadow_id,
                "engine":    engine,
                "symbol":    symbol,
                "direction": direction,   # LONG | SHORT
                "entry":     entry,
                "stop":      stop,
                "tp":        tp,
                "opened_at": ts,
                "reason":    reason,
            }
            _state["open"].append(trade)
        logger.log_info(
            f"SHADOW | NEW | {engine} {symbol} {direction} "
            f"entry={entry:.4f} stop={stop:.4f} tp={tp:.4f}"
        )
        _save()
    except Exception as exc:
        logger.log_warning(f"shadow_engine.record_shadow_signal error: {exc}")


def evaluate_shadows(prices: dict[str, float]) -> list[dict]:
    """
    Check all open shadow trades against current prices.
    prices: {symbol: current_price}

    Returns list of newly-closed shadow trade dicts.
    NEVER places orders.  NEVER modifies live state.
    """
    closed_now: list[dict] = []
    try:
        ts = datetime.now(timezone.utc).isoformat()
        still_open: list[dict] = []

        with _lock:
            open_copy = list(_state["open"])

        for t in open_copy:
            price = prices.get(t["symbol"])
            if price is None:
                still_open.append(t)
                continue

            hit_tp = hit_sl = False
            if t["direction"] == "LONG":
                if price >= t["tp"]:
                    hit_tp = True
                elif price <= t["stop"]:
                    hit_sl = True
            else:  # SHORT
                if price <= t["tp"]:
                    hit_tp = True
                elif price >= t["stop"]:
                    hit_sl = True

            if hit_tp or hit_sl:
                outcome = "TP" if hit_tp else "SL"
                exit_price = t["tp"] if hit_tp else t["stop"]

                if t["direction"] == "LONG":
                    pnl_pct = (exit_price - t["entry"]) / t["entry"] * 100
                else:
                    pnl_pct = (t["entry"] - exit_price) / t["entry"] * 100

                # Persist to shadow_trades DB table for comparative analytics
                try:
                    import shadow_analytics as _sa
                    _sa.record_shadow_trade(
                        engine=t["engine"],
                        symbol=t["symbol"],
                        direction=t["direction"],
                        entry_price=t["entry"],
                        exit_price=exit_price,
                        stop_price=t["stop"],
                        tp_price=t["tp"],
                        outcome=outcome,
                        pnl_pct=round(pnl_pct, 3),
                        opened_at=t.get("opened_at", ""),
                        closed_at=ts,
                        reason=t.get("reason", ""),
                    )
                except Exception as _sa_exc:
                    logger.log_warning(f"shadow_analytics.record failed (non-critical): {_sa_exc}")

                closed_trade = {
                    **t,
                    "closed_at":  ts,
                    "exit_price": exit_price,
                    "outcome":    outcome,
                    "pnl_pct":    round(pnl_pct, 3),
                }
                closed_now.append(closed_trade)

                with _lock:
                    eng = t["engine"]
                    if eng not in _state["closed"]:
                        _state["closed"][eng] = []
                    _state["closed"][eng].append(closed_trade)
                    _state["closed"][eng] = _state["closed"][eng][-200:]  # cap

                logger.log_info(
                    f"SHADOW | CLOSE | {t['engine']} {t['symbol']} "
                    f"{t['direction']} → {outcome} pnl={pnl_pct:+.2f}%"
                )
            else:
                still_open.append(t)

        with _lock:
            _state["open"] = still_open

        if closed_now:
            _save()

    except Exception as exc:
        logger.log_warning(f"shadow_engine.evaluate_shadows error: {exc}")

    return closed_now


def get_shadow_stats(engine: str) -> dict:
    """Return analytics for one shadow engine."""
    try:
        with _lock:
            trades = list(_state["closed"].get(engine, []))
            open_count = sum(1 for t in _state["open"] if t["engine"] == engine)

        n = len(trades)
        if n == 0:
            return {
                "engine": engine, "trades": 0, "open": open_count,
                "win_rate": 0.0, "total_pnl_pct": 0.0,
                "avg_pnl_pct": 0.0, "expectancy": 0.0,
            }

        wins   = [t for t in trades if t["outcome"] == "TP"]
        losses = [t for t in trades if t["outcome"] == "SL"]
        wr = len(wins) / n

        avg_win  = sum(t["pnl_pct"] for t in wins)   / len(wins)  if wins   else 0.0
        avg_loss = sum(t["pnl_pct"] for t in losses) / len(losses) if losses else 0.0
        total_pnl = sum(t["pnl_pct"] for t in trades)
        expectancy = wr * avg_win + (1 - wr) * avg_loss

        return {
            "engine":       engine,
            "trades":       n,
            "open":         open_count,
            "wins":         len(wins),
            "losses":       len(losses),
            "win_rate":     round(wr, 3),
            "avg_win_pct":  round(avg_win,  3),
            "avg_loss_pct": round(avg_loss, 3),
            "total_pnl_pct": round(total_pnl, 3),
            "avg_pnl_pct":  round(total_pnl / n, 3),
            "expectancy":   round(expectancy, 3),
        }
    except Exception as exc:
        logger.log_warning(f"shadow_engine.get_shadow_stats error: {exc}")
        return {"engine": engine, "trades": 0, "open": 0, "error": str(exc)}


def get_shadow_summary() -> dict:
    """Return shadow stats for all engines that have data."""
    try:
        with _lock:
            engines_with_data = set(_state["closed"].keys())
            for t in _state["open"]:
                engines_with_data.add(t["engine"])

        stats = {eng: get_shadow_stats(eng) for eng in sorted(engines_with_data)}

        # Aggregate open count
        with _lock:
            total_open = len(_state["open"])

        return {
            "engines":    stats,
            "total_open": total_open,
            "total_closed": sum(s.get("trades", 0) for s in stats.values()),
        }
    except Exception as exc:
        logger.log_warning(f"shadow_engine.get_shadow_summary error: {exc}")
        return {"engines": {}, "total_open": 0, "total_closed": 0}


def get_open_shadows() -> list[dict]:
    """Return list of currently open shadow trades."""
    try:
        with _lock:
            return list(_state["open"])
    except Exception:
        return []
