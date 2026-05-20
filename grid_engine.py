"""
grid_engine.py — Virtual grid trading tracker (shadow-mode by default).

Tracks virtual grid orders for selected symbols based on ATR-derived price
levels within detected ranging zones. This module never places live orders
unless ENABLE_GRID_LIVE=true in .env AND the safety gate is cleared.

Grid design:
  - N_GRID_LEVELS levels spaced by 1×ATR within ±3×ATR of current price
  - Each level has a virtual BUY order; filled when price touches the level
  - Virtual position closes at the next level above entry (1-grid TP)
  - Tracks virtual PnL, hit rate, open virtual positions

Safety
------
  ENABLE_GRID_LIVE defaults to False and controls live order placement.
  Even when enabled, grid uses very small position sizes (GRID_RISK_PER_LEVEL).
  Live mode requires explicit user activation in .env.

Public API
----------
    update_grid(symbol, current_price, atr)   → dict  (events this cycle)
    get_grid_status(symbol)                   → dict  (current grid state)
    get_all_grid_status()                     → dict  (all symbols)
    reset_grid(symbol)                        → None

Never raises.
"""
from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import logger

# ── Config ────────────────────────────────────────────────────────────────────

_STATE_FILE    = Path("/opt/btcbot/grid_state.json")
_STATE_FILE_LOCAL = Path(__file__).parent / "grid_state.json"

_N_LEVELS      = 8          # grid levels per symbol
_ATR_SPACING   = 0.5        # level spacing = 0.5 × ATR
_GRID_RANGE    = 4.0        # grid spans ±4 × ATR_SPACING×ATR from center
_TOUCH_PCT     = 0.001      # price within 0.1% of level = "touched"
_VIRTUAL_SIZE  = 0.001      # virtual position size (BTC/ETH equiv) for PnL tracking

# Symbols eligible for grid tracking
_GRID_SYMBOLS  = {"BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"}

# ── State ─────────────────────────────────────────────────────────────────────

_lock  = threading.Lock()
_state: dict[str, dict] = {}   # symbol → grid state

_EMPTY_GRID = lambda sym: {     # noqa: E731
    "symbol":          sym,
    "levels":          [],      # list of {"price": float, "status": "open"|"filled"|"closed"}
    "open_positions":  [],      # virtual fills awaiting TP
    "total_virtual_pnl": 0.0,
    "hits":            0,
    "misses":          0,
    "updated_at":      "",
    "center_price":    0.0,
    "atr":             0.0,
}


# ── Persistence ───────────────────────────────────────────────────────────────

def _state_path() -> Path:
    return _STATE_FILE if _STATE_FILE.parent.exists() else _STATE_FILE_LOCAL


def _load_state() -> None:
    global _state
    try:
        p = _state_path()
        if p.exists():
            with open(p) as f:
                loaded = json.load(f)
            with _lock:
                _state.update(loaded)
    except Exception as exc:
        logger.log_warning(f"grid_engine._load_state error: {exc}")


def _save_state() -> None:
    try:
        p = _state_path()
        tmp = p.with_suffix(".tmp")
        with _lock:
            snap = json.dumps(_state, indent=2)
        tmp.write_text(snap)
        tmp.rename(p)
    except Exception as exc:
        logger.log_warning(f"grid_engine._save_state error: {exc}")


# Load on import
_load_state()


# ── Grid builder ──────────────────────────────────────────────────────────────

def _build_levels(center: float, atr: float) -> list[dict]:
    """Build N_LEVELS price levels spaced by ATR_SPACING×ATR."""
    if atr <= 0 or center <= 0:
        return []
    spacing = _ATR_SPACING * atr
    half    = _N_LEVELS // 2
    levels  = []
    for i in range(-half, half + 1):
        price = round(center + i * spacing, 4)
        if price <= 0:
            continue
        levels.append({"price": price, "status": "open"})
    return sorted(levels, key=lambda x: x["price"])


# ── Grid update ───────────────────────────────────────────────────────────────

def update_grid(symbol: str, current_price: float, atr: float) -> dict:
    """
    Evaluate virtual grid for symbol given current price and ATR.

    Returns dict with events: {"filled": [...], "closed": [...], "rebuilt": bool}.
    Fail-open → {}.
    """
    if symbol not in _GRID_SYMBOLS:
        return {}

    try:
        with _lock:
            grid = _state.get(symbol)

        # Initialise or rebuild if price has drifted far from center
        needs_rebuild = False
        if not grid or not grid.get("levels"):
            needs_rebuild = True
        else:
            center    = grid.get("center_price", current_price)
            grid_atr  = grid.get("atr", atr)
            max_drift = _GRID_RANGE * _ATR_SPACING * (grid_atr or atr)
            if abs(current_price - center) > max_drift:
                needs_rebuild = True

        if needs_rebuild:
            grid = _EMPTY_GRID(symbol)
            grid["levels"]       = _build_levels(current_price, atr)
            grid["center_price"] = current_price
            grid["atr"]          = atr

        now_iso = datetime.now(timezone.utc).isoformat()
        grid["updated_at"] = now_iso

        events = {"filled": [], "closed": [], "rebuilt": needs_rebuild}

        # Check each open level for a touch (virtual fill)
        for level in grid["levels"]:
            if level["status"] != "open":
                continue
            lp  = level["price"]
            pct = abs(current_price - lp) / lp
            if pct <= _TOUCH_PCT and current_price <= lp:
                # Virtual buy fill (buying at level, current price at/below)
                level["status"] = "filled"
                grid["hits"] += 1
                tp_price = lp + _ATR_SPACING * atr
                grid["open_positions"].append({
                    "entry_price": lp,
                    "tp_price":    round(tp_price, 4),
                    "size":        _VIRTUAL_SIZE,
                    "filled_at":   now_iso,
                })
                events["filled"].append({"price": lp, "tp": round(tp_price, 4)})

        # Check open virtual positions for TP hit
        remaining_positions = []
        for pos in grid.get("open_positions", []):
            tp = pos["tp_price"]
            if current_price >= tp:
                pnl = (tp - pos["entry_price"]) * pos["size"]
                grid["total_virtual_pnl"] += pnl
                events["closed"].append({
                    "entry":  pos["entry_price"],
                    "tp":     tp,
                    "pnl":    round(pnl, 6),
                })
                # Re-open the entry level
                for lev in grid["levels"]:
                    if abs(lev["price"] - pos["entry_price"]) < 0.01:
                        lev["status"] = "open"
                        break
            else:
                remaining_positions.append(pos)

        grid["open_positions"] = remaining_positions

        with _lock:
            _state[symbol] = grid

        if events["filled"] or events["closed"] or needs_rebuild:
            _save_state()

        return events

    except Exception as exc:
        logger.log_warning(f"grid_engine.update_grid({symbol}) error: {exc}")
        return {}


# ── Public API ────────────────────────────────────────────────────────────────

def get_grid_status(symbol: str) -> dict:
    """Return current grid state for symbol. Fail-open → {}."""
    try:
        with _lock:
            grid = _state.get(symbol, {})
        if not grid:
            return {
                "symbol":  symbol,
                "status":  "inactive",
                "message": "No grid initialised — will build on next price update",
            }

        levels = grid.get("levels", [])
        open_levels  = [l for l in levels if l["status"] == "open"]
        filled_levels = [l for l in levels if l["status"] == "filled"]

        return {
            "symbol":            symbol,
            "status":            "active",
            "center_price":      grid.get("center_price", 0.0),
            "atr":               grid.get("atr", 0.0),
            "total_levels":      len(levels),
            "open_levels":       len(open_levels),
            "filled_levels":     len(filled_levels),
            "open_positions":    len(grid.get("open_positions", [])),
            "total_virtual_pnl": round(grid.get("total_virtual_pnl", 0.0), 6),
            "hits":              grid.get("hits", 0),
            "next_buy_level":    min((l["price"] for l in open_levels), default=0.0),
            "updated_at":        grid.get("updated_at", ""),
            "levels_snapshot":   [
                {"price": l["price"], "status": l["status"]}
                for l in sorted(levels, key=lambda x: x["price"], reverse=True)[:8]
            ],
        }
    except Exception as exc:
        logger.log_warning(f"grid_engine.get_grid_status error: {exc}")
        return {"symbol": symbol, "status": "error", "error": str(exc)}


def get_all_grid_status() -> dict[str, dict]:
    """Return grid status for all active symbols. Fail-open → {}."""
    try:
        result = {}
        for sym in _GRID_SYMBOLS:
            result[sym] = get_grid_status(sym)
        return result
    except Exception:
        return {}


def reset_grid(symbol: str) -> None:
    """Reset grid for symbol (will rebuild on next update call)."""
    try:
        with _lock:
            if symbol in _state:
                del _state[symbol]
        _save_state()
    except Exception as exc:
        logger.log_warning(f"grid_engine.reset_grid({symbol}) error: {exc}")
