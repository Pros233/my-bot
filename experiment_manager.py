"""
experiment_manager.py — Safe experiment framework.

Experiments are named configurations that shadow-trade a variant engine
or parameter set in isolation.  They auto-expire after a deadline and
never touch live trades.

Each experiment:
  - Has a unique name and a shadow engine tag (e.g. "EXP_PULLBACK_TIGHT")
  - Accumulates shadow P&L independently
  - Has a max_trades limit and an expiry date
  - Reports results via get_experiment_results()

Public API
----------
    create_experiment(name, engine, description, max_trades, expiry_days) → bool
    record_signal(experiment_name, symbol, direction, entry, stop, tp)    → bool
    get_experiment_results(name)       → dict
    get_all_experiments()              → dict
    expire_old_experiments()           → list[str]  (expired names)

Never raises.
"""
from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import logger
import shadow_engine as se

# ── Persistence ───────────────────────────────────────────────────────────────

_SAVE_PATHS = [
    Path("/opt/btcbot/experiment_manager.json"),
    Path("experiment_manager.json"),
]

_lock = threading.Lock()

# State: {name: {engine, description, created_at, expires_at, max_trades, active}}
_state: dict = {}


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
            _state.update(data)
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


# ── Public API ────────────────────────────────────────────────────────────────

def create_experiment(
    name:        str,
    engine:      str,
    description: str  = "",
    max_trades:  int  = 50,
    expiry_days: int  = 30,
) -> bool:
    """
    Create a new experiment.  Returns False if name already exists.
    The engine name is used as the shadow_engine tag, so results can be
    fetched via shadow_engine.get_shadow_stats(engine).
    """
    try:
        with _lock:
            if name in _state:
                return False

        now = datetime.now(timezone.utc)
        exp = {
            "name":        name,
            "engine":      engine,
            "description": description,
            "created_at":  now.isoformat(),
            "expires_at":  (now + timedelta(days=expiry_days)).isoformat(),
            "max_trades":  max_trades,
            "active":      True,
        }
        with _lock:
            _state[name] = exp
        _save()
        logger.log_info(
            f"EXPERIMENT | CREATE | {name} engine={engine} "
            f"max_trades={max_trades} expiry_days={expiry_days}"
        )
        return True
    except Exception as exc:
        logger.log_warning(f"experiment_manager.create_experiment error: {exc}")
        return False


def record_signal(
    experiment_name: str,
    symbol:          str,
    direction:       str,
    entry:           float,
    stop:            float,
    tp:              float,
) -> bool:
    """
    Record a shadow signal for an experiment.
    Returns False if experiment doesn't exist, is expired, or at max_trades.
    """
    try:
        with _lock:
            exp = _state.get(experiment_name)
        if exp is None or not exp.get("active"):
            return False

        # Check expiry
        expires_at = datetime.fromisoformat(exp["expires_at"])
        if datetime.now(timezone.utc) > expires_at:
            with _lock:
                _state[experiment_name]["active"] = False
            _save()
            return False

        # Check trade count
        stats = se.get_shadow_stats(exp["engine"])
        if stats.get("trades", 0) >= exp.get("max_trades", 50):
            with _lock:
                _state[experiment_name]["active"] = False
            _save()
            logger.log_info(f"EXPERIMENT | MAX_TRADES | {experiment_name}")
            return False

        se.record_shadow_signal(
            engine=exp["engine"],
            symbol=symbol,
            direction=direction,
            entry=entry,
            stop=stop,
            tp=tp,
            reason=f"exp:{experiment_name}",
        )
        return True
    except Exception as exc:
        logger.log_warning(f"experiment_manager.record_signal error: {exc}")
        return False


def get_experiment_results(name: str) -> dict:
    """Return results for one experiment (merged with shadow stats)."""
    try:
        with _lock:
            exp = dict(_state.get(name, {}))
        if not exp:
            return {"error": f"experiment '{name}' not found"}

        stats = se.get_shadow_stats(exp.get("engine", name))
        now   = datetime.now(timezone.utc)
        expires_at = datetime.fromisoformat(exp["expires_at"])
        days_left  = (expires_at - now).days

        return {
            **exp,
            "shadow_stats": stats,
            "days_remaining": max(0, days_left),
            "is_expired":    now > expires_at,
        }
    except Exception as exc:
        logger.log_warning(f"experiment_manager.get_experiment_results error: {exc}")
        return {"error": str(exc)}


def get_all_experiments() -> dict:
    """Return all experiments with their stats."""
    try:
        with _lock:
            names = list(_state.keys())
        return {name: get_experiment_results(name) for name in names}
    except Exception:
        return {}


def expire_old_experiments() -> list[str]:
    """Auto-expire any experiments past their deadline. Returns expired names."""
    expired = []
    try:
        now = datetime.now(timezone.utc)
        with _lock:
            names = list(_state.keys())
        for name in names:
            with _lock:
                exp = _state.get(name, {})
            if not exp.get("active"):
                continue
            try:
                if now > datetime.fromisoformat(exp["expires_at"]):
                    with _lock:
                        _state[name]["active"] = False
                    expired.append(name)
                    logger.log_info(f"EXPERIMENT | EXPIRED | {name}")
            except Exception:
                pass
        if expired:
            _save()
    except Exception as exc:
        logger.log_warning(f"experiment_manager.expire_old_experiments error: {exc}")
    return expired
