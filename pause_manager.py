"""
pause_manager.py — Auto-pause new entries on drawdown limits.

Tracks realized PnL from closed trades across daily, weekly, and
consecutive-loss windows. Writes a PAUSED file when a limit is breached.

Safe defaults:
  - is_paused() returns True (block entries) on any internal error
  - AUTO_PAUSE_ON_DRAWDOWN=false disables all tracking (always False)
  - Manually delete PAUSED to unpause immediately; bot detects it next cycle
  - Position management (check_position) always runs — only new entries blocked

State is persisted to PAUSE_STATE.json so a bot restart does not reset
running daily/weekly PnL counters.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import config
import logger

# ── File paths ────────────────────────────────────────────────────────────────
PAUSED_FILE = Path("PAUSED")
_STATE_FILE = Path("PAUSE_STATE.json")

# ── Module-level state (loaded once from disk on first use) ───────────────────
_state: dict = {}


# ── Key helpers ───────────────────────────────────────────────────────────────

def _day_key(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")


def _week_key(dt: datetime) -> str:
    iso = dt.isocalendar()
    return f"{iso[0]}-W{iso[1]:02d}"


# ── State persistence ─────────────────────────────────────────────────────────

def _default_state() -> dict:
    now = datetime.now(timezone.utc)
    return {
        "day_key": _day_key(now),
        "week_key": _week_key(now),
        "daily_pnl_usdt": 0.0,
        "weekly_pnl_usdt": 0.0,
        "consecutive_losses": 0,
        "balance": 10_000.0,
    }


def _load_state() -> dict:
    default = _default_state()
    if not _STATE_FILE.exists():
        return default
    try:
        data = json.loads(_STATE_FILE.read_text())
        return {**default, **data}
    except Exception:
        return default


def _save_state() -> None:
    try:
        _STATE_FILE.write_text(json.dumps(_state, indent=2))
    except Exception as exc:
        logger.log_warning(f"pause_manager: could not save state: {exc}")


def _init() -> None:
    global _state
    if not _state:
        _state = _load_state()


# ── Window resets ─────────────────────────────────────────────────────────────

def _reset_daily_if_new_day(now: datetime) -> bool:
    key = _day_key(now)
    if _state.get("day_key") != key:
        _state["day_key"] = key
        _state["daily_pnl_usdt"] = 0.0
        _save_state()
        return True
    return False


def _reset_weekly_if_new_week(now: datetime) -> bool:
    key = _week_key(now)
    if _state.get("week_key") != key:
        _state["week_key"] = key
        _state["weekly_pnl_usdt"] = 0.0
        _save_state()
        return True
    return False


# ── PAUSED file I/O ───────────────────────────────────────────────────────────

def _write_paused_file(reason: str, pause_until: Optional[datetime] = None) -> None:
    now = datetime.now(timezone.utc)
    balance = _state.get("balance", 10_000.0)
    daily_pct = _state["daily_pnl_usdt"] / balance if balance > 0 else 0.0
    weekly_pct = _state["weekly_pnl_usdt"] / balance if balance > 0 else 0.0
    data = {
        "paused": True,
        "reason": reason,
        "paused_at_utc": now.isoformat(),
        "pause_until_utc": pause_until.isoformat() if pause_until else "",
        "daily_pnl_pct": round(daily_pct, 6),
        "weekly_pnl_pct": round(weekly_pct, 6),
        "consecutive_losses": _state["consecutive_losses"],
    }
    try:
        PAUSED_FILE.write_text(json.dumps(data, indent=2))
    except Exception as exc:
        logger.log_warning(f"pause_manager: could not write PAUSED file: {exc}")


def _read_paused_file() -> Optional[dict]:
    try:
        return json.loads(PAUSED_FILE.read_text())
    except Exception:
        return None


def _do_auto_unpause(resume_type: str) -> None:
    """Delete PAUSED file and send resume alert."""
    try:
        PAUSED_FILE.unlink(missing_ok=True)
    except Exception:
        pass
    logger.log_info(f"AUTO-RESUMED | type={resume_type}")
    try:
        import alerts
        alerts.alert_resumed(resume_type)
    except Exception:
        pass


# ── Public API ────────────────────────────────────────────────────────────────

def update_balance(balance: float) -> None:
    """Call once per cycle from main.py with the current USDT balance."""
    if not config.AUTO_PAUSE_ON_DRAWDOWN:
        return
    try:
        _init()
        _state["balance"] = balance
        _save_state()
    except Exception as exc:
        logger.log_warning(f"pause_manager.update_balance error (non-critical): {exc}")


def record_close(realized_pnl_usdt: float, close_type: str) -> None:
    """
    Called from executor._clear_position after every trade close.
    Updates running PnL totals and checks whether any pause threshold
    has been breached. Writes PAUSED file and sends alert if so.
    """
    if not config.AUTO_PAUSE_ON_DRAWDOWN:
        return
    try:
        _init()
        now = datetime.now(timezone.utc)
        _reset_daily_if_new_day(now)
        _reset_weekly_if_new_week(now)

        _state["daily_pnl_usdt"] += realized_pnl_usdt
        _state["weekly_pnl_usdt"] += realized_pnl_usdt

        if realized_pnl_usdt < 0:
            _state["consecutive_losses"] += 1
        else:
            _state["consecutive_losses"] = 0

        _save_state()

        # Skip limit checks if already paused
        if PAUSED_FILE.exists():
            return

        balance = _state.get("balance", 10_000.0)
        daily_pct = _state["daily_pnl_usdt"] / balance if balance > 0 else 0.0
        weekly_pct = _state["weekly_pnl_usdt"] / balance if balance > 0 else 0.0
        consec = _state["consecutive_losses"]

        reason: Optional[str] = None
        pause_until: Optional[datetime] = None

        if daily_pct <= -config.MAX_DAILY_LOSS:
            reason = "daily_loss_limit"
            if config.AUTO_UNPAUSE_DAILY:
                tomorrow = (now + timedelta(days=1)).replace(
                    hour=0, minute=0, second=0, microsecond=0
                )
                pause_until = tomorrow
        elif weekly_pct <= -config.MAX_WEEKLY_LOSS:
            reason = "weekly_loss_limit"
            if config.AUTO_UNPAUSE_WEEKLY:
                # Next Monday 00:00 UTC
                days_until_monday = (7 - now.weekday()) % 7 or 7
                next_monday = (now + timedelta(days=days_until_monday)).replace(
                    hour=0, minute=0, second=0, microsecond=0
                )
                pause_until = next_monday
        elif consec >= config.MAX_CONSECUTIVE_LOSSES:
            reason = "consecutive_losses"
            if config.AUTO_UNPAUSE_CONSECUTIVE_LOSSES:
                tomorrow = (now + timedelta(days=1)).replace(
                    hour=0, minute=0, second=0, microsecond=0
                )
                pause_until = tomorrow

        if reason:
            _write_paused_file(reason, pause_until)
            logger.log_warning(
                f"PAUSE TRIGGERED | reason={reason} | "
                f"daily={daily_pct*100:.2f}% | weekly={weekly_pct*100:.2f}% | "
                f"consecutive={consec}"
            )
            try:
                import alerts
                alerts.alert_paused(reason, daily_pct, weekly_pct, consec)
            except Exception:
                pass

    except Exception as exc:
        logger.log_warning(f"pause_manager.record_close error (non-critical): {exc}")


def is_paused() -> bool:
    """
    Returns True if new entries should be blocked.
    Handles auto-unpause logic inline (time-based and day-boundary).
    Safe default: returns True (block entries) on any internal error.
    """
    if not config.AUTO_PAUSE_ON_DRAWDOWN:
        return False
    try:
        _init()
        if not PAUSED_FILE.exists():
            return False

        data = _read_paused_file()
        if data is None:
            return True  # corrupt file → stay paused

        reason = data.get("reason", "")
        pause_until_str = data.get("pause_until_utc", "")
        now = datetime.now(timezone.utc)

        # ── Time-based auto-unpause (daily limit with AUTO_UNPAUSE_DAILY) ────
        if pause_until_str:
            try:
                pause_until = datetime.fromisoformat(pause_until_str)
                if now >= pause_until:
                    _do_auto_unpause("auto_daily")
                    _state["day_key"] = _day_key(now)
                    _state["daily_pnl_usdt"] = 0.0
                    _save_state()
                    return False
            except Exception:
                pass

        # ── Day-boundary fallback for daily limit ─────────────────────────────
        if reason == "daily_loss_limit" and config.AUTO_UNPAUSE_DAILY:
            paused_at_str = data.get("paused_at_utc", "")
            try:
                paused_at = datetime.fromisoformat(paused_at_str)
                if _day_key(now) != _day_key(paused_at):
                    _do_auto_unpause("auto_daily")
                    _state["day_key"] = _day_key(now)
                    _state["daily_pnl_usdt"] = 0.0
                    _save_state()
                    return False
            except Exception:
                pass

        # ── Day-boundary auto-unpause for consecutive losses ──────────────────
        if reason == "consecutive_losses" and config.AUTO_UNPAUSE_CONSECUTIVE_LOSSES:
            paused_at_str = data.get("paused_at_utc", "")
            try:
                paused_at = datetime.fromisoformat(paused_at_str)
                if _day_key(now) != _day_key(paused_at):
                    _do_auto_unpause("auto_consecutive")
                    _state["consecutive_losses"] = 0
                    _save_state()
                    return False
            except Exception:
                pass

        return True

    except Exception as exc:
        logger.log_warning(f"pause_manager.is_paused error — defaulting to paused: {exc}")
        return True  # safe default


def pause_reason() -> str:
    """Return the reason string from PAUSED file, or '' if not paused."""
    try:
        data = _read_paused_file()
        return data.get("reason", "") if data else ""
    except Exception:
        return ""


def manual_unpause() -> bool:
    """
    Delete the PAUSED file and send a resume alert.
    Called by unpause_bot.sh. Returns True if bot was paused.
    """
    if not PAUSED_FILE.exists():
        print("Bot is not paused — nothing to do.")
        return False
    try:
        PAUSED_FILE.unlink()
        logger.log_info("MANUAL UNPAUSE | PAUSED file deleted")
        try:
            import alerts
            alerts.alert_resumed("manual")
        except Exception:
            pass
        print("Bot unpaused successfully.")
        return True
    except Exception as exc:
        print(f"Error deleting PAUSED file: {exc}")
        return False


def get_status() -> dict:
    """Return a status dict for pause_status.py."""
    _init()
    now = datetime.now(timezone.utc)
    _reset_daily_if_new_day(now)
    _reset_weekly_if_new_week(now)

    balance = _state.get("balance", 10_000.0)
    daily_pct = _state["daily_pnl_usdt"] / balance if balance > 0 else 0.0
    weekly_pct = _state["weekly_pnl_usdt"] / balance if balance > 0 else 0.0
    paused = PAUSED_FILE.exists()
    pause_data = _read_paused_file() if paused else {}

    return {
        "paused": paused,
        "reason": pause_data.get("reason", "") if paused else "",
        "paused_at_utc": pause_data.get("paused_at_utc", "") if paused else "",
        "pause_until_utc": pause_data.get("pause_until_utc", "") if paused else "",
        "daily_pnl_usdt": round(_state["daily_pnl_usdt"], 4),
        "daily_pnl_pct": round(daily_pct * 100, 3),
        "weekly_pnl_usdt": round(_state["weekly_pnl_usdt"], 4),
        "weekly_pnl_pct": round(weekly_pct * 100, 3),
        "consecutive_losses": _state["consecutive_losses"],
        "balance": round(balance, 2),
        "max_daily_loss_pct": config.MAX_DAILY_LOSS * 100,
        "max_weekly_loss_pct": config.MAX_WEEKLY_LOSS * 100,
        "max_consecutive_losses": config.MAX_CONSECUTIVE_LOSSES,
    }
