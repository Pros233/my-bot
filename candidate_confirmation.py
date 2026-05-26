"""
candidate_confirmation.py — 1H confirmation layer for 15-minute candidates.

Before any 15m candidate enters execution, this module verifies it against
the full set of 1H risk controls that protect the live account.

Gates (evaluated in order; first BLOCKING gate short-circuits):
  1. pause_gate            — pause_manager.is_paused() must be False
  2. max_open_trades_gate  — open_count must be < MAX_OPEN_TRADES
  3. confidence_gate       — confidence must not be DEFENSIVE (< 40)
  4. regime_gate           — 1H regime must not strongly oppose trade side
  5. volatility_gate       — 1H ATR% must not be chaotic (> threshold)
  6. avoidance_gate        — market_avoidance severity must be NONE / CAUTION
  7. spread_gate           — live spread must pass MAX_SPREAD_BPS
  8. btc_alignment_gate    — BTC alignment must pass for non-BTC symbols (soft)
  9. timeframe_context     — 1H context must support the 15m setup type (soft)
  10. exchange_filter_gate — estimated notional must meet exchange minimums

BLOCKING gates: confirmed=False immediately.
SOFT gates: penalise confirmation_score but do not block.
If confirmation_score < SCAN_15M_MIN_CONFIRMATION_SCORE → confirmed=False.

Public API
----------
    confirm_15m_candidate(
        candidate, market_states, balance, open_count,
        confidence_score, client, now_utc,
    ) -> ConfirmationResult

Never raises.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import config
import logger
import market_avoidance
import pause_manager
import trade_filters
import exchange_filters
from candidate_scanner_15m import Candidate15m

# ── Result type ────────────────────────────────────────────────────────────────

@dataclass
class ConfirmationResult:
    confirmed:          bool
    reason:             str
    confirmation_score: float        # 0-100
    blocking_gate:      str          # "" if no blocking gate fired
    gate_details:       dict = field(default_factory=dict)


# ── Constants ─────────────────────────────────────────────────────────────────

_CHAOTIC_ATR_PCT = 5.0          # 1H ATR% above this → chaotic, block entry
_CAUTION_ATR_PCT = 3.0          # 1H ATR% above this → elevated, soft penalty
_SOFT_PENALTY    = 15.0         # score deducted per soft gate failure

# Market avoidance severities that block the confirmation
_BLOCK_SEVERITIES = {
    market_avoidance.SEVERITY_WARNING,
    market_avoidance.SEVERITY_CRITICAL,
}

# 15m setup families: which 1H regime states support them
# TREND setups need at least a mild trend; RANGE setups need ranging market.
_TREND_SETUPS = {
    "15M_MICRO_BREAKOUT",
    "15M_MOMENTUM_CONTINUATION",
    "15M_VOL_EXPANSION",
}
_RANGE_SETUPS = {
    "15M_PULLBACK_RECLAIM",
    "15M_RANGE_BOUNCE",
}


# ── Main API ───────────────────────────────────────────────────────────────────

def confirm_15m_candidate(
    candidate: Candidate15m,
    market_states: dict,        # symbol -> ScanResult from 1H scan
    balance: float,
    open_count: int,
    confidence_score: float,
    client,
    now_utc: Optional[datetime] = None,
) -> ConfirmationResult:
    """
    Run all 1H risk gates against a 15m candidate.
    Returns ConfirmationResult. Never raises.
    """
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)

    score       = 100.0
    gate_detail = {}

    try:
        sym    = candidate.symbol
        side   = candidate.side        # "BUY" (only longs supported for now)
        setup  = candidate.setup_name

        min_conf_score = float(getattr(config, "SCAN_15M_MIN_CONFIRMATION_SCORE", 50.0))

        # ── Gate 1: pause_manager ─────────────────────────────────────────────
        if pause_manager.is_paused():
            reason = pause_manager.pause_reason() or "auto-pause active"
            gate_detail["pause_gate"] = reason
            return _blocked("pause_gate", f"bot paused: {reason}", score, gate_detail)

        gate_detail["pause_gate"] = "ok"

        # ── Gate 2: MAX_OPEN_TRADES ───────────────────────────────────────────
        max_open = int(getattr(config, "MAX_OPEN_TRADES", 3))
        if open_count >= max_open:
            gate_detail["max_open_trades_gate"] = f"{open_count}/{max_open}"
            return _blocked(
                "max_open_trades_gate",
                f"position slots full ({open_count}/{max_open})",
                score, gate_detail,
            )

        gate_detail["max_open_trades_gate"] = f"{open_count}/{max_open} ok"

        # ── Gate 3: confidence — block if DEFENSIVE ───────────────────────────
        try:
            import confidence_score as _cs
            conf_state = _cs.get_confidence_state(confidence_score)
        except Exception:
            conf_state = "NORMAL"

        if conf_state == "DEFENSIVE":
            gate_detail["confidence_gate"] = f"{confidence_score:.0f} DEFENSIVE"
            return _blocked(
                "confidence_gate",
                f"confidence DEFENSIVE ({confidence_score:.0f}) — no new entries",
                score, gate_detail,
            )

        gate_detail["confidence_gate"] = f"{confidence_score:.0f} {conf_state}"

        # ── Gate 4: 1H regime must not strongly oppose trade side ─────────────
        sym_state = market_states.get(sym)
        if sym_state is not None:
            trend = getattr(sym_state, "trend", "RANGING")
            adx   = float(getattr(sym_state, "adx", 0.0))

            opposing = False
            if side == "BUY" and trend in ("BEARISH", "STRONG_DOWN"):
                opposing = True
            elif side == "SELL" and trend in ("BULLISH", "STRONG_UP"):
                opposing = True

            # Only hard-block if ADX > threshold (strong trend), else soft penalise
            adx_trend = float(getattr(config, "ADX_TREND_THRESHOLD", 25.0))
            if opposing and adx > adx_trend:
                gate_detail["regime_gate"] = f"trend={trend} adx={adx:.1f} opposes {side}"
                return _blocked(
                    "regime_gate",
                    f"1H regime {trend} (ADX={adx:.1f}) strongly opposes {side}",
                    score, gate_detail,
                )
            elif opposing:
                score -= _SOFT_PENALTY
                gate_detail["regime_gate"] = f"warn: trend={trend} opposes {side} (ADX={adx:.1f}<{adx_trend}, soft)"
            else:
                gate_detail["regime_gate"] = f"ok trend={trend}"
        else:
            gate_detail["regime_gate"] = f"no 1H state for {sym} (skipped)"

        # ── Gate 5: 1H volatility must not be chaotic ─────────────────────────
        atr_pct = float(getattr(sym_state, "atr_pct", 0.0)) if sym_state else 0.0
        if atr_pct > _CHAOTIC_ATR_PCT:
            gate_detail["volatility_gate"] = f"ATR%={atr_pct:.2f} > {_CHAOTIC_ATR_PCT} CHAOTIC"
            return _blocked(
                "volatility_gate",
                f"1H ATR%={atr_pct:.2f} exceeds chaotic threshold {_CHAOTIC_ATR_PCT}",
                score, gate_detail,
            )
        elif atr_pct > _CAUTION_ATR_PCT:
            score -= _SOFT_PENALTY
            gate_detail["volatility_gate"] = f"elevated ATR%={atr_pct:.2f} (soft)"
        else:
            gate_detail["volatility_gate"] = f"ok ATR%={atr_pct:.2f}"

        # ── Gate 6: market avoidance (WARNING/CRITICAL → block) ───────────────
        try:
            _av_result = market_avoidance.check_avoidance(None, sym, now_utc)
            if _av_result.severity in _BLOCK_SEVERITIES:
                gate_detail["avoidance_gate"] = (
                    f"{_av_result.severity}: {'; '.join(_av_result.reasons)}"
                )
                return _blocked(
                    "avoidance_gate",
                    f"market avoidance {_av_result.severity}: {'; '.join(_av_result.reasons)}",
                    score, gate_detail,
                )
            gate_detail["avoidance_gate"] = f"ok severity={_av_result.severity}"
        except Exception as _av_exc:
            gate_detail["avoidance_gate"] = f"check failed (skipped): {_av_exc}"

        # ── Gate 7: spread (hard fail if SPREAD_FILTER enabled) ───────────────
        if getattr(config, "SPREAD_FILTER", True):
            try:
                _spread_r = trade_filters.filter_spread(sym, client)
                if _spread_r.hard_fail:
                    gate_detail["spread_gate"] = f"hard fail: {_spread_r.reason}"
                    return _blocked(
                        "spread_gate",
                        f"spread too wide: {_spread_r.reason}",
                        score, gate_detail,
                    )
                elif not _spread_r.passed:
                    score -= _SOFT_PENALTY
                    gate_detail["spread_gate"] = f"soft fail: {_spread_r.reason}"
                else:
                    gate_detail["spread_gate"] = "ok"
            except Exception as _sp_exc:
                gate_detail["spread_gate"] = f"check error (skipped): {_sp_exc}"
        else:
            gate_detail["spread_gate"] = "disabled"

        # ── Gate 8: BTC alignment (soft for non-BTC altcoins) ─────────────────
        if getattr(config, "BTC_ALIGNMENT_FILTER", True) and sym != "BTCUSDT":
            try:
                _btc_r = trade_filters.filter_btc_alignment(sym, market_states)
                if not _btc_r.passed:
                    score -= _SOFT_PENALTY
                    gate_detail["btc_alignment_gate"] = f"soft fail: {_btc_r.reason}"
                else:
                    gate_detail["btc_alignment_gate"] = "ok"
            except Exception as _ba_exc:
                gate_detail["btc_alignment_gate"] = f"check error (skipped): {_ba_exc}"
        else:
            gate_detail["btc_alignment_gate"] = "n/a (BTC or filter disabled)"

        # ── Gate 9: timeframe context — 1H trend/range supports setup ─────────
        if sym_state is not None:
            trend    = getattr(sym_state, "trend", "RANGING")
            adx      = float(getattr(sym_state, "adx", 0.0))
            adx_rng  = float(getattr(config, "ADX_RANGING_THRESHOLD", 20.0))
            adx_trnd = float(getattr(config, "ADX_TREND_THRESHOLD",   25.0))
            is_ranging = adx < adx_rng
            is_trending = adx >= adx_trnd

            if setup in _TREND_SETUPS and is_ranging:
                # Trend setup in a ranging market — soft penalise only
                score -= _SOFT_PENALTY
                gate_detail["timeframe_gate"] = (
                    f"soft: trend setup '{setup}' in RANGING market (ADX={adx:.1f})"
                )
            elif setup in _RANGE_SETUPS and is_trending:
                # Range setup in a strong trend — soft penalise only
                score -= _SOFT_PENALTY
                gate_detail["timeframe_gate"] = (
                    f"soft: range setup '{setup}' in TRENDING market (ADX={adx:.1f})"
                )
            else:
                gate_detail["timeframe_gate"] = (
                    f"ok setup={setup} trend={trend} ADX={adx:.1f}"
                )
        else:
            gate_detail["timeframe_gate"] = "no 1H state (skipped)"

        # ── Gate 10: exchange filter — estimated notional ─────────────────────
        try:
            risk_pct = float(getattr(config, "RISK_PER_TRADE", 0.01))
            entry    = float(candidate.entry_reference) or 1.0
            stop     = float(candidate.stop_reference)  or entry * 0.99
            stop_dist = abs(entry - stop)
            if stop_dist < 1e-8:
                stop_dist = entry * 0.01
            risk_usdt = balance * risk_pct
            qty_est   = risk_usdt / stop_dist

            _ex_r = exchange_filters.validate_order(client, sym, qty_est, entry)
            if not _ex_r.valid:
                gate_detail["exchange_filter_gate"] = f"invalid: {_ex_r.reason}"
                return _blocked(
                    "exchange_filter_gate",
                    f"exchange filter: {_ex_r.reason}",
                    score, gate_detail,
                )
            gate_detail["exchange_filter_gate"] = (
                f"ok qty_est={_ex_r.adjusted_qty} min_notional={_ex_r.min_notional}"
            )
        except Exception as _ef_exc:
            gate_detail["exchange_filter_gate"] = f"check error (skipped): {_ef_exc}"

        # ── Final score check ─────────────────────────────────────────────────
        score = max(0.0, score)
        if score < min_conf_score:
            return ConfirmationResult(
                confirmed=False,
                reason=(
                    f"confirmation score {score:.0f} < threshold {min_conf_score:.0f} "
                    f"(soft gate failures: see gate_details)"
                ),
                confirmation_score=score,
                blocking_gate="score_threshold",
                gate_details=gate_detail,
            )

        logger.log_info(
            f"CONFIRM_15M | {sym} | setup={setup} | score={score:.0f} | "
            + " | ".join(f"{k}={v}" for k, v in gate_detail.items())
        )

        return ConfirmationResult(
            confirmed=True,
            reason=f"all gates passed (score={score:.0f})",
            confirmation_score=score,
            blocking_gate="",
            gate_details=gate_detail,
        )

    except Exception as exc:
        logger.log_warning(f"confirm_15m_candidate error (non-critical): {exc}")
        return ConfirmationResult(
            confirmed=False,
            reason=f"confirmation error: {exc}",
            confirmation_score=0.0,
            blocking_gate="internal_error",
            gate_details={},
        )


# ── Internal helpers ───────────────────────────────────────────────────────────

def _blocked(gate: str, reason: str, score: float, details: dict) -> ConfirmationResult:
    logger.log_info(
        f"CONFIRM_15M_BLOCKED | gate={gate} | reason={reason} | score={score:.0f}"
    )
    return ConfirmationResult(
        confirmed=False,
        reason=reason,
        confirmation_score=score,
        blocking_gate=gate,
        gate_details=details,
    )
