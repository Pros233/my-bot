"""
trend_outcome_tracker.py — Track 1h / 4h / 24h return outcomes for trend signals
and compute continuation analytics once all horizons are filled.

Continuation metrics (stored on trend_signals):
  max_move_pct      — best positive return across all horizons
  min_move_pct      — worst (most negative) return across all horizons
  continuation_score
    1.0 = 4h > 1h > 0   (strong continuation)
    0.5 = 4h > 0 only   (partial continuation)
    0.0 = 4h ≤ 0        (failed to continue)
  reversal_score
    1.0 = 1h > 1% AND 4h < 0   (full reversal — fake pump)
    0.5 = 1h > 0 AND 4h < 25% of 1h gain  (momentum collapse)
    0.0 = no reversal pattern

Usage:
    python trend_outcome_tracker.py          # standalone CLI
    import trend_outcome_tracker; trend_outcome_tracker.track(client)
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path

import trend_scanner  # for _ensure_db and DB_PATH

HORIZONS = {
    "1h":  timedelta(hours=1),
    "4h":  timedelta(hours=4),
    "24h": timedelta(hours=24),
}

DB_PATH = trend_scanner.DB_PATH


def _compute_signal_metrics(returns: dict[str, float]) -> dict[str, float]:
    """
    Given a {horizon: return_pct} mapping, produce aggregated metrics.
    Called once all 3 horizons are recorded for a signal.
    """
    r1  = returns.get("1h",  0.0)
    r4  = returns.get("4h",  0.0)
    r24 = returns.get("24h", 0.0)

    all_rets = list(returns.values())
    max_move = max((v for v in all_rets if v > 0), default=0.0)
    min_move = min(all_rets, default=0.0)

    # Continuation: did the move strengthen by 4h?
    if r4 > r1 > 0:
        continuation = 1.0
    elif r4 > 0:
        continuation = 0.5
    else:
        continuation = 0.0

    # Reversal: did price spike at 1h then collapse by 4h?
    if r1 > 1.0 and r4 < 0:
        reversal = 1.0
    elif r1 > 0 and r4 < r1 * 0.25:
        reversal = 0.5
    else:
        reversal = 0.0

    return {
        "max_move_pct":       max_move,
        "min_move_pct":       min_move,
        "continuation_score": continuation,
        "reversal_score":     reversal,
    }


def track(client) -> int:
    """
    Record outcomes for all matured signal horizons and compute
    continuation metrics once a signal's 3 horizons are complete.

    Returns number of new outcome rows written.
    All failures are non-fatal — the trading bot continues regardless.
    """
    if not DB_PATH.exists():
        return 0

    # Ensure DB is initialised and migrated
    trend_scanner._ensure_db()

    now_utc = datetime.now(timezone.utc)
    written = 0

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row

        signals = conn.execute(
            "SELECT id, detected_at_utc, symbol, price, metrics_computed "
            "FROM trend_signals"
        ).fetchall()

        for sig in signals:
            sig_time = datetime.fromisoformat(sig["detected_at_utc"])
            if sig_time.tzinfo is None:
                sig_time = sig_time.replace(tzinfo=timezone.utc)

            # ── Record individual horizon outcomes ─────────────────────────
            for horizon_label, delta in HORIZONS.items():
                if now_utc < sig_time + delta:
                    continue  # not yet due

                already = conn.execute(
                    "SELECT 1 FROM trend_outcomes WHERE signal_id=? AND horizon=?",
                    (sig["id"], horizon_label),
                ).fetchone()
                if already:
                    continue

                try:
                    ticker = client.get_symbol_ticker(symbol=sig["symbol"])
                    current_price = float(ticker["price"])
                except Exception:
                    continue

                original_price = sig["price"]
                if original_price == 0:
                    continue

                return_pct = (current_price - original_price) / original_price * 100.0
                checked_at = now_utc.isoformat()

                conn.execute(
                    """
                    INSERT INTO trend_outcomes
                        (signal_id, checked_at_utc, horizon, original_price,
                         current_price, return_pct, created_at)
                    VALUES (?,?,?,?,?,?,?)
                    """,
                    (sig["id"], checked_at, horizon_label,
                     original_price, current_price, return_pct, checked_at),
                )
                written += 1

            # ── Compute aggregated metrics once all 3 horizons are done ───
            if sig["metrics_computed"]:
                continue

            completed = conn.execute(
                "SELECT horizon, return_pct FROM trend_outcomes WHERE signal_id=?",
                (sig["id"],),
            ).fetchall()

            if len(completed) < 3:
                continue

            returns = {row["horizon"]: row["return_pct"] for row in completed}
            m = _compute_signal_metrics(returns)

            conn.execute(
                """
                UPDATE trend_signals
                SET max_move_pct=?, min_move_pct=?,
                    continuation_score=?, reversal_score=?,
                    metrics_computed=1
                WHERE id=?
                """,
                (m["max_move_pct"], m["min_move_pct"],
                 m["continuation_score"], m["reversal_score"],
                 sig["id"]),
            )

        conn.commit()

    return written


def main() -> None:
    """CLI: track outcomes using live Binance client."""
    from pathlib import Path as P
    from dotenv import load_dotenv
    _vps_env = P("/opt/btcbot/.env")
    load_dotenv(dotenv_path=_vps_env if _vps_env.exists() else P(".env"))

    import config
    from binance.client import Client

    client = Client(
        api_key=config.BINANCE_API_KEY,
        api_secret=config.BINANCE_SECRET_KEY,
        testnet=config.TESTNET,
        requests_params={"timeout": 10},
    )
    n = track(client)
    print(f"Outcomes recorded: {n}")


if __name__ == "__main__":
    main()
