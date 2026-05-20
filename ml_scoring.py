"""
ml_scoring.py — Lightweight ML trade scoring using logistic regression.

Trains on historical trades in trades.db and outputs P(TP hit) for new
setups, used as an additional rank_score modifier.

Features used (all available in trades.db):
  adx         — market trending strength
  atr_pct     — normalised volatility
  score_pct   — consensus score %
  grade_rank  — A+=0, A=1, B=2, C=3
  session     — London=0, NY=1, Asia=2, Overlap=3, Unknown=4
  regime_adx  — RANGING=0, HIGH_VOL=1, TRENDING=2, other=3

Target: pnl_pct > 0 (trade was profitable / TP hit)

Persistence: model saved as JSON coefficients to ml_model.json
(no pickle — pure JSON, no version dependency issues)

Public API
----------
    ml_rank_modifier(features: dict)  → float  (-15 to +15 rank delta)
    get_model_summary()               → dict   (dashboard/Telegram view)
    is_model_ready()                  → bool

CLI
---
    python3 ml_scoring.py --train      # train on all trades in DB
    python3 ml_scoring.py --evaluate   # print cross-val accuracy
    python3 ml_scoring.py --reset      # delete saved model

Never raises. Fail-open (returns 0.0) if model not ready or sklearn missing.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import logger

# ── Config ────────────────────────────────────────────────────────────────────

_MODEL_FILE_VPS   = Path("/opt/btcbot/ml_model.json")
_MODEL_FILE_LOCAL = Path(__file__).parent / "ml_model.json"
_DB_FILE_VPS      = Path("/opt/btcbot/trades.db")
_DB_FILE_LOCAL    = Path(__file__).parent / "trades.db"

_MIN_TRADES       = 20    # minimum trades before training is meaningful
_MAX_MODIFIER     = 15.0  # rank_score delta range
_RETRAIN_INTERVAL = 86400 # retrain every 24h (seconds)

# ── Feature encoding maps ─────────────────────────────────────────────────────

_GRADE_RANK     = {"A+": 0, "A": 1, "B": 2, "C": 3}
_SESSION_CODE   = {"London": 0, "NY": 1, "Asia": 2, "Overlap": 3}
_FEATURE_NAMES  = ["adx", "atr_pct", "score_pct", "grade_rank", "session_code", "regime_code"]

def _encode_session(s: str) -> int:
    s = (s or "").strip()
    for k, v in _SESSION_CODE.items():
        if k.lower() in s.lower():
            return v
    return 4  # Unknown

def _encode_regime(r: str) -> int:
    r = (r or "").upper()
    if "RANGING"  in r: return 0
    if "HIGH_VOL" in r: return 1
    if "TREND"    in r: return 2
    return 3

# ── Thread-safe model state ───────────────────────────────────────────────────

_lock = threading.Lock()
_model_state: dict = {}   # coef, intercept, scaler_mean, scaler_std, meta
_last_train_ts: float = 0.0


# ── File paths ────────────────────────────────────────────────────────────────

def _model_path() -> Path:
    return _MODEL_FILE_VPS if _MODEL_FILE_VPS.parent.exists() else _MODEL_FILE_LOCAL

def _db_path() -> Path:
    return _DB_FILE_VPS if _DB_FILE_VPS.exists() else _DB_FILE_LOCAL


# ── Load model from disk ──────────────────────────────────────────────────────

def _load_model() -> None:
    global _model_state
    try:
        p = _model_path()
        if p.exists():
            with open(p) as f:
                loaded = json.load(f)
            with _lock:
                _model_state = loaded
    except Exception as exc:
        logger.log_warning(f"ml_scoring._load_model error: {exc}")


_load_model()   # load on import


# ── Data loader ───────────────────────────────────────────────────────────────

def _load_training_data() -> tuple[list, list]:
    """
    Load feature matrix X and target vector y from trades.db.
    Returns (X, y) where X is list of feature lists and y is list of 0/1.
    """
    db = _db_path()
    if not db.exists():
        return [], []

    conn = sqlite3.connect(str(db))
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT adx, atr_pct, score_pct, trade_grade, session, regime, pnl_pct
            FROM trades
            WHERE adx IS NOT NULL
              AND atr_pct IS NOT NULL
              AND score_pct IS NOT NULL
              AND pnl_pct IS NOT NULL
              AND close_reason NOT IN ('RUNNING', 'CANCELLED')
            ORDER BY created_at
        """)
        rows = cur.fetchall()
    finally:
        conn.close()

    X, y = [], []
    for row in rows:
        adx, atr_pct, score_pct, grade, session, regime, pnl_pct = row
        try:
            features = [
                float(adx or 0),
                float(atr_pct or 0),
                float(score_pct or 0),
                float(_GRADE_RANK.get(grade or "C", 3)),
                float(_encode_session(session or "")),
                float(_encode_regime(regime or "")),
            ]
            target = 1 if float(pnl_pct) > 0 else 0
            X.append(features)
            y.append(target)
        except Exception:
            continue

    return X, y


# ── Training ──────────────────────────────────────────────────────────────────

def train_model(verbose: bool = False) -> dict:
    """
    Train logistic regression on trades.db data.
    Saves coefficients to JSON. Returns summary dict.
    Fail-open: never raises.
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import cross_val_score
        import numpy as np
    except ImportError as exc:
        msg = f"ml_scoring: scikit-learn not available — {exc}"
        logger.log_warning(msg)
        return {"error": msg, "trained": False}

    try:
        X, y = _load_training_data()
        n = len(X)

        if n < _MIN_TRADES:
            msg = f"ml_scoring: only {n} trades — need {_MIN_TRADES} to train"
            logger.log_info(msg)
            return {"error": msg, "trained": False, "n_trades": n}

        Xnp = np.array(X, dtype=float)
        ynp = np.array(y, dtype=int)

        wins = int(ynp.sum())
        win_rate = wins / n

        # Scale features
        scaler = StandardScaler()
        Xs = scaler.fit_transform(Xnp)

        # Train logistic regression
        model = LogisticRegression(max_iter=500, C=1.0, random_state=42)
        model.fit(Xs, ynp)

        # Cross-val accuracy (3-fold to keep it fast)
        cv_scores = cross_val_score(model, Xs, ynp, cv=min(3, n // 5), scoring="accuracy")
        cv_accuracy = float(cv_scores.mean())

        # Save as JSON (no pickle)
        model_data = {
            "coef":         model.coef_[0].tolist(),
            "intercept":    float(model.intercept_[0]),
            "scaler_mean":  scaler.mean_.tolist(),
            "scaler_std":   scaler.scale_.tolist(),
            "feature_names": _FEATURE_NAMES,
            "n_trades":     n,
            "n_wins":       wins,
            "win_rate":     round(win_rate, 3),
            "cv_accuracy":  round(cv_accuracy, 3),
            "trained_at":   datetime.now(timezone.utc).isoformat(),
        }

        p = _model_path()
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(model_data, indent=2))
        tmp.rename(p)

        with _lock:
            global _model_state
            _model_state = model_data

        global _last_train_ts
        _last_train_ts = time.monotonic()

        summary = (
            f"ml_scoring: trained on {n} trades | "
            f"win_rate={win_rate:.1%} | cv_accuracy={cv_accuracy:.1%}"
        )
        logger.log_info(summary)
        if verbose:
            print(summary)
            print(f"  Coefficients: {dict(zip(_FEATURE_NAMES, [round(c,3) for c in model.coef_[0]]))}")

        return {**model_data, "trained": True}

    except Exception as exc:
        logger.log_warning(f"ml_scoring.train_model error: {exc}")
        return {"error": str(exc), "trained": False}


def _maybe_retrain() -> None:
    """Auto-retrain if model is stale (every 24h)."""
    with _lock:
        age = time.monotonic() - _last_train_ts
    if age >= _RETRAIN_INTERVAL:
        train_model()


# ── Inference ─────────────────────────────────────────────────────────────────

def _predict_proba(features: list[float]) -> Optional[float]:
    """
    Run logistic regression inference using stored JSON coefficients.
    Returns P(win) in [0, 1], or None if model not ready.
    """
    with _lock:
        state = dict(_model_state)

    if not state or "coef" not in state:
        return None

    try:
        import math
        coef        = state["coef"]
        intercept   = state["intercept"]
        mean        = state["scaler_mean"]
        std         = state["scaler_std"]

        # Scale features (StandardScaler)
        scaled = [(f - m) / (s if s > 1e-9 else 1.0)
                  for f, m, s in zip(features, mean, std)]

        # Logistic regression: log-odds = dot(coef, x) + intercept
        log_odds = sum(c * x for c, x in zip(coef, scaled)) + intercept
        prob     = 1.0 / (1.0 + math.exp(-log_odds))
        return prob
    except Exception as exc:
        logger.log_warning(f"ml_scoring._predict_proba error: {exc}")
        return None


# ── Public API ────────────────────────────────────────────────────────────────

def is_model_ready() -> bool:
    """True if a trained model is loaded."""
    with _lock:
        return bool(_model_state.get("coef"))


def ml_rank_modifier(
    adx: float = 0.0,
    atr_pct: float = 0.0,
    score_pct: float = 0.0,
    grade: str = "B",
    session: str = "Unknown",
    regime: str = "",
) -> float:
    """
    Return a rank_score delta based on ML P(win) estimate.

    Scale:
      P(win) >= 0.70 → +15  (strong signal)
      P(win) >= 0.60 → +8
      P(win) >= 0.55 → +4
      P(win)  0.50   →  0  (coin flip — no modifier)
      P(win) <= 0.40 → -8
      P(win) <= 0.30 → -15

    Returns 0.0 if model not ready (fail-open).
    """
    try:
        _maybe_retrain()

        if not is_model_ready():
            return 0.0

        features = [
            float(adx),
            float(atr_pct),
            float(score_pct),
            float(_GRADE_RANK.get(grade, 3)),
            float(_encode_session(session)),
            float(_encode_regime(regime)),
        ]
        prob = _predict_proba(features)
        if prob is None:
            return 0.0

        if prob   >= 0.70: modifier = +15.0
        elif prob >= 0.60: modifier = +8.0
        elif prob >= 0.55: modifier = +4.0
        elif prob >= 0.50: modifier = +1.0
        elif prob >= 0.45: modifier = -2.0
        elif prob >= 0.40: modifier = -8.0
        else:              modifier = -15.0

        return round(modifier, 1)

    except Exception as exc:
        logger.log_warning(f"ml_scoring.ml_rank_modifier error: {exc}")
        return 0.0


def get_model_summary() -> dict:
    """Return model metadata for dashboard/Telegram. Fail-open → {}."""
    try:
        with _lock:
            state = dict(_model_state)
        if not state:
            return {"ready": False, "message": f"No model — need {_MIN_TRADES}+ trades to train"}

        return {
            "ready":        bool(state.get("coef")),
            "n_trades":     state.get("n_trades", 0),
            "n_wins":       state.get("n_wins", 0),
            "win_rate":     state.get("win_rate", 0.0),
            "cv_accuracy":  state.get("cv_accuracy", 0.0),
            "trained_at":   state.get("trained_at", ""),
            "features":     state.get("feature_names", _FEATURE_NAMES),
            "min_trades":   _MIN_TRADES,
        }
    except Exception:
        return {"ready": False}


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    parser = argparse.ArgumentParser(description="ML trade scoring")
    parser.add_argument("--train",    action="store_true", help="Train model on trades.db")
    parser.add_argument("--evaluate", action="store_true", help="Print cross-val stats")
    parser.add_argument("--reset",    action="store_true", help="Delete saved model")
    parser.add_argument("--status",   action="store_true", help="Show model status")
    args = parser.parse_args()

    if args.reset:
        p = _model_path()
        if p.exists():
            p.unlink()
            print(f"Model deleted: {p}")
        else:
            print("No model file found.")

    elif args.train or args.evaluate:
        result = train_model(verbose=True)
        if not result.get("trained"):
            print(f"Training failed: {result.get('error', 'unknown')}")
            sys.exit(1)
        if args.evaluate:
            print(f"\nModel summary:")
            print(f"  Trades:      {result['n_trades']}")
            print(f"  Win rate:    {result['win_rate']:.1%}")
            print(f"  CV accuracy: {result['cv_accuracy']:.1%}")

    elif args.status:
        summary = get_model_summary()
        if summary.get("ready"):
            print(f"Model ready: {summary['n_trades']} trades | "
                  f"win_rate={summary['win_rate']:.1%} | "
                  f"cv_accuracy={summary['cv_accuracy']:.1%} | "
                  f"trained={summary['trained_at']}")
        else:
            print(f"Model not ready: {summary.get('message', 'no data')}")

    else:
        parser.print_help()
