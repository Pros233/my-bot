"""
trend_alerts.py — Telegram WATCH ONLY alerts for trending coins.

Only called by trend_scanner.py for A+ and A grades.
Never places or suggests placing a trade.
"""
from __future__ import annotations

import requests

import config

_CONTINUATION = {
    "A+": "HIGH",
    "A":  "MEDIUM-HIGH",
    "B":  "MEDIUM",
    "C":  "LOW",
}


def send_trend_alert(
    symbol: str,
    price: float,
    price_change_1h: float,
    price_change_4h: float,
    volume_spike: float,
    spread_pct: float,
    volatility_pct: float,
    sentiment: str,
    score: float,
    grade: str = "B",
    mtf_15m: str = "neutral",
    mtf_1h: str = "neutral",
    mtf_4h: str = "neutral",
) -> bool:
    """
    Send a graded Telegram trend alert.
    Returns True if message was accepted by Telegram.
    """
    if not config.ENABLE_TELEGRAM_ALERTS:
        return False
    if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
        return False

    base        = symbol.replace("USDT", "")
    vol_pct     = (volume_spike - 1.0) * 100.0
    cont_prob   = _CONTINUATION.get(grade, "VERY LOW")

    if score >= 85:
        momentum_quality = "strong"
    elif score >= 70:
        momentum_quality = "good"
    elif score >= 55:
        momentum_quality = "moderate"
    else:
        momentum_quality = "weak"

    text = (
        f"*TREND ALERT | {grade}*\n"
        f"Symbol: {base}USDT\n"
        f"15m: {mtf_15m}\n"
        f"1h: {mtf_1h}\n"
        f"4h: {mtf_4h}\n"
        f"Price: ${price:,.4f}\n"
        f"1h move: {price_change_1h:+.2f}%\n"
        f"4h move: {price_change_4h:+.2f}%\n"
        f"Vol spike: +{vol_pct:.0f}%\n"
        f"Spread: {spread_pct:.3f}%\n"
        f"Volatility: {volatility_pct:.2f}%\n"
        f"Momentum quality: {momentum_quality}\n"
        f"Sentiment: {sentiment}\n"
        f"Continuation probability: {cont_prob}\n"
        f"Action: *WATCH ONLY* — No trade placed."
    )

    try:
        url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage"
        r = requests.post(
            url,
            json={
                "chat_id": config.TELEGRAM_CHAT_ID,
                "text": text,
                "parse_mode": "Markdown",
            },
            timeout=10,
        )
        return r.ok
    except Exception:
        return False
