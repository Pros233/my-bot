"""
telegram_voice.py — Optional voice alert synthesis via gTTS.

Returns MP3 bytes suitable for telegram_bot.send_voice().
Never raises — returns None if gTTS is unavailable or synthesis fails.

Requires (optional): pip install gtts
"""
from __future__ import annotations

import io
from typing import Optional

import logger


def synthesize(text: str) -> Optional[bytes]:
    """
    Convert *text* to speech MP3 bytes using gTTS.
    Returns None if gTTS is not installed or synthesis fails.
    """
    try:
        from gtts import gTTS
    except ImportError:
        return None

    try:
        # Strip markdown formatting before TTS
        clean = (text
                 .replace("*", "")
                 .replace("`", "")
                 .replace("_", " ")
                 .replace("\n", ". "))
        # Truncate to 200 chars to keep voice messages short
        clean = clean[:200]

        buf = io.BytesIO()
        tts = gTTS(text=clean, lang="en", slow=False)
        tts.write_to_fp(buf)
        buf.seek(0)
        return buf.read()
    except Exception as exc:
        logger.log_warning(f"telegram_voice.synthesize failed: {exc}")
        return None


def trade_opened_voice(symbol: str, side: str, entry: float,
                       sl: float, tp: float) -> Optional[bytes]:
    """Pre-formatted voice for trade open."""
    base = symbol.replace("USDT", "")
    text = (
        f"Trade opened. {base} {side.lower()}. "
        f"Entry {entry:,.0f}. "
        f"Stop loss {sl:,.0f}. "
        f"Take profit {tp:,.0f}."
    )
    return synthesize(text)


def trade_closed_voice(symbol: str, pnl: float,
                       reason: str) -> Optional[bytes]:
    """Pre-formatted voice for trade close."""
    base = symbol.replace("USDT", "")
    direction = "profit" if pnl >= 0 else "loss"
    text = (
        f"Trade closed. {base}. "
        f"{direction} {abs(pnl):.2f} USDT. "
        f"Reason: {reason.replace('_', ' ')}."
    )
    return synthesize(text)
