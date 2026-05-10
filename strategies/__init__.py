"""
strategies package — each module exposes get_signal(df) -> int.

Signal convention:
  +1 = BUY
   0 = NEUTRAL
  -1 = SELL
"""
from .ema import get_signal as ema_signal
from .macd import get_signal as macd_signal
from .rsi import get_signal as rsi_signal
from .stochastic import get_signal as stochastic_signal
from .bollinger import get_signal as bollinger_signal
from .volume import get_signal as volume_signal
from .vwap import get_signal as vwap_signal

__all__ = [
    "ema_signal",
    "macd_signal",
    "rsi_signal",
    "stochastic_signal",
    "bollinger_signal",
    "volume_signal",
    "vwap_signal",
]
