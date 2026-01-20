from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict

def rule_combo_signals(features: Dict[str, pd.DataFrame], cfg: dict) -> Dict[str, pd.Series]:
    """Deterministic signals:
    - Trend: SMA fast > SMA slow bullish; opposite bearish.
    - Breakout confirmation adds conviction.
    - RSI extremes dampen (avoid overbought/oversold chase).
    - Volatility filter via ATR vs price.
    Returns signal in [-1, +1].
    """
    signals = {}
    params = cfg["strategy"]["params"]
    for sym, x in features.items():
        s = np.sign(x["sma_diff"])  # +1 if fast>slow else -1
        s += x["breakout_high"].astype(int) * 0.5
        s -= x["breakout_low"].astype(int)  * 0.5
        # dampen when RSI > 75 or < 25
        damp = np.where((x["rsi14"]>75) | (x["rsi14"]<25), 0.5, 1.0)
        # vol filter
        vol_filter = np.where((x["atr"] / x["close"]) > 0.08, 0.5, 1.0)  # if too volatile, scale down
        sig = s * damp * vol_filter
        sig = np.clip(sig, -1.0, 1.0)
        signals[sym] = pd.Series(sig, index=x.index, name="signal")
    return signals
