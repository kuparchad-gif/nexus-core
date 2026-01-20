from __future__ import annotations
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
from typing import Dict

def engineer_features(ohlcv: Dict[str, pd.DataFrame], cfg: dict) -> Dict[str, pd.DataFrame]:
    out = {}
    fast = cfg["strategy"]["params"]["lookback_fast"]
    slow = cfg["strategy"]["params"]["lookback_slow"]
    breakout_lb = cfg["strategy"]["params"]["breakout_lb"]
    atr_lb = cfg["strategy"]["params"]["atr_lb"]
    for sym, df in ohlcv.items():
        x = df.copy()
        x["ret1"] = x["close"].pct_change()
        x["sma_fast"] = x["close"].rolling(fast).mean()
        x["sma_slow"] = x["close"].rolling(slow).mean()
        x["sma_diff"] = (x["sma_fast"] - x["sma_slow"]) / (x["sma_slow"] + 1e-12)
        x["rsi14"] = RSIIndicator(x["close"], window=14).rsi()
        atr = AverageTrueRange(x["high"], x["low"], x["close"], window=atr_lb).average_true_range()
        x["atr"] = atr
        x["breakout_high"] = x["close"] > x["close"].rolling(breakout_lb).max().shift(1)
        x["breakout_low"]  = x["close"] < x["close"].rolling(breakout_lb).min().shift(1)
        x.dropna(inplace=True)
        out[sym] = x
    return out
