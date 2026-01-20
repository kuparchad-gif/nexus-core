from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict

def weights_from_signals(signals: Dict[str, pd.Series], prices: Dict[str, pd.DataFrame], cfg: dict) -> pd.DataFrame:
    """Map signals [-1,1] to portfolio weights with volatility targeting + caps."""
    target_vol = cfg["risk"]["target_vol_annual"]
    max_gross = cfg["risk"]["max_gross"]
    max_single = cfg["risk"]["max_single"]
    # align index (intersection of dates)
    idx = None
    for s in signals.values():
        idx = s.index if idx is None else idx.intersection(s.index)
    idx = idx.sort_values()
    syms = list(signals.keys())
    W = pd.DataFrame(index=idx, columns=syms, dtype=float)
    # simple z-like normalize and cap
    for sym in syms:
        W[sym] = signals[sym].reindex(idx).fillna(0.0) / len(syms)
    # scale to target vol proxy using price vol
    # compute daily vol proxy
    rets = {sym: prices[sym]["close"].reindex(idx).pct_change() for sym in syms}
    vol = pd.DataFrame(rets).ewm(span=60).std() * (252**0.5)
    scale = target_vol / (vol.mean(axis=1).clip(lower=1e-6))
    W = (W.T * scale).T
    # cap single name & gross
    W = W.clip(lower=-max_single, upper=max_single)
    gross = W.abs().sum(axis=1).clip(lower=1e-6)
    for t in W.index:
        if gross.loc[t] > max_gross:
            W.loc[t] *= (max_gross / gross.loc[t])
    return W.fillna(0.0)
