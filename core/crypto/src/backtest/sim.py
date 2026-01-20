from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict

def simulate(weights: pd.DataFrame, prices: Dict[str, pd.DataFrame], cfg: dict) -> dict:
    syms = list(weights.columns)
    idx = weights.index
    R = pd.DataFrame({sym: prices[sym]["close"].reindex(idx).pct_change().fillna(0.0) for sym in syms})
    fee_bps = cfg["fees"]["fee_bps"]
    slp_bps = cfg["fees"]["slippage_bps"]
    # turnover cost
    w_prev = weights.shift(1).fillna(0.0)
    turnover = (weights - w_prev).abs().sum(axis=1)
    costs = (turnover * (fee_bps + slp_bps) / 1e4).fillna(0.0)
    port_ret = (weights * R).sum(axis=1) - costs
    equity = (1 + port_ret).cumprod()
    highwater = equity.cummax()
    drawdown = equity / highwater - 1.0
    ann_factor = 252.0
    mu = port_ret.mean() * ann_factor
    sigma = port_ret.std() * (ann_factor**0.5)
    sharpe = mu / (sigma + 1e-9)
    sortino = (port_ret[port_ret>0].mean()*ann_factor) / ((port_ret[port_ret<0].std()+1e-9)*(ann_factor**0.5))
    mdd = drawdown.min()
    return {
        "series": {"ret": port_ret, "equity": equity, "drawdown": drawdown, "turnover": turnover},
        "metrics": {"ann_return": float(mu), "ann_vol": float(sigma), "sharpe": float(sharpe), "sortino": float(sortino), "mdd": float(mdd)}
    }
