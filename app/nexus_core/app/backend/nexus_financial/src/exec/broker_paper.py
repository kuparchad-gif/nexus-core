from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict

class PaperBroker:
    def __init__(self, initial_cash: float = 100000.0, fee_bps: float = 1.0, slippage_bps: float = 2.0):
        self.cash = initial_cash
        self.pos = {}  # symbol -> shares
        self.fee_bps = fee_bps
        self.slippage_bps = slippage_bps
        self.history = []

    def value(self, prices: Dict[str, float]) -> float:
        pv = self.cash
        for sym, sh in self.pos.items():
            pv += sh * prices.get(sym, 0.0)
        return pv

    def rebalance_to_weights(self, target_w: Dict[str, float], prices: Dict[str, float]):
        total = self.value(prices)
        for sym, w in target_w.items():
            tgt_val = total * w
            px = prices[sym]
            tgt_sh = int(tgt_val / px) if px > 0 else 0
            cur_sh = self.pos.get(sym, 0)
            delta = tgt_sh - cur_sh
            if delta != 0:
                # apply simple costs
                gross = abs(delta) * px
                cost = gross * (self.fee_bps + self.slippage_bps) / 1e4
                self.cash -= delta * px + cost
                self.pos[sym] = tgt_sh
                self.history.append({"sym": sym, "delta": delta, "px": px, "cost": cost})
