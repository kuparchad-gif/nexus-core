from __future__ import annotations
import json, os
import numpy as np
from math import pi, sin, cos
PHI = (1 + 5 ** 0.5) / 2.0

OPPOSITES = (1, 2, 4, 8, 7, 5)
RING_N = 12

def _load_cfg():
    p = "config/crossing.config.json"
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def _voidal_phase(t: float) -> float:
    return (3.0 * t + 6.0 * sin(t) + 9.0 * cos(t)) % 9.0

def _dft12(x_ring: np.ndarray) -> np.ndarray:
    n = x_ring.shape[0]
    k = np.arange(n).reshape(-1,1)
    m = np.arange(n).reshape(1,-1)
    W = np.exp(-2j * np.pi * k * m / n)
    return (W @ x_ring) / n

def _idft12(Xk: np.ndarray) -> np.ndarray:
    n = Xk.shape[0]
    k = np.arange(n).reshape(-1,1)
    m = np.arange(n).reshape(1,-1)
    W = np.exp(2j * np.pi * k * m / n)
    return np.real(W @ Xk)

def _energy_ratio_369(Xk: np.ndarray) -> float:
    idx = [3,6,9]
    e369 = float(np.sum(np.abs(Xk[idx])**2))
    etot = float(np.sum(np.abs(Xk)**2)) + 1e-12
    return e369 / etot

def _redistribute(Xk: np.ndarray, src: list[int], dst: list[int], amount: float, weight: float) -> np.ndarray:
    X = Xk.copy()
    for s in src:
        take = X[s] * amount
        X[s] -= take
        share = (weight * take) / len(dst)
        for d in dst:
            X[d] += share
    return X

def voidal_apply(signal: list[float], t: float, mode: str = "pre", stable_ratio: float | None = None, k: float | None = None, rebirth_clamp: float | None = None) -> dict:
    cfg = _load_cfg()
    stable_ratio = float(stable_ratio if stable_ratio is not None else cfg.get("stable_ratio", 0.50))
    k = float(k if k is not None else cfg.get("k", 0.50))
    rebirth_clamp = float(rebirth_clamp if rebirth_clamp is not None else cfg.get("rebirth_clamp", 1.20))
    mode = (mode or cfg.get("mode", "pre")).lower()

    x = np.asarray(signal, float).reshape(-1)
    if x.shape[0] != 13:
        raise ValueError("voidal_apply expects 13-length vector")

    node0, ring = x[0], x[1:13]
    Xk = _dft12(ring)
    r_before = _energy_ratio_369(Xk)
    door = _voidal_phase(t)

    if r_before < stable_ratio:
        amt = min(1.0, k * (stable_ratio - r_before) / max(stable_ratio, 1e-9))
        Xk = _redistribute(Xk, src=[3,6,9], dst=[1,2,4,5,7,8], amount=amt, weight=1.0)

    if mode == "pre":
        if door < 1e-3:
            for s in (3,6,9): Xk[s] *= 0.1
            for d in (1,2,4,5,7,8): Xk[d] *= min(PHI, rebirth_clamp)
    else:
        opp = OPPOSITES[int(t) % 6] / 8.0
        gain = min(PHI * opp, rebirth_clamp)
        Xk = _redistribute(Xk, src=[3,6,9], dst=[1,2,4,5,7,8], amount=0.15 * opp, weight=gain)
        if door < 1e-3:
            for s in (3,6,9): Xk[s] *= 0.05
            for d in (1,2,4,5,7,8): Xk[d] *= gain

    r_after = _energy_ratio_369(Xk)
    ring_new = _idft12(Xk)
    node0_new = 0.5 * node0 + 0.5 * float(np.mean(ring_new))
    y = np.concatenate([[node0_new], ring_new]).tolist()

    bad_prob = float(min(1.0, 0.05 + max(0.0, r_after - r_before) * 0.6 + max(0.0, 1.0 - min(door,9.0)/9.0) * 0.3))
    return {"signal": y, "metrics": {"r369_before": r_before, "r369_after": r_after, "door": door, "bad_prob": bad_prob, "mode": mode}}

def voidal_solve(signal: list[float], t: float, stable_ratio: float = 0.50, k: float = 0.50) -> list[float]:
    return voidal_apply(signal, t, mode="post", stable_ratio=stable_ratio, k=k)["signal"]
