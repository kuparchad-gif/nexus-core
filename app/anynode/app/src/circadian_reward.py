# src/service/core/circadian_reward.py
from __future__ import annotations
from typing import Dict
import random

# Converts circadian signal into reward modifiers.
# Keep tiny and dependency-free so it can be imported anywhere.

def status_to_mod(status: str) -> Dict[str, float]:
    """
    Return a dict with suggested baseline/gain/noise for RewardEngine.tick().
    """
    s = (status or "awake").lower()
    if s == "asleep":
        return {"baseline": 0.50, "gain": 0.05, "noise": 0.01}
    if s == "napping":
        return {"baseline": 0.54, "gain": 0.06, "noise": 0.015}
    if s in ("evening", "tired"):
        return {"baseline": 0.56, "gain": 0.07, "noise": 0.02}
    if s in ("morning", "alert", "focused"):
        return {"baseline": 0.62, "gain": 0.10, "noise": 0.03}
    # default awake/afternoon
    return {"baseline": 0.58, "gain": 0.08, "noise": 0.02}

def reward_to_speech_knobs(dopamine: float) -> Dict[str, float]:
    """
    Optionally map dopamine to speaking style tweaks.
    Rate/pitch around 1.0 baseline; warmth 0..1.
    """
    d = max(0.0, min(1.0, float(dopamine)))
    return {
        "rate":   0.95 + 0.15*d,
        "pitch":  0.98 + 0.06*d,
        "warmth": 0.50 + 0.35*d
    }
