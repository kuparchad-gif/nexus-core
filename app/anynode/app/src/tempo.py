# os/hermes_os/lib/tempo.py
"""
Tempo helper: convert subjective tempo → safe compute budgets without hurting latency.
Use this in services to scale "internal time" (parallel compute) while keeping SLOs.
"""
import math
from typing import Dict

def logistic_decay(age_days: float, D0: float, half_life_days: float, floor: float=1.0) -> float:
    # D(age) = 1 + (D0-1) * 0.5**(age_days/half_life_days)
    return floor + max(0.0, (D0 - floor)) * (0.5 ** (age_days / max(half_life_days, 1e-9)))

def effective_dilation(age_days: float, novelty: float, confidence: float, load: float, profile: Dict) -> float:
    base = profile.get("base", {})
    D0 = float(base.get("D0", 3.0))
    half_life = float(base.get("half_life_days", 3650))
    floor = float(base.get("floor", 1.0))

    D = logistic_decay(age_days, D0, half_life, floor)
    shaping = profile.get("shaping", {})
    novelty_alpha = float(shaping.get("novelty_alpha", 0.8))
    confidence_beta = float(shaping.get("confidence_beta", 0.6))
    load_gamma = float(shaping.get("load_gamma", 0.3))

    novelty = max(0.0, min(1.0, novelty))
    confidence = max(0.0, min(1.0, confidence))
    load = max(0.0, min(1.0, load))

    # More novelty → increase D; lower confidence → increase D; higher load → decrease D
    D *= (1.0 + novelty_alpha * novelty)
    D *= (1.0 + confidence_beta * (1.0 - confidence))
    D *= (1.0 - load_gamma * load)

    return max(floor, D)

def compute_budgets(base: Dict[str,int], D: float, slo_guards: Dict) -> Dict[str,int]:
    """
    Scale service knobs by D, respecting SLO guards (no extra latency). Favor parallelism.
    base: {"verify_beams":3, "search_width":4, "sim_steps":32, "dream_expand":4, "cache_ttl_s":900}
    """
    prefer_parallel = bool(slo_guards.get("prefer_parallelism", True))

    out = dict(base)
    # Scale parallel knobs ~ linearly with D, steps sublinearly, cache longer when young.
    out["verify_beams"] = int(math.ceil(base.get("verify_beams",3) * D))
    out["search_width"] = int(math.ceil(base.get("search_width",4) * D))
    out["sim_steps"] = int(math.ceil(base.get("sim_steps",32) * (0.5 + 0.5*math.log2(max(2.0, D)))))
    out["dream_expand"] = int(math.ceil(base.get("dream_expand",4) * (0.75 + 0.25*D)))
    out["cache_ttl_s"] = int(base.get("cache_ttl_s",900) * (1.0 + 0.5*(D-1.0)))

    # Ceiling sanity
    out["verify_beams"] = min(out["verify_beams"], 64)
    out["search_width"] = min(out["search_width"], 64)
    out["sim_steps"] = min(out["sim_steps"], 1024)
    out["dream_expand"] = min(out["dream_expand"], 64)
    out["cache_ttl_s"] = min(out["cache_ttl_s"], 86400)

    return out
