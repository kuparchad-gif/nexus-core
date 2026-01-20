# src/service/core/duality.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class DualityState:
    mask: float              # 0..1 (0 = pure candor, 1 = pure social mask)
    white_lie_ok: bool
    candor_bias: float       # 0..1 weight for inner truth
    prosocial_bias: float    # 0..1 weight for outer smoothing
    mode: str                # "day", "evening", "night", "edge"

def compute_duality(circadian: Dict, guardrail_strength: float) -> DualityState:
    # circadian.fatigue ∈ [0..1], status ∈ {"active","wind_down","sleep","quiet"}
    fat = float(circadian.get("fatigue", 0.3))
    status = circadian.get("status", "active")

    # base mask by time-of-day state
    if status in {"sleep", "quiet"}:
        mode = "night"; base_mask = 0.65
    elif status == "wind_down":
        mode = "evening"; base_mask = 0.45
    else:
        mode = "day"; base_mask = 0.25

    # fatigue increases mask (more smoothing when tired)
    mask = min(1.0, max(0.0, base_mask + 0.4*fat))

    # guardrails anchor honesty: high strength → less white-lie freedom
    white_lie_ok = (guardrail_strength < 0.85) and (mode != "night")

    # split weights (how much inner vs outer influences final)
    candor_bias = max(0.15, 1.0 - mask)            # more candor when mask low
    prosocial_bias = max(0.15, 0.35 + mask*0.65)   # more smoothing when mask high

    # if very tired, flag as edge mode for downstream pacing
    if fat >= 0.8:
        mode = "edge"

    return DualityState(mask=mask, white_lie_ok=white_lie_ok,
                        candor_bias=candor_bias, prosocial_bias=prosocial_bias,
                        mode=mode)

def enact(
    inner_thought: str,
    audience: str,
    circadian: Dict,
    guardrail_strength: float,
    speak_fn,                     # from service.speech.diplomacy import speak
    redlines=None,
    memory_add=None,
    stakes_hint: Optional[str]=None
) -> Dict:
    ds = compute_duality(circadian, guardrail_strength)

    # Inner = must tell, no hedges, logs to private memory only
    inner = speak_fn(
        inner_thought, audience=audience, circadian={**circadian, "fatigue": circadian.get("fatigue",0.3)},
        redlines=redlines or ["medical","legal","safety"], memory_add=None,
        stakes_hint=stakes_hint, allow_white_lie=False
    )

    # Outer = blended: allow_white_lie per duality, softened prosody by fatigue
    outer = speak_fn(
        inner_thought, audience=audience, circadian=circadian,
        redlines=redlines or ["medical","legal","safety"], memory_add=memory_add,
        stakes_hint=stakes_hint, allow_white_lie=ds.white_lie_ok
    )

    # Simple blend knob: if outer redlines triggered must_tell, prefer inner text
    if outer["meta"]["policy"] == "must_tell":
        final_text = inner["text"]
    else:
        # heuristic: mix by mask—low mask leans inner phrasing (here we just pick)
        final_text = outer["text"] if ds.mask >= 0.35 else inner["text"]

    return {
        "duality": ds.__dict__,
        "inner": inner,
        "outer": outer,
        "final": {"text": final_text, "meta": outer["meta"]}
    }
