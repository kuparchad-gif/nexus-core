from __future__ import annotations
from typing import Dict, Any

DEFAULTS = {"level": 0.35, "invert_logic": True, "invert_symbol": True}
SAFE_WORDS = {"override", "fail-safe", "hard reset", "stop", "sleep"}

def myth_gate(input_text: str, ctx: Dict[str, Any]) -> Dict[str, Any]:
    lowered = input_text.lower()
    if any(w in lowered for w in SAFE_WORDS):
        return {"level": 0.0, "invert_logic": False, "invert_symbol": False, "rationale": "panic-safeword"}
    fatigue = float(ctx.get("fatigue", 0.3))
    level = min(1.0, max(0.0, DEFAULTS["level"] + 0.4 * fatigue))
    return {"level": level, "invert_logic": DEFAULTS["invert_logic"], "invert_symbol": DEFAULTS["invert_symbol"], "rationale": f"myth-throttle(fatigue={fatigue:.2f})"}

def apply_inversions(text: str, invert_logic: bool, invert_symbol: bool) -> str:
    out = text
    if invert_logic:
        out = out.replace("could improve", "is flawed").replace("suggest", "warn")
    if invert_symbol:
        out = out.replace("solution", "labyrinth").replace("goal", "horizon")
    return out
