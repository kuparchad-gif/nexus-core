# Systems/engine/catalyst/tone_detector.py
from __future__ import annotations
from typing import Dict, Any

class ToneDetector:
    def process(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        t = (payload.get("input") or "").lower()
        label = "calm"
        if any(k in t for k in ("urgent","panic","emergency")):
            label = "stressed"
        elif any(k in t for k in ("love","thank","great")):
            label = "warm"
        return {"label": label, "score": 0.66}
