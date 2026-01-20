# Systems/engine/catalyst/narrative_engine.py
from __future__ import annotations
from typing import Dict, Any

class NarrativeEngine:
    def process(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        t = payload.get("input") or ""
        arc = "setup" if len(t) < 40 else ("rising" if len(t) < 120 else "complex")
        return {"arc": arc}
