from __future__ import annotations
from typing import Dict, Any

class AbstractInferencer:
    def process(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        t = (payload.get("input") or "").lower()
        # gate: only run if dream-triggered
        trigger = payload.get("trigger") == "dream"
        idea = "latent-symbol" if trigger else "idle"
        return {"idea": idea, "confidence": 0.42 if trigger else 0.1}
