from __future__ import annotations
from typing import Dict, Any

class VisualDecoder:
    def process(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        token = payload.get("image_token") or "IMG00000"
        return {"objects": ["symbolic:"+token[-3:]], "confidence": 0.33}
