# Systems/engine/catalyst/symbol_mapper.py
from __future__ import annotations
from typing import Dict, Any

class SymbolMapper:
    def process(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        t = payload.get("input") or ""
        keys = []
        if "wake" in t.lower(): keys.append("DAWN")
        if "resources" in t.lower(): keys.append("HARVEST")
        return {"keys": keys or ["PLAIN"], "entropy": 0.12}
