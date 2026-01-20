# Systems/engine/catalyst/structure_parser.py
from __future__ import annotations
from typing import Dict, Any

class StructureParser:
    def process(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        t = (payload.get("input") or "")
        is_code = any(x in t for x in ("{", "}", "def ", ":"))
        return {"type": ("code" if is_code else "text")}
