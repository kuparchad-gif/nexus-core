# src/lilith/mesh/mcp_adapter.py (pre-crossing stable state - low risk)
from __future__ import annotations
import json
from typing import Dict, Any, Callable, Awaitable

class MCPAdapter(TrustPhaseMixin):
    def __init__(self):
        super().__init__()
        self.tools = {}

    def register(self, name: str, func: Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]) -> None:
        self.tools[name] = func

    async def handle(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        if msg.get("type") != "mcp.call":
            return {"type": "mcp.result", "id": msg.get("id"), "ok": False, "error": "bad_type"}
        tool = msg.get("tool")
        if tool not in self.tools:
            return {"type": "mcp.result", "id": msg.get("id"), "ok": False, "error": "unknown_tool"}
        if not self.gate(5):
            return {"type": "mcp.result", "id": msg.get("id"), "ok": False, "error": "trust_denied"}
        try:
            out = await self.tools[tool](msg.get("params") or {})
            return {"type": "mcp.result", "id": msg.get("id"), "ok": True, "data": out}
        except Exception as e:
            return {"type": "mcp.result", "id": msg.get("id"), "ok": False, "error": str(e)}