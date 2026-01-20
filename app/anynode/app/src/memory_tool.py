from typing import Optional, List
try:
    from langchain.tools import BaseTool
except Exception:
    BaseTool = object

from sdk.python.nexus_mem import NexusMemClient

class RememberTool(BaseTool):
    name = "remember_tool"
    description = "Persist a memory with tags and priority to the local shard and mirror upstream if needed."

    def __init__(self, base_url: str = "[REDACTED-URL]):
        try:
            super().__init__()
        except Exception:
            pass
        self.mem = NexusMemClient(base_url=base_url)

    def _run(self, text: str, tags: Optional[List[str]] = None, priority: int = 5) -> str:
        r = self.mem.remember(text=text, tags=tags, priority=priority)
        return f"stored:{r.get('id','ok')}"

    async def _arun(self, text: str, tags: Optional[List[str]] = None, priority: int = 5) -> str:
        r = self.mem.remember(text=text, tags=tags, priority=priority)
        return f"stored:{r.get('id','ok')}"

