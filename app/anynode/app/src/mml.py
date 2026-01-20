import time
from typing import Any, Dict, Callable

class MML:
    """Minimal 'MML' harness to man a service or datastore.
    Responsibilities:
      - observe(): accept events and write to memory (Qdrant via callback)
      - reason(): summarize/recommend based on provider
      - act(): propose *signed* intents via core or route to Lillith
    Execution never bypasses signed-intent rules.
    """
    def __init__(self, *, name: str, role: str, recall: Callable, write: Callable, push_intent: Callable, provider: str="local"):
        self.name = name
        self.role = role
        self.recall = recall
        self.write = write
        self.push_intent = push_intent
        self.provider = provider

    async def observe(self, event: Dict[str, Any]) -> Dict[str, Any]:
        event = {**event, "mml": self.name, "role": self.role, "ts": time.time()}
        await self.write(event)
        return {"ok": True, "stored": True}

    async def reason(self, question: str, *, k: int = 20) -> Dict[str, Any]:
        docs = await self.recall({"q": question, "k": k})
        return {"ok": True, "answer": f"{self.name} sees {len(docs)} items", "context": docs[:3]}

    async def act(self, intent: Dict[str, Any]) -> Dict[str, Any]:
        reason = (intent.get("reason") or "").strip()
        if not reason:
            return {"ok": False, "error": "reason_required"}
        return await self.push_intent(intent)
