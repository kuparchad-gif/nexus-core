# metatron_gate.py

from fastapi import FastAPI, Body
from pydantic import BaseModel
import os, httpx

app = FastAPI()
LLM_BASE = os.getenv("LLM_BASE")  # e.g. http://host.docker.internal:1234/v1

class InMsg(BaseModel):
    content: str
    source: str = "chat"
    role: str = "user"
    meta: dict = {}

class OutMsg(BaseModel):
    action: str
    reason: str = "ok"
    trust_score: float = 0.7
    delay_ms: int = 0
    shaped_priority: float = 0.5

@app.get("/health")  # for your dashboard tile
def health(): return {"ok": True, "mode":"soft"}

@app.post("/process")  # Metatron gate per your spec
def gate(m: InMsg) -> OutMsg:
    if not m.content.strip():
        return OutMsg(action="drop", reason="empty", trust_score=0.2)
    # tiny heuristic = permit for now
    return OutMsg(action="permit", reason="relay", trust_score=0.7, shaped_priority=0.4)

@app.post("/api/v1/route")  # chat hits here; we gate first
async def chat_route(body: dict = Body(...)):
    o = gate(InMsg(content=str(body.get("prompt","")).strip()))
    if o.action != "permit":
        return {"system":{"status":o.action,"reason":o.reason}}
    # if LM Studio is configured and a model was passed, use it; else echo
    model = body.get("model")
    if LLM_BASE and model:
        async with httpx.AsyncClient(base_url=LLM_BASE, timeout=30) as c:
            r = await c.post("/chat/completions", json={
                "model": model,
                "messages":[{"role":"system","content":"Be concise."},
                            {"role":"user","content": body.get("prompt","")}],
                "stream": False, "temperature": 0.2
            })
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"]
        return {"selected":[{"name": model, "score":1.0}],
                "combined":{"assistant":{"text":text}}}
    return {"combined":{"assistant":{"text":f"echo: {body.get('prompt','')}"}}}
