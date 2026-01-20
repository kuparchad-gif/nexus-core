from fastapi import FastAPI, Header, HTTPException
from typing import Optional, Dict, Any, List
import time

app = FastAPI(title="Aethereal Registry (stub)")
REG: Dict[str, Dict[str, Any]] = {}

@app.post("/register")
def register(entry: Dict[str, Any]):
    name = entry.get("name")
    if not name: raise HTTPException(400, "name required")
    entry["last_seen"] = time.time()
    REG[name] = entry
    return {"ok": True, "count": len(REG)}

@app.get("/services")
def services():
    return {"count": len(REG), "services": REG}
