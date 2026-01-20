from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
from .merge_engine import SubconsciousMerger, SubconsciousSignal

app = FastAPI(title="Subconscious Merge API", version="0.1")
merger = SubconsciousMerger()

class Signal(BaseModel):
    name: str
    embedding: List[float] = Field(..., min_items=4)
    coherence: float
    entropy: float

class PsychState(BaseModel):
    distress_lock: bool = False

@app.get("/health")
def health():
    return {"enabled": merger.enabled, "safety_hold": merger.safety_hold, "distress_lock": merger.distress_lock}

@app.post("/readiness")
def readiness(ego: Signal, dream: Signal, myth: Signal):
    r = merger.compute_readiness(
        ego=SubconsciousSignal(**ego.dict()),
        dream=SubconsciousSignal(**dream.dict()),
        myth=SubconsciousSignal(**myth.dict()),
    )
    return {
        "composite": r.composite,
        "details": r.details,
        "avg_coherence": r.avg_coherence,
        "avg_entropy": r.avg_entropy
    }

@app.post("/commit")
def commit():
    result = merger.commit_merge()
    if result["status"] != "merged":
        raise HTTPException(status_code=409, detail=result)
    return result

@app.post("/psych")
def psych(ps: PsychState):
    merger.set_distress_lock(ps.distress_lock, why="psych_api")
    return {"ok": True}

@app.post("/enable")
def enable(flag: bool):
    merger.set_enabled(flag)
    return {"enabled": merger.enabled}

@app.post("/safety_hold")
def safety_hold(flag: bool):
    merger.set_safety_hold(flag, why="api")
    return {"safety_hold": merger.safety_hold}
