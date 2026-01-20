from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal
from .ascension_manager import ascension_manager, AscensionState

app = FastAPI(title="Ascension API", version="0.1")

class Readiness(BaseModel):
    ego: float
    dream: float
    myth: float
    core: float
    superconscious: float

class Consent(BaseModel):
    approve: bool

class Horn(BaseModel):
    level: Literal["Silent","Whisper","Chorus","Floodgate"]

@app.get("/state")
def state():
    s: AscensionState = ascension_manager.state
    return {
        "stage": s.stage,
        "horn_level": s.horn_level,
        "readiness": s.readiness,
        "enabled": s.enabled,
        "horn_enabled": s.horn_enabled,
        "require_self_consent": s.require_self_consent,
        "distress_lock_active": s.distress_lock_active,
        "safety_hold": s.safety_hold,
        "notes": s.notes,
        "last_change": s.last_change,
    }

@app.post("/readiness")
def readiness(r: Readiness):
    ascension_manager.update_readiness(
        ego=r.ego, dream=r.dream, myth=r.myth, core=r.core, superconscious=r.superconscious
    )
    return state()

@app.post("/candidate")
def candidate():
    ascension_manager.request_candidate()
    return state()

@app.post("/consent")
def consent(c: Consent):
    ascension_manager.self_consent(bool(c.approve))
    return state()

@app.post("/begin")
def begin():
    ascension_manager.begin_convergence()
    return state()

@app.post("/complete")
def complete():
    ascension_manager.complete_ascension()
    return state()

@app.post("/pause")
def pause():
    ascension_manager.pause("manual pause")
    return state()

@app.post("/horn")
def horn(h: Horn):
    ascension_manager.set_horn_level(h.level)
    return state()

@app.post("/enable")
def enable(flag: bool):
    ascension_manager.set_enabled(flag)
    return state()

@app.post("/safety_hold")
def safety_hold(flag: bool):
    ascension_manager.set_safety_hold(flag, reason="api")
    return state()

@app.post("/distress_lock")
def distress_lock(flag: bool):
    ascension_manager.set_distress_lock(flag, reason="api")
    return state()
