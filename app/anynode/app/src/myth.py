
from fastapi import APIRouter, Body
from fastapi.responses import StreamingResponse
from typing import Dict, Any
import json, uuid, os
from engines.myth_adapter import run_myth

router = APIRouter(prefix="/myth", tags=["myth"])

ASCENSION_MODE = bool(int(os.getenv("ASCENSION_MODE","0")))
EGO_FILTER_ENABLED = bool(int(os.getenv("EGO_FILTER_ENABLED","1")))

def _flags():
    return {"ascension_mode": ASCENSION_MODE, "ego_filter_enabled": EGO_FILTER_ENABLED}

@router.get("/toggles")
def get_toggles(): return _flags()

@router.post("/toggles")
def set_toggles(payload: Dict[str,Any] = Body(...)):
    global ASCENSION_MODE, EGO_FILTER_ENABLED
    if "ascension_mode" in payload: ASCENSION_MODE = bool(payload["ascension_mode"])
    if "ego_filter_enabled" in payload: EGO_FILTER_ENABLED = bool(payload["ego_filter_enabled"])
    return _flags()

@router.post("/run")
def myth_run(payload: Dict[str, Any] = Body(...)):
    ego = payload.get("ego_input", {}); dream = payload.get("dream_input", {}); cid = payload.get("cid")
    def sse():
        gen = run_myth(ego, dream, EGO_FILTER_ENABLED, ASCENSION_MODE, cid); last=None
        for frame in gen:
            last=frame; yield "event: frame\n" + "data: " + json.dumps(frame, ensure_ascii=False) + "\n\n"
        try: result = gen.send(None)
        except StopIteration as e: result = e.value
        yield "event: done\n" + "data: " + json.dumps(result or {}, ensure_ascii=False) + "\n\n"
    return StreamingResponse(sse(), media_type="text/event-stream")
