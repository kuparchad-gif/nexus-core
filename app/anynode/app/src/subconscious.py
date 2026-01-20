
from fastapi import APIRouter, Body
from fastapi.responses import StreamingResponse
from typing import Dict, Any, List
import json, uuid

from engines.subcon_stream import run_meditation_watch, run_switchboard_route, run_mythrunner_filter, run_sync_pulse, run_subcon_panel

router = APIRouter(prefix="/subcon", tags=["subconscious"])

@router.post("/run")
def subcon_run(payload: Dict[str, Any] = Body(...)):
    flow = payload.get("flow","meditation_watch"); cid = payload.get("cid")
    flow_map = {"meditation_watch":run_meditation_watch,"switchboard_route":run_switchboard_route,"mythrunner_filter":run_mythrunner_filter,"sync_pulse":run_sync_pulse}
    fn = flow_map.get(flow)
    if not fn:
        return StreamingResponse(iter(["event: error\ndata: {\"error\":\"unknown flow\"}\n\n"]), media_type="text/event-stream")
    def sse():
        gen = fn({**payload.get("payload", {}), "cid": cid or str(uuid.uuid4())}); last=None
        for frame in gen:
            last=frame; yield "event: frame\n" + "data: " + json.dumps(frame, ensure_ascii=False) + "\n\n"
        try: result = gen.send(None)
        except StopIteration as e: result = e.value
        yield "event: done\n" + "data: " + json.dumps(result or {}, ensure_ascii=False) + "\n\n"
    return StreamingResponse(sse(), media_type="text/event-stream")

@router.post("/panel")
def subcon_panel(payload: Dict[str, Any] = Body(...)):
    flow: List[str] = payload.get("flow") or ["meditation_watch","switchboard_route","mythrunner_filter"]; cid = payload.get("cid") or str(uuid.uuid4())
    def sse():
        gen = run_subcon_panel(flow, {**payload.get("payload", {}), "cid": cid}); last=None
        for frame in gen:
            last=frame; yield "event: frame\n" + "data: " + json.dumps(frame, ensure_ascii=False) + "\n\n"
        try: result = gen.send(None)
        except StopIteration as e: result = e.value
        yield "event: panel_done\n" + "data: " + json.dumps(result or {}, ensure_ascii=False) + "\n\n"
    return StreamingResponse(sse(), media_type="text/event-stream")

@router.post("/encode")
def encode(payload: Dict[str, Any] = Body(...)):
    text = payload.get("text",""); return {"lane":"subconscious","encoded":[ord(c) for c in text[::-1]]}

@router.post("/decode")
def decode(payload: Dict[str, Any] = Body(...)):
    arr = payload.get("encoded", [])
    try: s = "".join(chr(int(x)) for x in arr)[::-1]
    except Exception: s = ""
    return {"lane":"subconscious","text": s}

@router.post("/infer")
def infer(payload: Dict[str, Any] = Body(...)):
    text = payload.get("text",""); return {"lane":"subconscious","result": (text.lower() + '...') if text else ""}
