
from fastapi import APIRouter, Body
from fastapi.responses import StreamingResponse, PlainTextResponse
import json, uuid, os
from typing import Dict, Any, List
from pathlib import Path
from engines.niv_stream import load_template, run_template

TEMPLATES_ROOT = os.getenv("NIV_TEMPLATES_ROOT", "/mnt/data/nexus_thought_templates/json")
router = APIRouter(prefix="/conscious", tags=["conscious"])

@router.get("/templates", response_class=PlainTextResponse)
def list_templates():
    p = Path(TEMPLATES_ROOT)
    if not p.exists(): return "No template root found."
    return "\n".join(sorted([x.name for x in p.glob("*.json")]))

@router.post("/run")
def run(payload: Dict[str, Any] = Body(...)):
    ttype = payload.get("template_type")
    tpath = payload.get("template_path") or str(Path(TEMPLATES_ROOT) / f"{ttype}.json")
    template = load_template(tpath); cid = payload.get("cid") or str(uuid.uuid4())
    def sse():
        gen = run_template(template, input_overrides=payload.get("input"), cid=cid); last=None
        for frame in gen:
            last=frame; yield "event: frame\n" + "data: " + json.dumps(frame, ensure_ascii=False) + "\n\n"
        try: result = gen.send(None)
        except StopIteration as e: result = e.value
        yield "event: done\n" + "data: " + json.dumps(result or {}, ensure_ascii=False) + "\n\n"
    return StreamingResponse(sse(), media_type="text/event-stream")

@router.post("/panel")
def panel(payload: Dict[str, Any] = Body(...)):
    methods: List[str] = payload.get("methods") or ["pattern_recognition","first_principles","deductive_reasoning"]
    rinput = payload.get("input", {}); cid = payload.get("cid") or str(uuid.uuid4())
    def sse():
        summary = []
        for m in methods:
            tpath = str(Path(TEMPLATES_ROOT) / f"{m}.json"); template = load_template(tpath)
            gen = run_template(template, input_overrides=rinput, cid=cid); last=None
            for frame in gen:
                last=frame; frame["method"]=m; yield "event: frame\n" + "data: " + json.dumps(frame, ensure_ascii=False) + "\n\n"
            try: result = gen.send(None)
            except StopIteration as e: result = e.value
            summary.append({"method": m, "result": result})
        yield "event: panel_done\n" + "data: " + json.dumps({"cid":cid,"summary":summary}, ensure_ascii=False) + "\n\n"
    return StreamingResponse(sse(), media_type="text/event-stream")

@router.post("/encode")
def encode(payload: Dict[str, Any] = Body(...)):
    text = payload.get("text",""); return {"lane":"conscious","encoded":[ord(c) for c in text]}

@router.post("/decode")
def decode(payload: Dict[str, Any] = Body(...)):
    arr = payload.get("encoded", [])
    try: s = "".join(chr(int(x)) for x in arr)
    except Exception: s = ""
    return {"lane":"conscious","text": s}

@router.post("/infer")
def infer(payload: Dict[str, Any] = Body(...)):
    text = payload.get("text",""); return {"lane":"conscious","result": text.upper()}
