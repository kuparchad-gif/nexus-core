# nim_svc.py
from fastapi import FastAPI, Response, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import json, os, time, uuid
from pathlib import Path

from niv_stream import load_template, run_template

app = FastAPI(title="nim_svc (NIV Streaming)")

TEMPLATES_ROOT = os.getenv("NIV_TEMPLATES_ROOT", "/mnt/data/nexus_thought_templates/json")

class RunBody(BaseModel):
    template_type: str
    input: Optional[Dict[str, Any]] = None
    template_path: Optional[str] = None
    cid: Optional[str] = None

@app.get("/health")
def health():
    return {"ok": True, "ts": time.time(), "templates_root": TEMPLATES_ROOT}

@app.get("/templates", response_class=PlainTextResponse)
def list_templates():
    p = Path(TEMPLATES_ROOT)
    if not p.exists():
        return "No template root found."
    files = sorted([x.name for x in p.glob("*.json")])
    return "\n".join(files)

@app.post("/run")
def run(body: RunBody):
    # Choose a template file by path or by type under root
    if body.template_path:
        tpath = body.template_path
    else:
        tpath = str(Path(TEMPLATES_ROOT) / f"{body.template_type}.json")
    template = load_template(tpath)

    def sse_gen():
        # SSE prelude
        yield "retry: 500\n\n"
        try:
            gen = run_template(template, input_overrides=body.input, cid=body.cid or str(uuid.uuid4()))
            final_out = None
            for frame in gen:
                payload = json.dumps(frame, ensure_ascii=False)
                yield f"event: frame\ndata: {payload}\n\n"
            try:
                final_out = gen.send(None)  # Retrieve StopIteration.value (not actually works)
            except StopIteration as e:
                final_out = e.value
            if final_out is None:
                # reconstruct final from last frame
                final_out = {"result": frame.get("state_out",{}).get("result",""), "confidence": frame.get("state_out",{}).get("confidence",0.5)}
            yield f"event: done\ndata: {json.dumps(final_out, ensure_ascii=False)}\n\n"
        except Exception as e:
            err = {"error": str(e)}
            yield f"event: error\ndata: {json.dumps(err, ensure_ascii=False)}\n\n"
    return Response(sse_gen(), media_type="text/event-stream")



from fastapi import StreamingResponse

class PanelBody(BaseModel):
    riddle: str
    clues: list[str] | None = None
    methods: list[str] | None = None
    cid: Optional[str] = None
    templates_root: Optional[str] = None

@app.post("/panel")
def panel(body: PanelBody):
    """
    Stream a multi-method reasoning panel.
    For each method, we load {templates_root}/{method}.json and run it with standardized input.
    """
    root = Path(body.templates_root or os.getenv("NIV_TEMPLATES_ROOT", TEMPLATES_ROOT))
    methods = body.methods or [
        "deductive_reasoning","inductive_reasoning","abductive_reasoning",
        "analogy_based_thinking","lateral_thinking","systems_thinking",
        "first_principles","game_theory","pattern_recognition",
        "technological_profiling","reverse_engineering","red_teaming"
    ]
    cid = body.cid or str(uuid.uuid4())

    def sse():
        yield "retry: 500\n\n"
        summary = []
        for m in methods:
            tpath = root / f"{m}.json"
            if not tpath.exists():
                payload = {"method": m, "error": f"template not found: {tpath}"}
                yield f"event: method_error\ndata: {json.dumps(payload)}\n\n"
                continue
            template = load_template(str(tpath))
            input_override = dict(template.get("input", {}))
            # Map standardized inputs
            input_override.update({
                "riddle": body.riddle,
                "clues": body.clues or [],
                "context": body.riddle
            })
            last_frame = None
            frames = run_template(template, input_overrides=input_override, cid=cid)
            try:
                for frame in frames:
                    frame["method"] = m
                    yield "event: frame\n" + "data: " + json.dumps(frame, ensure_ascii=False) + "\n\n"
                    last_frame = frame
            except StopIteration as e:
                # not typically reached in Python generator consumption
                pass
            # Assemble per-method result from last_frame's state_out
            if last_frame:
                so = last_frame.get("state_out", {})
                summary.append({
                    "method": m,
                    "solution": so.get("solution", so.get("result","")),
                    "confidence": so.get("confidence", 0.5),
                    "notes": so.get("reasoning_summary","")
                })
        # Build a naive consensus
        # Count identical non-empty solutions
        consensus = ""
        best = 0
        counts = {}
        for row in summary:
            sol = (row.get("solution") or "").strip().lower()
            if not sol:
                continue
            counts[sol] = counts.get(sol, 0) + 1
            if counts[sol] > best:
                best = counts[sol]; consensus = sol
        # Recommend method order: start with pattern -> first_principles -> deductive -> red_team
        rec_order = ["pattern_recognition","first_principles","deductive_reasoning","red_teaming"]
        panel = {"cid": cid, "results": summary, "consensus": consensus, "recommendation": rec_order}
        yield "event: panel_done\n" + "data: " + json.dumps(panel, ensure_ascii=False) + "\n\n"
    return StreamingResponse(sse(), media_type="text/event-stream")
