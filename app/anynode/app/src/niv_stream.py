# niv_stream.py
import json, time, uuid, traceback
from pathlib import Path
from typing import Dict, Any, List, Generator, Optional

# Try to import your engine; if not available, we fallback to no-op transforms.
ENGINE = None
try:
    import thought_processing_engine as TPE  # must exist in PYTHONPATH or same folder
    ENGINE = TPE
except Exception:
    ENGINE = None

# Map engine_mode -> callable in engine (best-effort)
ENGINE_MODE_FUNCS = {
    "logical": getattr(ENGINE, "logical_reasoning", None) if ENGINE else None,
    "imaginative": getattr(ENGINE, "imagination", None) if ENGINE else None,
    "metacognitive": getattr(ENGINE, "metacognition", None) if ENGINE else None,
    "systemic": getattr(ENGINE, "systems_thinking", None) if ENGINE else None,
}

def load_template(path: str) -> Dict[str, Any]:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    # Validate minimal fields
    for k in ["id","type","input","subprocesses","output"]:
        if k not in data:
            raise ValueError(f"Template missing key: {k}")
    return data

def run_template(template: Dict[str, Any], input_overrides: Optional[Dict[str, Any]]=None, cid: Optional[str]=None) -> Generator[Dict[str, Any], None, Dict[str, Any]]:
    """
    Execute subprocesses, yielding NIV frames suitable for SSE.
    Returns final output dict as generator return value (StopIteration.value).
    """
    state = dict(template.get("input", {}))
    if input_overrides:
        state.update(input_overrides)
    steps = template.get("subprocesses", [])
    total = len(steps)
    out_spec = template.get("output", {})
    cid = cid or str(uuid.uuid4())

    for idx, sp in enumerate(steps, start=1):
        mode = sp.get("engine_mode") or "logical"
        step = sp.get("step")
        action = sp.get("action")
        ts_start = time.time()
        logs: List[str] = []
        status = "ok"
        state_out = dict(state)
        try:
            fn = ENGINE_MODE_FUNCS.get(mode)
            if callable(fn):
                # Unified adapter: fn(step=..., action=..., state=...) -> dict
                res = fn(step=step, action=action, state=state)
                if isinstance(res, dict):
                    state_out.update(res)
                else:
                    # if engine returns text or list, attach under 'result'
                    state_out["result"] = res
            else:
                # Fallback: no engine — simulate transformation
                state_out["last_step"] = step
                state_out["last_action"] = action
            # Optional: minimal normalization for common fields
            if "confidence" not in state_out:
                state_out["confidence"] = state_out.get("confidence", 0.5)
        except Exception as e:
            status = "error"
            logs.append(f"EXC: {e}")
            logs.append(traceback.format_exc())
        ts_end = time.time()
        frame = {
            "cid": cid,
            "idx": idx,
            "total": total,
            "type": template.get("type"),
            "mode": mode,
            "step": step,
            "action": action,
            "state_in": state,
            "state_out": state_out,
            "ts_start": ts_start,
            "ts_end": ts_end,
            "status": status,
            "logs": logs,
        }
        # Advance state for next step
        state = dict(state_out)
        yield frame

    # Finalize according to out_spec shape
    final = {}
    # If engine provides explicit fields, pick them; else map best-effort
    for k in out_spec.keys():
        final[k] = state.get(k, state.get("result", "" if isinstance(out_spec[k], str) else out_spec[k]))
    if "confidence" in out_spec:
        final["confidence"] = state.get("confidence", 0.5)
    return final

