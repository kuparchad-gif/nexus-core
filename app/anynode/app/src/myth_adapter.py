
from typing import Dict, Any, Generator
import time, uuid

def run_myth(ego_input: Dict[str,Any], dream_input: Dict[str,Any], filters_enabled: bool, ascension_mode: bool, cid: str|None=None):
    cid = cid or str(uuid.uuid4()); idx=0; total=2; t="myth_engine"; mode="metacognitive"
    state = {"ego_input":ego_input,"dream_input":dream_input}
    def frame(step, action, sout, status="ok"):
        return {"cid":cid,"idx":min(2,(1 if step!='filter' else 2)),"total":total,"type":t,"mode":mode,"step":step,"action":action,
                "state_in":state,"state_out":sout,"ts_start":time.time(),"ts_end":time.time(),"status":status,"logs":[]}
    yield frame("init","construct_mythrunner",{"have_mythrunner": False})
    if ascension_mode:
        out = {"filtered_content": ego_input or dream_input, "note":"ascension_bypass"}
        yield frame("filter","bypass_filter",{"result":out})
        return {"solution": out["filtered_content"], "reasoning_summary": "Ascension mode bypass", "confidence": 0.7, "next_data": ""}
    if not filters_enabled:
        out = {"filtered_content": ego_input or dream_input, "note":"filter_disabled"}
        yield frame("filter","disabled_or_missing",{"result":out})
        return {"solution": out["filtered_content"], "reasoning_summary": "Filter disabled", "confidence": 0.5, "next_data": ""}
    # default stub "filter"
    out = {"filtered_content": ego_input or dream_input, "note":"stub_filter"}
    yield frame("filter","apply_archetype_painrelief",{"result":out})
    return {"solution": out["filtered_content"], "reasoning_summary": "Myth filter (stub)", "confidence": 0.6, "next_data": ""}
