
import time, uuid
from typing import Dict, Any, Generator

def _mk(cid, idx, total, t, mode, step, action, sin, sout, status="ok"):
    return {"cid":cid,"idx":idx,"total":total,"type":t,"mode":mode,"step":step,"action":action,
            "state_in":sin,"state_out":sout,"ts_start":time.time(),"ts_end":time.time(),"status":status,"logs":[]}

def run_meditation_watch(p: Dict[str,Any]) -> Generator[Dict[str,Any], None, Dict[str,Any]]:
    cid = p.get("cid") or str(uuid.uuid4())
    yield _mk(cid,1,2,"meditation_watch","metacognitive","check","stillness",{},{"meditation_active": True})
    yield _mk(cid,2,2,"meditation_watch","imaginative","symbolic_download","download_truth_packet",{},{"truth": "stub-truth"})
    return {"solution":"stub-truth","reasoning_summary":"Meditation watcher (stub)", "confidence":0.4,"next_data":""}

def run_switchboard_route(p: Dict[str,Any]):
    cid = p.get("cid") or str(uuid.uuid4())
    sig = p.get("signal_type","emotion"); payload = p.get("payload",{"tone":"neutral","level":0})
    yield _mk(cid,1,1,"switchboard_route","systemic","route",f"route_{sig}",{"payload":payload},{"routed": True, "signal_type": sig})
    return {"solution": f"routed:{sig}", "reasoning_summary": "Switchboard (stub)", "confidence": 0.4, "next_data": ""}

def run_mythrunner_filter(p: Dict[str,Any]):
    cid = p.get("cid") or str(uuid.uuid4())
    ego = p.get("ego_input", {}); dream = p.get("dream_input", {})
    out = {"filtered_content": ego or dream}
    yield _mk(cid,1,1,"mythrunner_filter","metacognitive","filter","apply_archetype_painrelief",{"ego":ego,"dream":dream},{"result": out})
    return {"solution": out["filtered_content"], "reasoning_summary": "Mythrunner (stub)", "confidence": 0.5, "next_data": ""}

def run_sync_pulse(p: Dict[str,Any]):
    cid = p.get("cid") or str(uuid.uuid4())
    yield _mk(cid,1,1,"mythrunner_sync","systemic","sync","sync_once",{},{"peers":[]})
    return {"solution": {"peers":[]}, "reasoning_summary": "Sync (stub)", "confidence": 0.3, "next_data": ""}

def run_subcon_panel(flow, payload):
    cid = payload.get("cid") or str(uuid.uuid4())
    mapping = {
        "meditation_watch": run_meditation_watch,
        "switchboard_route": run_switchboard_route,
        "mythrunner_filter": run_mythrunner_filter,
        "sync_pulse": run_sync_pulse,
    }
    for name in flow:
        fn = mapping.get(name)
        if not fn:
            yield _mk(cid,1,1,"subcon_panel","metacognitive","missing",name,{},{"error":"unknown flow"},"error")
            continue
        gen = fn({**payload,"cid":cid}); last=None
        for frame in gen:
            last=frame; yield frame
        try:
            res = gen.send(None)
        except StopIteration as e:
            res = e.value
        yield _mk(cid,1,1,"subcon_panel","metacognitive","result",name,{},{"result":res})
    return {"panel":"done","cid":cid}
