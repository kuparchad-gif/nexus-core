from __future__ import annotations
import os
from typing import Dict, Optional

def check_local(proposal: str, route: str="local/planner", payload: Optional[Dict]=None, mode: Optional[str]=None) -> Dict:
    try:
        from Systems.engine.subconscious.viren.viren_service import get_viren
        v = get_viren()
        if mode:
            v.set_mode(mode)
        return v.check(proposal, route=route, payload=payload or {})
    except Exception as e:
        return {"error": f"local viren unavailable: {e}"}

def check_http(proposal: str, route: str="local/planner", payload: Optional[Dict]=None, mode: Optional[str]=None) -> Dict:
    url = os.environ.get("VIREN_URL","http://127.0.0.1:8031/check")
    try:
        import requests
        res = requests.post(url, json={"proposal": proposal, "route": route, "payload": payload or {}, "mode": mode})
        res.raise_for_status()
        return res.json()
    except Exception as e:
        return {"error": f"http viren unavailable: {e}"}
