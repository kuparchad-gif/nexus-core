
# src/service/core/vendors.py
from __future__ import annotations
from typing import Dict, Any, Tuple
from time import sleep
import random

class LocalServices:
    def __init__(self):
        # lazy binding; replace with real imports when services are ready
        pass

    def planner(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        # placeholder planning
        goals = payload.get("goals", ["stabilize", "optimize"])
        return {"plan": [{"step": "analyze_state"}, {"step": "schedule_decay_job"}, {"step": "report"}],
                "goals": goals}

    def lillith_core(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        # pretend to run a core action
        return {"result": "ok", "action": payload.get("action", "noop")}

    def guardian_review(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        # pretend guardian offers a verdict
        return {"verdict": "approve_with_conditions", "conditions": ["log", "uuid", "cooldown_5m"]}

    def execute(self, service: str, payload: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        svc = service.lower()
        if svc == "planner":
            return ("planner", self.planner(payload))
        if svc == "lillith":
            return ("lillith", self.lillith_core(payload))
        if svc == "guardian":
            return ("guardian", self.guardian_review(payload))
        raise ValueError(f"Unknown local service: {service}")

class RemoteAPI:
    def call(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        # Simulation: latency + mock response
        sleep(random.uniform(0.05, 0.2))
        return {"remote_url": url, "status": 200, "echo": payload}
