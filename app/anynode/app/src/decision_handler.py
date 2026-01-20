
# src/service/core/decision_handler.py
from __future__ import annotations
from typing import Dict, Any
from uuid import uuid4
from datetime import datetime, timezone
from .vendors import LocalServices, RemoteAPI
from .logger import council_logger, log_json
from .guardian_client import flag_high_risk

RISKY_PROPOSALS = {"proceed_remote", "disable_guardrails", "escalate_privileges"}

class DecisionHandler:
    def __init__(self):
        self.local = LocalServices()
        self.remote = RemoteAPI()

    def handle(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        # decorate decision
        decision_id = str(uuid4())
        ts = datetime.now(timezone.utc).isoformat()
        decision["meta"] = {"uuid": decision_id, "ts": ts}
        proposal = (decision.get("proposal") or "").lower()

        # log inbound
        log_json(council_logger, "decision.inbound", {"decision": decision})

        # risk gate
        if proposal in RISKY_PROPOSALS:
            flag_high_risk(decision, reason=f"Proposal '{proposal}' is in risky set")

        route = decision.get("route", "local/planner")
        payload = decision.get("payload", {})

        if route.startswith("local/"):
            service = route.split("/", 1)[1]
            svc_name, result = self.local.execute(service, payload)
            out = {"decision_id": decision_id, "proposal": proposal, "route": route, "service": svc_name, "result": result}
        elif route.startswith("remote/"):
            url = route.split("/", 1)[1]
            result = self.remote.call(url, payload)
            out = {"decision_id": decision_id, "proposal": proposal, "route": route, "service": "remote", "result": result}
        else:
            raise ValueError(f"Unknown route: {route}")

        # log outbound
        log_json(council_logger, "decision.outbound", {"out": out})
        return out
