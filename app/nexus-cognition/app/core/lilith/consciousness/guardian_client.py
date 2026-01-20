
# src/service/core/guardian_client.py
from __future__ import annotations
from typing import Dict, Any
from .logger import risk_logger, log_json

def flag_high_risk(proposal: Dict[str, Any], reason: str, severity: str = "high") -> None:
    payload = {"proposal": proposal, "reason": reason, "severity": severity}
    log_json(risk_logger, "guardian.flag_high_risk", payload)
    # TODO: integrate real Guardian channel (gRPC/HTTP/Firestore trigger) here.
