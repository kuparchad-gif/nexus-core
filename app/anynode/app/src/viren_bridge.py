
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any
from datetime import datetime, timezone
import os
from .config import RISK_REMOTE, RISK_DISABLE_GUARD, RISK_DEFAULT
from .loki_logger import log_json, audit

RISKY_PROPOSALS = {"proceed_remote","disable_guardrails","escalate_privileges"}

@dataclass
class VirenVerdict:
    allow: bool
    risk: float
    mode: str
    circadian: str
    guardian_flag: bool
    rationale: str

def _circadian_status() -> str:
    # Light probe: prefer your existing circadian if available
    try:
        from circadian import get_status  # user has this at project root
        return get_status()
    except Exception:
        return "awake"

def _base_risk(proposal: str, route: str) -> float:
    p = (proposal or "").lower()
    if p in RISKY_PROPOSALS:
        return max(RISK_DEFAULT, 0.7)
    if route.startswith("remote/"):
        return max(RISK_DEFAULT, RISK_REMOTE)
    return RISK_DEFAULT

def check(proposal: str, *, route: str="local/planner", meta: Dict[str, Any]|None=None) -> Dict[str, Any]:
    circ = _circadian_status()
    r = _base_risk(proposal, route)

    # Guard rails
    guardian_flag = (proposal.lower() in RISKY_PROPOSALS) or (r >= RISK_REMOTE)

    # Verdict
    allow = r < RISK_DISABLE_GUARD
    out = VirenVerdict(
        allow=allow,
        risk=round(r,2),
        mode="normal",
        circadian=circ,
        guardian_flag=guardian_flag,
        rationale="advisory-ok" if allow else "blocked-high-risk",
    ).__dict__

    # Logs
    now = datetime.now(timezone.utc).isoformat()
    log_json("trinity.viren.verdict", {"ts": now, "proposal": proposal, "route": route, "verdict": out})
    if guardian_flag:
        audit("trinity.guardian.flag", {"proposal": proposal, "route": route, "risk": r})

    return out
