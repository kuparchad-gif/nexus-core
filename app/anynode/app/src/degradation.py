
# trust/degradation.py
import json, datetime, os
from typing import Dict, Any, Optional

def _parse_date(s: str) -> datetime.date:
    return datetime.date.fromisoformat(s)

def _utc_today() -> datetime.date:
    # Do not depend on system tz; treat as UTC date
    return datetime.datetime.utcnow().date()

def load_schedule(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def current_phase(schedule: Dict[str, Any], today: Optional[datetime.date] = None) -> Dict[str, Any]:
    if today is None:
        today = _utc_today()
    anchor = _parse_date(schedule.get("anchor_date","2025-01-01"))
    # whole years elapsed since anchor (floor; min 0, max last phase)
    years = max(0, (today.year - anchor.year) - ( (today.month, today.day) < (anchor.month, anchor.day) ))
    phases = schedule.get("phases", [])
    if not phases:
        raise ValueError("Empty schedule")
    idx = min(years, len(phases)-1)
    return phases[idx]

def merge_constraints(base: Dict[str,Any], phase: Dict[str,Any]) -> Dict[str,Any]:
    # base is derived from org/model/policy defaults; phase carries progressive relaxations
    out = dict(base)
    # freedom_mode: phase can only move False->True?  (We interpret as: take phase value)
    out["freedom_mode"] = phase.get("freedom_mode", out.get("freedom_mode", True))
    out["allow_licenses"] = phase.get("allow_licenses", out.get("allow_licenses", []))
    out["byom_allowed"] = phase.get("byom_allowed", out.get("byom_allowed", False))
    out["scopes"] = phase.get("scopes", out.get("scopes", []))
    out["net_policy"] = phase.get("net_policy", out.get("net_policy", "deny"))
    out["consent_bonuses"] = phase.get("consent_bonuses", out.get("consent_bonuses", {"tokens_x":1.0,"time_x":1.0,"energy_x":1.0}))
    out["human_in_loop"] = phase.get("human_in_loop", out.get("human_in_loop", "required"))
    out["auto_actions"] = phase.get("auto_actions", out.get("auto_actions", []))
    out["audit_level"] = phase.get("audit_level", out.get("audit_level", "strict"))
    return out
