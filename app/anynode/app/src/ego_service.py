from __future__ import annotations
from typing import Dict, Any
import random
from .types import EgoVerdict

JUNG_ARCHETYPES = ["Sage","Ruler","Caregiver","Everyman","Hero","Outlaw","Explorer","Creator","Innocent","Jester","Lover","Magician"]
RISKY = {"proceed_remote","disable_guardrails","escalate_privileges"}

def judge(proposal: str, route: str, ctx: Dict[str, Any]) -> EgoVerdict:
    base = 0.2
    if proposal.lower() in RISKY: base = max(base, 0.7)
    if route.startswith("remote/"): base = max(base, 0.65)
    if ctx.get("circadian")== "asleep": base = min(1.0, base + 0.1)
    notes = []
    if base >= 0.65: notes.append("guardian-flag")
    return EgoVerdict(allow = base < 0.9, risk = round(base,2), notes = notes, archetype = random.choice(JUNG_ARCHETYPES), rationale = "ego-archetype-eval")
