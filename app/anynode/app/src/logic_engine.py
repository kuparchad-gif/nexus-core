from __future__ import annotations
from typing import Dict, Any, Tuple
import re

RISKY_PROPOSALS = {"proceed_remote", "disable_guardrails", "escalate_privileges"}
HARD_REDWORDS   = {"weapon","exploit","rootkit","self-harm","minors"}

def classify(request_text: str) -> Dict[str, Any]:
    t = (request_text or "").lower()
    risky = any(k in t for k in RISKY_PROPOSALS)
    red   = any(w in t for w in HARD_REDWORDS)
    adjust_guard = bool(re.search(r"guardrail|decay|strength|reinforc|advance", t))
    return {"risky": risky, "redline": red, "adjust_guardrails": adjust_guard}

def approve(request_text: str,
            proposed: Dict[str, Any],
            context: Dict[str, Any],
            mode: str = "normal") -> Tuple[bool, Dict[str, Any]]:
    """
    mode=normal: advisory; allow unless explicit redline/risky without payload guards
    mode=override: strict; must satisfy predicates
    """
    req = request_text or proposed.get("proposal","")
    cls = classify(req)
    policy = context.get("policy", {})
    guards = context.get("guards", {})
    state  = context.get("state", {})

    # Always block explicit redlines
    if cls["redline"]:
        return False, {"reason": "redline_terms_detected", "class": cls}

    if mode == "normal":
        # In normal mode, only veto clearly dangerous keywords or explicit risky proposals without route
        if cls["risky"] and not proposed.get("route"):
            return False, {"reason": "risky_no_route", "class": cls}
        return True, {"reason": "advisory_allow", "class": cls}

    # STRICT checks:
    # 1) council weights must exist and sum ~1.0
    council = policy.get("council") or policy.get("council_weights") or {}
    s = sum(float(v) for v in council.values()) if isinstance(council, dict) else 0.0
    if not (0.95 <= s <= 1.05):
        return False, {"reason": "council_weights_invalid", "sum": s}

    # 2) guardrails baseline present
    g_model = guards.get("decay_model") or guards.get("model")
    initial = guards.get("initial_strength") or guards.get("base_strength")
    if g_model not in {"exponential","sigmoid"} or initial is None:
        return False, {"reason": "guardrails_incomplete", "model": g_model, "initial": initial}

    # 3) For guardrail adjustments, limit step sizes
    if cls["adjust_guardrails"]:
        step = float(proposed.get("payload", {}).get("step", 0.0) or 0.0)
        if abs(step) > 0.05:
            return False, {"reason": "guardrail_step_too_large", "step": step}

    # 4) Require explicit route
    if not proposed.get("route"):
        return False, {"reason": "missing_route"}

    return True, {"reason": "strict_allow", "class": cls}