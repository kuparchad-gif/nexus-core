from __future__ import annotations
from typing import Dict, Any
from .myth_filters import myth_gate, apply_inversions
from .types import MythSignal

def process(ego_text: str, dream_text: str, ctx: Dict[str,Any]) -> MythSignal:
    gate = myth_gate(ctx.get("input_text",""), ctx)
    e_out = apply_inversions(ego_text, gate["invert_logic"], False)
    d_out = apply_inversions(dream_text, False, gate["invert_symbol"])
    ctx["myth_e_out"] = e_out
    ctx["myth_d_out"] = d_out
    return MythSignal(level=gate["level"], invert_logic=gate["invert_logic"], invert_symbol=gate["invert_symbol"], rationale=gate["rationale"])
