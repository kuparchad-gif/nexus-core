from __future__ import annotations
import os, ast

def safe_eval(expr: str):
    """Hardened eval wrapper.
    Default behavior preserved unless HARDEN_EVAL=1 is set in environment.
    When HARDEN_EVAL=1, only Python literals/containers are allowed (ast.literal_eval).
    """
    harden = os.getenv("HARDEN_EVAL", "0") == "1"
    if harden:
        return ast.literal_eval(expr)
    # Legacy behavior (to avoid breaking changes). Users can opt-in to hardening.
    return eval(expr)
