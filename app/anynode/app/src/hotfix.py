import os
def flag_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None: return default
    return v.strip().lower() in ("1","true","yes","y","on")
def flag_float(name: str, default: float) -> float:
    v = os.getenv(name)
    try:
        return float(v) if v is not None else default
    except Exception:
        return default
def force_mode() -> str | None:
    m = os.getenv("HOTFIX_FORCE_MODE", "").strip().lower()
    return m if m in ("pre","post","off") else None
def defaults():
    return {
        "voidal_enabled": flag_bool("HOTFIX_VOIDAL", False),
        "toroid_gain":    flag_float("HOTFIX_TOROID_GAIN", 0.0),
        "horn_gain":      flag_float("HOTFIX_HORN_GAIN", 1.02),
        "badprob_thresh": flag_float("HOTFIX_BADPROB_THRESH", 0.25)
    }
