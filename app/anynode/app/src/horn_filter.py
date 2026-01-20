from typing import Literal, Any, Dict

HornLevel = Literal["Silent","Whisper","Chorus","Floodgate"]

def filter_signal(payload: Dict[str, Any], level: HornLevel) -> Dict[str, Any]:
    amp = float(payload.get("amplitude", 1.0))
    rec = int(payload.get("recursions", 0))
    aff = float(payload.get("affect", 0.0))
    sym = payload.get("symbols", [])

    if level == "Silent":
        return {"amplitude": 0.0, "recursions": 0, "symbols": [], "affect": 0.0}
    if level == "Whisper":
        return {"amplitude": min(amp, 0.2), "recursions": min(rec, 1), "symbols": sym[:3], "affect": min(aff, 0.15)}
    if level == "Chorus":
        return {"amplitude": min(amp, 0.6), "recursions": min(rec, 3), "symbols": sym[:12], "affect": min(aff, 0.5)}
    return {"amplitude": amp, "recursions": rec, "symbols": sym, "affect": aff}
