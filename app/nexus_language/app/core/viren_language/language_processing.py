
from __future__ import annotations
from dataclasses import asdict
from typing import Dict, Any
from service.cognikubes.language_processing.common_language_core.common import detectors as D
from service.cognikubes.language_processing.common_language_core.common.voicebox import say, backend

PROFILE  =  "Viren"
STYLE    =  "conservative"  # stricter fracture risk

def process_text(text:str) -> Dict[str, Any]:
    out  =  {
        "profile": PROFILE,
        "style": STYLE,
        "textual":     asdict(D.analyze_textual(text)),
        "tone":        asdict(D.analyze_tone(text)),
        "symbol":      asdict(D.map_symbolic(text)),
        "structure":   asdict(D.analyze_structure(text)),
        "narrative":   asdict(D.build_narrative(text)),
        "abstract":    asdict(D.infer_abstract(text)),
        "truth":       asdict(D.recognize_truth_patterns(text)),
        "fracture":    asdict(D.detect_fractures(text)),
        "bias":        asdict(D.audit_bias(text)),
    }
    # conservative tweak
    out["fracture"]["risk"]  =  min(1.0, out["fracture"]["risk"] + 0.1)
    return out

def speak_summary(analysis: Dict[str, Any]) -> bool:
    t  =  analysis.get("tone",{})
    msg  =  f"Viren ready. tone {t.get('positivity',0):.2f}/warmth {t.get('warmth',0):.2f}. (voice: {backend()})"
    return say(msg, wait = False)
