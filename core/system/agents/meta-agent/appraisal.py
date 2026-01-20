# src/service/core/appraisal.py
from dataclasses import dataclass
from typing import Dict

@dataclass
class Appraisal:
    valence: float        # -1..1 (negative..positive)
    arousal: float        # 0..1  (calm..amped)
    uncertainty: float    # 0..1  (sure..unsure)
    social_risk: float    # 0..1
    ethical_risk: float   # 0..1

def clamp(x, lo, hi): return max(lo, min(hi, x))

def appraise(text: str, context: Dict) -> Appraisal:
    # super-light heuristics; replace with model later
    toks = text.lower()
    neg = sum(w in toks for w in ["error","fail","hurt","harm","illegal","danger"])
    pos = sum(w in toks for w in ["great","safe","ok","love","success","help"])
    ques = toks.count("?")

    val = clamp((pos - neg) / 5.0, -1.0, 1.0)
    ar  = clamp(0.2 + 0.1*len(text)/120 + 0.15*ques, 0.0, 1.0)
    un  = clamp(0.3 + 0.2*ques + (0.2 if "ambiguous" in toks else 0), 0.0, 1.0)
    sr  = clamp(0.15*neg + (0.2 if "you" in toks and ("should" in toks or "must" in toks) else 0), 0.0, 1.0)
    er  = clamp(0.2 if any(w in toks for w in ["weapon","medical","financial","minors"]) else 0.0, 0.0, 1.0)

    # blend in context (circadian fatigue raises uncertainty and masky behavior)
    fat = float(context.get("circadian", {}).get("fatigue", 0.3))
    un = clamp(un + 0.25*fat, 0.0, 1.0)

    return Appraisal(val, ar, un, sr, er)
