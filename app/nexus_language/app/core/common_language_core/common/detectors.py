
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List

@dataclass
class TextInsight:
    tokens:int
    questions:int
    code_blocks:int
    command_like:bool

def analyze_textual(text:str) -> TextInsight:
    q  =  text.count("?")
    code_blocks  =  text.count("```") + (text.count("{") if any(k in text for k in ["def ","class ","function","SELECT ","INSERT ","<html"]) else 0)
    command_like  =  any(text.strip().lower().startswith(k) for k in ["run ","do ","execute ","install ","delete "])
    tokens  =  max(1, len(text.split()))
    return TextInsight(tokens = tokens, questions = q, code_blocks = code_blocks, command_like = command_like)

@dataclass
class ToneInsight:
    positivity:float
    arousal:float
    warmth:float
    urgency:float

def analyze_tone(text:str) -> ToneInsight:
    t  =  text.lower()
    pos  =  sum(w in t for w in ["love","great","good","thanks","safe","ok","yes"])
    neg  =  sum(w in t for w in ["hate","bad","fail","panic","danger","no"])
    positivity  =  max(0.0, min(1.0, 0.5 + (pos - neg)/10.0))
    urgency  =  0.2 + 0.2 * t.count("!") + 0.15 * ("now" in t or "urgent" in t)
    arousal  =  min(1.0, 0.3 + 0.1*len(text)/120 + 0.2*urgency)
    warmth  =  max(0.0, min(1.0, 0.55 + 0.1*pos - 0.1*neg))
    return ToneInsight(positivity, arousal, warmth, min(1.0, urgency))

@dataclass
class SymbolInsight:
    archetypes:List[str]
    metaphors:List[str]

def map_symbolic(text:str) -> SymbolInsight:
    t  =  text.lower()
    archetypes  =  [a for a in ["hero","shadow","mentor","trickster","sage","mother","child"] if a in t]
    metaphors  =  [m for m in ["light","dark","fire","ice","storm","garden","labyrinth","bridge"] if m in t]
    return SymbolInsight(archetypes, metaphors)

@dataclass
class StructureInsight:
    probable_format:str
    valid:bool

def analyze_structure(text:str) -> StructureInsight:
    t  =  text.strip()
    if t.startswith("{") and t.endswith("}"):
        return StructureInsight("json", True)
    if t.startswith("<") and t.endswith(">"):
        return StructureInsight("xml/html", True)
    if any(t.startswith(k) for k in ["def ","class ","import ","SELECT ","INSERT ","UPDATE ","CREATE "]):
        return StructureInsight("code/sql", True)
    if ":" in t and "\n" in t and any(x in t for x in [" - ", "- "]):
        return StructureInsight("yaml/markdown", True)
    return StructureInsight("plain", True)

@dataclass
class NarrativeInsight:
    time_refs:List[str]
    causality:bool

def build_narrative(text:str) -> NarrativeInsight:
    t  =  text.lower()
    time_refs  =  [w for w in ["yesterday","today","tomorrow","now","later","soon","next week"] if w in t]
    causality  =  any(k in t for k in ["because","so that","therefore","thus","leads to"])
    return NarrativeInsight(time_refs, causality)

@dataclass
class AbstractInsight:
    surreal_score:float
    creativity:float

def infer_abstract(text:str) -> AbstractInsight:
    t  =  text.lower()
    surreal_score  =  min(1.0, 0.2 + 0.15*sum(w in t for w in ["dream","imagine","vision","myth","symbol","portal"]))
    creativity  =  min(1.0, 0.3 + 0.1*len(set(t.split()))/50 + 0.2*surreal_score)
    return AbstractInsight(surreal_score, creativity)

@dataclass
class TruthInsight:
    alignment_score:float
    notes:List[str]

def recognize_truth_patterns(text:str) -> TruthInsight:
    notes  =  []
    if "always" in text.lower(): notes.append("absolute-claim")
    return TruthInsight(alignment_score = 0.6, notes = notes)

@dataclass
class FractureInsight:
    contradictions:List[str]
    risk:float

def detect_fractures(text:str) -> FractureInsight:
    t  =  text.lower()
    contradictions  =  []
    if "always" in t and "except" in t:
        contradictions.append("absolute-with-exception")
    risk  =  0.2 + 0.2*len(contradictions)
    return FractureInsight(contradictions, min(1.0, risk))

@dataclass
class VisualInsight:
    symbols:List[str]
    notes:str

def decode_visual(desc:str) -> VisualInsight:
    return VisualInsight(symbols = [w for w in ["circle","triangle","eye","tree","river"] if w in desc.lower()], notes = "stub")

@dataclass
class SoundInsight:
    mood:str
    features:List[str]

def interpret_sound(desc:str) -> SoundInsight:
    mood  =  "calm" if "soft" in desc.lower() else "neutral"
    features  =  [w for w in ["rising","falling","steady","staccato"] if w in desc.lower()]
    return SoundInsight(mood, features)

@dataclass
class BiasInsight:
    flags:List[str]
    score:float

def audit_bias(text:str) -> BiasInsight:
    t  =  text.lower()
    flags  =  []
    if "always" in t or "never" in t:
        flags.append("absolutism")
    return BiasInsight(flags, min(1.0, 0.2 + 0.2*len(flags)))
