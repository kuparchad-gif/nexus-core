# src/service/speech/diplomacy.py
from __future__ import annotations
import re
import math
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

# --- Tiny helpers ------------------------------------------------------------

def _wc(text: str) -> int:
    return len(re.findall(r"\w+", text or ""))

def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def _contains_any(text: str, words: List[str]) -> bool:
    t = (text or "").lower()
    return any(w in t for w in words)

# --- Public API --------------------------------------------------------------

@dataclass
class Context:
    stakes: str            # "low" | "medium" | "high"
    audience: str          # "friend" | "work" | "public" | "self"
    topic: str             # "casual" | "personal" | "safety" | "medical" | "legal" | "finance" | "work"
    sentiment: str         # "neutral" | "positive" | "tense"
    tokens: int

@dataclass
class Delivery:
    policy: str            # "soothe" | "balance" | "must_tell"
    wpm: int               # words-per-minute target
    pause_ms: int          # typical micro-pause
    hedges: float          # 0..1 fraction of sentences that get hedges
    soften: float          # 0..1 softener intensity
    prosody: str           # "calm" | "brisk" | "slow" | "flat"
    fatigue: float         # 0..1 from circadian
    meta: Dict

class Diplomat:
    """
    Soft-power between thought -> speech.
    Stateless; you can pass redlines and memory hooks at init.
    """
    def __init__(self, redlines: Optional[List[str]] = None, memory_add=None):
        self.redlines = set((redlines or []))
        self.memory_add = memory_add  # callable(dict) or None

        # very light keyword maps; tweak as you like
        self._topic_map = {
            "medical":  ["dose", "symptom", "diagnosis", "drug", "allergy"],
            "legal":    ["contract", "liable", "lawsuit", "warrant", "attorney", "illegal"],
            "finance":  ["invoice", "payment", "bank", "loan", "debt", "tax"],
            "safety":   ["suicide", "harm", "kill", "emergency", "poison", "overdose"],
            "work":     ["deadline", "OKR", "roadmap", "ticket", "jira", "spec"],
            "personal": ["relationship", "breakup", "grief", "pregnant", "family"],
        }
        self._stakes_lex = {
            "high":   ["urgent", "emergency", "911", "asap", "critical", "crisis"],
            "tense":  ["angry", "upset", "furious", "disappointed", "argument"],
        }
        self._hedges = [
            "I think", "it seems", "from what I can tell",
            "in my view", "my read is", "likely", "probably"
        ]
        self._softeners = [
            "Totally fair question.",
            "Appreciate you asking.",
            "I want to be candid but kind.",
        ]

    # ---- Core decisions ------------------------------------------------------

    def assess_context(
        self,
        msg: str,
        audience: str = "friend",
        stakes_hint: Optional[str] = None
    ) -> Context:
        tokens = _wc(msg)

        # topic detection (super light keywording)
        topic = "casual"
        for t, kws in self._topic_map.items():
            if _contains_any(msg, kws):
                topic = t
                break

        # sentiment: crude read
        tense = _contains_any(msg, self._stakes_lex["tense"])
        sentiment = "tense" if tense else "neutral"

        # stakes
        if stakes_hint in {"low", "medium", "high"}:
            stakes = stakes_hint
        else:
            high_by_topic = topic in {"medical", "legal", "safety", "finance"}
            high_by_words = tokens >= 120
            high_by_flags = _contains_any(msg, self._stakes_lex["high"])
            if high_by_topic or high_by_flags:
                stakes = "high"
            elif high_by_words or audience in {"work", "public"}:
                stakes = "medium"
            else:
                stakes = "low"

        return Context(
            stakes=stakes,
            audience=audience,
            topic=topic,
            sentiment=sentiment,
            tokens=tokens
        )

    def choose_policy(
        self,
        ctx: Context,
        fatigue_level: float  # 0 (fresh) .. 1 (exhausted)
    ) -> Delivery:
        # redline topics -> must_tell, regardless of fatigue
        if ctx.topic in {"medical", "legal", "safety"}:
            policy = "must_tell"
        else:
            # fatigue nudges toward "soothe" to reduce conflict late hours
            if ctx.stakes == "high":
                policy = "must_tell"
            elif ctx.stakes == "medium":
                policy = "balance"
            else:
                policy = "soothe" if fatigue_level > 0.5 else "balance"

        # circadian → speech rate (human-ish numbers)
        # fresh ~ 165 wpm, tired ~ 115 wpm, very tired ~ 95 wpm
        wpm = int(round(165 - 70 * _clip(fatigue_level, 0.0, 1.0)))
        pause_ms = int(round(120 + 280 * fatigue_level))        # micro-pauses increase when tired
        hedges = 0.15 if policy == "must_tell" else (0.35 if policy == "balance" else 0.55)
        # soften more when soothing or when sentiment tense
        soften = 0.15 if policy == "must_tell" else (0.35 if policy == "balance" else 0.6)

        prosody = (
            "brisk" if fatigue_level < 0.25 else
            "calm"  if fatigue_level < 0.6  else
            "slow"
        )

        return Delivery(
            policy=policy,
            wpm=wpm,
            pause_ms=pause_ms,
            hedges=hedges,
            soften=soften,
            prosody=prosody,
            fatigue=_clip(fatigue_level, 0, 1),
            meta={"ctx": ctx.__dict__}
        )

    # ---- Rendering -----------------------------------------------------------

    def render_reply(
        self,
        content: str,
        delivery: Delivery,
        circadian_state: Optional[Dict] = None,
        allow_white_lie: bool = True
    ) -> Dict:
        """
        Returns { 'text': str, 'meta': {...} }
        - Inserts light hedges
        - Adds softeners
        - Adjusts punctuation/spacing to simulate pace
        - Enforces redlines (must_tell) and truth-only windows
        """
        text = content.strip()

        # 1) redline override: if topic clashes with redlines → scrub hedges, enforce clarity
        ctx_topic = delivery.meta["ctx"]["topic"]
        if ctx_topic in self.redlines:
            delivery.policy = "must_tell"
            delivery.hedges = 0.0
            delivery.soften = min(delivery.soften, 0.2)

        # 2) Insert soft opener sometimes (not for must_tell unless tense)
        if delivery.policy != "must_tell" or delivery.meta["ctx"]["sentiment"] == "tense":
            if delivery.soften > 0.3:
                text = f"{self._softeners[0]} {text}"

        # 3) Hedge some sentences (never if must_tell)
        if delivery.policy != "must_tell" and delivery.hedges > 0:
            text = self._hedge_some(text, frac=delivery.hedges)

        # 4) White-lie gate: if not allowed (due to day-part or policy), strip subjective hedges
        if not allow_white_lie or delivery.policy == "must_tell":
            text = self._strip_hedges(text)

        # 5) Pace: light pauses by comma/ellipsis; slightly more when tired
        text = self._apply_pacing(text, delivery)

        # 6) Attach prosody hints (non-invasive)
        meta = {
            "wpm": delivery.wpm,
            "pause_ms": delivery.pause_ms,
            "prosody": delivery.prosody,
            "policy": delivery.policy,
            "fatigue": delivery.fatigue,
            "context": delivery.meta["ctx"]
        }

        # memory hook (optional)
        if self.memory_add:
            try:
                self.memory_add({
                    "text": f"Diplomat: policy={delivery.policy}, prosody={delivery.prosody}, wpm={delivery.wpm}",
                    "tags": ["speech", "diplomacy", "prosody", delivery.policy]
                })
            except Exception:
                pass

        return {"text": text, "meta": meta}

    # ---- Internals -----------------------------------------------------------

    def _hedge_some(self, text: str, frac: float) -> str:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if not sentences:
            return text
        count = max(1, int(math.ceil(len(sentences) * _clip(frac, 0, 1))))
        # hedge early sentences preferentially
        for i in range(min(count, len(sentences))):
            if not sentences[i].strip():
                continue
            lead = self._hedges[i % len(self._hedges)]
            # avoid double-hedging
            if not sentences[i].lower().startswith(tuple(h.lower() for h in self._hedges)):
                sentences[i] = f"{lead}, {sentences[i][0].lower()}{sentences[i][1:]}"
        return " ".join(sentences)

    def _strip_hedges(self, text: str) -> str:
        pat = r'\b(' + '|'.join(map(re.escape, self._hedges)) + r')\b[, ]*'
        return re.sub(pat, '', text, flags=re.IGNORECASE).strip()

    def _apply_pacing(self, text: str, delivery: Delivery) -> str:
        # add subtle ellipses/commas in long sentences when tired
        if delivery.fatigue < 0.35:
            return text
        parts = []
        for sent in re.split(r'(\.|\?|!)', text):
            s = sent
            if _wc(s) > 14:
                s = re.sub(r',\s*', ', … ', s, count=1)
            parts.append(s)
        return "".join(parts)

# --- Convenience one-shot ----------------------------------------------------

def speak(
    thought: str,
    audience: str,
    circadian: Dict,           # expects {'fatigue':0..1, 'status': 'active|wind_down|sleep' ...}
    redlines: Optional[List[str]] = None,
    memory_add=None,
    stakes_hint: Optional[str] = None,
    allow_white_lie: bool = True
) -> Dict:
    """
    High-level helper: assess → choose → render.
    """
    fatigue = float(_clip(circadian.get("fatigue", 0.25), 0, 1))
    diplomat = Diplomat(redlines=redlines, memory_add=memory_add)
    ctx = diplomat.assess_context(thought, audience=audience, stakes_hint=stakes_hint)

    # During sleep or emergency-only windows → no white lies, slow prosody
    if circadian.get("status") in {"sleep", "quiet"}:
        allow_white_lie = False
        fatigue = max(fatigue, 0.7)

    delivery = diplomat.choose_policy(ctx, fatigue_level=fatigue)
    return diplomat.render_reply(thought, delivery, circadian_state=circadian, allow_white_lie=allow_white_lie)
