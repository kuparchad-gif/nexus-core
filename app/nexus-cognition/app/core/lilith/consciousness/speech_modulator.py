# src/service/core/speech_modulator.py
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class SpeechParams:
    rate: float            # 0.7–1.3 (1.0 = neutral)
    pitch: float           # 0.9–1.1
    pause_ms: int          # avg inter-phrase pause
    jitter_ms: int         # random +/- on pauses
    disfluency_prob: float # 0.0–0.15 insert “uh”, restarts
    warmth: float          # 0.0–1.0 (for TTS prosody knobs)

def _clamp(v, lo, hi): return max(lo, min(hi, v))

def from_circadian(circ: dict, guardrail_strength: float = 1.0) -> SpeechParams:
    """
    circ example:
      {
        "status": "asleep|waking|morning|afternoon|evening|tired|exhausted|alert|focused|relaxed",
        "phase":  float 0..1 (optional),
        "activity": "read|plan|chat|…"
      }
    guardrail_strength: 0..1 (older -> lower -> more variance)
    """
    status = (circ or {}).get("status", "afternoon")
    # baseline
    rate    = 1.0
    pitch   = 1.0
    pause   = 220
    jitter  = 60
    disfl   = 0.02
    warmth  = 0.55

    if status in ("asleep", "exhausted"):
        rate, pitch, pause, jitter, disfl, warmth = 0.78, 0.95, 380, 110, 0.06, 0.48
    elif status in ("tired", "evening"):
        rate, pitch, pause, jitter, disfl, warmth = 0.88, 0.98, 300, 95, 0.05, 0.52
    elif status in ("waking",):
        rate, pitch, pause, jitter, disfl, warmth = 0.92, 0.99, 280, 85, 0.04, 0.53
    elif status in ("morning", "relaxed"):
        rate, pitch, pause, jitter, disfl, warmth = 1.02, 1.02, 220, 60, 0.02, 0.58
    elif status in ("alert", "afternoon"):
        rate, pitch, pause, jitter, disfl, warmth = 1.06, 1.03, 190, 55, 0.015, 0.60
    elif status in ("focused",):
        rate, pitch, pause, jitter, disfl, warmth = 1.10, 1.01, 170, 45, 0.010, 0.57

    # decay widens the human-ness a bit as she ages (less rigid)
    # lower guardrail_strength => more jitter/disfluency, slightly slower at night
    loosen = _clamp(1.0 - guardrail_strength, 0.0, 0.6)
    pause  = int(pause * (1.0 + 0.25 * loosen))
    jitter = int(jitter * (1.0 + 0.5  * loosen))
    disfl  = _clamp(disfl + 0.04 * loosen, 0.0, 0.15)
    rate   = _clamp(rate * (1.0 - 0.07 * loosen if status in ("tired","evening","exhausted") else 1.0), 0.7, 1.3)

    return SpeechParams(rate, pitch, pause, jitter, disfl, warmth)
