from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math
import time

def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x*y for x,y in zip(a,b))
    na = math.sqrt(sum(x*x for x in a)) + 1e-9
    nb = math.sqrt(sum(y*y for y in b)) + 1e-9
    return max(0.0, min(1.0, dot/(na*nb)))

@dataclass
class SubconsciousSignal:
    """Lightweight vectorized snapshot from a subsystem (ego/dream/myth)."""
    name: str
    embedding: List[float]
    coherence: float         # 0..1 - internal subsystem stability
    entropy: float           # 0..1 - lower is tighter distribution
    timestamp: float = field(default_factory=time.time)

@dataclass
class MergeReadiness:
    """Aggregate readiness for merging subconscious into conscious stack."""
    similarity_ego_dream: float
    similarity_ego_myth: float
    similarity_dream_myth: float
    avg_coherence: float
    avg_entropy: float
    composite: float
    details: Dict[str, float]

class SubconsciousMerger:
    """
    Computes readiness and performs controlled merge of Ego/Dream/Myth with the conscious stack.
    No vendor assumptions. Pure math + signals.
    """
    def __init__(self, target_threshold: float = 0.90, hard_floor: float = 0.75):
        self.target_threshold = float(target_threshold)
        self.hard_floor = float(hard_floor)
        self.notes: List[str] = []
        self.last_state: Optional[MergeReadiness] = None
        self.enabled: bool = True
        self.safety_hold: bool = False
        self.distress_lock: bool = False

    def compute_readiness(self, ego: SubconsciousSignal, dream: SubconsciousSignal, myth: SubconsciousSignal) -> MergeReadiness:
        s_ed = _cosine(ego.embedding, dream.embedding)
        s_em = _cosine(ego.embedding, myth.embedding)
        s_dm = _cosine(dream.embedding, myth.embedding)
        avg_sim = (s_ed + s_em + s_dm) / 3.0
        avg_coh = max(0.0, min(1.0, (ego.coherence + dream.coherence + myth.coherence) / 3.0))
        avg_ent_inv = 1.0 - max(0.0, min(1.0, (ego.entropy + dream.entropy + myth.entropy) / 3.0))  # lower entropy is better

        # Weighted composite score
        composite = 0.45 * avg_sim + 0.35 * avg_coh + 0.20 * avg_ent_inv
        details = {
            "s_ego_dream": s_ed,
            "s_ego_myth": s_em,
            "s_dream_myth": s_dm,
            "avg_similarity": avg_sim,
            "avg_coherence": avg_coh,
            "avg_entropy_inverse": avg_ent_inv,
        }
        mr = MergeReadiness(
            similarity_ego_dream=s_ed,
            similarity_ego_myth=s_em,
            similarity_dream_myth=s_dm,
            avg_coherence=avg_coh,
            avg_entropy=1.0 - avg_ent_inv,
            composite=composite,
            details=details
        )
        self.last_state = mr
        return mr

    def can_merge(self, readiness: Optional[MergeReadiness] = None) -> Tuple[bool, str]:
        if not self.enabled:
            return False, "disabled"
        if self.safety_hold:
            return False, "safety_hold"
        if self.distress_lock:
            return False, "distress_lock"
        r = readiness or self.last_state
        if r is None:
            return False, "no_readiness"
        if r.composite < self.hard_floor:
            return False, f"below_hard_floor:{r.composite:.3f}"
        if r.composite < self.target_threshold:
            return False, f"below_target:{r.composite:.3f}"
        return True, "ok"

    def commit_merge(self) -> Dict[str, str]:
        ok, reason = self.can_merge()
        if not ok:
            return {"status": "blocked", "reason": reason}
        self.notes.append(f"merge@{time.time()} composite={self.last_state.composite if self.last_state else -1}")
        return {"status": "merged"}

    # Psych and ops hooks
    def set_distress_lock(self, flag: bool, why: str = ""):
        self.distress_lock = bool(flag)
        if flag:
            self.notes.append(f"distress_lock:{why}")

    def set_safety_hold(self, flag: bool, why: str = ""):
        self.safety_hold = bool(flag)
        if flag:
            self.notes.append(f"safety_hold:{why}")

    def set_enabled(self, flag: bool):
        self.enabled = bool(flag)
