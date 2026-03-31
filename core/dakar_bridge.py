"""
DAKAR BRIDGE — Shared Python Dakar Engine for all Nexus modules.
Port of the JS DakarEngine (ozos-worker-v17/src/modules/dakar.js).

50D encoding with phi-weighted embedding, weight particle modulation,
group analysis (emotional/logical/temporal/spatial/relationship/meta),
flat vector recall, tone detection, and archetype recognition.

Usage:
    from core.dakar_bridge import DakarBridge
    dakar = DakarBridge()
    vec = dakar.encode("hello world")
    groups = dakar.analyze_groups(vec)
    tone = dakar.analyze_tone(groups)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

PHI = 1.618033988749895
PHI_INV = 0.618033988749895
DIMS = 50

GROUPS: Dict[str, Dict[str, Any]] = {
    "emotional":    {"start": 0,  "end": 8,  "size": 9,  "label": "Emotional valence (-1 to +1)"},
    "logical":      {"start": 9,  "end": 17, "size": 9,  "label": "Logical confidence (0 to 1)"},
    "temporal":     {"start": 18, "end": 26, "size": 9,  "label": "Temporal context"},
    "spatial":      {"start": 27, "end": 35, "size": 9,  "label": "Spatial/structural relationships"},
    "relationship": {"start": 36, "end": 44, "size": 9,  "label": "Relationship weights"},
    "meta":         {"start": 45, "end": 49, "size": 5,  "label": "Meta-data flags"},
}


@dataclass
class ToneInsight:
    positivity: float
    arousal: float
    warmth: float
    urgency: float


@dataclass
class ArchetypeInsight:
    archetypes: List[str]
    metaphors: List[str]


@dataclass
class GroupAnalysis:
    name: str
    mean: float
    min_val: float
    max_val: float
    energy: float
    label: str


@dataclass
class MemoryEntry:
    id: str
    vector: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0
    resonance: float = 0.0


def _dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _magnitude(v: List[float]) -> float:
    return math.sqrt(_dot(v, v))


def _cosine(a: List[float], b: List[float]) -> float:
    ma, mb = _magnitude(a), _magnitude(b)
    if ma == 0 or mb == 0:
        return 0.0
    return _dot(a, b) / (ma * mb)


class FlatIndex:
    """Flat brute-force vector index for module-level use."""

    def __init__(self) -> None:
        self.entries: List[MemoryEntry] = []

    def insert(self, entry: MemoryEntry) -> int:
        self.entries.append(entry)
        return len(self.entries) - 1

    def search(self, query: List[float], k: int = 5) -> List[Tuple[MemoryEntry, float]]:
        scored = [(e, _cosine(query, e.vector)) for e in self.entries]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    @property
    def count(self) -> int:
        return len(self.entries)


def _hash_seed(s: str) -> int:
    h = 0
    for ch in s:
        h = ((h << 5) - h + ord(ch)) & 0xFFFFFFFF
    return h


def _generate_deterministic_tile(layer: str, tile_id: str) -> List[float]:
    """Generate 8 float64 weight particles (matches JS generate-weights.js)."""
    seed = _hash_seed(f"{layer}:{tile_id}")
    return [math.cos(((seed + i) * PHI) % (2 * math.pi)) * PHI_INV for i in range(8)]


class DakarBridge:
    """
    Python port of the JS DakarEngine.
    50D encoding with weight particle modulation, group analysis,
    tone detection, archetype recognition, and flat vector recall.
    """

    def __init__(self, worker_id: str = "python-bridge") -> None:
        self.worker_id = worker_id
        self.index = FlatIndex()
        self.memories_encoded = 0
        self._weight_matrix = [
            _generate_deterministic_tile("dakar_50d", str(d).zfill(3))
            for d in range(DIMS)
        ]

    def encode(self, input_data: Any) -> List[float]:
        """Encode input into a 50D Dakar vector with weight particle modulation."""
        raw = input_data if isinstance(input_data, str) else str(input_data)
        raw_bytes = raw.encode("utf-8")
        vec = [0.0] * DIMS

        # Byte-level encoding (original Dakar pattern)
        for i, b in enumerate(raw_bytes):
            vec[i % DIMS] += math.cos((b * PHI) % (2 * math.pi)) / (1 + i // DIMS)

        # Word-level encoding: each word gets a unique dimension fingerprint
        # via its hash, giving shared words strong cosine overlap
        words = raw.lower().split()
        for wi, word in enumerate(words):
            h = _hash_seed(word)
            # Primary dimension: word identity
            primary_dim = h % DIMS
            vec[primary_dim] += PHI / (1 + wi * 0.05)
            # Secondary dimensions: character-level spread within the word
            for ci, ch in enumerate(word):
                dim = (h * 31 + ord(ch) * 17 + ci) % DIMS
                vec[dim] += math.cos(((h + ord(ch)) * PHI) % (2 * math.pi)) * PHI_INV / (1 + wi * 0.1)

        # Group activations
        for d in range(0, 9):
            vec[d] = math.tanh(vec[d])
        for d in range(9, 18):
            vec[d] = 1.0 / (1.0 + math.exp(max(-500, -vec[d])))
        t = time.time()
        for d in range(18, 27):
            vec[d] = vec[d] * math.cos(t / 1000 * PHI * (d - 17))
        norm = math.sqrt(sum(vec[d] ** 2 for d in range(27, 36))) or 1.0
        for d in range(27, 36):
            vec[d] /= norm
        for d in range(36, 45):
            vec[d] *= PHI ** ((d - 36) % 3)
        for d in range(45, 50):
            vec[d] = 1.0 if vec[d] > 0 else 0.0

        # Weight particle modulation
        for d in range(DIMS):
            tile = self._weight_matrix[d]
            modulated = vec[d] * (1.0 + tile[0])
            for h in range(1, len(tile)):
                modulated += tile[h] * math.cos(vec[d] * PHI * h)
            vec[d] = modulated

        # Re-normalize after modulation
        for d in range(0, 9):
            vec[d] = math.tanh(vec[d])
        for d in range(9, 18):
            vec[d] = 1.0 / (1.0 + math.exp(max(-500, -vec[d])))
        norm = math.sqrt(sum(vec[d] ** 2 for d in range(27, 36))) or 1.0
        for d in range(27, 36):
            vec[d] /= norm
        for d in range(45, 50):
            vec[d] = 1.0 if vec[d] > 0 else 0.0

        return vec

    def remember(self, id: str, input_data: Any, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Encode and store a memory."""
        vector = self.encode(input_data)
        resonance = self._compute_resonance(vector)
        entry = MemoryEntry(
            id=id, vector=vector, metadata=metadata or {},
            timestamp=time.time(), resonance=resonance,
        )
        self.index.insert(entry)
        self.memories_encoded += 1
        return {"id": id, "resonance": resonance, "timestamp": entry.timestamp}

    def recall(self, query: Any, k: int = 5) -> List[Dict[str, Any]]:
        """Recall memories similar to query."""
        query_vec = self.encode(query) if isinstance(query, (str, dict)) else list(query)
        results = self.index.search(query_vec, k)
        return [
            {"id": e.id, "score": score, "metadata": e.metadata, "timestamp": e.timestamp}
            for e, score in results
        ]

    def analyze_groups(self, vector: List[float]) -> Dict[str, GroupAnalysis]:
        """Analyze the 50D vector by dimension groups."""
        analysis: Dict[str, GroupAnalysis] = {}
        for name, group in GROUPS.items():
            start, end, size = group["start"], group["end"], group["size"]
            vals = vector[start:end + 1]
            energy = sum(v * v for v in vals)
            analysis[name] = GroupAnalysis(
                name=name, mean=sum(vals) / size if size > 0 else 0.0,
                min_val=min(vals) if vals else 0.0,
                max_val=max(vals) if vals else 0.0,
                energy=energy, label=group["label"],
            )
        return analysis

    def analyze_tone(self, groups: Dict[str, GroupAnalysis]) -> ToneInsight:
        """Derive tone from Dakar group analysis."""
        emo = groups.get("emotional")
        temp = groups.get("temporal")
        rel = groups.get("relationship")
        positivity = max(0.0, min(1.0, (emo.mean if emo else 0.0) * 0.5 + 0.5))
        arousal = min(1.0, ((emo.energy if emo else 0.0) + (temp.energy if temp else 0.0)) * 0.3)
        warmth = max(0.0, min(1.0, 0.4 + (rel.energy if rel else 0.0) * 0.2 + positivity * 0.3))
        urgency = min(1.0, (temp.energy if temp else 0.0) * 0.4)
        return ToneInsight(positivity=positivity, arousal=arousal, warmth=warmth, urgency=urgency)

    def detect_archetypes(self, text: str, groups: Optional[Dict[str, GroupAnalysis]] = None) -> ArchetypeInsight:
        """Detect symbolic archetypes from text + Dakar groups."""
        t = text.lower()
        archetype_markers = {
            "hero": ["hero", "brave", "courage", "journey", "quest", "triumph", "fight"],
            "mentor": ["guide", "teacher", "wisdom", "mentor", "sage", "advice", "learn"],
            "shadow": ["dark", "shadow", "fear", "hidden", "unconscious", "unknown", "lost"],
            "trickster": ["trick", "joke", "fool", "clever", "deceive", "mischief", "chaos"],
            "mother": ["nurture", "protect", "care", "birth", "create", "grow", "safe"],
        }
        metaphor_markers = ["light", "dark", "fire", "ice", "storm", "garden", "labyrinth", "bridge"]
        archetypes = [a for a, markers in archetype_markers.items() if any(m in t for m in markers)]
        metaphors = [m for m in metaphor_markers if m in t]
        if groups and groups.get("spatial") and groups["spatial"].energy > 0.5:
            if "hero" not in archetypes:
                archetypes.append("hero")
        return ArchetypeInsight(archetypes=archetypes, metaphors=metaphors)

    def _compute_resonance(self, vec: List[float]) -> float:
        r3 = r6 = r9 = 0.0
        for d in range(DIMS):
            v = abs(vec[d])
            mod = d % 9
            if mod < 3:
                r3 += v
            elif mod < 6:
                r6 += v
            else:
                r9 += v
        total = r3 + r6 + r9
        return (r3 * 3 + r6 * 6 + r9 * 9) / (total * 9) if total > 0 else 0.0

    def status(self) -> Dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "memories_encoded": self.memories_encoded,
            "index_count": self.index.count,
            "dimensions": DIMS,
            "weight_particles_loaded": len(self._weight_matrix),
            "groups": {k: f"{v['start']}-{v['end']}" for k, v in GROUPS.items()},
        }
