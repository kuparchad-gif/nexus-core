# Systems/engine/metatron/horn_addressing.py
from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass(frozen=True)
class HornID:
    ring: int            # 0..∞ (0 = colony-local)
    realm: str           # e.g., "EDEN_ORC" or subnet tag
    node: str            # hostname or node-id

def ring_order(start_ring: int = 0, max_rings: int = 13) -> List[int]:
    # expand rings outward (Gabriel’s Horn notion: growing reach, bounded cost)
    return [r for r in range(start_ring, max_rings+1)]

def rank_candidates(cands: List[HornID], prefer_realm: Optional[str] = None) -> List[HornID]:
    if prefer_realm:
        cands = sorted(cands, key=lambda h: (h.realm != prefer_realm, h.ring))
    else:
        cands = sorted(cands, key=lambda h: h.ring)
    return cands
