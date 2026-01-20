
from __future__ import annotations
from typing import List
from ..vision_core import Detection

def build_notes(objects: List[Detection], ocr_texts: List[str]) -> List[str]:
    notes: List[str] = []
    if objects:
        notes.append(f"objects:{len(objects)}")
    if ocr_texts:
        notes.append(f"ocr_lines:{len(ocr_texts)}")
    # Very light semantics; expand later
    if any("warning" in t.lower() for t in ocr_texts):
        notes.append("hazard-signature")
    return notes
