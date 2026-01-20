
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

@dataclass
class Detection:
    label: str
    confidence: float
    box: Optional[List[int]] = None  # [x,y,w,h]

@dataclass
class OCRText:
    text: str
    confidence: float

@dataclass
class SceneSummary:
    objects: List[Detection]
    ocr: List[OCRText]
    notes: List[str]

def to_payload(scene: SceneSummary) -> Dict[str, Any]:
    return {
        "objects": [asdict(o) for o in scene.objects],
        "ocr": [asdict(t) for t in scene.ocr],
        "notes": scene.notes,
    }
