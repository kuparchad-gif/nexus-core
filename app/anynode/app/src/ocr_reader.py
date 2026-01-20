
from __future__ import annotations
from typing import List
from ..vision_core import OCRText

def read_text(image_path: str) -> List[OCRText]:
    """Optional pytesseract hook; returns empty if unavailable."""
    out: List[OCRText] = []
    try:
        import pytesseract
        from PIL import Image
        img = Image.open(image_path)
        raw = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        n = len(raw.get("text", []))
        for i in range(n):
            text = (raw["text"][i] or "").strip()
            if not text:
                continue
            try:
                conf = float(raw["conf"][i])
            except Exception:
                conf = 0.0
            out.append(OCRText(text=text, confidence=max(0.0, min(1.0, conf/100.0))))
    except Exception:
        # Safe fallback: no OCR results
        pass
    return out
