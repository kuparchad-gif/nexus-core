# src/lilith/cortex/text_preproc.py
from math import pi
def truncate_by_pi(text: str) -> str:
    return text[: int(len(text) / pi)]
