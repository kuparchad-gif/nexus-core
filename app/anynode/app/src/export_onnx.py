# path: backend/app/inference/onnx_runtime.py
from __future__ import annotations
import logging
from pathlib import Path
from typing import List

import numpy as np

logger = logging.getLogger(__name__)

# ONNX is optional — we’ll try to load it if present; otherwise we stub predict().
try:
    import onnxruntime as ort  # type: ignore
except Exception:  # onnxruntime not installed
    ort = None  # type: ignore

_SESSION = None
_MODEL_PATH = Path(__file__).parents[1] / "models" / "model.onnx"


def _lazy_session():
    """Create and cache an ONNX Runtime session if the model and ORT are available."""
    global _SESSION
    if _SESSION is None and ort is not None and _MODEL_PATH.exists():
        logger.info("Loading ONNX model: %s", _MODEL_PATH)
        _SESSION = ort.InferenceSession(str(_MODEL_PATH))  # default providers
    return _SESSION


def predict(text_input: str) -> List[List[float]]:
    """
    Return a 2-class probability vector [[p0, p1]].

    If ONNX + model file exist, runs real inference.
    Otherwise, returns a deterministic stub based on input length.
    """
    sess = _lazy_session()
    if sess is not None:
        # Minimal demo: map text length to a fixed-size numeric input (1x10)
        # Real code would tokenize/encode properly to match your model.
        x = np.zeros((1, 10), dtype=np.float32)
        x[0, 0] = min(len(text_input), 1000) / 1000.0
        inputs = {sess.get_inputs()[0].name: x}
        outputs = sess.run(None, inputs)[0]
        # Ensure nested list return
        return np.asarray(outputs, dtype=np.float32).tolist()

    # Fallback stub (no ORT or no model file)
    L = float(len(text_input))
    p0 = max(0.0, min(1.0, (L % 100) / 100.0))
    p1 = 1.0 - p0
    return [[p0, p1]]
