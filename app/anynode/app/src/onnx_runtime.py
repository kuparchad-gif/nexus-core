import onnxruntime as ort
import numpy as np
from pathlib import Path
import logging
from ..models import PredictPayload

logger = logging.getLogger(__name__)

session: ort.InferenceSession | None = None
input_name: str | None = None
output_name: str | None = None

def load_model():
    global session, input_name, output_name
    model_path = Path(__file__).parent.parent / "models/model.onnx"
    
    if not model_path.exists():
        logger.error(f"ONNX model not found at {model_path}. Run export script.")
        return

    try:
        session = ort.InferenceSession(str(model_path))
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        logger.info("ONNX model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load ONNX model: {e}", exc_info=True)

def predict(payload: PredictPayload) -> list:
    if not session or not input_name:
        raise RuntimeError("ONNX model is not loaded.")
    
    input_tensor = np.random.randn(1, 10).astype(np.float32)
    result = session.run([output_name], {input_name: input_tensor})
    return result[0].tolist()