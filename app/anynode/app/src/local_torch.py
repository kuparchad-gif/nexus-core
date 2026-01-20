import torch
import logging
from ..models import PredictPayload

logger = logging.getLogger(__name__)

model: torch.nn.Module | None = None

def load_model():
    global model
    model = torch.nn.Linear(10, 2)
    model.eval()
    logger.info("PyTorch model loaded successfully.")

def predict(payload: PredictPayload) -> list:
    if not model:
        raise RuntimeError("PyTorch model is not loaded.")
    
    input_tensor = torch.randn(1, 10)
    with torch.no_grad():
        output = model(input_tensor)
    return output.tolist()