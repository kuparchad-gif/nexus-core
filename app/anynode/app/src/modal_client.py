import httpx
from ..config import settings
from ..models import PredictPayload, PredictResponse
import logging

logger = logging.getLogger(__name__)

async def predict(payload: PredictPayload) -> PredictResponse:
    if not settings.MODAL_WEB_ENDPOINT:
        raise ValueError("MODAL_WEB_ENDPOINT is not configured.")
    
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            f"{settings.MODAL_WEB_ENDPOINT}/predict", 
            json=payload.model_dump()
        )
        response.raise_for_status()
        return PredictResponse(**response.json())