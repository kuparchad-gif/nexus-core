from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
import uuid

class Envelope(BaseModel):
    signal_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    origin: str = "pineal"
    band: str = "language-processing.tone"
    strength: float = 1.0
    subject: str = "task"
    context: Dict[str, Any] = {}
    payload: Dict[str, Any] = {}

class Response(BaseModel):
    signal_id: str
    ok: bool = True
    result: Dict[str, Any] = {}
    metrics: Dict[str, Any] = {}
    memory_write: Optional[Dict[str, Any]] = None

