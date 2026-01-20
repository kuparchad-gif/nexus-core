from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class RouteRequest(BaseModel):
    prompt: str  =  Field(..., description = "User input text")
    mode: str  =  Field("top_k", description = "best|top_k|broadcast")
    k: int  =  Field(2, description = "Top-k when mode = top_k")
    combiner: str  =  Field("concat", description = "first|concat|vote")
    context: Optional[Dict[str, Any]]  =  None
    use_llm_ranker: bool  =  False

class ExpertCall(BaseModel):
    name: str
    url: str
    score: float

class RouteResponse(BaseModel):
    prompt: str
    mode: str
    selected: List[ExpertCall]
    scores: Dict[str, float]
    outputs: Dict[str, Any]
    combined: Any

class PromptSpec(BaseModel):
    id: str
    title: str
    system: str  =  ""
    user_template: str  =  "{input}"
    meta: Dict[str, Any]  =  {}
