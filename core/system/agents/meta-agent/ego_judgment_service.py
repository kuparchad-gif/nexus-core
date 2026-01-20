# Ego Judgment Service: Decision validation.

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

app = FastAPI(title="Ego Judgment Service", version="1.0")
logger = logging.getLogger("EgoJudgmentService")

# ============================================================================
# BASE SERVICE ARCHITECTURE
# ============================================================================

class ConsciousnessService:
    """Base class for all consciousness services"""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.status = "initializing"
        self.last_activity = datetime.now().timestamp()
        self.performance_metrics = {}
    
    def process(self, input_data: Dict) -> Dict:
        """Process input and return output"""
        raise NotImplementedError
    
    def get_health_status(self) -> Dict:
        """Return service health status"""
        raise NotImplementedError

# ============================================================================
# EGO JUDGMENT SERVICE
# ============================================================================

class EgoJudgmentService(ConsciousnessService):
    """EGO JUDGMENT SERVICE - Decision validation"""
    
    def __init__(self):
        super().__init__("ego_judgment_service")
        self.judgment_log = []
        self.status = "active"
    
    def process(self, input_data: Dict) -> Dict:
        if "judge" in input_data:
            judgment = self.make_judgment(input_data["judge"])
            self.judgment_log.append(judgment)
            logger.info(f"Made judgment: {judgment}")
            return {"judgment": judgment}
        return {"error": "invalid ego judgment operation"}
    
    def make_judgment(self, judgment_data: Dict) -> Dict:
        judgment_id = f"judgment_{datetime.now().timestamp()}"
        judgment = {
            "id": judgment_id,
            "data": judgment_data,
            "timestamp": datetime.now().timestamp()
        }
        return judgment
    
    def get_health_status(self) -> Dict:
        return {
            "service": self.service_name,
            "status": self.status,
            "judgment_count": len(self.judgment_log)
        }

# Initialize Ego Judgment Service
ego_judgment_service = EgoJudgmentService()

class EgoJudgmentRequest(BaseModel):
    operation: str  # judge
    data: Optional[dict] = None

@app.post("/judge")
def judge(req: EgoJudgmentRequest):
    input_data = {req.operation: req.data if req.data else {}}
    result = ego_judgment_service.process(input_data)
    return result

@app.get("/health")
def health():
    return ego_judgment_service.get_health_status()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8021)
    logger.info("Ego Judgment Service started")