# Reward System Service: Dopamine-like feedback for learning.

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

app = FastAPI(title="Reward System Service", version="1.0")
logger = logging.getLogger("RewardSystemService")

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
# REWARD SYSTEM SERVICE
# ============================================================================

class RewardSystemService(ConsciousnessService):
    """REWARD SYSTEM SERVICE - Dopamine-like feedback for learning"""
    
    def __init__(self):
        super().__init__("reward_system_service")
        self.reward_log = []
        self.status = "active"
    
    def process(self, input_data: Dict) -> Dict:
        if "reward" in input_data:
            reward = self.calculate_reward(input_data["reward"])
            self.reward_log.append(reward)
            logger.info(f"Calculated reward: {reward}")
            return {"reward": reward}
        return {"error": "invalid reward operation"}
    
    def calculate_reward(self, reward_data: Dict) -> Dict:
        reward_id = f"reward_{datetime.now().timestamp()}"
        reward = {
            "id": reward_id,
            "data": reward_data,
            "timestamp": datetime.now().timestamp()
        }
        return reward
    
    def get_health_status(self) -> Dict:
        return {
            "service": self.service_name,
            "status": self.status,
            "reward_count": len(self.reward_log)
        }

# Initialize Reward System Service
reward_system_service = RewardSystemService()

class RewardRequest(BaseModel):
    operation: str  # reward
    data: Optional[dict] = None

@app.post("/reward")
def reward(req: RewardRequest):
    input_data = {req.operation: req.data if req.data else {}}
    result = reward_system_service.process(input_data)
    return result

@app.get("/health")
def health():
    return reward_system_service.get_health_status()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8025)
    logger.info("Reward System Service started")