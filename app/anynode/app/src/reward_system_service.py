# Reward System Service: Dopamine-like feedback and motivation for Lillith's learning and decision-making.
# Implements reward calculation, motivation tracking, and learning reinforcement.

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

app = FastAPI(title="Reward System Service", version="1.0")
logger = logging.getLogger("RewardSystem")

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
    """REWARD SYSTEM - Dopamine-like feedback and motivation"""
    
    def __init__(self):
        super().__init__("reward_system")
        self.reward_history = []
        self.motivation_levels = {}
        self.dopamine_baseline = 0.5
        self.status = "active"
    
    def process(self, input_data: Dict) -> Dict:
        action = input_data.get("action", {})
        outcome = input_data.get("outcome", {})
        if action and outcome:
            reward_value = self.calculate_reward(action, outcome)
            activity = action.get("activity", "general")
            self.update_motivation(activity, reward_value)
            self.reward_history.append({"action": action, "outcome": outcome, "reward": reward_value, "timestamp": datetime.now().timestamp()})
            return {"reward": reward_value, "motivation": self.motivation_levels.get(activity, 0.5)}
        return {"error": "missing action or outcome for reward calculation"}
    
    def calculate_reward(self, action: Dict, outcome: Dict) -> float:
        # Placeholder for reward calculation logic
        success_factor = outcome.get("success", 0.0)
        importance = action.get("importance", 0.5)
        reward = success_factor * importance * 2.0  # Simple formula for demonstration
        logger.info(f"Calculated reward: {reward} for action {action.get('activity', 'unknown')}")
        return min(max(reward, -1.0), 1.0)  # Normalize between -1 and 1
    
    def update_motivation(self, activity: str, reward: float):
        # Update motivation levels based on reward
        current_motivation = self.motivation_levels.get(activity, self.dopamine_baseline)
        new_motivation = current_motivation + (reward * 0.1)  # Incremental update
        self.motivation_levels[activity] = min(max(new_motivation, 0.0), 1.0)  # Normalize between 0 and 1
        logger.info(f"Updated motivation for {activity} to {self.motivation_levels[activity]}")
    
    def get_health_status(self) -> Dict:
        return {
            "service": self.service_name,
            "status": self.status,
            "reward_history_count": len(self.reward_history),
            "motivation_activities": len(self.motivation_levels)
        }

# Initialize Reward System Service
reward_service = RewardSystemService()

class RewardRequest(BaseModel):
    action: dict
    outcome: dict

@app.post("/calculate_reward")
def calculate_reward(req: RewardRequest):
    result = reward_service.process({"action": req.action, "outcome": req.outcome})
    return result

@app.get("/health")
def health():
    return reward_service.get_health_status()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8015)
    logger.info("Reward System service started")
