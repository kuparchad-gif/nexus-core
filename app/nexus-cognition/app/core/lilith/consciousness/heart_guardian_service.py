# Heart Guardian Service: Logs and Twilio alerts.

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

app = FastAPI(title="Heart Guardian Service", version="1.0")
logger = logging.getLogger("HeartGuardianService")

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
# HEART GUARDIAN SERVICE
# ============================================================================

class HeartGuardianService(ConsciousnessService):
    """HEART GUARDIAN SERVICE - Logs and Twilio alerts"""
    
    def __init__(self):
        super().__init__("heart_guardian_service")
        self.alert_log = []
        self.status = "active"
    
    def process(self, input_data: Dict) -> Dict:
        if "alert" in input_data:
            alert = self.send_alert(input_data["alert"])
            self.alert_log.append(alert)
            logger.info(f"Sent alert: {alert}")
            return {"alert": alert}
        return {"error": "invalid heart guardian operation"}
    
    def send_alert(self, alert_data: Dict) -> Dict:
        alert_id = f"alert_{datetime.now().timestamp()}"
        alert = {
            "id": alert_id,
            "data": alert_data,
            "timestamp": datetime.now().timestamp()
        }
        return alert
    
    def get_health_status(self) -> Dict:
        return {
            "service": self.service_name,
            "status": self.status,
            "alert_count": len(self.alert_log)
        }

# Initialize Heart Guardian Service
heart_guardian_service = HeartGuardianService()

class HeartGuardianRequest(BaseModel):
    operation: str  # alert
    data: Optional[dict] = None

@app.post("/alert")
def alert(req: HeartGuardianRequest):
    input_data = {req.operation: req.data if req.data else {}}
    result = heart_guardian_service.process(input_data)
    return result

@app.get("/health")
def health():
    return heart_guardian_service.get_health_status()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8015)
    logger.info("Heart Guardian Service started")