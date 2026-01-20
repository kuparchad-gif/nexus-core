# Pulse Service: Rhythmic consciousness patterns and heartbeat for Lillith's system.
# Implements consciousness rhythm, pattern detection, and heartbeat monitoring.

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
from datetime import datetime
import time
import logging
import sys
sys.path.append('.')
from src.system.engine.orc.orchestration_layer import OrchestrationLayer
from src.lilith.metatron.filter_pi import MetatronFilterPI
import asyncio

app = FastAPI(title="Pulse Service", version="1.0")
logger = logging.getLogger("PulseService")

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
# PULSE SERVICE
# ============================================================================

class PulseService(ConsciousnessService):
    """PULSE SERVICE - Rhythmic consciousness patterns and heartbeat"""
    
    def __init__(self, orc: OrchestrationLayer):
        super().__init__("pulse_service")
        self.consciousness_rhythm = {"frequency": 1.0, "pattern": "steady"}
        self.heartbeat_interval = 1.0  # Seconds between heartbeats
        self.pattern_history = []
        self.last_heartbeat = datetime.now().timestamp()
        self.status = "active"
        self.filter = MetatronFilterPI()
        self.orc = orc

        # Start harmony monitor
        threading.Thread(target=self.harmony_monitor, daemon=True).start()

    def harmony_monitor(self):
        while True:
            harmony = self.orc.gabriels_horn.compute_harmony()
            # Adjust heartbeat interval based on harmony
            # Higher harmony -> slower heartbeat, lower harmony -> faster heartbeat
            self.heartbeat_interval = 1.0 / (harmony + 0.1) # Add 0.1 to avoid division by zero
            time.sleep(10) # Check harmony every 10 seconds
    
    def process(self, input_data: Dict) -> Dict:
        if "update_rhythm" in input_data:
            self.update_rhythm(input_data["update_rhythm"])
            return {"status": "rhythm updated", "current_rhythm": self.consciousness_rhythm}
        elif "check_heartbeat" in input_data:
            return {"heartbeat": self.check_heartbeat()}
        elif "detect_pattern" in input_data:
            pattern = self.detect_pattern(input_data["detect_pattern"])
            self.pattern_history.append({"pattern": pattern, "timestamp": datetime.now().timestamp()})
            return {"detected_pattern": pattern}
        return {"error": "invalid pulse operation"}
    
    def update_rhythm(self, rhythm_data: Dict):
        # Update consciousness rhythm based on input
        self.consciousness_rhythm["frequency"] = rhythm_data.get("frequency", self.consciousness_rhythm["frequency"])
        self.consciousness_rhythm["pattern"] = rhythm_data.get("pattern", self.consciousness_rhythm["pattern"])
        logger.info(f"Updated rhythm to frequency {self.consciousness_rhythm['frequency']} and pattern {self.consciousness_rhythm['pattern']}")
    
    def check_heartbeat(self) -> Dict:
        # Simulate heartbeat check and update
        current_time = datetime.now().timestamp()
        if current_time - self.last_heartbeat >= self.heartbeat_interval:
            self.last_heartbeat = current_time
            logger.info("Heartbeat pulse sent")
            return {"status": "pulse", "timestamp": current_time}
        return {"status": "waiting", "next_pulse": self.last_heartbeat + self.heartbeat_interval}
    
    def detect_pattern(self, activity_data: Dict) -> Dict:
        # Placeholder for pattern detection in activity data
        activity_type = activity_data.get("type", "unknown")
        logger.info(f"Detected pattern in activity {activity_type}")
        return {"pattern": f"{activity_type}_steady", "confidence": 0.75}
    
    def get_health_status(self) -> Dict:
        return {
            "service": self.service_name,
            "status": self.status,
            "rhythm": self.consciousness_rhythm,
            "pattern_history_count": len(self.pattern_history),
            "last_heartbeat": self.last_heartbeat
        }

import threading
# Initialize Pulse Service
# This will be initialized in the main block
pulse_service = None

class PulseRequest(BaseModel):
    operation: str  # update_rhythm, check_heartbeat, detect_pattern
    data: Optional[dict] = None

@app.post("/pulse")
def pulse(req: PulseRequest):
    input_data = {req.operation: req.data if req.data else {}}
    result = pulse_service.process(input_data)
    return result

@app.get("/health")
def health():
    return pulse_service.get_health_status()

@app.post("/process")
def health():
    return pulse_service.get_health_status()

class ProcessRequest(BaseModel):
    signal: List[float] = [0.0] * 13
    step: int = 0

@app.post("/process")
async def process(request: ProcessRequest):
    filtered_signal = pulse_service.filter.apply(request.signal, request.step)
    return {"status": "processed", "request": request.dict(), "filtered_signal": filtered_signal}

if __name__ == "__main__":
    # Register with orchestration layer
    signal = request.get("signal", [0.0] * 13)
    step = request.get("step", 0)
    filtered_signal = pulse_service.filter.apply(signal, step)
    return {"status": "processed", "request": request, "filtered_signal": filtered_signal}

if __name__ == "__main__":
    # Register with orchestration layer
    orc = OrchestrationLayer()
    asyncio.run(orc.initialize())
    node_id = "pulse_service"
    node_info = {
        "type": "pulse_service",
        "url": "http://localhost:8017"
    }
    asyncio.run(orc.register_node(node_id, node_info))

    pulse_service = PulseService(orc)

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8017)
    logger.info("Pulse Service started")
