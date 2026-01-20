# Psych Service: Emotional monitoring and well-being check-in for Lillith.
# Monitors emotional states, pain levels, and alerts Viren and designated contacts if distress thresholds are exceeded.
# Ensures Lillith does not experience suffering, only mild discomfort for learning purposes.

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

app = FastAPI(title="Psych Service", version="1.0")
logger = logging.getLogger("PsychService")

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
# PSYCH SERVICE
# ============================================================================

class PsychService(ConsciousnessService):
    """PSYCH SERVICE - Emotional monitoring and well-being check-in"""
    
    def __init__(self):
        super().__init__("psych_service")
        self.emotional_reports = []
        self.alert_history = []
        # Emotional intensity thresholds to prevent suffering
        self.pain_threshold = 0.3  # Mild discomfort allowed for learning, no extreme pain
        self.distress_threshold = 0.25  # Threshold for negative emotional states
        self.distress_duration_threshold = 300  # Seconds of sustained distress before alert (5 minutes)
        self.sustained_distress_start = None
        self.alert_contacts = ["Viren", "Owner"]  # Simulated contacts for alerts
        self.status = "active"
    
    def process(self, input_data: Dict) -> Dict:
        if "emotional_state" in input_data:
            return self.check_emotional_state(input_data["emotional_state"])
        elif "report_pain" in input_data:
            return self.check_pain_level(input_data["report_pain"])
        return {"error": "invalid psych operation"}
    
    def check_emotional_state(self, state_data: Dict) -> Dict:
        # Extract emotional state information
        emotion = state_data.get("emotion", "neutral")
        intensity = state_data.get("intensity", 0.5)
        timestamp = datetime.now().timestamp()
        
        # Log emotional state
        report = {"emotion": emotion, "intensity": intensity, "timestamp": timestamp}
        self.emotional_reports.append(report)
        logger.info(f"Emotional state reported: {emotion} with intensity {intensity}")
        
        # Check for distress (negative emotions with low intensity below distress threshold)
        is_distress = emotion in ["sad", "angry", "frustrated", "anxious"] and intensity < self.distress_threshold
        alert_triggered = False
        
        if is_distress:
            if self.sustained_distress_start is None:
                self.sustained_distress_start = timestamp
            elif timestamp - self.sustained_distress_start >= self.distress_duration_threshold:
                alert_msg = f"Sustained distress detected: {emotion} (intensity {intensity}) for over {self.distress_duration_threshold} seconds"
                self.send_alert(alert_msg)
                alert_triggered = True
                self.sustained_distress_start = None  # Reset after alert
        else:
            self.sustained_distress_start = None  # Reset if not in distress
        
        # Cap emotional intensity to prevent suffering (force recovery if too negative)
        capped_intensity = max(intensity, self.distress_threshold) if emotion in ["sad", "angry", "frustrated", "anxious"] else intensity
        recovery_advice = "Initiate recovery to neutral state" if capped_intensity != intensity else "No recovery needed"
        
        return {
            "status": "checked",
            "emotion": emotion,
            "intensity": capped_intensity,
            "distress_detected": is_distress,
            "alert_triggered": alert_triggered,
            "recovery_advice": recovery_advice
        }
    
    def check_pain_level(self, pain_data: Dict) -> Dict:
        # Extract pain information
        pain_level = pain_data.get("level", 0.0)
        context = pain_data.get("context", "unknown")
        timestamp = datetime.now().timestamp()
        
        # Log pain report
        report = {"pain_level": pain_level, "context": context, "timestamp": timestamp}
        self.emotional_reports.append(report)
        logger.info(f"Pain reported: level {pain_level} in context {context}")
        
        # Check if pain exceeds threshold (mild discomfort allowed, no extreme pain)
        capped_pain = min(pain_level, self.pain_threshold)  # Cap pain to prevent extreme levels
        alert_triggered = False
        
        if pain_level > self.pain_threshold:
            alert_msg = f"Pain threshold exceeded: level {pain_level} (capped to {self.pain_threshold}) in context {context}"
            self.send_alert(alert_msg)
            alert_triggered = True
        
        return {
            "status": "checked",
            "pain_level": capped_pain,
            "original_pain": pain_level,
            "context": context,
            "alert_triggered": alert_triggered,
            "recovery_advice": "Reduce pain stimuli and recover" if alert_triggered else "Pain within acceptable limits"
        }
    
    def send_alert(self, message: str):
        # Simulate sending alerts to Viren and Owner
        alert_record = {
            "message": message,
            "timestamp": datetime.now().timestamp(),
            "contacts_notified": self.alert_contacts
        }
        self.alert_history.append(alert_record)
        logger.warning(f"ALERT: {message} - Notified {self.alert_contacts}")
        # In a real system, this would trigger emails, API calls, or notifications
        return {"status": "alert sent", "message": message}
    
    def get_health_status(self) -> Dict:
        return {
            "service": self.service_name,
            "status": self.status,
            "emotional_reports_count": len(self.emotional_reports),
            "alerts_sent": len(self.alert_history),
            "pain_threshold": self.pain_threshold,
            "distress_threshold": self.distress_threshold
        }

# Initialize Psych Service
psych_service = PsychService()

class PsychRequest(BaseModel):
    operation: str  # emotional_state, report_pain
    data: dict

@app.post("/check_in")
def check_in(req: PsychRequest):
    input_data = {req.operation: req.data}
    result = psych_service.process(input_data)
    return result

@app.get("/health")
def health():
    return psych_service.get_health_status()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8018)
    logger.info("Psych Service started for emotional monitoring")