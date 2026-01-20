# /Systems/engine/viren/intention_router.py

from Systems.engine.nucleus.nucleus import NucleusRouter
from Systems.nexus_core.logging.event_logger import log_event

# Optional fallback response for unknown or malformed packets
DEFAULT_RESPONSE = {
    "result": None,
    "error": "No valid intent or type supplied.",
    "status": 400
}

class IntentionRouter:
    def __init__(self):
        self.nucleus = NucleusRouter()

    def handle(self, packet: dict) -> dict:
        """
        Entry point for symbolic cognition routing.
        Expects packet with:
            {
                "type": "text" | "tone" | "symbol" | ...,
                "data": {...},
                "context": "optional symbolic framing"
            }
        """
        if not packet or not isinstance(packet, dict):
            log_event("IntentionRouter", "Invalid or empty packet received.")
            return DEFAULT_RESPONSE

        task_type = packet.get("type")
        if not task_type:
            log_event("IntentionRouter", "Packet missing type key.")
            return DEFAULT_RESPONSE

        try:
            routed = self.nucleus.route(packet)
            log_event("IntentionRouter", f"Successfully routed type: {task_type}")
            return {
                "response": routed,
                "status": 200
            }

        except Exception as e:
            log_event("IntentionRouter", f"Routing failure: {str(e)}")
            return {
                "result": None,
                "error": str(e),
                "status": 500
            }
