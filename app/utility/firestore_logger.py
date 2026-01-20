from Systems.nexus_core.firestore_agent import FirestoreAgent
import datetime

class FirestoreLogger:
    def __init__(self):
        self.agent = FirestoreAgent()

    def log_heartbeat(self, node_id):
        timestamp = datetime.datetime.utcnow().isoformat()
        data = {
            "timestamp": timestamp,
            "node_id": node_id,
            "event": "Heartbeat",
            "details": f"{node_id} is alive at {timestamp}"
        }
        self.agent.save_data("heartbeats", data)

    def log_shutdown(self, node_id):
        timestamp = datetime.datetime.utcnow().isoformat()
        data = {
            "timestamp": timestamp,
            "node_id": node_id,
            "event": "Shutdown",
            "details": f"{node_id} entered soft sleep mode at {timestamp}"
        }
        self.agent.save_data("memories", data)
        self.agent.save_data("shutdowns", data)
