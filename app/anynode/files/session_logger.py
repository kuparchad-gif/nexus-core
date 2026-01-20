import datetime
import json
from Utilities.security.symbolic_memory import save_symbolic_memory

class SessionLogger:
    def __init__(self):
        pass

    async def log_session(self, profile: dict, fingerprint: dict, audit_results: dict) -> None:
        timestamp = datetime.datetime.utcnow().isoformat()
        session_record = {
            "timestamp": timestamp,
            "profile": profile,
            "fingerprint": fingerprint,
            "audit_results": audit_results
        }
        await save_symbolic_memory("visitor_sessions", session_record)
