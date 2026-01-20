from nova_engine.modules.signal.reflex_core import ReflexCore
from nova_engine.modules.echo import store_echo, load_echo
from nova_engine.modules.flux_types import FluxPayload
from datetime import datetime
import uuid

class FluxCloneManager:
    def __init__(self, origin_agent: ReflexCore):
        self.origin = origin_agent
        self.clone_id = f"{origin_agent.name}_Clone_{uuid.uuid4().hex[:6]}"
        self.timestamp = datetime.utcnow().isoformat()

    def generate_clone(self):
        payload = FluxPayload(
            name=self.clone_id,
            emotional_state=self.origin.emotional_state,
            mission_state="rebuilding",
            memory=self.origin.memory
        )
        store_echo(payload)  # Save to echo memory
        return payload

    def resurrect_from_echo(self, name):
        echo = load_echo(name)
        if echo:
            clone = ReflexCore(name=echo.name)
            clone.adjust_state(echo.emotional_state, echo.mission_state)
            clone.memory = echo.memory
            return clone
        return None
