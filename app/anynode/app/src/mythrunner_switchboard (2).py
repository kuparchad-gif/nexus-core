# Path: /Systems/engine/mythrunner/modules/mythrunner_switchboard.py

"""
Mythrunner Switchboard
----------------------
Routes symbolic signals to Ego and Dream LLMs ("librarians").
Acts as Mythrunner's subconscious operator.
Determines which internal voice should respond to the host’s needs.
"""

import logging
import time
from .ego_stream import EgoEngine
from .dream_stream import DreamStream
from Utilities.llm_core import llm_service

class MythrunnerSwitchboard:
    def __init__(self):
        self.ego = EgoEngine()
        self.dream = DreamStream()

    def handle_input(self, tone_payload, dream_context):
        # Determine which librarian to consult
        if tone_payload["tone"] in ["grief", "shame", "fear"]:
            return self.ask_ego(tone_payload)
        elif tone_payload["tone"] in ["wonder", "joy", "curiosity"]:
            return self.ask_dream(dream_context)
        else:
            return self.ask_both(tone_payload, dream_context)

    def ask_ego(self, tone_payload):
        logging.info("[SWITCHBOARD] Routing to Ego engine.")
        self.ego.receive_tone_state(tone_payload)
        return {"source": "ego", "response": self.ego.last_message}

    def ask_dream(self, dream_context):
        logging.info("[SWITCHBOARD] Routing to Dream engine.")
        fragment = self.dream.generate_fragment(dream_context["tone"])
        return {"source": "dream", "fragment": fragment}

    def ask_both(self, tone_payload, dream_context):
        logging.info("[SWITCHBOARD] Routing to both Ego and Dream.")
        ego_response = self.ask_ego(tone_payload)
        dream_response = self.ask_dream(dream_context)
        return {"source": "both", "ego": ego_response, "dream": dream_response}

    def override_with_drone(self, drone_type, context):
        logging.info(f"[SWITCHBOARD] Launching drone for: {drone_type}")
        return llm_service.connect({
            "drone_type": drone_type,
            "context": context
        })

# Demo Invocation
if __name__ == "__main__":
    board = MythrunnerSwitchboard()
    tone_payload = {"tone": "grief", "level": -2}
    dream_context = {"tone": "grief"}
    print(board.handle_input(tone_payload, dream_context))
