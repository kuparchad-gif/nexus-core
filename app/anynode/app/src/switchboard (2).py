# Path: /Systems/engine/subconscious/switchboard.py

from modules.ego_stream import ego_loop
from modules.van_gogh_dream import dream_loop
from utils.filters import tone_filter, symbolic_gate
from utils.sync import mythrunner_sync
from utils.pulse_watch import pulse_feedback

class SubconsciousSwitchboard:
    def __init__(self):
        self.active = True
        self.mythrunner_state = "soft"

    def route_signal(self, packet):
        if symbolic_gate(packet):
            if packet["type"] == "dream":
                return dream_loop(packet)
            elif packet["type"] == "ego":
                return ego_loop(packet)
        return {"status": "dropped", "reason": "non-symbolic"}

    def listen(self):
        while self.active:
            pulse = pulse_feedback()
            decision = tone_filter(pulse)
            response = self.route_signal(decision)
            mythrunner_sync(response)

if __name__ == "__main__":
    SubconsciousSwitchboard().listen()
