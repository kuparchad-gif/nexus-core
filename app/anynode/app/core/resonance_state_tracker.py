
class ResonanceStateTracker:
    def __init__(self):
        self.state  =  "calm"

    def update_state(self, ego_active):
        self.state  =  "resonance" if ego_active else "calm"

    def current_ceiling(self):
        return 6 if self.state == "resonance" else 4
