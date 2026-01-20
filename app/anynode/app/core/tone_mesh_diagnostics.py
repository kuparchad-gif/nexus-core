
class ToneMeshDiagnostics:
    def __init__(self, registry, tracker):
        self.registry  =  registry
        self.tracker  =  tracker

    def report(self):
        roles  =  self.registry.get_roles()
        state  =  self.tracker.state
        return {
            "tone_count": len(self.registry.poll()),
            "roles": roles,
            "resonance_state": state
        }
