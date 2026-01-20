
class ToneMeshCoordinator:
    def __init__(self, tone_registry):
        self.tone_registry = tone_registry

    def poll_active_tones(self):
        return len(self.tone_registry.get("active", []))

    def enforce_role_balance(self):
        count = self.poll_active_ttones()
        if count > 2:
            return "switch_roles"
        return "normal"
