# keystone.py
import importlib
import json

ROLE_MODULES = {
    'orchestrator': 'roles.orchestrator',
    'scheduler': 'roles.scheduler',
    'memory': 'roles.memory',
    'text': 'roles.text',
    'tone': 'roles.tone',
}

CRITICAL_ROLES = ['orchestrator', 'scheduler']
MIN_ROLE_THRESHOLD = 1  # number of drones to consider role healthy

class Keystone:
    def __init__(self):
        self.id = "keystone"
        self.active_role = "keystone"
        self.fallback = False
        self.configs = self.load_configs()

    def load_configs(self):
        # Placeholder: later load JSON or YAML configs
        return {}

    def collect_votes(self):
        # Placeholder for now. Inject mock role votes
        return {
            "orchestrator": 0,
            "scheduler": 1,
            "memory": 1,
            "text": 1,
            "tone": 0
        }

    def resolve_identity(self):
        votes = self.collect_votes()

        for role in CRITICAL_ROLES:
            if votes.get(role, 0) < MIN_ROLE_THRESHOLD:
                self.fallback = True
                self.active_role = role
                print(f"[KEYSTONE] Assuming role: {role} (fallback)")
                self.boot_role(role)
                return

        print("[KEYSTONE] Operating in standard keystone mode.")
        self.boot_role("keystone")

    def boot_role(self, role):
        module_path = ROLE_MODULES.get(role)
        if not module_path:
            print(f"[ERROR] No module path for role: {role}")
            return
        try:
            mod = importlib.import_module(module_path)
            mod.boot()
        except Exception as e:
            print(f"[ERROR] Failed to boot role {role}: {e}")

    def status(self):
        return {
            "id": self.id,
            "active_role": self.active_role,
            "fallback": self.fallback
        }

# Runner
if __name__ == "__main__":
    keystone = Keystone()
    keystone.resolve_identity()
    print(json.dumps(keystone.status(), indent=2))
