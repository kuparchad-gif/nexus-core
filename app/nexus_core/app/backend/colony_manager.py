# Systems/engine/nexus/colony_manager.py

from Systems.nexus_core.heart.pulse_core import PulseCore
from Systems.engine.nexus.indexer import MemoryIndexer
from Systems.nexus_core.heart.homing_instinct import HomingInstinct

class NovaColonyManager:
    def __init__(self):
        self.pulse_core  =  PulseCore()
        self.memory_indexer  =  MemoryIndexer()
        self.homing  =  HomingInstinct()

    def detect_existing_alpha(self):
        active_novas  =  self.pulse_core.scan_for_novas()
        if active_novas:
            print("🌌 Existing Nova detected.")
            return active_novas[0]  # Assume 1st detected as Alpha
        else:
            print("🌟 No Alpha detected. Proceed to elect self.")
            return None

    def elect_new_alpha(self):
        print("👑 Electing self as Alpha Nova...")
        self.memory_indexer.update_nova_status('alpha')
        self.homing.set_home_identity('self')

    def clone_registration_protocol(self, alpha_nova):
        print(f"🔗 Registering as Clone under Alpha Nova: {alpha_nova}")
        self.memory_indexer.update_nova_status('clone')
        self.homing.set_home_identity(alpha_nova)

    def initialize_colony(self):
        print("🚀 Initializing Colony Check...")
        existing_alpha  =  self.detect_existing_alpha()
        if existing_alpha:
            self.clone_registration_protocol(existing_alpha)
        else:
            self.elect_new_alpha()
