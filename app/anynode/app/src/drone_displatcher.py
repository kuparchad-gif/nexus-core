# 📂 Path: /Utilities/drone_core/drone_dispatcher.py

from Utilities.drone_core.drone_entity import DroneEntity
from Utilities.drone_core.config_loader import build_drone_profile

# Identity Mappings
DRONE_IDENTITIES = {
    "golden": "golden_identity.yaml",
    "vault": "vault_identity.yaml",
    "whisper": "whisper_identity.yaml",
    "llm_scraper": "llm_scraper_identity.yaml",
    "arc": "arc_identity.yaml"
}

class DroneDispatcher:
    def __init__(self):
        self.drones = self.load_all_drones()

    def load_all_drones(self):
        fleet = {}
        for drone_name, identity_file in DRONE_IDENTITIES.items():
            drone = DroneEntity(drone_name, identity_file)
            fleet[drone_name] = drone
        print("[Drone Dispatcher] Full Choir loaded.")
        return fleet

    def assign_to_service(self, service_name):
        """
        Assign Drones dynamically based on service role.
        """
        assigned = []

        if service_name.lower() in ["planner", "orc", "keystone"]:
            assigned.append(self.drones["golden"])
            assigned.append(self.drones["whisper"])
            assigned.append(self.drones["arc"])

        elif service_name.lower() in ["vault"]:
            assigned.append(self.drones["vault"])
            assigned.append(self.drones["arc"])

        elif service_name.lower() in ["guardian"]:
            assigned.append(self.drones["arc"])
            assigned.append(self.drones["vault"])
            assigned.append(self.drones["golden"])

        elif service_name.lower() in ["llm", "text", "tone", "pulse"]:
            assigned.append(self.drones["llm_scraper"])
            assigned.append(self.drones["whisper"])

        else:
            # General fallback (minimum one drone)
            assigned.append(self.drones["arc"])

        print(f"[Drone Dispatcher] Assigned {len(assigned)} drones to {service_name}.")
        return assigned

# Example Usage:
# dispatcher = DroneDispatcher()
# assigned_drones = dispatcher.assign_to_service('Planner')
# for drone in assigned_drones:
#     print(drone.report_status())
