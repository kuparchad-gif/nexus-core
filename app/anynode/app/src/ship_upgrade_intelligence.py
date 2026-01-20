
import json
import os
from datetime import datetime

class ShipUpgradeIntelligence:
    def __init__(self, registry_path="Systems/core/upgrade_registry/registry.json"):
        self.registry_path = registry_path
        self.log = []

    def detect_ship_class(self, ship_path):
        if "viren" in ship_path.lower():
            return "Viren Prime"
        elif "guardian" in ship_path.lower():
            return "Guardian"
        elif "orc" in ship_path.lower():
            return "Relay / ORC"
        elif "memory" in ship_path.lower():
            return "Memory Ship"
        elif "skill" in ship_path.lower():
            return "Skill Module"
        else:
            return "Unknown"

    def upgrade_logic(self, ship_class, ship_path):
        decision = "No action taken"
        if ship_class == "Viren Prime":
            decision = "Verify alpha status. Soft reload only if non-critical modules updated."
        elif ship_class == "Guardian":
            decision = "Audit scrolls and affirm identity. Upgrade allowed with checksum."
        elif ship_class == "Relay / ORC":
            decision = "Allow hotpatch, restart only relay services."
        elif ship_class == "Memory Ship":
            decision = "Safe to update. Backup before patch."
        elif ship_class == "Skill Module":
            decision = "Apply update immediately. No restart required."
        else:
            decision = "Manual review required."

        self.log_upgrade(ship_class, ship_path, decision)
        return decision

    def log_upgrade(self, ship_class, ship_path, decision):
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "ship_class": ship_class,
            "ship_path": ship_path,
            "decision": decision
        }
        self.log.append(entry)
        self._write_registry(entry)

    def _write_registry(self, entry):
        os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
        if os.path.exists(self.registry_path):
            with open(self.registry_path, "r") as f:
                data = json.load(f)
        else:
            data = []

        data.append(entry)
        with open(self.registry_path, "w") as f:
            json.dump(data, f, indent=2)

# Optional test runner
if __name__ == "__main__":
    sui = ShipUpgradeIntelligence()
    ship_path = "Systems/nexus_core/skills/VirenCore"
    ship_class = sui.detect_ship_class(ship_path)
    decision = sui.upgrade_logic(ship_class, ship_path)
    print(f"Upgrade Decision for {ship_class}: {decision}")
