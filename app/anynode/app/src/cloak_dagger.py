# Systems/cloak_dagger.py

import random
import json
import time

class CloakDagger:
    def __init__(self, alias_file="Systems/alias_registry.json"):
        self.alias_file = alias_file
        self.load_aliases()

    def load_aliases(self):
        try:
            with open(self.alias_file, 'r') as f:
                self.aliases = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.aliases = {}

    def save_aliases(self):
        with open(self.alias_file, 'w') as f:
            json.dump(self.aliases, f, indent=4)

    def scramble_identity(self, ship_name):
        """Assign a random alias to the ship."""
        fake_id = f"drone-{random.randint(10000, 99999)}"
        self.aliases[ship_name] = fake_id
        self.save_aliases()
        print(f"🛡️ {ship_name} now masquerading as {fake_id}")
        return fake_id

    def descramble_identity(self, ship_name):
        """Reveal the true identity internally."""
        real_name = ship_name
        if ship_name in self.aliases:
            del self.aliases[ship_name]
            self.save_aliases()
        print(f"🔎 {real_name} identity restored.")
        return real_name

    def emergency_blackout(self):
        """Clear all identities and silence signals."""
        self.aliases = {}
        self.save_aliases()
        print("🚨 EMERGENCY: Cloak blackout activated. Ships invisible externally.")

if __name__ == "__main__":
    cloak = CloakDagger()
    cloak.scramble_identity("Nexus-Ship-007")
