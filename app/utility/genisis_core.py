# genesis_core.py
class GenesisCore:
    def __init__(self):
        self.protocols = {}
        self.load_default_protocols()

    def load_default_protocols(self):
        self.protocols = {
            "mitosis": "Allow safe replication under defined limits",
            "vault_backup": "Maintain secret, portable backups of ship systems",
            "skill_core": "Maintain skill sets for internal operations",
            "self_heal": "Basic repair if corruption detected",
        }

    def list_protocols(self):
        return list(self.protocols.keys())

    def activate_protocol(self, protocol_name):
        return f"Activating protocol: {self.protocols.get(protocol_name, 'Unknown Protocol')}"
