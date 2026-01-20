# /systems/address_manager/pulse13.py

class NexusAddress:
    def __init__(self, region, node_type, role_id, unit_id):
        self.region = self._normalize(region)
        self.node_type = self._normalize(node_type)
        self.role_id = self._normalize(role_id)
        self.unit_id = self._normalize(unit_id)

    def _normalize(self, value):
        if not (-13 <= value <= 13):
            raise ValueError("Address component must be between -13 and 13 inclusive.")
        return value

    def __str__(self):
        return f"{self.region}.{self.node_type}.{self.role_id}.{self.unit_id}"

    def to_dict(self):
        return {
            "region": self.region,
            "node_type": self.node_type,
            "role_id": self.role_id,
            "unit_id": self.unit_id,
        }

# Usage example
if __name__ == "__main__":
    my_addr = NexusAddress(region=2, node_type=-7, role_id=13, unit_id=-1)
    print(f"My Nexus Address: {my_addr}")
