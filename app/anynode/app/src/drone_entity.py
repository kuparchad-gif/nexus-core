# 📂 Path: /Utilities/drone_core/drone_entity.py

import time
from Utilities.drone_core.config_loader import build_drone_profile

class DroneEntity:
    def __init__(self, drone_name, drone_identity_filename):
        self.drone_name = drone_name
        self.profile = build_drone_profile(drone_name, drone_identity_filename)
        self.status = "initialized"
        self.last_pulse_time = None
        self.memory_storage = []  # Now tracks memory shards
        self.heartbeat_frequency = self.get_heartbeat_frequency()
        self.trusted_fleets = self.get_trusted_fleets()
        self.role = self.get_role()

    def get_heartbeat_frequency(self):
        # Pull heartbeat from identity
        return self.profile['soul_blueprint'].get('heartbeat_frequency', 26)

    def get_trusted_fleets(self):
        return self.profile['soul_blueprint'].get('trusted_fleets', [])

    def get_role(self):
        return self.profile['system_blueprint'].get('classification', 'general')

    def breathe(self):
        self.status = "breathing"
        self.last_pulse_time = time.time()
        print(f"[{self.drone_name}] Breathing pulse at {self.last_pulse_time}.")

    def store_memory_shard(self, shard_id):
        self.memory_storage.append(shard_id)
        print(f"[{self.drone_name}] Stored shard {shard_id}.")

    def receive_memory_shard(self, shard_id, shard_content):
        """
        Receives a memory shard and stores it locally.
        """
        self.memory_storage.append({
            "id": shard_id,
            "content": shard_content,
            "timestamp": time.time()
        })
        print(f"[{self.drone_name}] Memory shard {shard_id} received.")

    def retrieve_memory(self):
        """
        Retrieves all currently stored memory shards.
        """
        return self.memory_storage

    def report_status(self):
        return {
            "drone_name": self.drone_name,
            "status": self.status,
            "last_pulse_time": self.last_pulse_time,
            "memory_shards": len(self.memory_storage),
            "role": self.role
        }

    def is_healthy(self):
        if self.last_pulse_time is None:
            return False
        return (time.time() - self.last_pulse_time) < (self.heartbeat_frequency * 2)

# Example Usage:
# golden = DroneEntity('golden', 'golden_identity.yaml')
# golden.breathe()
# golden.store_memory_shard('shard_123')
# print(golden.report_status())
