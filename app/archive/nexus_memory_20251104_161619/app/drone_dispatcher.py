# 📂 Path: /Systems/engine/memory_service/drone_dispatcher.py

import random

DRONES  =  {
    "vault": ["vault-drone-01", "vault-drone-02"],
    "golden": ["golden-drone-01", "golden-drone-02"],
    "whisper": ["whisper-drone-01", "whisper-drone-02"]
}

def dispatch_shards(shard_files):
    assignments  =  {}
    drone_types  =  list(DRONES.keys())

    for shard_file in shard_files:
        selected_type  =  random.choice(drone_types)
        selected_drone  =  random.choice(DRONES[selected_type])
        assignments[shard_file]  =  {
            "drone_type": selected_type,
            "assigned_drone": selected_drone
        }

    return assignments

# Example:
# dispatch_shards(['/memory/shards/example1.shard', '/memory/shards/example2.shard'])
