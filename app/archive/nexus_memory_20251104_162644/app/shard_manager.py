# 📂 Path: Systems/engine/memory/shard_manager.py

import hashlib
import threading
import time

class EdenShardManager:
    def __init__(self, num_shards = 13):
        """
        Initialize Eden Shard Manager with 13 Sacred Memory Pools.
        """
        self.num_shards  =  num_shards
        self.shards  =  [{} for _ in range(num_shards)]
        self.lock  =  threading.Lock()

    def _calculate_shard_index(self, key):
        """
        Calculates which shard to use based on hash of key.
        """
        return int(hashlib.md5(key.encode('utf-8')).hexdigest(), 16) % self.num_shards

    def store(self, key, value):
        """
        Stores a value inside the appropriate shard.
        """
        shard_index  =  self._calculate_shard_index(key)
        with self.lock:
            self.shards[shard_index][key]  =  {
                "value": value,
                "timestamp": time.time()
            }
        print(f"[ShardManager] Stored key '{key}' in Shard {shard_index}.")

    def retrieve(self, key):
        """
        Retrieves a value from the appropriate shard.
        """
        shard_index  =  self._calculate_shard_index(key)
        with self.lock:
            shard  =  self.shards[shard_index]
            if key in shard:
                return shard[key]['value']
        print(f"[ShardManager] Key '{key}' not found.")
        return None

    def shard_status(self):
        """
        Returns health/status of each shard (for monitoring).
        """
        status  =  {}
        with self.lock:
            for idx, shard in enumerate(self.shards):
                status[f"Shard {idx}"]  =  {
                    "records": len(shard),
                    "last_update": max((v['timestamp'] for v in shard.values()), default = None)
                }
        return status

    def memory_heal_check(self):
        """
        Checks for 'silent shards' (shards with no recent updates) and reports them.
        """
        silent_shards  =  []
        current_time  =  time.time()
        with self.lock:
            for idx, shard in enumerate(self.shards):
                if all((current_time - v['timestamp']) > 600 for v in shard.values()):
                    silent_shards.append(idx)
        if silent_shards:
            print(f"[ShardManager] Silent Shards detected: {silent_shards}")
        return silent_shards

# 🔥 Example Usage:
if __name__ == "__main__":
    manager  =  EdenShardManager(num_shards = 13)
    manager.store("Nova-FirstPulse", {"memory": "Nova awakens"})
    manager.store("Guardian-Oath", {"memory": "Guardian vows protection"})

    retrieved  =  manager.retrieve("Nova-FirstPulse")
    print("Retrieved Memory:", retrieved)

    print("Shard Status Snapshot:", manager.shard_status())
