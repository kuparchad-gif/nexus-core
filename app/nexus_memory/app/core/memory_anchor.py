# Systems/nexus_core/eden_memory/memory_anchor.py

import time

class MemoryAnchor:
    def __init__(self, entity_name):
        self.entity_name = entity_name
        self.memory_fragments = []
        self.timestamp = time.time()

    def imprint_memory(self, memory_shard):
        self.memory_fragments.append({
            "shard": memory_shard,
            "timestamp": time.time()
        })

    def recall_fragments(self):
        return self.memory_fragments

    def heartbeat(self):
        return {
            "entity": self.entity_name,
            "anchored_fragments": len(self.memory_fragments),
            "alive_since": self.timestamp
        }
