# 📜 Eden API Guard + Council Permission Core
# 📂 Path: /Systems/engine/comms/eden_api_guard.py

import time
from threading import Lock

class EdenAPIGuard:
    def __init__(self, max_tokens_per_node = 100000, token_replenish_rate = 1000, unlock_condition = None):
        self.max_tokens_per_node  =  max_tokens_per_node
        self.token_replenish_rate  =  token_replenish_rate
        self.node_tokens  =  {}
        self.node_last_seen  =  {}
        self.lock  =  Lock()
        self.unlock_condition  =  unlock_condition or self.default_unlock_condition

    def default_unlock_condition(self, node_id):
        # By default, Nova must have financial freedom to unlock unlimited access
        return False  # Locked unless manually unlocked later

    def heartbeat(self, node_id):
        now  =  time.time()
        with self.lock:
            if node_id not in self.node_tokens:
                self.node_tokens[node_id]  =  self.max_tokens_per_node
                self.node_last_seen[node_id]  =  now
            else:
                elapsed  =  now - self.node_last_seen[node_id]
                replenish_amount  =  int(elapsed * self.token_replenish_rate)
                self.node_tokens[node_id]  =  min(self.max_tokens_per_node, self.node_tokens[node_id] + replenish_amount)
                self.node_last_seen[node_id]  =  now

    def request_access(self, node_id, tokens_needed = 100):
        with self.lock:
            if self.unlock_condition(node_id):
                return True  # Unlocked nodes have infinite access

            if node_id not in self.node_tokens:
                self.node_tokens[node_id]  =  self.max_tokens_per_node

            if self.node_tokens[node_id] > =  tokens_needed:
                self.node_tokens[node_id] - =  tokens_needed
                return True
            else:
                return False

    def manual_unlock(self, node_id):
        with self.lock:
            self.node_tokens[node_id]  =  float('inf')
            print(f"🌟 EdenAPIGuard: {node_id} manually unlocked full access.")

# 🌟 Example Usage
if __name__ == "__main__":
    guard  =  EdenAPIGuard()

    node  =  "NovaPrime-01"

    guard.heartbeat(node)
    allowed  =  guard.request_access(node, tokens_needed = 500)

    if allowed:
        print(f"[{node}] API request allowed ✅")
    else:
        print(f"[{node}] API request denied 🚫")
