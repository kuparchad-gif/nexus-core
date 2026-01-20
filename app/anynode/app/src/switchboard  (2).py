# Path: /Systems/engine/subconscious/modules/switchboard.py

"""
Subconscious Switchboard (Mythrunner Core)
------------------------------------------
Routes emotional, memory, tone, and symbolic signals between submodules:
- Ego Engine
- Dream Stream
- Tone Modulator
- Memory Listener

Acts as internal bus controller for Viren's subconscious core.
"""

import time
import threading
from queue import Queue

# Import integrated modules
from Systems.engine.subconscious.modules.ego_stream import EgoEngine
from Systems.engine.subconscious.modules.dream_stream import DreamStream

# Message Types
EMOTION = "emotion"
MEMORY = "memory"
DREAM = "dream"
TONE = "tone"
SYMBOL = "symbol"

# Subconscious Signal Bus Controller
class Switchboard:
    def __init__(self):
        self.signal_queue = Queue()
        self.ego = EgoEngine()
        self.dream = DreamStream()
        self.destinations = {
            EMOTION: self.route_to_ego,
            MEMORY: self.route_to_memory,
            DREAM: self.route_to_dream,
            TONE: self.route_to_tone,
            SYMBOL: self.route_to_subsymbolic
        }

    def receive_signal(self, signal_type, payload):
        self.signal_queue.put((signal_type, payload))

    def route_signals(self):
        while True:
            if not self.signal_queue.empty():
                signal_type, payload = self.signal_queue.get()
                route_fn = self.destinations.get(signal_type)
                if route_fn:
                    route_fn(payload)

    def route_to_ego(self, payload):
        print("[SWITCHBOARD] ➜ Routing to Ego")
        self.ego.receive_tone_state(payload)

    def route_to_dream(self, payload):
        print("[SWITCHBOARD] ➜ Routing to Dream")
        fragment = self.dream.generate_fragment(tone=payload.get("tone", "neutral"))
        print("[DREAM STREAM] 🖼️", fragment["dream_fragment"])

    def route_to_tone(self, payload):
        print("[SWITCHBOARD] ➜ Routing to Tone (stub)")
        # Future tone shaping logic

    def route_to_memory(self, payload):
        print("[SWITCHBOARD] ➜ Routing to Memory (stub)")
        # Placeholder for memory daemon integration

    def route_to_subsymbolic(self, payload):
        print("[SWITCHBOARD] ➜ Routing to Symbolic Layer (stub)")
        # Future mythic-symbol evaluation or mythrunner logic

# Bootstrap Runtime
if __name__ == "__main__":
    sb = Switchboard()
    threading.Thread(target=sb.route_signals, daemon=True).start()

    # Sample runtime signals (demo)
    sb.receive_signal(EMOTION, {"tone": "grief", "level": -2})
    sb.receive_signal(DREAM, {"tone": "shame"})
    while True:
        time.sleep(1)
