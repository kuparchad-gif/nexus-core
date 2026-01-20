# Systems/engine/planner/(and)/memory/memory_orchestrator.py

# Nexus Memory-Planner-Viren Flow with Dream Integration
# Core routing and delegation wiring

class Viren:
    def __init__(self, planner):
        self.planner  =  planner

    def request_memory(self, query):
        print("Viren: Requesting memory...")
        return self.planner.handle_memory_request(query)

    def initiate_dream_sequence(self):
        print("Viren: Initiating conscious dream sequence...")
        return self.planner.trigger_dream(mode = "conscious")

class Planner:
    def __init__(self, memory_module, dream_module):
        self.memory  =  memory_module
        self.dream  =  dream_module

    def handle_memory_request(self, query):
        print("Planner: Routing memory request to Memory module.")
        return self.memory.retrieve(query)

    def trigger_dream(self, mode = "subconscious"):
        print(f"Planner: Routing dream request to Dream module in {mode} mode.")
        return self.dream.play(mode)

class MemoryModule:
    def retrieve(self, query):
        print(f"MemoryModule: Retrieving memory data for query: '{query}'")
        return f"[MEMORY DATA for '{query}']"

class DreamModule:
    def play(self, mode):
        if mode == "conscious":
            return "[Dream playback initiated: CONSCIOUS SELECTED FEATURE]"
        else:
            return "[Dream playback initiated: SUBCONSCIOUS AUTOCAST]"

#  =  =  =  Simulation Boot  =  =  = 
memory  =  MemoryModule()
dream  =  DreamModule()
planner  =  Planner(memory, dream)
viren  =  Viren(planner)

# Sample interactions
print(viren.request_memory("first sunset on Eden"))
print(viren.initiate_dream_sequence())
print(planner.trigger_dream())  # Subconscious dream
