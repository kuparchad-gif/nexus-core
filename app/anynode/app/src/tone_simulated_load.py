
import random
import time

class ToneSimulatedLoad:
    def __init__(self, tone_agents):
        self.tone_agents = tone_agents
        self.emotions = ["joy", "fear", "sadness", "hope", "guilt", "awe"]

    def simulate(self, rounds=5):
        for _ in range(rounds):
            agent = random.choice(self.tone_agents)
            emotion = random.choice(self.emotions)
            intensity = random.randint(1, 10)
            print(f"Simulating: {agent.agent_id} emitting {emotion} at {intensity}")
            output = agent.emit(emotion, intensity)
            print("Result:", output)
            time.sleep(1)
