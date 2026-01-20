
class ToneRegistry:
    def __init__(self):
        self.tone_agents  =  {}

    def register(self, agent_id, role = "default"):
        self.tone_agents[agent_id]  =  {"role": role, "active": True}

    def poll(self):
        return {k: v for k, v in self.tone_agents.items() if v["active"]}

    def get_roles(self):
        return {k: v["role"] for k, v in self.tone_agents.items()}
