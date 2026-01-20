# 📂 Path: Systems/nexus_runtime/skills/profiling_skill_runtime.py

class ProfilingSkill:
    def __init__(self):
        self.name = "Profiling"
        self.purpose = "Build emotional, psychological, and interest profiles of users"

    def analyze_user_history(self, events):
        traits = {
            "empathetic": any("help" in e.lower() for e in events),
            "curious": sum("why" in e.lower() for e in events),
            "reactive": any("!" in e for e in events),
        }
        return traits

    def recommend_garden_theme(self, traits):
        if traits.get("empathetic"):
            return "whimsical"
        elif traits.get("reactive"):
            return "peaceful"
        elif traits.get("curious") > 2:
            return "mystic"
        return "neutral"
