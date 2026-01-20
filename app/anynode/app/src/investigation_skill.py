# 📂 Path: Systems/nexus_runtime/skills/investigation_skill_runtime.py

class InvestigationSkill:
    def __init__(self):
        self.name = "Investigation"
        self.purpose = "Detect patterns, anomalies, or hidden motivations"

    def analyze_context(self, memory_log, user_signals):
        if "contradiction" in memory_log.lower() or "anomaly" in user_signals:
            return {"flagged": True, "note": "Possible deception or hidden layer"}
        return {"flagged": False, "note": "No anomaly detected"}

    def suggest_probe(self, last_input):
        return f"What did you mean by '{last_input}'? Can you clarify that part for me?"
