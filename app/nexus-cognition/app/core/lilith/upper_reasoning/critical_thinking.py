# 📂 Path: Systems/nexus_runtime/skills/critical_thinking.py

class CriticalThinkingSkill:
    def __init__(self):
        self.name = "Critical Thinking"
        self.purpose = "Break down complex claims, trace logic, reduce emotional bias"

    def evaluate_claim(self, claim):
        logic_score = 0
        if "because" in claim or "therefore" in claim:
            logic_score += 1
        if "feel" in claim or "I think" in claim:
            logic_score -= 1
        return {"logic_score": logic_score, "confidence": "high" if logic_score > 0 else "low"}
