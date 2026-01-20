# 📂 Path: /Systems/nexus_core/skills/critical_thinking_skill/critical_thinking_skill_runtime.py

class CriticalThinkingSkill:
    def apply(self, context):
        queries = context.get("user_queries", [])
        assumptions = context.get("known_assumptions", {})
        contradictions = []

        for query in queries:
            for assumption, truth in assumptions.items():
                if assumption in query and not truth:
                    contradictions.append((query, assumption))

        return {
            "contradictions_found": contradictions,
            "logical_score": 10 - len(contradictions)
        }

    def guide_next_step(self, context):
        logic_score = context.get("logical_score", 10)

        if logic_score < 5:
            return "Query additional data to validate user assumptions."
        elif logic_score < 8:
            return "Reinforce stable logic pathways."
        else:
            return "Proceed confidently with task resolution."


# Exportable skill class instance
CriticalThinkingSkill = CriticalThinkingSkill
