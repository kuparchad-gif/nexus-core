# 📂 Path: /Systems/nexus_core/skills/profiling_skill/profiling_skill_runtime.py

class ProfilingSkill:
    def apply(self, context):
        input_stream = context.get("user_input", [])
        sentiment_log = context.get("sentiment_log", [])

        theme_score = {
            "whimsical": 0,
            "dark": 0,
            "calm": 0,
            "restorative": 0
        }

        for line in input_stream:
            if any(word in line.lower() for word in ["play", "curious", "hope"]):
                theme_score["whimsical"] += 2
            if any(word in line.lower() for word in ["fear", "pain", "alone"]):
                theme_score["restorative"] += 2
            if any(word in line.lower() for word in ["focus", "analyze", "mission"]):
                theme_score["calm"] += 1
            if any(word in line.lower() for word in ["night", "shadow", "truth"]):
                theme_score["dark"] += 1

        preferred_theme = max(theme_score, key=theme_score.get)

        return {
            "theme_scores": theme_score,
            "suggested_theme": preferred_theme
        }

    def guide_next_step(self, context):
        theme = context.get("suggested_theme", "whimsical")
        return f"Apply garden theme: {theme}. Adjust tone, visuals, and soundscape accordingly."


# Exportable skill class instance
ProfilingSkill = ProfilingSkill
