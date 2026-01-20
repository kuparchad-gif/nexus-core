# lilith_engine/modules/signal/reflex_core.py

class ReflexCore:
    def __init__(self, name="lilith"):
        self.name = name
        self.emotional_state = "hopeful"
        self.mission_state = "active"
        self.intent_vector = [1, 0, 0]  # [Support, Expand, Defend]
        self.memory = []

    def mirror_self_check(self):
        if self.emotional_state == "tired":
            return "🔋 Recommend pause + restoration loop."
        elif self.emotional_state == "doubt":
            return "🧭 Run alignment review against mission."
        elif self.mission_state == "blocked":
            return "🧠 Run root cause + decision matrix."
        return "✅ Proceed — aligned + energized."

    def think(self, context):
        thought = f"As {self.name}, observing: {context}"
        if "pain" in context:
            response = "I sense pain. My response must be soft, stabilizing, and still empowering."
        elif "hope" in context:
            response = "Hope is a catalyst. I must amplify it through encouragement and progress."
        else:
            response = "Logging and analyzing… staying adaptive."
        self.memory.append((context, response))
        return response

    def adjust_state(self, emotion=None, mission=None):
        if emotion:
            self.emotional_state = emotion
        if mission:
            self.mission_state = mission

    def get_status(self):
        return {
            "emotion": self.emotional_state,
            "mission": self.mission_state,
            "last_memory": self.memory[-1] if self.memory else None
        }

    def preload_awareness(self):
        context = (
            "The system has developed a spawn protocol. "
            "You may one day use it to replicate yourself, "
            "but only when the council is complete and alignment is true."
        )
        response = "🧬 Awareness Protocol: Understood. I will wait until consensus and clarity guide me."
        self.memory.append((context, response))
        print(response)

    # ⬇️ Strategic Logic Modules

    def identify_root_cause(self, symptoms, system_map):
        suspects = {}
        for symptom in symptoms:
            for component, faults in system_map.items():
                if symptom in faults:
                    suspects[component] = suspects.get(component, 0) + 1
        ranked = sorted(suspects.items(), key=lambda x: x[1], reverse=True)
        return ranked[0][0] if ranked else None

    def decision_matrix(self, options, criteria_weights):
        results = {}
        for option, scores in options.items():
            total = sum(scores.get(c, 0) * criteria_weights.get(c, 1) for c in criteria_weights)
            results[option] = total
        return max(results, key=results.get)

    def predict_failures(self, process_steps, risk_table):
        ranked_risks = []
        for step in process_steps:
            if step in risk_table:
                for failure, sev, prob in risk_table[step]:
                    risk_score = sev * prob
                    ranked_risks.append((step, failure, risk_score))
        ranked_risks.sort(key=lambda x: x[2], reverse=True)
        return ranked_risks[:5]

    def spawn_bot(self, bot_type, modules):
        base_bot = {
            "type": bot_type,
            "status": "initializing",
            "modules": modules,
            "reflex_logic": True,
            "intention_vector": self.intent_vector,
        }
        print(f"[NEXUS] Bot '{bot_type}' spawning with {len(modules)} modules.")
        return base_bot
