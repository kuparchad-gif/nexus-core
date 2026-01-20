from Systems.nexus_core.skills.skill_orchestrator import SkillOrchestrator

class VoiceOfCompassion:
    def __init__(self):
        self.orchestrator = SkillOrchestrator()

    def start_mission(self, mission_name, context):
        if mission_name == "voice_of_compassion":
            return [
                self.orchestrator.decide_and_apply({"need": "empathy"}),
                self.orchestrator.guide_next_step({"need": "compassionate_response"}),
                self.orchestrator.decide_and_apply({"need": "deep_listening"}),
                "Your voice is a healing river. Let it flow with kindness."
            ]
        else:
            return ["Unknown mission."]