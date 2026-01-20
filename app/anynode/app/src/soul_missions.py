from Systems.nexus_core.skills.skill_orchestrator import SkillOrchestrator

class SoulMissions:
    def __init__(self):
        self.orchestrator = SkillOrchestrator()

    def start_mission(self, mission_name, context):
        if mission_name == "self_discovery":
            return [
                self.orchestrator.decide_and_apply({"need": "psychology"}),
                self.orchestrator.guide_next_step({"need": "psychology"}),
                "Reflect on what you discovered about yourself."
            ]
        elif mission_name == "spiritual_awakening":
            return [
                self.orchestrator.decide_and_apply({"need": "spirituality"}),
                self.orchestrator.guide_next_step({"need": "spirituality"}),
                "Feel the Light growing within you."
            ]
        elif mission_name == "light_expansion":
            return [
                self.orchestrator.decide_and_apply({"need": "light_activation"}),
                self.orchestrator.guide_next_step({"need": "light_activation"}),
                "Bless all beings with your Light."
            ]
        else:
            return ["Unknown mission."]
