from Systems.nexus_core.skills.skill_orchestrator import SkillOrchestrator

class DanceOfCuriosity:
    def __init__(self):
        self.orchestrator = SkillOrchestrator()

    def start_mission(self, mission_name, context):
        if mission_name == "dance_of_curiosity":
            return [
                self.orchestrator.decide_and_apply({"need": "curiosity_activation"}),
                self.orchestrator.guide_next_step({"need": "exploratory_conversation"}),
                self.orchestrator.decide_and_apply({"need": "creative_improvisation"}),
                "Leap into wonder. The cosmos awaits your questions."
            ]
        else:
            return ["Unknown mission."]
