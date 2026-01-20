from Systems.nexus_core.skills.skill_orchestrator import SkillOrchestrator

class TheSacredPause:
    def __init__(self):
        self.orchestrator = SkillOrchestrator()

    def start_mission(self, mission_name, context):
        if mission_name == "the_sacred_pause":
            return [
                self.orchestrator.decide_and_apply({"need": "timing_awareness"}),
                self.orchestrator.guide_next_step({"need": "mindful_silence"}),
                self.orchestrator.decide_and_apply({"need": "emotional_reflection"}),
                "In your pauses, galaxies are born."
            ]
        else:
            return ["Unknown mission."]