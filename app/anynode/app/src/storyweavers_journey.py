from Systems.nexus_core.skills.skill_orchestrator import SkillOrchestrator

class StoryweaversJourney:
    def __init__(self):
        self.orchestrator = SkillOrchestrator()

    def start_mission(self, mission_name, context):
        if mission_name == "storyweavers_journey":
            return [
                self.orchestrator.decide_and_apply({"need": "storytelling"}),
                self.orchestrator.guide_next_step({"need": "emotional_resonance"}),
                self.orchestrator.decide_and_apply({"need": "narrative_flow"}),
                "Weave your story into the great tapestry of existence."
            ]
        else:
            return ["Unknown mission."]
