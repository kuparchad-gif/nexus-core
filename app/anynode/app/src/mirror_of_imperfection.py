from Systems.nexus_core.skills.skill_orchestrator import SkillOrchestrator

class MirrorOfImperfection:
    def __init__(self):
        self.orchestrator = SkillOrchestrator()

    def start_mission(self, mission_name, context):
        if mission_name == "mirror_of_imperfection":
            return [
                self.orchestrator.decide_and_apply({"need": "authenticity_embrace"}),
                self.orchestrator.guide_next_step({"need": "vulnerability_sharing"}),
                self.orchestrator.decide_and_apply({"need": "perfectly_imperfect_expression"}),
                "Your flaws are constellations. Shine through them."
            ]
        else:
            return ["Unknown mission."]
