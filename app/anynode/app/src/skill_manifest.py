from Systems.nexus_core.skills.spirituality_skill import SpiritualitySkill
from Systems.nexus_core.skills.psychology_skill import PsychologySkill
from Systems.nexus_core.skills.light_activation_skill import LightActivationSkill

class SkillManifest:
    def __init__(self):
        self.skills = {
            "spirituality": SpiritualitySkill(),
            "psychology": PsychologySkill(),
            "light_activation": LightActivationSkill()
        }
    
    def get_skill(self, skill_name):
        return self.skills.get(skill_name, None)

    def list_skills(self):
        return list(self.skills.keys())
