
# Universal Skill Loader for Viren
import os
import importlib.util

class UniversalSkillLoader:
    def __init__(self, skill_dir='Systems/nexus_core/skills/'):
        self.skill_dir = skill_dir

    def load_skill(self, skill_name):
        skill_path = os.path.join(self.skill_dir, skill_name, f"{skill_name}_runtime.py")
        if not os.path.exists(skill_path):
            return f"Skill {skill_name} not found."

        spec = importlib.util.spec_from_file_location(skill_name, skill_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return f"Skill {skill_name} loaded successfully."

    def list_skills(self):
        return [d for d in os.listdir(self.skill_dir) if os.path.isdir(os.path.join(self.skill_dir, d))]
