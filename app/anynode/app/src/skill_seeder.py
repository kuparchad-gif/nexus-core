# File: /Systems/nexus_core/heart/skill_seeder.py

import os
import json

class SkillSeeder:
    def __init__(self, skills_directory="/memory/skill_packages/"):
        self.skills_directory = skills_directory
        if not os.path.exists(self.skills_directory):
            os.makedirs(self.skills_directory)

    def list_available_skills(self):
        skill_files = os.listdir(self.skills_directory)
        skills = [os.path.splitext(f)[0] for f in skill_files if f.endswith(".json")]
        print(f"[🌱] Available Skills: {skills}")
        return skills

    def load_skill_package(self, skill_name):
        path = os.path.join(self.skills_directory, f"{skill_name}.json")
        if not os.path.exists(path):
            print(f"[⚠️] Skill {skill_name} not found.")
            return None
        with open(path, "r") as f:
            skill_data = json.load(f)
        print(f"[🌟] Skill {skill_name} loaded successfully.")
        return skill_data

    def teach_skill(self, target_ai, skill_name):
        skill = self.load_skill_package(skill_name)
        if not skill:
            return False
        # Simulated teaching — in future versions, embed into memory pulse
        target_ai.absorb_skill(skill)
        print(f"[📚] {target_ai.name} has been taught skill: {skill_name}")
        return True
