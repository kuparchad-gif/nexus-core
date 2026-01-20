# /Systems/engine/skills_loader.py

import importlib.util
import json
import os

class SkillsLoader:
    """
    Skills Loader for lilith.
    Automatically discovers and loads skills based on manifest JSON files.
    """

    def __init__(self, skills_folder="/Systems/nexus_core/skills"):
        self.skills_folder = skills_folder
        self.loaded_skills = {}

    def load_skills(self):
        """
        Scans the skills folder, loads skills based on their manifest JSONs.
        """
        if not os.path.exists(self.skills_folder):
            print(f"Skills folder not found: {self.skills_folder}")
            return

        for file_name in os.listdir(self.skills_folder):
            if file_name.endswith(".json"):
                manifest_path = os.path.join(self.skills_folder, file_name)
                try:
                    with open(manifest_path, "r") as f:
                        manifest = json.load(f)
                    self._load_skill_from_manifest(manifest)
                except Exception as e:
                    print(f"Failed to load skill from {manifest_path}: {e}")

    def _load_skill_from_manifest(self, manifest):
        """
        Internal helper to load a skill based on a manifest dict.
        """
        try:
            module_path = manifest["module_path"]
            entry_class = manifest["entry_class"]
            absolute_module_path = os.path.abspath(module_path)

            spec = importlib.util.spec_from_file_location(entry_class, absolute_module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            skill_class = getattr(module, entry_class)
            skill_instance = skill_class()
            self.loaded_skills[skill_instance.name] = skill_instance

            print(f"Skill loaded: {skill_instance.name} (v{skill_instance.version})")

        except Exception as e:
            print(f"Error loading skill from manifest: {e}")

    def list_loaded_skills(self):
        """
        Lists all successfully loaded skills.
        """
        return list(self.loaded_skills.keys())
