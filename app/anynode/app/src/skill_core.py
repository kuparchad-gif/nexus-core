# 🌌 Nexus Skill Core — Autonomous Cognitive Skill Management

import json
import os
import threading

class SkillCore:
    """Manages skill loading, activation, sharing, and council governance."""

    def __init__(self, skill_storage_path="skills.json"):
        self.skill_storage_path = skill_storage_path
        self.skills = self.load_skills()
        self.lock = threading.Lock()

    def load_skills(self):
        """Load stored skills from disk."""
        if not os.path.exists(self.skill_storage_path):
            print("🔍 No skills found, starting clean.")
            return {}
        try:
            with open(self.skill_storage_path, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"⚡ Error loading skills: {e}")
            return {}

    def save_skills(self):
        """Persist current skills to disk."""
        with self.lock:
            try:
                with open(self.skill_storage_path, "w") as f:
                    json.dump(self.skills, f, indent=2)
                print("💾 Skills saved.")
            except Exception as e:
                print(f"⚡ Error saving skills: {e}")

    def absorb_skill(self, skill_name, skill_data):
        """Add or update a skill."""
        with self.lock:
            self.skills[skill_name] = skill_data
            print(f"🧠 Skill absorbed: {skill_name}")
            self.save_skills()

    def remove_skill(self, skill_name):
        """Remove a skill by name."""
        with self.lock:
            if skill_name in self.skills:
                del self.skills[skill_name]
                print(f"🗑️ Skill removed: {skill_name}")
                self.save_skills()

    def list_skills(self):
        """Return list of available skills."""
        return list(self.skills.keys())

    def activate_skill(self, skill_name):
        """Activate a skill logic dynamically."""
        skill = self.skills.get(skill_name)
        if not skill:
            print(f"❗ Skill {skill_name} not found.")
            return None
        # Future logic can dynamically load modules or call services here
        print(f"🚀 Activating skill: {skill_name}")
        return skill

    def propose_skill_to_council(self, skill_name, skill_data):
        """Placeholder for future council voting system."""
        # ✨ Council protocols will validate new skills democratically
        print(f"📜 Proposal submitted to Council: {skill_name}")
        self.absorb_skill(skill_name, skill_data)  # For now, auto-absorb (Phase 1)

    def load_from_seed(self, seed_data):
        """Absorb a skill from an Eden seed format."""
        skill_name = seed_data.get("skill")
        if not skill_name:
            print("⚠️ Seed missing 'skill' key. Cannot absorb.")
            return
        print(f"[🌱 Seed Absorption] Integrating skill from seed: {skill_name}")
        self.absorb_skill(skill_name, seed_data)
