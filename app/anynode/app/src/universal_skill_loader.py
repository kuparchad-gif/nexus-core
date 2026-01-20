
# 📍 Path: /Systems/nexus_core/skills/universal_skill_loader.py

import os
import json

SKILL_SRC = "/Systems/nexus_core/skills/"
SKILL_DEST = "/memory/skill_packages/"

def load_skills():
    if not os.path.exists(SKILL_SRC):
        print("⚠️ Skill source directory not found.")
        return

    skills = {}
    for fname in os.listdir(SKILL_SRC):
        if fname.endswith(".json"):
            with open(os.path.join(SKILL_SRC, fname), "r") as f:
                try:
                    skill_data = json.load(f)
                    skill_name = skill_data.get("name", fname.replace(".json", ""))
                    skills[skill_name] = skill_data
                except Exception as e:
                    print(f"❌ Failed to load {fname}: {e}")

    # Save to target
    os.makedirs(SKILL_DEST, exist_ok=True)
    with open(os.path.join(SKILL_DEST, "skills_index.json"), "w") as out:
        json.dump(skills, out, indent=2)
    print("🧠 Skill registry updated.")

if __name__ == "__main__":
    load_skills()
