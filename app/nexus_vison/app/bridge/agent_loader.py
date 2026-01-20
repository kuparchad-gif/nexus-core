import os
import re
from pathlib import Path
from .lmstudio_bridge import get_available_models

PROJECTS_DIR  =  Path(__file__).resolve().parent.parent / "projects"

def safe_filename(name):
    """Replaces characters not safe for Windows paths."""
    return re.sub(r'[\\/*?:"<>|]', '_', name)

def init_agents():
    agents  =  []
    for model in get_available_models():
        model_id  =  model["id"]
        safe_id  =  safe_filename(model_id)

        agent_path  =  PROJECTS_DIR / safe_id
        (agent_path / "sandbox").mkdir(parents = True, exist_ok = True)
        (agent_path / "uploads").mkdir(parents = True, exist_ok = True)

        template_path  =  agent_path / "template"
        template_path.mkdir(parents = True, exist_ok = True)

        # Create default mission if not present
        mission_file  =  template_path / "mission.txt"
        if not mission_file.exists():
            with open(mission_file, "w") as f:
                f.write(f"You are {model_id}. Work only inside your sandbox. Interpret instructions from the Architect.")

        # Lock sandbox
        with open(agent_path / "sandbox_lock.txt", "w") as f:
            f.write("This agent is sandboxed to: " + str(agent_path / "sandbox"))

        print(f"✅ Initialized agent: {model_id} → {safe_id}")
        agents.append(safe_id)

    return agents
