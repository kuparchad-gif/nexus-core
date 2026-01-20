import os
import sys
import requests
import re
from pathlib import Path
from datetime import datetime
from pathlib import Path
from session_manager import load_sessions, save_session, append_to_latest_session

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, "..")
BRIDGE_DIR = os.path.join(ROOT_DIR, "bridge")
PUBLIC_DIR = os.path.join(ROOT_DIR, "public")
MEMORY_DIR = os.path.join(ROOT_DIR, "memory")
DB_PATH = os.path.join(MEMORY_DIR, "engineers.db")
MODELS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "models"))
LOADED_MODELS_JSON = os.path.join(PUBLIC_DIR, "loaded_models.json")
LOAD_LIST = os.path.join(BASE_DIR, 'models_to_load.txt')
MANIFEST_PATH = r"C:\\Engineers\\root\\models\\model_manifest.json"
LOG_STREAM_FILE = r"C:\\Engineers\\root\\logs\\lms_prompt_debug.log"
ENV_CONTEXT_FILE = os.path.join(BASE_DIR, 'environment_context.json')
MAX_LOG_SIZE_MB = 20
LOAD_WAIT_SECS = 6
LMS_EXECUTABLE = "lms"
LMS_CWD = r"C:\\Engineers\\root\\Local\\Programs\\LM Studio\\resources\\app\\.webpack"
ANSI_SUPPORTED = sys.stdout.isatty() and os.name != "nt"
LMSTUDIO_BASE_URL = "http://localhost:1313"
PROJECTS_DIR = Path(__file__).resolve().parent.parent / "projects"

def safe_filename(name):
    """Replaces characters not safe for Windows paths."""
    return re.sub(r'[\\/*?:"<>|]', '_', name)

def init_agents():
    agents = []
    for model in get_available_models():
        model_id = model["id"]
        safe_id = safe_filename(model_id)

        agent_path = PROJECTS_DIR / safe_id
        (agent_path / "sandbox").mkdir(parents=True, exist_ok=True)
        (agent_path / "uploads").mkdir(parents=True, exist_ok=True)

        template_path = agent_path / "template"
        template_path.mkdir(parents=True, exist_ok=True)

        # Create default mission if not present
        mission_file = template_path / "mission.txt"
        if not mission_file.exists():
            with open(mission_file, "w") as f:
                f.write(f"You are {model_id}. Work only inside your sandbox. Interpret instructions from the Architect.")

        # Lock sandbox
        with open(agent_path / "sandbox_lock.txt", "w") as f:
            f.write("This agent is sandboxed to: " + str(agent_path / "sandbox"))

        print(f"✅ Initialized agent: {model_id} → {safe_id}")
        agents.append(safe_id)

    return agents
