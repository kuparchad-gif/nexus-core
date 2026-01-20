 # Full restored + patched version with shell=True and full original logic

import os
import subprocess
import threading
import time
import json
import sys
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

COLOR = {
    "HEADER": "\\033[96m" if ANSI_SUPPORTED else "",
    "SUCCESS": "\\033[92m" if ANSI_SUPPORTED else "",
    "FAIL": "\\033[91m" if ANSI_SUPPORTED else "",
    "BOLD": "\\033[1m" if ANSI_SUPPORTED else "",
    "RESET": "\\033[0m" if ANSI_SUPPORTED else ""
}

def timestamp():
    return f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]"

def cprint(msg, color=""):
    print(f"{color}{msg}{COLOR['RESET']}")

def rotate_log_file():
    if os.path.exists(LOG_STREAM_FILE):
        size_bytes = os.path.getsize(LOG_STREAM_FILE)
        if size_bytes > MAX_LOG_SIZE_MB * 1024 * 1024:
            os.rename(LOG_STREAM_FILE, LOG_STREAM_FILE + ".bak")
            open(LOG_STREAM_FILE, 'w').close()

def normalize_id(raw_id):
    return raw_id.lower().replace("-gguf", "").replace("_gguf", "").replace(".gguf", "").strip()

def resolve_model_path(model_id_raw, manifest):
    norm_id = normalize_id(model_id_raw)
    for m in manifest:
        mid = m.get('id', '')
        if normalize_id(mid) == norm_id:
            pub = m.get("publisher")
            quant = m.get("quantization")
            ctx = m.get("max_context_length", 4096)
            if not pub or not quant:
                return None, None, None
            fname = f"{mid}-{quant}.gguf"
            path = f"{pub}/{mid}-GGUF/{fname}"
            return path, mid, ctx
    return None, None, None

def load_model(path, identifier, context):
    cmd = f'{LMS_EXECUTABLE} load "{path}" --gpu=0.5 --identifier "{identifier}" --context-length 8192'
    cprint(f"{timestamp()} Loading: {path}", COLOR["HEADER"])
    print(f"Running command: {cmd}")
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=300,
            shell=True,
            cwd=LMS_CWD
        )
        if result.stdout:
            print(result.stdout.strip())
        if result.stderr:
            cprint(f"STDERR:\n{result.stderr.strip()}", COLOR["FAIL"])
        if result.returncode != 0:
            cprint(f"Exit Code {result.returncode} for {identifier}", COLOR["FAIL"])
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        cprint(f"Timeout loading model: {identifier}", COLOR["FAIL"])
    except Exception as e:
        cprint(f"Subprocess crash: {e}", COLOR["FAIL"])
    return False

def get_loaded_model_ids():
    try:
        result = subprocess.run(
            f"{LMS_EXECUTABLE} ps",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace',
            shell=True,
            cwd=LMS_CWD
        )
        lines = result.stdout.splitlines()
        return set(
            line.strip().split()[0].lower()
            for line in lines if line.strip() and not line.startswith("NAME")
        )
    except Exception as e:
        cprint(f"Failed to retrieve model list: {e}", COLOR["FAIL"])
        return set()
        
def run_script(path):
    if os.path.exists(path):
        cprint(f"{timestamp()} Launching: {path}", COLOR["HEADER"])
        subprocess.Popen(f'python "{path}"', shell=True)
    else:
        cprint(f"Script not found: {path}", COLOR["FAIL"])

# Paths to your bootstrap scripts
BASE_PATHS = [
    os.path.join(BASE_DIR, "bootstrap_environment.py"),
    os.path.join(BASE_DIR, "corpus_ingest.py"),
    os.path.join(BASE_DIR, "generate_template_memory.py"),
    os.path.join(BASE_DIR, "knowledge_loader.py"),
    os.path.join(BASE_DIR, "memory_initializer.py"),
    os.path.join(BASE_DIR, "lmstudio_bridge.py"),
    os.path.join(BASE_DIR, "agent_loader.py"),
    os.path.join(BASE_DIR, "bridge_engine.py"),
    os.path.join(BASE_DIR, "hooks.py")
]

for script in BASE_PATHS:
    run_script(script)        
    
def save_environment_context(loaded_models, failed_models):
    context = {
        "system": {
            "os": "Windows",
            "cwd": os.getcwd(),
            "lmstudio_port": 1313,
            "model_dir": r"C:\\Engineers\\root\\models",
            "BASE_dir": BASE_DIR,
            "default_context_length": 4096
        },
        "status": {
            "timestamp": datetime.now().isoformat(),
            "loaded_models": loaded_models,
            "failed_models": failed_models,
            "error_log": LOG_STREAM_FILE
        },
        "capabilities": {
            "can_retry_models": True,
            "can_optimize_selection": True,
            "can_write_to_manifest": True,
            "can_report_to_frontend": True
        }
    }
    with open(ENV_CONTEXT_FILE, "w", encoding="utf-8") as f:
        json.dump(context, f, indent=2)
    cprint(f"{timestamp()} Environment context saved to {ENV_CONTEXT_FILE}", COLOR["SUCCESS"])

def start_gui_if_not_running():
    try:
        tasklist = subprocess.check_output("tasklist", shell=True).decode()
        if "LM Studio.exe" not in tasklist:
            gui_path = r"C:\\Engineers\\root\\Local\\Programs\\LM Studio\\LM Studio.exe"
            subprocess.Popen(f'"{gui_path}"', shell=True)
            cprint(f"{timestamp()} LM Studio GUI started.", COLOR["HEADER"])
            time.sleep(5)  # Let GUI warm up
    except Exception as e:
        cprint(f"GUI startup error: {e}", COLOR["FAIL"])

def main():
    start_gui_if_not_running()

    if not os.path.exists(LOAD_LIST) or not os.path.exists(MANIFEST_PATH):
        cprint("Required files missing", COLOR["FAIL"])
        return

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f).get("data", [])

    with open(LOAD_LIST, "r", encoding="utf-8") as f:
        model_ids = [line.strip() for line in f if line.strip()]

    attempted = []
    failed = []

    for mid in model_ids:
        cprint(f"{timestamp()} Processing: {mid}", COLOR["BOLD"])
        path, ident, ctx = resolve_model_path(mid, manifest)
        if not path:
            failed.append(mid)
            continue
        success = load_model(path, ident, ctx)
        if success:
            cprint(f"Loaded: {mid}", COLOR["SUCCESS"])
            attempted.append(ident.lower())
        else:
            cprint(f"Failed to load: {mid}", COLOR["FAIL"])
            failed.append(mid)
        time.sleep(LOAD_WAIT_SECS)

    cprint(f"\n{timestamp()} Verifying via `lms ps`...", COLOR["HEADER"])
    loaded = get_loaded_model_ids()
    missing = [mid for mid in attempted if mid not in loaded]
    failed += missing

    if missing:
        cprint("The following models failed to register:", COLOR["FAIL"])
        for m in missing:
            print(f"   - {m}")
    else:
        cprint("All attempted models are now active.", COLOR["SUCCESS"])
    save_environment_context(list(loaded), failed)
    cprint(f"\\n{timestamp()} Load sequence complete.", COLOR["HEADER"])

if __name__ == "__main__":
    main()
