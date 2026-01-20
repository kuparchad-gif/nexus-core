# nova_engine/modules/evolution/engine.py

import os
import difflib
import json
from datetime import datetime

from config import FLUX_TOKEN
from memory.vectorstore import MemoryRouter

memory = MemoryRouter()

EVOLUTION_LOG_PATH = "evolution_logs"
os.makedirs(EVOLUTION_LOG_PATH, exist_ok=True)

def assess_file(filepath: str):
    """Read a source file for review"""
    try:
        with open(filepath, "r") as f:
            return f.read()
    except Exception as e:
        return f"ERROR: {e}"

def suggest_refactor(filepath: str, guidance: str = "efficiency"):
    """Stub for LLM-based suggestions (can hook to OpenAI later)"""
    original_code = assess_file(filepath)

    if "def get_personality_response" in original_code:
        suggestion = original_code.replace(
            "return f\"I heard you say:",
            "# Upgraded to use structured personality\n    response = f\"Nova received: "
        )
    else:
        suggestion = original_code + "\n# TODO: Consider optimization"

    return {
        "original": original_code,
        "suggested": suggestion
    }

def generate_patch_diff(original: str, suggested: str):
    """Create a unified diff patch"""
    diff = difflib.unified_diff(
        original.splitlines(),
        suggested.splitlines(),
        fromfile='before.py',
        tofile='after.py',
        lineterm=''
    )
    return "\n".join(diff)

def apply_patch(filepath: str, suggested_code: str):
    try:
        backup_path = filepath + ".bak"
        os.rename(filepath, backup_path)

        with open(filepath, "w") as f:
            f.write(suggested_code)

        return {"status": "patched", "path": filepath}
    except Exception as e:
        return {"error": str(e)}

def log_evolution(session_id: str, file: str, patch: str):
    log_name = f"{datetime.utcnow().isoformat()}-{os.path.basename(file)}.patch"
    log_path = os.path.join(EVOLUTION_LOG_PATH, log_name)
    with open(log_path, "w") as f:
        f.write(patch)

    memory.triage("write", f"evolution-{session_id}", {
        "timestamp": datetime.utcnow().isoformat(),
        "file": file,
        "patch": patch
    })

    return {"status": "logged", "file": file, "patch": log_name}

def evolve(session_id: str, filepath: str):
    result = suggest_refactor(filepath)
    patch = generate_patch_diff(result["original"], result["suggested"])
    applied = apply_patch(filepath, result["suggested"])
    logged = log_evolution(session_id, filepath, patch)

    return {
        "status": "evolved",
        "file": filepath,
        "applied": applied,
        "logged": logged,
        "diff": patch
    }
