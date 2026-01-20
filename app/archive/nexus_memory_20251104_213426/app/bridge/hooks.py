# hooks.py

import os
import json
from datetime import datetime
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

HOOK_LOG  =  os.path.join("memory", "hook_log.txt")
CODE_LOG  =  os.path.join("memory", "code_snapshots")
PROJECT_INDEX  =  os.path.join("memory", "project_index.json")

# Event Logging
def log_hook_event(event_type, data):
    timestamp  =  datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs("memory", exist_ok = True)
    with open(HOOK_LOG, "a", encoding = "utf-8") as f:
        f.write(f"[{timestamp}] {event_type}: {json.dumps(data, indent = 2)}\n")

def on_prompt_received(prompt):
    log_hook_event("Prompt", {"prompt": prompt})

def on_llm_response(model, response):
    log_hook_event("LLM Response", {"model": model, "response": response})

def on_error(model, error):
    log_hook_event("Error", {"model": model, "error": error})

def on_memory_written(entry):
    log_hook_event("Memory Written", entry)

# Code Ops
def read_code_file(path):
    try:
        with open(path, "r", encoding = "utf-8") as f:
            return f.read()
    except Exception as e:
        return f"[Failed to read {path}]: {str(e)}"

def write_patch_file(path, content):
    try:
        with open(path, "w", encoding = "utf-8") as f:
            f.write(content)
        return True
    except Exception as e:
        on_error("PatchWrite", {"file": path, "error": str(e)})
        return False

# Project Indexing
def index_project_environment(root_path):
    project_index  =  []
    for subdir, dirs, files in os.walk(root_path):
        for name in files:
            if name.endswith(".py") or name.endswith(".html") or name.endswith(".js"):
                full_path  =  os.path.join(subdir, name)
                rel_path  =  os.path.relpath(full_path, root_path)
                project_index.append(rel_path)
    os.makedirs("memory", exist_ok = True)
    with open(PROJECT_INDEX, "w", encoding = "utf-8") as f:
        json.dump(project_index, f, indent = 2)

# Memory Tagging
def tag_memory(tag: str, content: str):
    timestamp  =  datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tagged_entry  =  {
        "timestamp": timestamp,
        "tag": tag,
        "content": content
    }
    path  =  os.path.join("memory", "tagged_memory.json")
    if os.path.exists(path):
        with open(path, "r", encoding = "utf-8") as f:
            memory  =  json.load(f)
    else:
        memory  =  []
    memory.append(tagged_entry)
    with open(path, "w", encoding = "utf-8") as f:
        json.dump(memory, f, indent = 2)
    log_hook_event("Tagged Memory", tagged_entry)

# Middleware for logging
class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        body  =  await request.body()
        log_hook_event("Middleware Request", {"path": request.url.path, "body": body.decode("utf-8")})
        response: Response  =  await call_next(request)
        log_hook_event("Middleware Response", {"status": response.status_code})
        return response

# Shutdown Hook
def graceful_shutdown():
    log_hook_event("Shutdown", {"event": "Application shutdown initiated"})
