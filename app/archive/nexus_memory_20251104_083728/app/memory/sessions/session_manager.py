import uuid
import json
from pathlib import Path

SESSIONS_DIR  =  Path("C:/Projects/engineering_department/eden_engineering/memory/sessions")
SESSIONS_DIR.mkdir(parents = True, exist_ok = True)

def create_session():
    session_id  =  str(uuid.uuid4())
    session_file  =  SESSIONS_DIR / f"{session_id}.json"
    with open(session_file, "w", encoding = "utf-8") as f:
        json.dump({"session_id": session_id, "messages": []}, f, indent = 2)
    return session_id

def append_message(session_id, role, content):
    session_file  =  SESSIONS_DIR / f"{session_id}.json"
    if not session_file.exists():
        raise FileNotFoundError(f"Session not found: {session_id}")
    with open(session_file, "r", encoding = "utf-8") as f:
        session_data  =  json.load(f)
    session_data["messages"].append({"role": role, "content": content})
    with open(session_file, "w", encoding = "utf-8") as f:
        json.dump(session_data, f, indent = 2)

def get_session_messages(session_id):
    session_file  =  SESSIONS_DIR / f"{session_id}.json"
    if not session_file.exists():
        raise FileNotFoundError(f"Session not found: {session_id}")
    with open(session_file, "r", encoding = "utf-8") as f:
        session_data  =  json.load(f)
    return session_data["messages"]

if __name__ == "__main__":
    sid  =  create_session()
    append_message(sid, "user", "Hello, engineer.")
    append_message(sid, "system", "Welcome to the project workspace.")
    print(get_session_messages(sid))
