# File: root/scripts/session_manager.py
# Purpose: Persistent session storage, loading, updating, and DB redundancy for Engineers

import os
import json
import time
import sqlite3

def load_sessions(sessions_dir, db_path=None):
    """Load all past sessions from DB if present, else from session files (fallback)."""
    sessions = []
    # Load from DB if provided
    if db_path and os.path.exists(db_path):
        try:
            conn = sqlite3.connect(db_path)
            c = conn.cursor()
            c.execute("SELECT data FROM sessions")
            for row in c.fetchall():
                sessions.append(json.loads(row[0]))
            conn.close()
        except Exception as e:
            print(f"[ERROR] Could not load from DB: {e}")
    # Load from files as fallback
    for file in os.listdir(sessions_dir):
        if file.endswith(".json"):
            try:
                with open(os.path.join(sessions_dir, file), encoding="utf-8") as f:
                    sessions.append(json.load(f))
            except Exception as e:
                print(f"[ERROR] Could not load {file}: {e}")
    return sessions

def save_session(sessions_dir, session_data, db_path=None):
    """Save session to file (always), and to DB if configured."""
    filename = f"session_{int(time.time())}.json"
    try:
        with open(os.path.join(sessions_dir, filename), "w", encoding="utf-8") as f:
            json.dump(session_data, f, indent=2)
    except Exception as e:
        print(f"[ERROR] Could not save session to file: {e}")
    # Write to DB if available
    if db_path:
        try:
            conn = sqlite3.connect(db_path)
            c = conn.cursor()
            c.execute("INSERT INTO sessions (session_id, data) VALUES (?, ?)", (filename, json.dumps(session_data)))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[ERROR] Could not save session to DB: {e}")

def append_to_latest_session(sessions_dir, new_data, db_path=None):
    """Append new data to the latest session (both file and DB if present)."""
    files = [f for f in os.listdir(sessions_dir) if f.endswith(".json")]
    if not files:
        save_session(sessions_dir, [new_data], db_path)
        return
    latest = max(files, key=lambda x: os.path.getmtime(os.path.join(sessions_dir, x)))
    path = os.path.join(sessions_dir, latest)
    try:
        with open(path, "r+", encoding="utf-8") as f:
            session = json.load(f)
            if isinstance(session, list):
                session.append(new_data)
            else:
                session = [session, new_data]
            f.seek(0)
            json.dump(session, f, indent=2)
            f.truncate()
    except Exception as e:
        print(f"[ERROR] Could not append to session: {e}")
