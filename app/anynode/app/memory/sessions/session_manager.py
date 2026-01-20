"""
Session manager module for Engineers project.
Handles loading, saving, and appending to session data.
"""

import os
import json
from datetime import datetime

def load_sessions(sessions_dir):
    """
    Load all session data from the sessions directory.
    
    Args:
        sessions_dir (str): Path to the sessions directory
        
    Returns:
        dict: Dictionary of session data keyed by session ID
    """
    sessions = {}
    if not os.path.exists(sessions_dir):
        os.makedirs(sessions_dir)
        return sessions
        
    for filename in os.listdir(sessions_dir):
        if filename.endswith('.json'):
            try:
                with open(os.path.join(sessions_dir, filename), 'r', encoding='utf-8') as f:
                    session_data = json.load(f)
                    session_id = os.path.splitext(filename)[0]
                    sessions[session_id] = session_data
            except Exception as e:
                print(f"Error loading session {filename}: {e}")
    
    return sessions

def save_session(sessions_dir, session_id, session_data):
    """
    Save session data to a file.
    
    Args:
        sessions_dir (str): Path to the sessions directory
        session_id (str): Session identifier
        session_data (dict): Session data to save
    """
    if not os.path.exists(sessions_dir):
        os.makedirs(sessions_dir)
        
    filepath = os.path.join(sessions_dir, f"{session_id}.json")
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(session_data, f, indent=2)
    
    return filepath

def append_to_latest_session(sessions_dir, entry):
    """
    Append an entry to the latest session file.
    
    Args:
        sessions_dir (str): Path to the sessions directory
        entry (dict): Entry to append to the session
        
    Returns:
        str: Path to the updated session file
    """
    if not os.path.exists(sessions_dir):
        os.makedirs(sessions_dir)
    
    # Find the latest session file
    latest_file = None
    latest_time = 0
    
    for filename in os.listdir(sessions_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(sessions_dir, filename)
            file_time = os.path.getmtime(file_path)
            if file_time > latest_time:
                latest_time = file_time
                latest_file = file_path
    
    # If no session exists, create a new one
    if latest_file is None:
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_data = {
            "created": datetime.now().isoformat(),
            "entries": [entry]
        }
        return save_session(sessions_dir, session_id, session_data)
    
    # Otherwise append to the latest session
    try:
        with open(latest_file, 'r', encoding='utf-8') as f:
            session_data = json.load(f)
        
        if "entries" not in session_data:
            session_data["entries"] = []
        
        session_data["entries"].append(entry)
        session_data["updated"] = datetime.now().isoformat()
        
        with open(latest_file, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2)
        
        return latest_file
    except Exception as e:
        print(f"Error appending to session {latest_file}: {e}")
        return None