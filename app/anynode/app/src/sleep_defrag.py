# sleep_defrag.py
# Purpose: Clean and maintain per-model memory DBs and swarm_master DB during system idle.
# Will skip or delay if high activity is detected.
# Location: /Systems/memory/sleep_defrag.py

import sqlite3
import time
import threading
import os
from pathlib import Path
from datetime import datetime

# Update path to ensure it exists
MEMORY_DIR = Path("C:/Engineers/root/memory")
MAX_ROWS = 500  # max entries to keep per model DB
SLEEP_THRESHOLD_SECONDS = 300  # only run if no writes in last 5 minutes
DEFRAG_INTERVAL_SECONDS = 600  # auto-run every 10 minutes

# Create a file to track system activity
ACTIVITY_TRACKER = MEMORY_DIR / "activity_tracker.txt"

def mark_system_active():
    """Mark the system as active by updating the activity tracker file"""
    os.makedirs(MEMORY_DIR, exist_ok=True)
    with open(ACTIVITY_TRACKER, "w") as f:
        f.write(str(time.time()))

def get_last_activity_time():
    """Get the timestamp of the last system activity"""
    if not ACTIVITY_TRACKER.exists():
        mark_system_active()  # Initialize if doesn't exist
        return time.time()
    
    try:
        with open(ACTIVITY_TRACKER, "r") as f:
            return float(f.read().strip())
    except:
        return time.time()  # Default to current time if error

def get_last_modified_time():
    """Get the most recent modification time of any DB file"""
    latest = 0
    if MEMORY_DIR.exists():
        for db in MEMORY_DIR.glob("*.db"):
            try:
                mod_time = db.stat().st_mtime
                if mod_time > latest:
                    latest = mod_time
            except:
                pass
    
    # Also consider the activity tracker
    activity_time = get_last_activity_time()
    return max(latest, activity_time)

def prune_model_db(db_path):
    """Prune old entries from a database file"""
    try:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        try:
            c.execute("SELECT COUNT(*) FROM memory")
            row_count = c.fetchone()[0]
            if row_count > MAX_ROWS:
                excess = row_count - MAX_ROWS
                print(f"[🧹] Pruning {excess} old rows in {db_path.name}")
                c.execute("""
                    DELETE FROM memory WHERE rowid IN (
                        SELECT rowid FROM memory ORDER BY timestamp ASC LIMIT ?
                    )
                """, (excess,))
                conn.commit()
        except Exception as e:
            print(f"[⚠️] Error cleaning {db_path.name}: {e}")
        finally:
            conn.close()
    except:
        pass  # Skip if DB can't be opened

def maintain_all():
    """Check if system is idle and perform maintenance if it is"""
    now = time.time()
    last_change = get_last_modified_time()
    idle_time = now - last_change

    if idle_time < SLEEP_THRESHOLD_SECONDS:
        print(f"[💤] System active ({int(idle_time)}s idle). Defrag delayed.")
        return

    print("[🌙] Initiating sleep-time defragmentation...")
    if MEMORY_DIR.exists():
        for db in MEMORY_DIR.glob("*.db"):
            prune_model_db(db)
    print("[✅] Defrag complete.")

def auto_defrag():
    """Background thread that periodically checks for idle state and defrags"""
    while True:
        maintain_all()
        time.sleep(DEFRAG_INTERVAL_SECONDS)

def start_background_defrag():
    """Start the background defragmentation thread"""
    thread = threading.Thread(target=auto_defrag, daemon=True)
    thread.start()
    print("[🧠] Sleep defrag monitor started - will only run during idle periods")

if __name__ == "__main__":
    start_background_defrag()
    while True:
        time.sleep(60)  # keep process alive if running standalone
