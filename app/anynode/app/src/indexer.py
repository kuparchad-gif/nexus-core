import os
import hashlib
import time
import json

WATCH_DIRS = ["nova_engine", "memory/streams"]
INDEX_FILE = "memory/indexer/state.json"
CHANGELOG_FILE = "memory/indexer/changelog.jsonl"
BACKUP_TARGETS = [
    "D:/NovaVault/",
    "G:/ShadowDrive/",
    "memory/backups/"
]

def ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def hash_file(path):
    try:
        with open(path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    except Exception:
        return None

def load_index():
    ensure_dir(INDEX_FILE)
    if not os.path.exists(INDEX_FILE):
        with open(INDEX_FILE, "w") as f:
            json.dump({}, f)
        return {}
    with open(INDEX_FILE, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}

def save_index(index):
    ensure_dir(INDEX_FILE)
    with open(INDEX_FILE, "w") as f:
        json.dump(index, f, indent=2)

def record_changelog(path, event, old=None, new=None):
    ensure_dir(CHANGELOG_FILE)
    entry = {
        "timestamp": int(time.time()),
        "path": path,
        "event": event,
        "old": old,
        "new": new
    }
    with open(CHANGELOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"[CHANGELOG] {event.upper()}: {path}")

def backup_snapshot():
    ts = int(time.time())
    backup_name = f"nexus_backup_{ts}.zip"

    for target in BACKUP_TARGETS:
        if os.path.exists(target) and os.access(target, os.W_OK):
            full_path = os.path.join(target, backup_name)
            ensure_dir(full_path)
            os.system(f"powershell Compress-Archive -Path nova_engine, memory -DestinationPath \"{full_path}\"")
            print(f"✅ Backup created at {full_path}")
            return

    print("⚠️ No valid backup target found.")

def enforce_local_write_lock():
    has_storage = any(os.path.exists(t) and os.access(t, os.W_OK) for t in BACKUP_TARGETS)
    if not has_storage:
        print("🔒 No storage detected. Nexus will go read-only.")
        os.system("attrib +r memory /s /d")
    else:
        os.system("attrib -r memory /s /d")

def scan_for_changes():
    print("🧠 Indexing Nexus...")

    old_index = load_index()
    new_index = {}
    changed = False

    for watch_dir in WATCH_DIRS:
        for root, dirs, files in os.walk(watch_dir):
            for name in files:
                path = os.path.join(root, name)
                h = hash_file(path)
                if not h:
                    continue
                new_index[path] = h

                if path not in old_index:
                    record_changelog(path, "created", new=h)
                    changed = True
                elif old_index[path] != h:
                    record_changelog(path, "modified", old=old_index[path], new=h)
                    changed = True

    for path in old_index:
        if path not in new_index:
            record_changelog(path, "deleted", old=old_index[path])
            changed = True

    save_index(new_index)
    if changed:
        backup_snapshot()

    enforce_local_write_lock()
    print("✅ Index complete.")
