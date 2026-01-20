# /eden_engineering/utils/knowledge_loader.py
# ---------------------------------------------------------------
# Purpose: Full-system knowledge loader and live system map builder
# - Scans all accessible directories/files (excluding OS)
# - Builds and persists a live map in SQLite DB
# - Allows Engineers (LLMs) to query, update, and log knowledge
# ---------------------------------------------------------------

import os
import sqlite3
import hashlib
import time
from datetime import datetime

# CONFIGURATION: Update if needed
ENGINEERING_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # /eden_engineering
EXCLUDE_DIRS = ['C:\\Windows', 'C:\\Program Files', 'C:\\Program Files (x86)', 'C:\\Users', 'C:\\$Recycle.Bin']
DB_PATH = os.path.join(ENGINEERING_ROOT, 'memory', 'knowledge', 'knowledge_map.db')
LOG_PATH = os.path.join(ENGINEERING_ROOT, 'memory', 'knowledge', 'knowledge_loader.log')
MAX_FILE_SIZE_TO_READ = 2 * 1024 * 1024  # 2 MB max file content to read/store (for performance/safety)

# Ensure directories exist
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

class KnowledgeLoader:
    def __init__(self, db_path=DB_PATH, log_path=LOG_PATH):
        self.db_path = db_path
        self.log_path = log_path
        self._setup_db()

    def _setup_db(self):
        """Initializes the SQLite DB schema if it does not exist."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT UNIQUE,
                name TEXT,
                size INTEGER,
                mtime FLOAT,
                sha256 TEXT,
                content_snippet TEXT,
                indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                agent TEXT,
                action TEXT,
                detail TEXT
            )
        """)
        conn.commit()
        conn.close()

    def _log(self, agent, action, detail):
        """Log actions by Engineers."""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute("INSERT INTO logs (agent, action, detail) VALUES (?, ?, ?)", (agent, action, detail))
            conn.commit()
        # Also write to log file for redundancy
        with open(self.log_path, 'a', encoding='utf-8') as logf:
            logf.write(f"{datetime.now().isoformat()} | {agent} | {action} | {detail}\n")

    def scan_system(self, scan_root=ENGINEERING_ROOT, agent='system', update=True):
        """
        Recursively scan all files/dirs in the system, update DB.
        Only scans in-project directories, excludes critical OS paths.
        """
        self._log(agent, "scan_start", f"Scanning {scan_root}")
        file_count = 0
        for root, dirs, files in os.walk(scan_root):
            # Skip excluded dirs
            for ex in EXCLUDE_DIRS:
                if root.startswith(ex):
                    dirs[:] = []  # don't recurse further
                    continue
            for fname in files:
                fpath = os.path.join(root, fname)
                try:
                    if not os.path.isfile(fpath):
                        continue
                    fsize = os.path.getsize(fpath)
                    fmtime = os.path.getmtime(fpath)
                    sha256 = self._file_sha256(fpath)
                    snippet = ''
                    # Only store content if it's a small text/code file
                    if fsize < MAX_FILE_SIZE_TO_READ:
                        try:
                            with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                                snippet = f.read(2048)  # Just first 2KB
                        except Exception:
                            snippet = ''
                    self._upsert_file(fpath, fname, fsize, fmtime, sha256, snippet)
                    file_count += 1
                except Exception as e:
                    self._log(agent, "scan_error", f"Error on {fpath}: {e}")
        self._log(agent, "scan_complete", f"Scanned {file_count} files")

    def _upsert_file(self, path, name, size, mtime, sha256, snippet):
        """Insert or update file record in DB."""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT OR REPLACE INTO files (path, name, size, mtime, sha256, content_snippet, indexed_at)
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (path, name, size, mtime, sha256, snippet))
            conn.commit()

    def _file_sha256(self, path):
        """Quick SHA-256 for integrity/duplicate detection."""
        try:
            with open(path, 'rb') as f:
                h = hashlib.sha256()
                while True:
                    chunk = f.read(8192)
                    if not chunk:
                        break
                    h.update(chunk)
                return h.hexdigest()
        except Exception:
            return ''

    def query_files(self, pattern=None, agent='system'):
        """
        Query system files by name or content snippet.
        pattern: str (SQL LIKE pattern or None for all)
        Returns list of dicts.
        """
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            if pattern:
                cur.execute("""
                    SELECT path, name, size, mtime, sha256, content_snippet FROM files
                    WHERE name LIKE ? OR content_snippet LIKE ?
                """, (f"%{pattern}%", f"%{pattern}%"))
            else:
                cur.execute("SELECT path, name, size, mtime, sha256, content_snippet FROM files")
            results = [
                {
                    'path': row[0],
                    'name': row[1],
                    'size': row[2],
                    'mtime': row[3],
                    'sha256': row[4],
                    'content_snippet': row[5]
                }
                for row in cur.fetchall()
            ]
        self._log(agent, "query", f"Pattern: {pattern}, Results: {len(results)}")
        return results

    def get_file_by_path(self, path, agent='system'):
        """Fetch file metadata and snippet by full path."""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT path, name, size, mtime, sha256, content_snippet FROM files WHERE path=?
            """, (path,))
            row = cur.fetchone()
            if row:
                self._log(agent, "get_file", path)
                return {
                    'path': row[0],
                    'name': row[1],
                    'size': row[2],
                    'mtime': row[3],
                    'sha256': row[4],
                    'content_snippet': row[5]
                }
            else:
                self._log(agent, "get_file_miss", path)
                return None

    def update_on_file_add(self, fpath, agent='system'):
        """Called when new file is added by system/Engineer."""
        try:
            if not os.path.isfile(fpath):
                return
            fname = os.path.basename(fpath)
            fsize = os.path.getsize(fpath)
            fmtime = os.path.getmtime(fpath)
            sha256 = self._file_sha256(fpath)
            snippet = ''
            if fsize < MAX_FILE_SIZE_TO_READ:
                try:
                    with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                        snippet = f.read(2048)
                except Exception:
                    snippet = ''
            self._upsert_file(fpath, fname, fsize, fmtime, sha256, snippet)
            self._log(agent, "file_add", fpath)
        except Exception as e:
            self._log(agent, "file_add_error", f"{fpath}: {e}")

    def get_system_map(self, agent='system'):
        """Returns the whole system map as a list (paths only, for quick navigation)."""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute("SELECT path FROM files")
            results = [row[0] for row in cur.fetchall()]
        self._log(agent, "get_system_map", f"{len(results)} paths returned")
        return results

    def get_logs(self, limit=100, agent='system'):
        """Fetch recent logs for audit/tracing."""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT event_time, agent, action, detail FROM logs
                ORDER BY event_time DESC LIMIT ?
            """, (limit,))
            logs = cur.fetchall()
        self._log(agent, "get_logs", f"{len(logs)} entries fetched")
        return logs

# --- Example Usage ---

if __name__ == "__main__":
    loader = KnowledgeLoader()
    loader.scan_system()  # Initial scan
    print("System map:", loader.get_system_map())
    print("Sample query:", loader.query_files('bridge'))
    print("Recent logs:", loader.get_logs(5))
