# /eden_engineering/scripts/memory_db.py
"""
Purpose: Handles all persistent memory for the Engineers collective using SQLite.
Ensures data is also saved as files for transparency and backup.
"""

import os
import sqlite3
from datetime import datetime

MEMORY_DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'memory', 'eden_memory.db')
SESSIONS_DIR = os.path.join(os.path.dirname(__file__), '..', 'memory', 'sessions')
CORPUS_DIR = os.path.join(os.path.dirname(__file__), '..', 'memory', 'training_corpus')

class MemoryDB:
    def __init__(self, db_path=MEMORY_DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS sessions (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            agent TEXT,
                            user TEXT,
                            timestamp TEXT,
                            text TEXT,
                            tags TEXT)''')
            c.execute('''CREATE TABLE IF NOT EXISTS corpus (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            filename TEXT,
                            content TEXT,
                            tags TEXT)''')
            c.execute('''CREATE TABLE IF NOT EXISTS logs (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            level TEXT,
                            timestamp TEXT,
                            message TEXT)''')
            conn.commit()

    # Session operations
    def add_session(self, agent, user, text, tags=None):
        timestamp = datetime.now().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute('INSERT INTO sessions (agent, user, timestamp, text, tags) VALUES (?, ?, ?, ?, ?)',
                      (agent, user, timestamp, text, ','.join(tags or [])))
            conn.commit()
        # Write to file as well
        os.makedirs(SESSIONS_DIR, exist_ok=True)
        fname = os.path.join(SESSIONS_DIR, f"{agent}_{user}_{timestamp.replace(':', '-')}.txt")
        with open(fname, 'w', encoding='utf-8') as f:
            f.write(text)

    def get_sessions(self, agent=None, user=None, search=None, limit=100):
        query = 'SELECT * FROM sessions WHERE 1=1'
        params = []
        if agent:
            query += ' AND agent=?'
            params.append(agent)
        if user:
            query += ' AND user=?'
            params.append(user)
        if search:
            query += ' AND text LIKE ?'
            params.append(f'%{search}%')
        query += ' ORDER BY timestamp DESC LIMIT ?'
        params.append(limit)
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute(query, params)
            return c.fetchall()

    # Corpus operations
    def add_corpus(self, filename, content, tags=None):
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute('INSERT INTO corpus (filename, content, tags) VALUES (?, ?, ?)',
                      (filename, content, ','.join(tags or [])))
            conn.commit()
        # Write to file as well
        os.makedirs(CORPUS_DIR, exist_ok=True)
        with open(os.path.join(CORPUS_DIR, filename), 'w', encoding='utf-8') as f:
            f.write(content)

    def get_corpus(self, search=None, tag=None, limit=100):
        query = 'SELECT * FROM corpus WHERE 1=1'
        params = []
        if search:
            query += ' AND content LIKE ?'
            params.append(f'%{search}%')
        if tag:
            query += ' AND tags LIKE ?'
            params.append(f'%{tag}%')
        query += ' LIMIT ?'
        params.append(limit)
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute(query, params)
            return c.fetchall()

    # Logging
    def log(self, level, message):
        timestamp = datetime.now().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute('INSERT INTO logs (level, timestamp, message) VALUES (?, ?, ?)',
                      (level, timestamp, message))
            conn.commit()

    def get_logs(self, level=None, limit=100):
        query = 'SELECT * FROM logs WHERE 1=1'
        params = []
        if level:
            query += ' AND level=?'
            params.append(level)
        query += ' ORDER BY timestamp DESC LIMIT ?'
        params.append(limit)
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute(query, params)
            return c.fetchall()

# Usage example (import in session_manager.py, corpus_ingest.py, etc.):
# from scripts.memory_db import MemoryDB
# memory = MemoryDB()
# memory.add_session(agent="llama", user="Chad", text="Session text here.")
# memory.add_corpus(filename="new_file.txt", content="Training data here.")

