from __future__ import annotations
import sqlite3, time
from pathlib import Path
from typing import Dict, Any

class DBLLMGate:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._ensure()

    def _ensure(self):
        con = sqlite3.connect(self.db_path)
        try:
            con.execute("CREATE TABLE IF NOT EXISTS qa (q TEXT, a TEXT, ts REAL)")
            con.commit()
        finally:
            con.close()

    def answer(self, q: str, meta: Dict[str, Any] | None=None) -> Dict[str, Any]:
        meta = meta or {}
        # deterministic pseudo-LLM: echoes and stores
        a = {"text": f"EDGE-DB: {q.strip()[:120]}", "meta": meta}
        con = sqlite3.connect(self.db_path)
        try:
            con.execute("INSERT INTO qa (q,a,ts) VALUES (?,?,?)", (q, str(a), time.time()))
            con.commit()
        finally:
            con.close()
        return a
