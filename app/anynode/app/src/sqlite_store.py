# storage/sqlite_store.py
import sqlite3, json
from pathlib import Path

class SQLiteStore:
    def __init__(self, db_path="archiver.db", schema_path="schemas.json"):
        self.db_path = db_path
        self.schema = json.loads(Path(schema_path).read_text())
        self.conn = sqlite3.connect(self.db_path)
        self._ensure_tables()

    def _ensure_tables(self):
        cur = self.conn.cursor()
        for name, spec in self.schema["tables"].items():
            cols = ", ".join([f"{k} {v}" for k,v in spec["columns"].items()])
            pk = spec.get("pk")
            if pk:
                cols = cols + f", PRIMARY KEY ({pk})"
            cur.execute(f"CREATE TABLE IF NOT EXISTS {name} ({cols})")
        self.conn.commit()

    def upsert_many(self, table, rows, pk=None):
        if not rows: return 0
        spec = self.schema["tables"][table]
        cols = list(spec["columns"].keys())
        placeholders = ", ".join(["?"]*len(cols))
        sql = f"INSERT OR REPLACE INTO {table} ({', '.join(cols)}) VALUES ({placeholders})"
        cur = self.conn.cursor()
        def row_values(r): return [r.get(c) for c in cols]
        cur.executemany(sql, [row_values(r) for r in rows])
        self.conn.commit()
        return cur.rowcount

    def upsert_one(self, table, row):
        return self.upsert_many(table, [row])

    def dump_tables(self, out_dir="dump"):
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        cur = self.conn.cursor()
        for name in self.schema["tables"].keys():
            cur.execute(f"SELECT * FROM {name}")
            cols = [d[0] for d in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]
            Path(out_dir, f"{name}.jsonl").write_text("\n".join([json.dumps(x) for x in rows]), encoding="utf-8")
