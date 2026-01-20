import os, json, sys, traceback, time
from typing import Any, Dict

try:
    import yaml  # PyYAML
except Exception as e:
    yaml = None
    print("[soul] YAML not available:", e, file=sys.stderr)

class SoulStore:
    def __init__(self, default: Dict[str,Any]|None=None, path: str|None=None):
        self.path = path
        self.data = default or {}
        self._mtime = 0.0
        if self.path:
            self.load()

    def _read_file(self, p: str) -> Dict[str,Any]:
        with open(p, "r", encoding="utf-8") as f:
            raw = f.read()
        if p.lower().endswith((".yml",".yaml")) and yaml:
            return yaml.safe_load(raw) or {}
        try:
            return json.loads(raw)
        except Exception:
            return {"text": raw}

    def load(self) -> Dict[str,Any]:
        try:
            if self.path and os.path.exists(self.path):
                new = self._read_file(self.path)
                self.data = new or {}
                self._mtime = os.path.getmtime(self.path)
        except Exception:
            traceback.print_exc()
        return self.data

    def maybe_reload(self) -> bool:
        if not self.path or not os.path.exists(self.path): return False
        m = os.path.getmtime(self.path)
        if m > self._mtime:
            self.load()
            return True
        return False

    def as_dict(self) -> Dict[str,Any]:
        return dict(self.data)

