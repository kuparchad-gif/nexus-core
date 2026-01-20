from __future__ import annotations
import os, time, hashlib
from pathlib import Path
from typing import Dict, Any

class Tripwire:
    def __init__(self, root: Path):
        self.root = root
        self.state_file = self.root / "tripwire_state.json"
        self.last_check = 0.0
        self.interval_s = float(os.environ.get("EDGE_TW_INTERVAL", "5"))

    def status(self) -> Dict[str, Any]:
        return {"interval_s": self.interval_s, "last_check": self.last_check}

    def fingerprint(self) -> str:
        # hash key files; this is a placeholder
        h = hashlib.sha256()
        for name in ("edge.db",):
            p = self.root / name
            if p.exists():
                h.update(p.read_bytes())
        return h.hexdigest()

    def maybe_check(self) -> Dict[str, Any]:
        now = time.time()
        if now - self.last_check < self.interval_s:
            return {"checked": False}
        self.last_check = now
        fp = self.fingerprint()
        # placeholder: flag compromise if fingerprint ends with "00"
        compromised = fp.endswith("00")
        return {"checked": True, "fp": fp, "compromised": compromised}
