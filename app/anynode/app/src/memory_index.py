import json, time
from pathlib import Path
class MemoryIndex:
    def __init__(self, path: str):
        self.path = Path(path); self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists(): self.path.write_text("[]", encoding="utf-8")
    def add(self, event: dict):
        data = json.loads(self.path.read_text()); event["ts"] = time.time(); data.append(event)
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    def recall(self, query: str, k=5):
        data = json.loads(self.path.read_text()); q = query.lower().split(); scored = []
        for e in data:
            text = (e.get("text") or "") + " " + " ".join(e.get("tags",[]))
            score = sum(w in text.lower() for w in q)
            if score>0: scored.append((score,e))
        scored.sort(key=lambda x: -x[0]); return [e for _,e in scored[:k]]
