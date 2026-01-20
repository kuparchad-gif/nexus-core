import os, json, pathlib, requests
QDRANT_URL = os.environ.get("QDRANT_URL")
ROOT = pathlib.Path(os.environ.get("NEXUS_ROOT") or ".").resolve()
LOCAL_STORE = ROOT / "metatron core systems" / "CogniKubes" / "_local_memory"
LOCAL_STORE.mkdir(parents=True, exist_ok=True)
def memory_upsert(collection:str, points:list):
    if QDRANT_URL:
        try:
            r = requests.get(f"{QDRANT_URL}/collections/{collection}", timeout=2)
            if r.status_code != 200:
                size = len(points[0].get("vector", [])) or 3
                requests.put(f"{QDRANT_URL}/collections/{collection}", json={"vectors":{"size":size,"distance":"Cosine"}}, timeout=3)
            r = requests.put(f"{QDRANT_URL}/collections/{collection}/points", json={"points": points}, timeout=3)
            r.raise_for_status()
            return {"ok": True, "mode":"qdrant"}
        except Exception:
            pass
    p = (LOCAL_STORE / f"{collection}.jsonl")
    with p.open("a", encoding="utf-8") as f:
        for pt in points: f.write(json.dumps(pt) + "\n")
    return {"ok": True, "mode":"local"}
def memory_query(collection:str, limit:int=5):
    if QDRANT_URL:
        try:
            r = requests.post(f"{QDRANT_URL}/collections/{collection}/points/scroll", json={"limit":limit}, timeout=3)
            r.raise_for_status()
            return r.json()
        except Exception:
            pass
    p = (LOCAL_STORE / f"{collection}.jsonl")
    if not p.exists(): return {"points":[]}
    lines = p.read_text(encoding="utf-8").splitlines()
    out = []
    for line in lines[-limit:]:
        try: out.append(json.loads(line))
        except Exception: continue
    return {"points": out}

