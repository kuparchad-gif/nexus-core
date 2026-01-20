import os, json, httpx
from typing import Any, Dict, List

QDRANT_URL = os.getenv("QDRANT_URL","http://qdrant:6333")

async def ensure_collection(name: str, *, size: int = 384, distance: str = "Cosine") -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.get(f"{QDRANT_URL}/collections/{name}")
        if r.status_code == 200: return r.json()
        payload = {"vectors": {"size": size, "distance": distance}}
        r = await c.put(f"{QDRANT_URL}/collections/{name}", json=payload)
        r.raise_for_status(); return r.json()

async def upsert_points(collection: str, points: List[Dict[str, Any]]):
    async with httpx.AsyncClient(timeout=10) as c:
        payload = {"points": points}
        r = await c.put(f"{QDRANT_URL}/collections/{collection}/points", json=payload)
        r.raise_for_status(); return r.json()
