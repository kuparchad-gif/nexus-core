import os
from typing import List, Dict, Any, Optional

try:
    import httpx as _http
except Exception:  # pragma: no cover
    import requests as _http  # type: ignore

DEFAULT_BASE = os.getenv("MEM_BASE_URL", "http://127.0.0.1:8303")

class NexusMemClient:
    def __init__(self, base_url: str = DEFAULT_BASE, timeout: float = 2.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def get_context(self, budget_tokens: int = 800, topics: Optional[List[str]] = None) -> Dict[str, Any]:
        params = {"budget_tokens": str(budget_tokens)}
        if topics:
            params["topics"] = ",".join(topics)
        url = f"{self.base_url}/context"
        resp = _http.get(url, params=params, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def remember(self, text: str, tags: Optional[List[str]] = None, priority: int = 5, ttl_seconds: Optional[int] = None) -> Dict[str, Any]:
        payload = {"role": "system", "text": text, "priority": priority}
        if tags: payload["tags"] = tags
        if ttl_seconds is not None: payload["ttl_seconds"] = ttl_seconds
        url = f"{self.base_url}/remember"
        resp = _http.post(url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def mirror(self, text: str, vector: Optional[List[float]] = None, tags: Optional[List[str]] = None, priority: int = 8) -> Dict[str, Any]:
        payload = {"text": text, "tags": tags or [], "priority": priority}
        if vector: payload["vector"] = vector
        url = f"{self.base_url}/mirror"
        resp = _http.post(url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def health(self) -> Dict[str, Any]:
        resp = _http.get(f"{self.base_url}/healthz", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()
