
from typing import List, Dict, Any

# Stubs: replace with real client SDK calls when keys exist.
def publish_youtube(assets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{"platform": "youtube", "id": f"yt_{a.get('kind')}_{i}"} for i, a in enumerate(assets)]

def publish_tiktok(assets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{"platform": "tiktok", "id": f"tk_{a.get('kind')}_{i}"} for i, a in enumerate(assets)]

def publish_x(assets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{"platform": "x", "id": f"x_{a.get('kind')}_{i}"} for i, a in enumerate(assets)]

def publish_reddit(assets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{"platform": "reddit", "id": f"rd_{a.get('kind')}_{i}"} for i, a in enumerate(assets)]

PLATFORMS = {
    "youtube": publish_youtube,
    "tiktok": publish_tiktok,
    "x": publish_x,
    "reddit": publish_reddit,
}

def publish(platforms: List[str], assets: List[Dict[str, Any]]):
    results = []
    for p in platforms:
        fn = PLATFORMS.get(p)
        if not fn: continue
        results.extend(fn(assets))
    return results
