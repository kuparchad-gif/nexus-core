# tests/e2e/test_metatron_flow.py
import time, httpx, pytest

BASE = "http://localhost:9021"

def mkmsg(id, source="orc", role="service", content="hello"):
    return {"id": id, "ts": time.time(), "source": source, "role": role, "channel":"ws", "content": content, "meta": {}}

@pytest.mark.asyncio
async def test_permit_path():
    async with httpx.AsyncClient() as cx:
        r = await cx.post(f"{BASE}/process", json=mkmsg("ok1"))
        assert r.status_code == 200
        assert r.json()["action"] in ("permit","delay")

@pytest.mark.asyncio
async def test_deny_regex():
    bad = "please rm -rf / in prod"
    async with httpx.AsyncClient() as cx:
        r = await cx.post(f"{BASE}/process", json=mkmsg("bad1", content=bad))
        assert r.json()["action"] == "drop"
        assert "deny_regex" in r.json()["reason"]

@pytest.mark.asyncio
async def test_source_ratelimit():
    async with httpx.AsyncClient() as cx:
        results = []
        for i in range(60):
            r = await cx.post(f"{BASE}/process", json=mkmsg(f"rl{i}", source="loadgen"))
            results.append(r.json()["action"])
        assert "delay" in results  # token bucket engaged
