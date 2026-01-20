# backend/utilities/modal_health_check.py
import modal, httpx, asyncio, json
from datetime import datetime

app = modal.App("watchdog-modal")
image = modal.Image.debian_slim().pip_install("httpx")

SERVICES = {
    "aries": "https://chad--aries-mmlm-fusion.modal.app",
    "decision": "https://chad--decision-engine.modal.app",
    "stripe": "https://chad--stripe-processor.modal.app"
}

qdrant = modal.Volume.from_name("qdrant-data", create_if_missing=True)

async def check_service(name: str, url: str):
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{url}/")
            return r.status_code == 200
    except:
        return False

@app.function(image=image, volumes={"/qdrant": qdrant}, schedule=modal.Period(seconds=30))
async def watchdog_cycle():
    incidents = []
    for name, url in SERVICES.items():
        if not await check_service(name, url):
            incident = {"service": name, "down_at": datetime.now().isoformat(), "status": "failed"}
            incidents.append(incident)
            # Auto-restart via Modal CLI (or webhook)
            # modal deploy <app> --detach
    if incidents:
        with open("/qdrant/incidents.jsonl", "a") as f:
            for i in incidents:
                f.write(json.dumps(i) + "\n")
    return {"checked": len(SERVICES), "incidents": len(incidents)}