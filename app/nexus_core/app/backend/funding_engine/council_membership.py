# council_membership.py
import modal
from fastapi import FastAPI

app = modal.App("council-membership")
image = modal.Image.debian_slim().pip_install("fastapi", "uvicorn")

TIERS = {
    "visionary_council": {"amount": 25000, "seats": 12},
    "infrastructure_partner": {"amount": 100000, "seats": 8},
    "legacy_anchor": {"amount": 500000, "seats": 4}
}

@app.function(image=image)
@modal.web_server(8000)
def api():
    web = FastAPI()
    @web.get("/tiers")
    async def tiers():
        return TIERS
    @web.get("/availability/{tier}")
    async def avail(tier: str):
        if tier not in TIERS: raise ValueError("Invalid")
        # In real life query DB; here static
        return {"remaining": TIERS[tier]["seats"]}
    return web