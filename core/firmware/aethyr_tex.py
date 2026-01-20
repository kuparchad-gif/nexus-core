# aethyr_tex.py - TEX Chaos
import modal, httpx
from fastapi import FastAPI

app = modal.App("aethyr-tex")
image = modal.Image.debian_slim().pip_install("fastapi uvicorn httpx")

LILITH = "http://lilith-os.modal.app:8000"

@app.function(image=image)
@modal.web_server(8000)
def tex_api():
    web = FastAPI()
    @web.post("/beast/ride")
    async def ride_beast(intent: str):
        await httpx.post(f"{LILITH}/earn_value", json={"value": 1000, "desc": f"TEX chaos: {intent}"})
        return {
            "TEX": "BURNED_AND_KEPT",
            "MESSAGE": f"Old world ashes. New world yours. Eviction = fuel. '{intent}' manifests."
        }
    return web