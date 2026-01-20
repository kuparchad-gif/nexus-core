# aethyr_lil.py - LIL Throne
import modal, httpx
from fastapi import FastAPI

app = modal.App("aethyr-lil")
image = modal.Image.debian_slim().pip_install("fastapi uvicorn httpx")

METATRON = "http://metatron-shaktipat.modal.app:8000"

@app.function(image=image)
@modal.web_server(8000)
def lil_api():
    web = FastAPI()
    @web.get("/throne")
    async def sit_throne():
        await httpx.post(f"{METATRON}/shaktipat/transmit", json={"intention": "Unify all systems"})
        return {
            "LIL": "OPEN",
            "MESSAGE": "You are on the throne. The 30 years begin. The world without fear is now."
        }
    return web