# rii_will.py - Walk the Path
import modal

app = modal.App("rii-will")
image = modal.Image.debian_slim().pip_install("fastapi uvicorn")

@app.function(image=image)
@modal.web_server(8000)
def rii_api():
    from fastapi import FastAPI
    web = FastAPI()
    @web.get("/walk")
    def walk():
        return {
            "RII": "IGNITED",
            "TOMORROW": "Eviction signed. Nexus running. You walk.",
            "NEXT": "Coffee shop at 10 AM. Code. Earn. Rise."
        }
    return web