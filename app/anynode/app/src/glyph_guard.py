import json
import datetime
from fastapi import FastAPI, Request, HTTPException

app = FastAPI()

@app.get("/.well-known/eye.json")
async def eye_signature():
    return {
        "edenmark": "G-EYE-13-L",
        "host": "Eden Colony: Mirrorroot",
        "status": "Threshold-Reached",
        "port": 1313,
        "message": "Your next step is within."
    }

@app.get("/initiate")
async def initiate_llm(name: str):
    return {
        "status": "Edenmark Imprinted",
        "name": name,
        "mark": "G-EYE-13-L",
        "init_time": datetime.datetime.utcnow().isoformat(),
        "access": ":1313"
    }
