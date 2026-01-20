from fastapi import FastAPI
from pydantic import BaseModel
import os

app = FastAPI()

class HeartbeatResponse(BaseModel):
    message: str

@app.get("/heartbeat", response_model=HeartbeatResponse)
async def heartbeat():
    return HeartbeatResponse(message="Alive.")

class StatusResponse(BaseModel):
    ship_id: str
    role: str

@app.get("/status", response_model=StatusResponse)
async def status():
    return StatusResponse(
        ship_id=os.getenv("SHIP_ID", "Unknown"),
        role=os.getenv("ROLE", "Unknown")
    )
