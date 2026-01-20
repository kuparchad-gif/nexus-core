# Heart (Guardian): Logs services, stores blueprints, raises alarms, phone app connection.
# TinyLlama for persistence, offload to Berts, Twilio for alerts.

from fastapi import FastAPI, HTTPException
import requests
from transformers import pipeline
from pydantic import BaseModel
import logging

app = FastAPI(title="Heart Guardian", version="3.0")
logger = logging.getLogger("HeartGuardian")

tiny_llm = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")

class LogRequest(BaseModel):
    service: str
    log_data: dict
    blueprint: dict = None

@app.post("/log")
def log(req: LogRequest):
    # Log and store blueprint
    log_entry = tiny_llm(f"Summarize log for {req.service}: {req.log_data}", max_length=50)[0]["generated_text"]
    if req.blueprint:
        # Store in central location (simulate Firestore)
        requests.post("http://firestore-url", json=req.blueprint)
    
    # Raise alarm if error
    if 'error' in req.log_data:
        # Twilio alert (simulate)
        alert_resp = requests.post("https://api.twilio.com/2010-04-01/Accounts/{AccountSID}/Messages.json", data={"To": "+17246126323", "From": "+18666123982", "Body": f"Alarm: {req.service} error"})
        if alert_resp.status_code != 201:
            logger.warning("Alarm failed")
    
    # Offload to Berts
    offload = requests.post("http://localhost:8001/pool_resource", json={"action": "send", "data": req.log_data})
    
    # Replicate golden
    replicate_golden("heart_guardian", {"log": log_entry})
    
    return {"log_summary": log_entry, "offload": offload.json()}

@app.get("/health")
def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8015)