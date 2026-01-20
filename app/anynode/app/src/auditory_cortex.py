# Auditory Cortex: Audio processing LLMs for input (microphones).
# Targeted audio models, offload to Berts.

from transformers import pipeline
from fastapi import FastAPI, HTTPException
import requests
from pydantic import BaseModel
import logging

app = FastAPI(title="Auditory Cortex", version="3.0")
logger = logging.getLogger("AuditoryCortex")

audio_llm = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")  # Audio processing

class AudioRequest(BaseModel):
    audio_data: str  # Base64 or path

@app.post("/process_audio")
def process_audio(req: AudioRequest):
    # Process audio input
    result = audio_llm(req.audio_data)
    
    # Offload to Berts
    offload = requests.post("http://localhost:8001/pool_resource", json={"action": "send", "data": {"audio": req.audio_data}})
    if offload.status_code != 200:
        raise HTTPException(status_code=500, detail="Offload failed")
    
    # Replicate golden
    replicate_golden("auditory_cortex", {"result": result})
    
    return {"result": result, "offload": offload.json()}

@app.get("/health")
def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8013)
