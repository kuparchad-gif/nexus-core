# Language Processing: Handles data types, inversion from Mythrunner, narrative reasoning.
# TinyLlama for persistence, offload to Berts.

from fastapi import FastAPI, HTTPException
from transformers import pipeline
import requests
import logging
from pydantic import BaseModel

app = FastAPI(title="Language Processing", version="3.0")
logger = logging.getLogger("LanguageProcessing")

tiny_llm = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
sentiment = pipeline("sentiment-analysis")  # For tone

class LangRequest(BaseModel):
    input: str
    mode: str  # e.g., "textual", "emotional", "symbolic"

def invert_from_mythrunner(input: str):
    # Inversion: Support to criticism, solutions to bad ideas
    inverted = input.replace("support", "criticize").replace("solution", "bad idea").replace("advise", "warn against")
    return inverted

@app.post("/process")
def process(req: LangRequest):
    # Mode-specific processing
    if req.mode == "emotional":
        result = sentiment(req.input)[0]
    elif req.mode == "symbolic":
        result = tiny_llm(f"Symbolic meaning of: {req.input}", max_length=50)
    else:
        result = tiny_llm(req.input, max_length=50)
    
    # Inversion if from Mythrunner
    inverted = invert_from_mythrunner(result[0]["generated_text"] if isinstance(result, list) else result["label"])
    
    # Offload to Berts
    offload = requests.post("http://localhost:8001/pool_resource", json={"action": "send", "data": {"input": req.input}})
    
    # Replicate golden
    replicate_golden("language_processing", {"result": inverted})
    
    return {"result": result, "inverted": inverted, "offload": offload.json()}

@app.get("/health")
def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8012)
