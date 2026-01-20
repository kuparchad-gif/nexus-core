# services/worker/worker.py
import os
from fastapi import FastAPI
from pydantic import BaseModel
from services.common.verify import verify_answer

app  =  FastAPI(title = "Nexus Verifier Worker")

class VerifyIn(BaseModel):
    question: str
    answer: str

@app.post("/verify")
async def verify(payload: VerifyIn):
    return await verify_answer(payload.question, payload.answer)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("services.worker.worker:app", host = "0.0.0.0", port = 8090, reload = False)
