from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Aethereal Trader API")

class PayoutRequest(BaseModel):
    amount: float
    currency: str = "USD"
    destination: str
    class Config:
        schema_extra = {"example": {"amount": 100.0, "currency": "USD", "destination": "user123"}}

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/payout")
def payout(req: PayoutRequest):
    if req.amount <= 0:
        raise HTTPException(status_code=400, detail="Amount must be positive")
    if req.currency not in ["USD", "ETH"]:
        raise HTTPException(status_code=400, detail="Unsupported currency")
    return {"status":"initiated","amount":req.amount,"currency":req.currency,"destination":req.destination,"txid":"SIM-"+str(abs(hash(req.destination))%10**8)}

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.get("/ready")
def ready():
    # simple readiness: process up
    return {"ready": True}
