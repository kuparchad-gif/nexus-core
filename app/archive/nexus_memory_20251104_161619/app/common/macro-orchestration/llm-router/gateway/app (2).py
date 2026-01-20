# C:\Projects\Stacks\nexus-metatron\services\gateway\app.py
from fastapi import FastAPI
import os
app  =  FastAPI(title = os.getenv("SERVICE_NAME", "service"))
@app.get("/")
def root(): return {"service": os.getenv("SERVICE_NAME","service"), "ok": True}
@app.get("/health")
def health(): return {"status": "ok", "service": os.getenv("SERVICE_NAME","service")}
