# app.py — generic CKube service stub (MCP-ready placeholder hooks)
import os, ray
from fastapi import FastAPI
app  =  FastAPI(title = os.getenv('SERVICE_NAME','service'))
@app.get('/health')
def health(): return {'status':'ok','service':os.getenv('SERVICE_NAME','service')}
@ray.remote
def _ping(x): return x
@app.post('/act')
def act(payload: dict):
    ray.init(address = 'auto', ignore_reinit_error = True, namespace = 'metatron')
    return ray.get(_ping.remote({'ok':True,'payload':payload}))
