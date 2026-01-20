from fastapi import FastAPI, Request
import os, httpx
app = FastAPI(title = 'llm-proxy')
OLLAMA = os.getenv('OLLAMA_URL',''); LMSTUDIO = os.getenv('LMSTUDIO_URL',''); OPENAI_BASE = os.getenv('OPENAI_BASE_URL','https://api.openai.com/v1'); OPENAI_KEY = os.getenv('OPENAI_API_KEY',''); DEFAULT_MODEL = os.getenv('DEFAULT_MODEL','qwen2.5:3b')
@app.get("/health")
async def h():
    return {"status": "ok"}
async def up(u):
    try: async with httpx.AsyncClient(timeout = 2) as cx: r = await cx.get(u); return r.status_code<500
    except: return False
async def pick():
    if OLLAMA and await up(f'{OLLAMA}/api/tags'): return 'ollama'
    if LMSTUDIO and await up(f'{LMSTUDIO.replace("/v1","")}/health'): return 'lmstudio'
    if OPENAI_KEY: return 'openai'
    return 'none'
@app.post('/v1/chat/completions')
async def chat(req:Request):
    body = await req.json(); model = body.get('model') or DEFAULT_MODEL; b = await pick()
    if b =  = 'ollama':
        async with httpx.AsyncClient(timeout = 120) as cx:
            r = await cx.post(f'{OLLAMA}/api/chat', json = {'model':model,'messages':body.get('messages',[]),'stream':False}); r.raise_for_status(); j = r.json()
        text = j.get('message',{}).get('content','')
        return {'id':'chatcmpl-ollama','object':'chat.completion','choices':[{'index':0,'message':{'role':'assistant','content':text},'finish_reason':'stop'}]}
    if b =  = 'lmstudio':
        async with httpx.AsyncClient(timeout = 120) as cx: r = await cx.post(LMSTUDIO+'/chat/completions', json = {**body,'model':model}); r.raise_for_status(); return r.json()
    if b =  = 'openai':
        async with httpx.AsyncClient(timeout = 120) as cx: r = await cx.post(OPENAI_BASE+'/chat/completions', headers = {'Authorization':f'Bearer {OPENAI_KEY}'}, json = {**body,'model':model}); r.raise_for_status(); return r.json()
    return {'error':'no_backend_available','hint':'Start Ollama or LM Studio, or set OPENAI_API_KEY.'}
