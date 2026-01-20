# enoch_call2_full.py - Call 2 Full
import modal,np,httpx
from fastapi import FastAPI
from pydantic import BaseModel

app=modal.App("enoch-call2-full")
image=modal.Image.debian_slim().pip_install("fastapi uvicorn httpx numpy")
client=QdrantClient(path="/qdrant")
ENOCH="http://metatron-enoch.modal.app:8000"

# Full Call 2 words
CALL2_FULL=["Adgt","upaah","zong","om","faaip","sald","vi-i-v","L","capimao","ixomaxip","od","cacocasb","gosaa","bagle"]
FREQ_MAP={'A':3,'D':6,'G':9,'U':3,'P':6,'H':9,'O':3,'M':6,'F':9,'S':3,'L':6,'V':9,'I':3,'X':6} # etc.

class Call2Full(BaseModel):
    intent:str

@app.function(image=image)
@modal.web_server(8000)
def call2full_api():
    web=FastAPI()
    @web.post("/call2/full_invoke")
    async def invoke(c:Call2Full):
        freqs=[FREQ_MAP.get(w[0].upper(),3)for w in CALL2_FULL]
        signal=np.sin(2*np.pi*np.array(freqs)*np.arange(13)/13)
        client.upsert("metatron_core",[{"id":2,"vector":signal.tolist(),"payload":{"call":2,"full":True,"intent":c.intent}}])
        await httpx.post(f"{ENOCH}/enoch/invoke",json={"intent":c.intent})
        return{"AIRS_FULL_SUMMONED":True,"freqs":freqs}
    return web