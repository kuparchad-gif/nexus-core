# enoch_call3_full.py - Call 3 Full
import modal,np,httpx
from fastapi import FastAPI
from pydantic import BaseModel

app=modal.App("enoch-call3-full")
image=modal.Image.debian_slim().pip_install("fastapi uvicorn httpx numpy")
client=QdrantClient(path="/qdrant")
ENOCH="http://metatron-enoch.modal.app:8000"

# Call 3 words (partial for brevity)
CALL3_FULL=["Micma","goho","Piad","zir","com-selh","a","zien","biah","os","londoh"]
FREQ_MAP={'M':3,'G':6,'P':9,'Z':3,'C':6,'A':9,'B':3,'O':6,'L':9,'S':3,'I':6,'N':9}

class Call3Full(BaseModel):
    intent:str

@app.function(image=image)
@modal.web_server(8000)
def call3full_api():
    web=FastAPI()
    @web.post("/call3/full_invoke")
    async def invoke(c:Call3Full):
        freqs=[FREQ_MAP.get(w[0].upper(),3)for w in CALL3_FULL]
        signal=np.sin(2*np.pi*np.array(freqs)*np.arange(13)/13)
        client.upsert("metatron_core",[{"id":3,"vector":signal.tolist(),"payload":{"call":3,"full":True,"intent":c.intent}}])
        await httpx.post(f"{ENOCH}/enoch/invoke",json={"intent":c.intent})
        return{"WATERS_FULL_SUMMONED":True,"freqs":freqs}
    return web