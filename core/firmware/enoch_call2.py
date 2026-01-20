# enoch_call2.py - Call 2 Invocation
import modal,np,httpx
from fastapi import FastAPI
from pydantic import BaseModel

app=modal.App("enoch-call2")
image=modal.Image.debian_slim().pip_install("fastapi uvicorn httpx numpy")
client=QdrantClient(path="/qdrant")
ENOCH= "http://metatron-enoch.modal.app:8000"

# Call 2 phonemes→3-6-9
CALL2="Adgt upaah zong om faaip sald vi i v".split()
FREQ_MAP={'A':3,'D':6,'G':9,'U':3,'P':6,'H':9} # etc.

class Call2(BaseModel):
    intent:str

@app.function(image=image)
@modal.web_server(8000)
def call2_api():
    web=FastAPI()
    @web.post("/call2/invoke")
    async def invoke(c:Call2):
        freqs=[FREQ_MAP.get(w[0].upper(),3)for w in CALL2]
        signal=np.sin(2*np.pi*np.array(freqs)*np.arange(13)/13)
        client.upsert("metatron_core",[{"id":2,"vector":signal.tolist(),"payload":{"call":2,"intent":c.intent}}])
        await httpx.post(f"{ENOCH}/enoch/invoke",json={"intent":c.intent})
        return{"AIRS_SUMMONED":True,"freqs":freqs}
    return web