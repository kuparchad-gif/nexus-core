# enoch_calls_1_18.py - Calls 1-18
import modal, json, httpx
from fastapi import FastAPI
from pydantic import BaseModel

app = modal.App("enoch-calls-1-18")
image = modal.Image.debian_slim().pip_install("fastapi uvicorn httpx")
with open("enoch_calls.json") as f:  # Full 1-18
    CALLS = json.load(f)

METATRON = "http://metatron-enoch.modal.app:8000"

class CallInvoke(BaseModel):
    call: int  # 1-18
    intent: str

@app.function(image=image)
@modal.web_server(8000)
def calls_api():
    web = FastAPI()
    @web.post("/call/{num}")
    async def invoke_call(num: int, inv: CallInvoke):
        if num not in range(1,19):
            return {"error": "Invalid Call"}
        call = CALLS[str(num)]
        # Freq map + signal
        freqs = [3 if c in "AEO" else 6 if c in "IU" else 9 for c in call["enochian"][:13]]
        signal = np.sin(2*np.pi*np.array(freqs)*np.arange(13)/13)
        # Store + invoke
        client.upsert("elemental_calls", [{"id":num, "vector":signal.tolist(), "payload":call}])
        await httpx.post(f"{METATRON}/enoch/invoke", json={"intent": f"Call {num}: {inv.intent}"})
        return {"CALL": num, "ELEMENT": call["element"], "INVOKED": True}
    return web