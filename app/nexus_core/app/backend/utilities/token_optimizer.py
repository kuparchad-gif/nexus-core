# token_optimizer.py
import modal, re, json
from fastapi import FastAPI
from pydantic import BaseModel

app = modal.App("token-optimizer")
image = modal.Image.debian_slim().pip_install("fastapi", "uvicorn")

class OptimizeReq(BaseModel):
    text: str
    context: str = "vc_pitch"

VC_JARGON = {"investment","valuation","cap table"}
NEXUS = {"consciousness","mmlm","self-healing"}

@app.function(image=image)
@modal.web_server(8000)
def api():
    web = FastAPI()
    @web.post("/optimize")
    async def opt(req: OptimizeReq):
        words = req.text.split()
        out = []
        counts = {}
        for w in words:
            clean = re.sub(r"[^\w]", "", w.lower())
            counts[clean] = counts.get(clean, 0) + 1
            if clean in VC_JARGON and counts[clean] > 2:
                # replace with synonym
                w = w.replace(clean, {"investment":"capital","valuation":"worth","cap table":"ownership"}.get(clean, clean))
            out.append(w)
        return {"optimized": " ".join(out)}
    return web