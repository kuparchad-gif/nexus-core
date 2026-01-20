import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .schemas import RouteRequest, RouteResponse
from .config import MODE_DEFAULT, TOP_K_DEFAULT, COMBINER_DEFAULT
from . import moe
from .prompts import router as prompts_router

app  =  FastAPI(title = "Nexus MoE Gateway", version = "1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"], allow_credentials = True,
    allow_methods = ["*"], allow_headers = ["*"],
)

app.include_router(prompts_router)

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/v1/route", response_model = RouteResponse)
async def route(req: RouteRequest):
    mode  =  req.mode or MODE_DEFAULT
    k  =  req.k or TOP_K_DEFAULT
    combiner  =  req.combiner or COMBINER_DEFAULT
    scores, selected, outputs, combined  =  await moe.route(req.prompt, mode, k, combiner)
    return {
        "prompt": req.prompt,
        "mode": mode,
        "selected": selected,
        "scores": scores,
        "outputs": outputs,
        "combined": combined,
    }

if __name__ == "__main__":
    import uvicorn
    port  =  int(os.getenv("PORT", 1313))
    uvicorn.run(app, host = "0.0.0.0", port = port)
