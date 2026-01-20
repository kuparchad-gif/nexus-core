from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os, hmac, hashlib, yaml
from src.lilith.policy.trust_phase import TrustPhaseMixin

# We use OpenAI-compatible clients for vLLM/TGI/Ollama
from openai import OpenAI

JWT_SECRET = os.getenv("NEXUS_JWT_SECRET", "dev-secret")

def _verify_jwt(authorization: Optional[str]) -> None:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization.split(" ",1)[1].strip()
    sig = hmac.new(JWT_SECRET.encode(), b"AUTH", hashlib.sha256).hexdigest()
    if not hmac.compare_digest(token, sig):
        raise HTTPException(status_code=401, detail="Invalid token")

def _load_models_cfg() -> dict:
    path = os.getenv("AETHEREAL_MODELS_YAML", "config/models.yaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _mk_client(provider: str, cfg: dict) -> OpenAI:
    prov = cfg["providers"][provider]
    base_url = os.getenv(prov["base_url_env"])
    api_key  = os.getenv(prov["api_key_env"], "nokey")
    if not base_url:
        raise HTTPException(status_code=500, detail=f"{provider} base URL not set ({prov['base_url_env']})")
    return OpenAI(base_url=base_url, api_key=api_key)

class ChatMsg(BaseModel):
    role: str
    content: str

class ChatReq(BaseModel):
    messages: List[ChatMsg]
    route: str = "chat.general"   # task route key
    model: Optional[str] = None   # force a model name
    provider: Optional[str] = None
    max_tokens: int = 512
    temperature: float = 0.6
    tools_allowed: bool = False

class EmbedReq(BaseModel):
    inputs: List[str]
    route: str = "embed.default"
    model: Optional[str] = None
    provider: Optional[str] = None

app = FastAPI(title="Gema Gateway")

class GemaGateway(TrustPhaseMixin):
    def __init__(self):
        super().__init__()
        self.cfg = _load_models_cfg()

GW = GemaGateway()

@app.get("/policy")
def policy(authorization: Optional[str] = Header(default=None)):
    _verify_jwt(authorization)
    return GW.policy_dict()

@app.post("/chat")
def chat(req: ChatReq, authorization: Optional[str] = Header(default=None)):
    _verify_jwt(authorization)
    # honor policy gate for tools exposure
    if req.tools_allowed and not GW.gates()["can_open_tools"]:
        raise HTTPException(status_code=403, detail="Tool use gated by policy (requires policy_level<=10)")

    cfg = GW.cfg
    provider = req.provider or cfg.get("default_provider", "vllm")
    model = req.model or cfg.get("default_model", "gemma-2-9b-it")
    # route-aware selection
    if req.route in cfg.get("routes", {}):
        # pick first available candidate by trying to create a client
        last_error = None
        for cand in cfg["routes"][req.route]:
            try:
                prov = cand.get("provider", provider)
                mdl  = cand.get("model", model)
                client = _mk_client(prov, cfg)
                # single-shot non-stream for simplicity
                resp = client.chat.completions.create(
                    model=mdl,
                    messages=[m.model_dump() for m in req.messages],
                    max_tokens=req.max_tokens,
                    temperature=req.temperature,
                )
                text = resp.choices[0].message.content
                return {"provider": prov, "model": mdl, "text": text}
            except Exception as e:
                last_error = str(e)
        raise HTTPException(status_code=502, detail=f"All candidates failed for route {req.route}: {last_error}")
    else:
        # simple default path
        client = _mk_client(provider, cfg)
        resp = client.chat.completions.create(
            model=model,
            messages=[m.model_dump() for m in req.messages],
            max_tokens=req.max_tokens,
            temperature=req.temperature,
        )
        text = resp.choices[0].message.content
        return {"provider": provider, "model": model, "text": text}

@app.post("/embed")
def embed(req: EmbedReq, authorization: Optional[str] = Header(default=None)):
    _verify_jwt(authorization)
    cfg = GW.cfg
    provider = req.provider or cfg.get("default_provider", "vllm")
    model = req.model or (cfg.get("routes", {}).get(req.route, [{}])[0].get("model") if cfg.get("routes", {}).get(req.route) else None)
    model = model or "bge-large-en-v1.5"
    client = _mk_client(provider, cfg)
    # OpenAI embeddings API compatible
    resp = client.embeddings.create(model=model, input=req.inputs)
    return {"provider": provider, "model": model, "vectors": [e.embedding for e in resp.data]}
