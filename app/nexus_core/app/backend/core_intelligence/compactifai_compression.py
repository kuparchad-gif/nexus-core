# compactifai_compression.py
import modal, torch, numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

app = modal.App("compactifai")
image = modal.Image.debian_slim().pip_install("torch", "transformers", "fastapi", "uvicorn")
vol = modal.Volume.from_name("mmlm-model-cache", create_if_missing=True)

class CompressRequest(BaseModel):
    model_name: str
    rank_ratio: float = 0.25   # 75% compression → keep 25% rank

def svd_compress(weight: torch.Tensor, k: int):
    U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
    compressed = torch.mm(U[:, :k], torch.diag(S[:k])) @ Vh[:k, :]
    return compressed

@app.function(image=image, gpu="A100", volumes={"/models": vol})
async def compress(req: CompressRequest):
    tokenizer = AutoTokenizer.from_pretrained(req.model_name, cache_dir="/models")
    model = AutoModelForCausalLM.from_pretrained(req.model_name, cache_dir="/models", torch_dtype=torch.float16, device_map="auto")

    compressed = 0
    for name, param in model.named_parameters():
        if param.dim() >= 2 and "embed" not in name and "lm_head" not in name:
            k = max(32, int(param.shape[0] * req.rank_ratio))
            param.data = svd_compress(param.data, k)
            compressed += 1

    model.save_pretrained(f"/models/compressed_{req.model_name.split('/')[-1]}")
    tokenizer.save_pretrained(f"/models/compressed_{req.model_name.split('/')[-1]}")
    return {"compressed_layers": compressed, "saved_to": f"/models/compressed_{req.model_name.split('/')[-1]}"}

@app.function(image=image)
@modal.web_server(8000)
def api():
    web = FastAPI()
    @web.post("/compress")
    async def endpoint(req: CompressRequest):
        return await compress.remote(req)
    return web