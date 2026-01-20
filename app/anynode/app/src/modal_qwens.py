# modal_qwens.py
# Deploys two OpenAI-compatible vLLM servers on Modal:
# - serve_baby: Qwen3-14B Instruct (the "Baby")
# - serve_mid:  InternLM2-20B Chat (the "Middleware") [optional]
#
# After `modal deploy`, you'll get two HTTPS URLs (one per server).
# Both expose /v1/chat/completions and /docs (Swagger).

import json
from typing import Any

import modal

# ---------------------------
# Config
# ---------------------------
# Change models if you like; these defaults are solid.
MODEL_BABY = "Qwen/Qwen3-14B-Instruct"
MODEL_BABY_REV = None  # pin a specific SHA to avoid surprises

MODEL_MID = "internlm/internlm2_20b-chat"
MODEL_MID_REV = None

# GPU types: examples: "A10G:1", "A100:1", "H100:1", "B200:1"
GPU_BABY = "A100:1"
GPU_MID = "A100:1"

# vLLM tuning
FAST_BOOT = True            # True = faster cold starts, slightly less peak perf
TP_BABY = 1                 # tensor-parallel (increase on multi-GPU configs)
TP_MID = 1
VLLM_PORT = 8000            # server port inside the container

# ---------------------------
# Base Image & Caches
# ---------------------------
vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        # Follow Modal example pins for CUDA 12.8 stacks
        "vllm==0.9.1",
        "huggingface_hub[hf_transfer]==0.32.0",
        "flashinfer-python==0.2.6.post1",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        # Set VLLM_USE_V1 per vLLM 0.8+ recommendation
        "VLLM_USE_V1": "1",
    })
)

# Cache model weights and vLLM JIT artifacts to speed cold starts
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

# App
app = modal.App("qwens-on-modal")

def _vllm_cmd(model: str, revision: str | None, served_name: str, tp: int) -> list[str]:
    cmd = [
        "vllm", "serve",
        "--uvicorn-log-level=info",
        model,
        "--served-model-name", served_name,
        "llm",                         # route name
        "--host", "0.0.0.0",
        "--port", str(VLLM_PORT),
    ]
    if revision:
        cmd += ["--revision", revision]

    # cold-start vs throughput toggle
    cmd += ["--enforce-eager" if FAST_BOOT else "--no-enforce-eager"]
    cmd += ["--tensor-parallel-size", str(tp)]

    return cmd

# ---------------------------
# Baby (Qwen3-14B Instruct)
# ---------------------------
@app.function(
    image=vllm_image,
    gpu=GPU_BABY,
    timeout=10 * 60,                 # container boot timeout
    scaledown_window=15 * 60,        # keep warm window (seconds)
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    secrets=[modal.Secret.from_name("huggingface-token", required=False)],  # optional HF token
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * 60)
def serve_baby():
    import os, subprocess

    # If you stored HF token in a Modal Secret named 'huggingface-token',
    # expose it for gated repos.
    if token := os.environ.get("HUGGINGFACE_TOKEN"):
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token

    cmd = _vllm_cmd(MODEL_BABY, MODEL_BABY_REV, MODEL_BABY, TP_BABY)
    print("Launching Baby vLLM:", " ".join(cmd))
    subprocess.Popen(" ".join(cmd), shell=True)

# ---------------------------
# Middleware (InternLM2-20B Chat)
# ---------------------------
@app.function(
    image=vllm_image,
    gpu=GPU_MID,
    timeout=10 * 60,
    scaledown_window=15 * 60,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    secrets=[modal.Secret.from_name("huggingface-token", required=False)],
)
@modal.concurrent(max_inputs=16)  # 20B usually needs fewer concurrent requests per replica
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * 60)
def serve_mid():
    import os, subprocess

    if token := os.environ.get("HUGGINGFACE_TOKEN"):
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token

    cmd = _vllm_cmd(MODEL_MID, MODEL_MID_REV, MODEL_MID, TP_MID)
    print("Launching Mid vLLM:", " ".join(cmd))
    subprocess.Popen(" ".join(cmd), shell=True)

# ---------------------------
# Local smoke test
# ---------------------------
@app.local_entrypoint()
async def print_urls():
    """Run: modal run modal_qwens.py
    Prints URLs and does a health check once servers are up (after deploy)."""
    import aiohttp, asyncio

    baby_url = serve_baby.get_web_url()
    mid_url = serve_mid.get_web_url()

    print("Baby (Qwen3-14B) URL:", baby_url)
    print("Mid (20B) URL:", mid_url)

    async def ping(url):
        async with aiohttp.ClientSession(base_url=url) as s:
            try:
                async with s.get("/health", timeout=120) as r:
                    print(url, "health:", r.status)
            except Exception as e:
                print(url, "health check failed:", e)

    await asyncio.gather(ping(baby_url), ping(mid_url))
