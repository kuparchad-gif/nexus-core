# modal_hermes.py
# Aethereal Nexus — Hermes on Modal (1.0 API)
#
# - Builds llama.cpp server in the image (OpenBLAS + CURL)
# - Mounts volume "hermes-models" at /models (put your GGUF there)
# - On startup, chooses MODEL_PATH or first *.gguf (or downloads MODEL_URL)
# - Spawns llama-server on 127.0.0.1:8001
# - Exposes FastAPI: /health, /v1/models, /v1/chat/completions
#
# Env vars (optional):
#   MODEL_PATH     e.g. /models/hermes7b.Q4_K_M.gguf
#   MODEL_URL      http(s)://.../model.gguf  (fallback if /models is empty)
#   CTX_SIZE       default 4096
#   BATCH_SIZE     default 128
#   CHAT_TEMPLATE  e.g. "chatml"
#
# Modal 1.0 notes:
# - Use @app.function(image=..., volumes=..., max_containers=..., scaledown_window=...)
# - Stack with @asgi_app() (no kwargs)

from __future__ import annotations

import asyncio
import os
import pathlib
import shlex
import signal
import subprocess
from contextlib import asynccontextmanager
from typing import Optional

import aiohttp
import fastapi
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

import modal
from modal import asgi_app

app = modal.App("aethereal-nexus-hermes")

# ---------- Build image with llama.cpp (server) ----------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git",
        "cmake",
        "build-essential",
        "libopenblas-dev",
        "libcurl4-openssl-dev",
        "wget",
        "curl",
        "ca-certificates",
    )
    .pip_install("fastapi", "uvicorn[standard]", "aiohttp")
    .run_commands(
        # Fetch llama.cpp
        "git clone --depth 1 https://github.com/ggerganov/llama.cpp /opt/llama.cpp",
        # Configure
        "cmake -S /opt/llama.cpp -B /opt/llama.cpp/build "
        "-DLLAMA_BUILD_SERVER=ON -DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS -DLLAMA_CURL=ON",
        # Build
        "cmake --build /opt/llama.cpp/build --config Release -j $(nproc)",
    )
)

# Volume with your GGUF(s)
models_vol = modal.Volume.from_name("hermes-models", create_if_missing=True)

LLAMA_BIN = "/opt/llama.cpp/build/bin/llama-server"
LLAMA_URL = "http://127.0.0.1:8001"


def _find_model_file(explicit: Optional[str]) -> Optional[str]:
    """Pick a GGUF from explicit path or first *.gguf in /models."""
    if explicit:
        p = pathlib.Path(explicit)
        if p.exists() and p.is_file():
            return str(p)
    for p in sorted(pathlib.Path("/models").glob("*.gguf")):
        if p.is_file():
            return str(p)
    return None


async def _download_model(url: str, dest_path: str) -> None:
    """Download a GGUF into /models (resumeless, simple)."""
    tmp_path = dest_path + ".part"
    async with aiohttp.ClientSession() as s:
        async with s.get(url, timeout=aiohttp.ClientTimeout(total=None)) as r:
            r.raise_for_status()
            with open(tmp_path, "wb") as f:
                async for chunk in r.content.iter_chunked(1 << 20):
                    f.write(chunk)
    os.replace(tmp_path, dest_path)


async def _llm_ready(session: aiohttp.ClientSession, timeout_s: float = 2.0) -> bool:
    try:
        async with session.get(f"{LLAMA_URL}/v1/models", timeout=timeout_s) as resp:
            return resp.status == 200
    except Exception:
        return False


async def _wait_llm_ready(session: aiohttp.ClientSession, retries: int = 60, delay: float = 1.0) -> None:
    for _ in range(retries):
        if await _llm_ready(session):
            return
        await asyncio.sleep(delay)
    raise RuntimeError("llama-server never became ready")


async def _start_llama_if_needed() -> str:
    """
    Ensure a model is present, then start llama-server as a subprocess.
    Returns the selected model path.
    """
    model_path_env = os.getenv("MODEL_PATH", "").strip()
    model_url_env = os.getenv("MODEL_URL", "").strip()
    ctx = int(os.getenv("CTX_SIZE", "4096"))
    bsz = int(os.getenv("BATCH_SIZE", "128"))
    chat_tmpl = os.getenv("CHAT_TEMPLATE", "").strip()

    # Ensure model exists
    model_path = _find_model_file(model_path_env)
    if not model_path and model_url_env:
        pathlib.Path("/models").mkdir(parents=True, exist_ok=True)
        dest = "/models/" + pathlib.Path(model_url_env).name
        await _download_model(model_url_env, dest)
        model_path = dest

    if not model_path:
        raise RuntimeError(
            "No model file found in /models and no MODEL_URL provided. "
            "Upload a GGUF to the 'hermes-models' volume or set MODEL_URL."
        )

    # Check if already running
    try:
        async with aiohttp.ClientSession() as s:
            if await _llm_ready(s, timeout_s=0.25):
                return model_path
    except Exception:
        pass

    # Build llama-server command (CPU-only, -ngl 0; avoid mmap issues with --no-mmap)
    cmd = [
        LLAMA_BIN,
        "-m",
        model_path,
        "--host",
        "127.0.0.1",
        "--port",
        "8001",
        "--no-mmap",
        "-ngl",
        "0",
        "--ctx-size",
        str(ctx),
        "--batch-size",
        str(bsz),
    ]
    if chat_tmpl:
        cmd += ["--chat-template", chat_tmpl]

    # Launch background process; avoid PIPE buffering by discarding output
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,  # new process group for clean kill
        text=False,
    )
    os.environ["LLAMA_PID"] = str(proc.pid)
    return model_path


def _stop_llama():
    pid = os.environ.get("LLAMA_PID")
    if not pid:
        return
    try:
        # Kill process group first (best-effort)
        os.killpg(int(pid), signal.SIGTERM)
    except Exception:
        try:
            os.kill(int(pid), signal.SIGTERM)
        except Exception:
            pass
    finally:
        os.environ.pop("LLAMA_PID", None)


# ---------- FastAPI with lifespan ----------

@asynccontextmanager
async def lifespan(app_: fastapi.FastAPI):
    session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60))
    app_.state.http = session
    model_path = await _start_llama_if_needed()
    await _wait_llm_ready(session)
    app_.state.model_path = model_path
    try:
        yield
    finally:
        try:
            await session.close()
        finally:
            _stop_llama()


def build_api() -> fastapi.FastAPI:
    api = fastapi.FastAPI(title="Hermes (Modal)", lifespan=lifespan)

    @api.get("/")
    async def root():
        return {"ok": True, "service": "Hermes (Modal)", "llama_url": LLAMA_URL}

    @api.get("/health")
    async def health():
        return {"ok": True, "llama_url": LLAMA_URL}

    @api.get("/v1/models")
    async def list_models(request: Request):
        try:
            async with request.app.state.http.get(f"{LLAMA_URL}/v1/models") as r:
                # Proxy verbatim (JSON when possible)
                if r.headers.get("content-type", "").startswith("application/json"):
                    return JSONResponse(status_code=r.status, content=await r.json())
                return JSONResponse(status_code=r.status, content={"text": await r.text()})
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"LLM unreachable: {e}")

    @api.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        try:
            payload = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON body")

        try:
            async with request.app.state.http.post(f"{LLAMA_URL}/v1/chat/completions", json=payload) as r:
                if r.headers.get("content-type", "").startswith("application/json"):
                    return JSONResponse(status_code=r.status, content=await r.json())
                return JSONResponse(status_code=r.status, content={"text": await r.text()})
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"LLM unreachable: {e}")

    return api


# ---------- Modal function + ASGI wrapper (Modal 1.0) ----------

@app.function(
    image=image,
    volumes={"/models": models_vol},   # mount volume at /models
    max_containers=1,                  # replaces concurrency_limit
    scaledown_window=180,              # replaces container_idle_timeout
)
@asgi_app()  # no kwargs in Modal 1.0
def web():
    return build_api()
