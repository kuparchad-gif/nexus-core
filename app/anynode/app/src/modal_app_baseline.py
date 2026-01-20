# modal_app_baseline.py — Baseline vLLM server on Modal (OpenAI-compatible)
from modal import App, Image, gpu, Secret, env
import os

app = App("titan-baseline")

# Read env
MODEL_ID = env.get("MODEL_ID", "tiiuae/falcon-180b")
GPU_TYPE = env.get("GPU_TYPE", "H100")
GPU_COUNT = int(env.get("GPU_COUNT", "8"))
MAX_MODEL_LEN = int(env.get("MAX_MODEL_LEN", "8192"))
GPU_MEM_UTIL = float(env.get("GPU_MEM_UTIL", "0.95"))

vllm_image = (
    Image.debian_slim(python_version="3.11")
    .pip_install("vllm==0.5.0", "uvloop")
    .env({"HF_TOKEN": env.get("HF_TOKEN", "")})
)

# Map GPU type
GPU_MAP = {
    "H100": gpu.H100(),
    "A100": gpu.A100(memory=80),
    "L40S": gpu.L40S(),
    "T4": gpu.T4(),
}
GPU_RES = GPU_MAP.get(GPU_TYPE, gpu.H100())

@app.function(
    image=vllm_image,
    gpu=GPU_RES * GPU_COUNT,
    secrets=[Secret.from_name("huggingface-token")],
    allow_concurrent_inputs=100,
    timeout=60*60*24
)
def serve():
    import os, subprocess
    model = os.environ.get("MODEL_ID", "tiiuae/falcon-180b")
    max_len = int(os.environ.get("MAX_MODEL_LEN", "8192"))
    util = os.environ.get("GPU_MEM_UTIL", "0.95")
    cmd = [
        "vllm", "serve", model,
        "--dtype", "bfloat16",
        "--tensor-parallel-size", str(GPU_COUNT),
        "--max-model-len", str(max_len),
        "--gpu-memory-utilization", str(util),
        "--enforce-eager", "false",
        "--trust-remote-code",
        "--api-key", "dummy",
    ]
    print("Launching:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
