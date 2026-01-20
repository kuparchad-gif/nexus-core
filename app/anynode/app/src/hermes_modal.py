import modal
image = (modal.Image.debian_slim()
    .apt_install("git","build-essential","cmake","wget","curl","ca-certificates")
    .run_commands("git clone --depth 1 https://github.com/ggml-org/llama.cpp /opt/llama.cpp",
                  "cmake -S /opt/llama.cpp -B /opt/llama.cpp/build",
                  "cmake --build /opt/llama.cpp/build -j")
    .pip_install("fastapi","uvicorn"))
app = modal.App("hermes-modal")
@app.function(image=image, cpu=4.0, timeout=600)
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI, Body
    import os, subprocess, urllib.request
    MODEL_URL = os.environ.get("MODEL_URL","https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/q4_k_m.gguf")
    MODEL_PATH = "/root/model.gguf"
    def ensure_model():
        if not os.path.exists(MODEL_PATH):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    app = FastAPI(title="Hermes (Modal)", version="1.0")
    @app.get("/v1/models")
    def models(): return {"data":[{"id":"hermes-cpu-gguf","object":"model"}]}
    @app.post("/v1/chat/completions")
    def chat(payload: dict = Body(...)):
        ensure_model()
        messages = payload.get("messages", [])
        prompt = messages[-1].get("content","") if messages else ""
        out = subprocess.check_output(["/opt/llama.cpp/build/bin/llama-cli","-m",MODEL_PATH,"-p",prompt,"-n","200","--temp","0.8"], text=True)
        return {"choices":[{"message":{"role":"assistant","content":out}}]}
