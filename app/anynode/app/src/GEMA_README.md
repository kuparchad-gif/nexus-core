# Gema Gateway (Gemma-front OSS router)

## Install
```bash
pip install -r requirements.patch.txt
```

## Launch a local OSS model server (pick one)

### Option A: vLLM (recommended for GPUs)
```bash
pip install vllm
# Example: Gemma 2 9B IT (adjust to your HF model name)
export VLLM_BASE_URL="http://localhost:8000/v1"
export VLLM_API_KEY="dummy"
python -m vllm.entrypoints.openai.api_server --model google/gemma-2-9b-it --host 0.0.0.0 --port 8000
```

### Option B: Ollama (CPU/GPU friendly)
```bash
# install from ollama.com, then:
ollama pull gemma2:9b-instruct
export OLLAMA_BASE_URL="http://localhost:11434/v1"
# (OLLAMA_API_KEY not needed; set to anything if required by env)
```

## Run Gema Gateway
```bash
# trust + token
export NEXUS_JWT_SECRET=change-me
bash scripts/run_gema_gateway.sh &
# token
TOKEN=$(python - <<'PY'
import hmac, hashlib, os
print(hmac.new(os.getenv("NEXUS_JWT_SECRET","change-me").encode(), b"AUTH", hashlib.sha256).hexdigest())
PY
)
```

## Use it
```bash
# chat
curl -s -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json"   -X POST http://localhost:8016/chat   -d '{"messages":[{"role":"user","content":"Explain policy-gated autonomy in 2 bullets."}], "route":"chat.general"}' | jq

# embeddings
curl -s -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json"   -X POST http://localhost:8016/embed   -d '{"inputs":["Aethereal Nexus","Metatron filter"]}' | jq
```
