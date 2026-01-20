# OSS Tiering Patch (GPT-OSS → Llama/Mistral)

This patch sets **GPT-OSS (OLMo)** as Tier 1 and **Llama/Mistral** as Tier 2 fallbacks.

## 1) Drop-in
Unzip into your project root (overwrites `config/models.yaml` and adds scripts).

## 2) Start OSS backends
```bash
bash scripts/run_oss_servers.sh
# (separate terminal) ensure ollama is up for Tier 2:
# ollama serve
# ollama pull llama3.1:8b-instruct
# ollama pull mistral:7b-instruct
```

## 3) Start Gema Gateway
```bash
export NEXUS_JWT_SECRET=change-me
bash scripts/run_gema_gateway.sh &   # serves :8016
```

## 4) Test
```bash
TOKEN=$(python - <<'PY'
import hmac, hashlib, os
print(hmac.new(os.getenv("NEXUS_JWT_SECRET","change-me").encode(), b"AUTH", hashlib.sha256).hexdigest())
PY
)

# chat (Tier 1 → OLMo; Tier 2 → Llama/Mistral if Tier 1 unavailable)
curl -s -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json"   -X POST http://localhost:8016/chat   -d '{"messages":[{"role":"user","content":"Summarize policy-gated autonomy in 2 bullets."}], "route":"chat.general"}' | jq
```

## Notes
- **Model Names:** If `allenai/OLMo-7B-Instruct` differs on your HF registry, edit `config/models.yaml` and the vLLM command.
- **Embeddings:** `BAAI/bge-large-en-v1.5` requires vLLM with embeddings support; otherwise use Ollama's `embeddings` endpoint or switch to `nomic-embed-text` and a compatible server.
- **Licenses:** All listed models are OSS-friendly; verify licenses before production.
