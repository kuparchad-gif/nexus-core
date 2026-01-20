# UCE Router (Inference Router)

FastAPI router that proxies OpenAI-style endpoints to your llama.cpp backends.
- Aggregates `/v1/models`
- Proxies `/v1/chat/completions` and `/v1/completions`
- Exposes Prometheus `/metrics`
- Optional NATS relay on subject `uce.req`

## Compose snippet

```yaml
  uce-router:
    build: ./cognikubes/uce-router
    environment:
      ROUTER_NAME: uce-router
      NATS_URL: nats://nats:4222
      BACKENDS: |
        [
          {"id":"7b","url":"http://llama-7b:8001","default":true},
          {"id":"14b","url":"http://llama-14b:8002"}
        ]
    ports: ["8007:8007"]
    profiles: ["base","router","firmware","inference"]
    depends_on:
      nats:
        condition: service_started
      llama-7b:
        condition: service_started
      llama-14b:
        condition: service_started
```

## Build & Run

```powershell
podman compose -p nexus -f .\podman-compose.yaml up -d uce-router
Invoke-WebRequest http://localhost:8007/health -UseBasicParsing | Select-Object -Expand Content
```

## Quick test

```powershell
# Non-streamed chat
$chat = @{
  model = "14b"
  messages = @(@{role="user";content="Say hi in 6 words."})
  max_tokens = 32
} | ConvertTo-Json -Depth 6
Invoke-RestMethod -Method Post -Uri "http://localhost:8007/v1/chat/completions" -Body $chat -ContentType "application/json"
```
