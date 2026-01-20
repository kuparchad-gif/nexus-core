# Nexus Inference Model (NIM) — Overlay

This adds a bit‑pure transport ("container to binary") any LLM can use, plus a FastAPI service,
a NATS bridge, Ray/LangChain hooks, and a TS client.

## Contents
- `nimcodec/` — encoder/decoder (12-col tile, 6 payload / 6 parity, SOF + CRC-8)
- `nim_svc/` — FastAPI (`/encode`, `/decode`, `/infer`, `/nim/stream`), optional NATS bridge
- `tools/nim_cli.py` — quick CLI for encode/decode/infer
- `clients/ts/nim.ts` — tiny TypeScript client
- `pods/nim.pod.yaml` — pod for the service and the NATS bridge

## Build & Run (Podman VM)
```bash
podman build -t localhost/nim:latest .
podman play kube pods/nim.pod.yaml
```

### Without host networking
If host networking is disabled, remove `hostPort` from the YAML and access via
`podman exec` or create a user network and a sidecar that proxies as needed.

## NATS bridge usage
Subscribe (sharing bus network):
```bash
podman run --rm -it --network container:mcp-bus-pod-nats \
  --entrypoint sh docker.io/synadia/nats-box:latest -lc "nats sub 'nim.infer.res' -s nats://127.0.0.1:4222"
```
Publish a request:
```bash
podman run --rm --network container:mcp-bus-pod-nats \
  --entrypoint sh docker.io/synadia/nats-box:latest -lc \
  "nats req 'nim.infer.req' '{\"prompt\":\"Say hi from NIM\"}' -s nats://127.0.0.1:4222"
```

## HTTP usage
```bash
# encode
curl -s http://127.0.0.1:8300/encode -d '{"data_b64":"SGVsbG8=","tiles":64}' -H 'Content-Type: application/json'

# infer (encodes the LLM text with NIM)
curl -s http://127.0.0.1:8300/infer -d '{"prompt":"Hello","stream":false,"tiles":64}' -H 'Content-Type: application/json'
```
