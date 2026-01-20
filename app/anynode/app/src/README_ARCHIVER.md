# Archiver (Qdrant-only)

A minimal, fast Archiver microcell that writes events to **Qdrant** and makes them searchable (vector + payload). It uses your **Gema Gateway /embed** to get embeddings.

## Env
- `QDRANT_URL` (default `http://localhost:6333`)
- `QDRANT_API_KEY` (optional)
- `ARCHIVER_COLLECTION` (default `nexus_events`)
- `EMBED_URL` (default `http://localhost:8016/embed`)
- `EMBED_TOKEN` (optional, e.g., JWT from your gateway)
- `ARCHIVER_HMAC_KEY` (default `NEXUS_JWT_SECRET`)

## Run as a microcell
```bash
export MICROCELL_NAME=archiver-01
export MICROCELL_APP=src.aethereal.archiver.service:app
export MICROCELL_CAPS=archive,search,timeline
export QDRANT_URL=http://localhost:6333
export EMBED_URL=http://localhost:8016/embed
bash scripts/run_cell.sh 9020
```

## Endpoints
- `GET /health`
- `POST /archive` — body: `{source, topic, tags[], severity, policy_level, phase, payload{...}}`
- `POST /search` — body: `{q, top_k, tags?, source?}`
- `POST /timeline` — body: `{since?, until?, limit}`
- `GET /verify_chain` — tamper-evident check using HMAC over canonical payloads

## Curl
```bash
curl -s -X POST http://localhost:9020/archive -H 'Content-Type: application/json' -d '{
  "source":"loki","topic":"export","tags":["logs"],"severity":"INFO","payload":{"count":42}
}' | jq

curl -s -X POST http://localhost:9020/search -H 'Content-Type: application/json' -d '{
  "q":"error export logs", "top_k":5
}' | jq
```
