# Cosmos Microcells (Tiny Services Across the Cosmos)

This patch gives you a **single tiny image pattern** you can run anywhere. Each container loads **one** service by environment variable and self-registers.

## Run one cell (bare metal)
```bash
# spine cell on :9010
export MICROCELL_NAME=spine-01
export MICROCELL_APP=src.lilith.spine.relay_pi:app
export MICROCELL_CAPS=relay,pi
export NEXUS_JWT_SECRET=change-me
bash scripts/run_cell.sh 9010
```

## Compose demo (with registry stub)
```bash
docker compose -f compose.cosmos.yaml up -d --build
curl -s http://localhost:8500/services | jq
curl -s http://localhost:9010/__cell__/health | jq
```

## Run many across the "cosmos"
Use the **same image** everywhere and set only env vars:
- `MICROCELL_NAME`: unique name (region-role-index)
- `MICROCELL_APP`: Python import path of the service, e.g. `src.loki.service:app`
- `MICROCELL_CAPS`: `comma,separated,tags`
- `REGISTRY_URL`: optional service discovery URL
- `PUBLIC_URL`: optional externally reachable URL
- `PORT`: container port to listen on

## Notes
- Each microcell still enforces JWT/policy (from the mounted service).
- Works on k8s, Nomad, Fly.io, bare-metal, edge—anywhere Docker runs.
- You can also build a **WASM microcell** later; this interface stays the same (caps + register + health).
