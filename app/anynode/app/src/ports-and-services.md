# Ports & Services (overlay v3)

## Core
- Loki (HTTP): 3100
- Stem Router (HTTP/gRPC): 7070 / 7071
- Pineal (HTTP): 7080
- Lilith WS Bridge (WS/HTTP): 7090

## Memory / Vector
- Qdrant: 6333 (HTTP), 6334 (gRPC)

## Orchestration
- Ray Dashboard: 8265
- Ray Client: 10001

## Notes
- All CogniKubes communicate with Stem via stem-adapter over gRPC/HTTP (see `stem-adapters/`).
- Default org/tenant for Loki testing: `fake` (header `X-Scope-OrgID: fake`).

