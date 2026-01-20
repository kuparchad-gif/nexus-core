Apply into 'nexus-metatron':

1) Copy `packages/nexus_common` into repo `packages/`.
2) Copy `services/*/mml_agent.py` into each service dir, then edit each `app.py` to mount:
    from .mml_agent import router as mml_router
    app.include_router(mml_router)
3) Copy `services/qdrant-mml/**` into repo; merge `infra/compose.yaml.patch`.
4) Add to `nexus-core/routes.yaml` (optional external access):
    - prefix: /api/qdrant-mml
      upstream: http://qdrant-mml:8007
      strip_prefix: true
5) Ensure all services have `QDRANT_URL=http://qdrant:6333` in compose.
6) Rebuild:
    docker compose -f infra/compose.yaml up -d --build

Smoke:
- POST http://localhost:1313/api/memory/mml/observe  {"line":"memory started"}
- POST http://localhost:1313/api/edge/mml/observe    {"line":"edge firewall engaged"}
- GET  http://localhost:1313/api/qdrant-mml/health
