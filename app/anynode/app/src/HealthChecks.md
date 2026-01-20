# Runbook — Health Checks

- Viren: `GET /health` should return `{ ok: true }`.
- Loki: `GET /ready` → 200.
- Promtail: accepts pushes on :3500 and forwards to Loki.
- Grafana: reachable at :3000, dashboard loads.

Scripts:
```powershell
./scripts/health.ps1
```
