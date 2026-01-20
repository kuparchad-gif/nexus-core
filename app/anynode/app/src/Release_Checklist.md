# Runbook — Release Checklist

- [ ] Update `services/viren/requirements.txt` if deps change
- [ ] Edit `policies.yaml` (review thresholds)
- [ ] `./scripts/deploy.ps1 -Rebuild`
- [ ] Verify `GET /health`, Loki `/ready`, Grafana dashboard
- [ ] Smoke test `/authorize` (allow + deny paths)
- [ ] Commit/Tag: `sprint-pack-v1.X`
