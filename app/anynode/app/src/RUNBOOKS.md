# Runbooks
## Bring‑up
1) Copy .env.example → .env; set CONSENT_TOKEN
2) ./scripts/deploy.ps1
3) Check Grafana dashboard (Viren Receipts)

## Health Checks
- Viren /health → { ok: true }
- Loki /ready → 200
- Promtail accepts pushes on :3500
- Grafana :3000 up

## Incident Playbook
- Denies spike: Explore query `{app="viren"} | json | decision="deny"`; inspect reason; adjust policies.yaml
- Redis down: receipts still in stdout; restart Redis
- NATS down: HTTP path works; restart NATS; watch Viren logs
- Loki/Promtail down: restart promtail, then loki

## Backup/Restore
- Loki volume loki-data
- Grafana volume grafana-storage
- Redis receipts export: LRANGE viren:receipts 0 -1
