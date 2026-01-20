# Runbook — Incident Playbook

## Spike in denies
- Query Grafana: `{app="viren"} | json | decision="deny"`
- Sample receipts from Redis:
  ```powershell
  podman exec -it neexus_redis_1 redis-cli LRANGE viren:receipts 0 20
  ```
- Check `policies.yaml` for thresholds; adjust if appropriate; rebuild or restart viren.

## Redis down
- Viren logs will warn, but receipts still go to stdout/Promtail. Restart Redis:
  ```powershell
  podman restart neexus_redis_1
  ```

## NATS disconnected
- Viren will continue serving HTTP; NATS worker logs a warning. Verify NATS:
  ```powershell
  podman logs Nexus_nats-1 --tail 200
  ```

## Loki or Promtail down
- Check Promtail → Loki push path; restart services:
  ```powershell
  podman restart promtail; podman restart loki
  ```

## Grafana login issues
- Ensure `.env` credentials; if needed, reset container:
  ```powershell
  podman rm -f grafana; ./scripts/deploy.ps1 -Rebuild
  ```
