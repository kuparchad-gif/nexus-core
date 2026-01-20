# Runbook — Backup & Restore

## Loki (boltdb-shipper on filesystem)
- Backup volume `loki-data` or container path `/loki`.
- Restore by reattaching the volume to a new Loki container.

## Grafana
- Dashboards are provisioned at startup; for dynamic assets, backup `grafana-storage` volume.

## Viren receipts
- Redis dump or export list:
  ```powershell
  podman exec -it neexus_redis_1 redis-cli SAVE
  podman exec -it neexus_redis_1 redis-cli LRANGE viren:receipts 0 -1 > receipts.jsonl
  ```
