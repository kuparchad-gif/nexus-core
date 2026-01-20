# Inventory (Live & Static)

## Expected containers
| Name            | Purpose                       | Image                         | Ports    |
|-----------------|-------------------------------|-------------------------------|----------|
| viren           | consent/safety API + NATS wrk | built from services/viren     | 8715/tcp |
| loki            | log database                  | grafana/loki:2.9.6            | 3100/tcp |
| promtail        | log shipper (Push API)        | grafana/promtail:2.9.6        | 3500/tcp |
| grafana         | dashboards                    | grafana/grafana:10.4.5        | 3000/tcp |
| Nexus_nats-1*   | NATS bus (external)           | nats:2.10 (JetStream)         | 4222     |
| neexus_redis_1* | Redis (external)              | redis:7                       | 6379     |

`*` optional but expected to exist in your environment.

## Live inventory script
Run:
```powershell
./scripts/inventory.ps1
```
It prints container states, mapped ports, and endpoint health results.
