# Nexus Sprint Pack — Ops Cheatsheet

## Bring-up / Rebuild / Tear-down
```powershell
# Bring up (prefers podman-compose, falls back to python -m podman_compose)
./scripts/deploy.ps1

# Force rebuild images
./scripts/deploy.ps1 -Rebuild

# Down (if you need it)
$Compose = (Get-Command podman-compose -ErrorAction SilentlyContinue) ? 'podman-compose' : 'python -m podman_compose'
& $Compose -f podman-compose.yaml down
```

## Logs (tail)
```powershell
podman logs -f viren
podman logs -f loki
podman logs -f promtail
podman logs -f grafana
```

## Health checks
```powershell
iwr http://localhost:8715/health | % Content
iwr http://localhost:3100/ready | % StatusCode
iwr http://localhost:3000/login -UseBasicParsing | % StatusCode
```

## Test Viren (HTTP)
```powershell
$Body = @{
  action = "purchase"
  subject = "credits"
  amount = 125
  consent_token = $env:CONSENT_TOKEN
  stats = @{ meditation_minutes = 600; metaphor_count = 15 }
} | ConvertTo-Json

irm "http://localhost:8715/authorize" -Method POST -ContentType "application/json" -Body $Body
```

## Test Viren (NATS)
```powershell
nats req viren.authorize.test '{"action":"purchase","amount":50}'
```

## Redis receipts (optional)
```powershell
podman exec -it neexus_redis_1 redis-cli LLEN viren:receipts
podman exec -it neexus_redis_1 redis-cli LINDEX viren:receipts 0
```

## Grafana
- URL: http://localhost:3000
- Default user/pass: admin / admin (change in `.env` as GRAFANA_ADMIN_*)
- Dashboard: **Viren Receipts**

## Compose service names and containers
- viren → FastAPI safety/consent gate (8715)
- loki  → log store (3100)
- promtail → push receiver (3500)
- grafana → dashboards (3000)

## One-liner to add NATS/Redis (if missing)
```powershell
podman run -d --name Nexus_nats-1 -p 4222:4222 -p 8222:8222 nats:2.10 -js
podman run -d --name neexus_redis_1 -p 6379:6379 redis:7
```
