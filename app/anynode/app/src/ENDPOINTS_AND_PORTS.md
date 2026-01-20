# Endpoints & Ports

## HTTP Endpoints
### Viren (http://localhost:8715)
- GET `/health` → `{ ok: true, name: "viren" }`
- GET `/alive`  → `{ alive: true }`
- POST `/authorize` → body: JSON request, returns `{decision, reason, receipt_id}`

### Loki (http://localhost:3100)
- GET `/ready` → 200 when ready

### Promtail Push API (via Promtail @ :3500 → forwards to Loki)
- POST `http://localhost:3500/loki/api/v1/push` → `{streams:[{stream:{labels},values:[[ns_ts, log]]}]}`

### Grafana (http://localhost:3000)
- UI login (admin/admin unless overridden)

## Messaging
- NATS subject (subscribe/req-reply): `viren.authorize.*`

## Ports (host → container)
| Service  | Host Port | Container Port |
|----------|-----------|----------------|
| viren    | 8715      | 8715           |
| loki     | 3100      | 3100           |
| promtail | 3500      | 3500           |
| grafana  | 3000      | 3000           |
| nats*    | 4222/8222 | 4222/8222      |
| redis*   | 6379      | 6379           |
