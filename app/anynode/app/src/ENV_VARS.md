# Environment Variable Registry

| Var                     | Default                                        | Notes |
|-------------------------|------------------------------------------------|-------|
| VIREN_NAME              | viren                                          | Service instance label |
| NATS_URL                | nats://Nexus_nats-1:4222                       | NATS endpoint (JetStream enabled) |
| REDIS_URL               | redis://neexus_redis_1:6379/0                  | Redis for receipts |
| CONSENT_TOKEN           | CHANGE_ME                                      | Required for destructive actions; set strong value |
| ASCENSION_HOURS_REQ     | 480                                            | Meditation minutes threshold |
| ASCENSION_METAPHORS_REQ | 12                                             | Metaphor count threshold |
| VIREN_LOG_LEVEL         | INFO                                           | Logging level |
| PROMTAIL_PUSH_URL       | http://promtail:3500/loki/api/v1/push          | Promtail push receiver (internal URL) |
| GRAFANA_ADMIN_USER      | admin                                          | Grafana admin user |
| GRAFANA_ADMIN_PASSWORD  | admin                                          | Grafana admin password |
