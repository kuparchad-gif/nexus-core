# SLOs & Alerts (draft)
- Availability: ≥ 99.5% Viren HTTP /health over 7d
- Latency: p95 ≤ 150ms for /authorize (non‑destructive)
- Error budget: alert if denies/allow ratio > 0.35 over 15m
- Loki ingestion: alert if no {app="viren"} logs in 5m
