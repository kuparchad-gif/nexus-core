# Organizer Bots

Two microcells to **survey → plan → rename → rewire** your cosmos.

## Run (compose overlay)
```bash
docker compose -f compose.cosmos.organizer.yaml up -d --build
```

## Flow
1. **Survey** (get a plan):
```bash
curl -s http://localhost:9061/survey?colony=alpha | jq > plan.json
```
2. **Apply** (rename + wire + start missing):
```bash
curl -s -X POST http://localhost:9062/apply -H 'Content-Type: application/json' --data-binary @plan.json | jq
```

- Surveyor groups containers by capability → role and proposes names like `viren-alpha-01`.
- Organizer stops/recreates containers with `MICROCELL_NAME=<desired>`, sets labels `{role,colony}`, then calls:
  - **WireCrawler** `/wire` to build caps→urls,
  - **StarterCrawler** `/start_missing` to trigger start hooks,
  - **Archiver** `/archive` to record every step.
