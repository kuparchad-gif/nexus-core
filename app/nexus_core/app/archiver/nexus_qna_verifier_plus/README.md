# Nexus Q&A Scraper — PLUS (Slack, GitHub, Reddit, StackOverflow, Discourse) + MNLI Verifier

**Connectors**: Discord, IRC, Slack (Socket Mode), GitHub (webhook + poller), Reddit, Stack Overflow, Discourse.  
**Verifier**: MNLI (roberta-large-mnli by default) + web evidence via Tavily/Brave/Bing.  
**Store**: Qdrant (vectors + payload).

## Start
```bash
cp .env.example .env
docker compose up -d         # qdrant + api + verifier
# Then in separate shells, start any connectors you want, e.g.
python connectors/discord/bot.py
python connectors/slack/socket_mode.py
python connectors/github/poller.py
python connectors/reddit/stream.py
python connectors/stackoverflow/poller.py
python connectors/discourse/poller.py
```
GitHub webhook (recommended): point to `http://<host>:8080/webhooks/github` and set `GITHUB_WEBHOOK_SECRET`.

## Query
```bash
curl -s http://localhost:8080/query -H 'content-type: application/json' -d '{"q":"How to rotate Nginx logs?"}'
```

## Tuning
- `.env` → `VERIFY_MIN_SOURCES`, `VERIFY_MIN_COVERAGE`, `VERIFY_MAX_AGE_DAYS`, `DOMAIN_ALLOWLIST`  
- `VERIFIER_BACKEND=mnli|stub` (use `stub` on tiny dev boxes)  
- Swap MNLI model with `MNLI_MODEL_NAME` if you need lighter weights.
