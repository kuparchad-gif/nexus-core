# Env Probe + Sidecar + Archiver (Drop-in Pack)

## Layout
- `config/env.modes.json` — defaults for local/compose/k8s
- `scripts/env_probe.py` — detects env, writes `sidecar/.env.resolved`, **reports to Archiver**
- `sidecar/` — universal MCP sidecar (modern/legacy SDK auto-wire)
- `src/archiver/archiver_stub.py` — simple Archiver event sink on :9020

## Install
### Archiver (optional for smoke)
```
pip install fastapi uvicorn
python -m uvicorn src.archiver.archiver_stub:app --host 0.0.0.0 --port 9020
```

### Sidecar
```
cd sidecar
cp -n .env.local.example .env 2>/dev/null || copy .env.local.example .env
npm install
# Windows (Chrome path if needed):
# export PUPPETEER_EXECUTABLE_PATH="/c/Program Files/Google/Chrome/Application/chrome.exe"
npm run dev
# -> predev runs env_probe (reports to Archiver) then starts sidecar on :8088
```

### Health
```
curl -s http://localhost:8088/health | python -m json.tool
# See appended events:
# PowerShell:
type ..\..\archiver_events.jsonl
# Git Bash (from repo root):
tail -n 5 archiver_events.jsonl
```

## For Compose/K8s
Use:
```
ENV_MODE=compose python scripts/env_probe.py --out sidecar/.env.resolved --report
# or
ENV_MODE=k8s python scripts/env_probe.py --out sidecar/.env.resolved --report
```
Then `npm run dev` in `sidecar/`.
