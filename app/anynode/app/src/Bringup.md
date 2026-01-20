# Runbook — Bring-up

1) Copy `.env.example` → `.env` and set a strong `CONSENT_TOKEN`.
2) Start services:
   ```powershell
   ./scripts/deploy.ps1
   ```
3) Verify health:
   ```powershell
   iwr http://localhost:8715/health | % Content
   iwr http://localhost:3100/ready | % StatusCode
   Start-Process http://localhost:3000
   ```
4) Send a test authorization request; confirm Grafana logs.
