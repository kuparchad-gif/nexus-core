# Browser Receiver + Console (Nexus)

This is an expanded version of the Browser Receiver that now includes a **web console ("shell")**
to interact with **Viren** and NATS from your browser.

**Service:** `browser-receiver` on **:8818**

## Capabilities
- POST `/contact` → publish to `mesh.think.chat.request` (Nexus Envelope v1); receive deltas via `/stream/{req_id}` using SSE.
- **Console ("shell")** at `/console`:
  - **Viren panel**: `/api/viren/alive`, `/api/viren/authorize` proxy calls.
  - **NATS panel**: publish via `/api/nats/publish`; subscribe via WebSocket `/ws/nats?subject=<pattern>`.
- **Admin token** for console/API: header `X-Admin-Token: <token>` (set `ADMIN_TOKEN` in env).

## Files
```
.\compose.browser-receiver.yaml
.\wired\servicesrowser_receiver.env.example
.\cognikubes
exus_core\servicesrowser_receiver\Dockerfile
.\cognikubes
exus_core\servicesrowser_receiverequirements.txt
.\cognikubes
exus_core\servicesrowser_receiver\Start-BrowserReceiver.ps1
.\cognikubes
exus_core\servicesrowser_receiverpp\main.py
.\cognikubes
exus_core\servicesrowser_receiverpp
ats_client.py
.\cognikubes
exus_core\servicesrowser_receiverpp
exus_envelope.py
.\cognikubes
exus_core\servicesrowser_receiverpp\static\index.html
.\cognikubes
exus_core\servicesrowser_receiverpp\staticpp.js
.\cognikubes
exus_core\servicesrowser_receiverpp\static\console.html
.\cognikubes
exus_core\servicesrowser_receiverpp\static\console.js
```

## Quick Start
1) Unzip into `C:\Projects\Stacks\nexus-metatron\`
2) Copy & edit env:
```powershell
Copy-Item .\wired\services\browser_receiver.env.example .\wired\services\browser_receiver.env
notepad .\wired\services\browser_receiver.env
```
3) Run:
```powershell
.\cognikubes
exus_core\servicesrowser_receiver\Start-BrowserReceiver.ps1
```
4) Use:
- Public form: http://localhost:8818/
- Console: http://localhost:8818/console  (your browser will prompt for the **Admin Token**)

## Hosted vs Local
- **Local-only** (default): safer attack surface; ideal during build-out.
- **Hosted (public)**: front with Nginx/Cloudflare, enable TLS, rate limit, and require Viren consent + `ADMIN_TOKEN` for console.
