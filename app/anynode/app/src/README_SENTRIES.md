# Sentries Pack (v3)
Adds three per-village microservices:
- **module-scribe** (8870): captures boot/env metadata; optionally tails `/var/log/module` and pushes to Loki; always publishes metadata envelopes to **Memory Gateway**.
- **coldstore-agent** (8875): scans `/data/cold` for non-critical files (older than `MIN_AGE_MINUTES`), tags with **Dewey-like** codes, ships to **Memory Gateway** as blobs. Sends a metadata envelope first.
- **viren-desk** (8880): opens/escalates support calls for **Viren**; publishes to NATS (`viren.support`, `viren.spawn`) and records to Redis.

All archival paths respect the **Memory-first** mandate (`memory-gateway` in-between).

## Usage
```powershell
# From: CogniKube-Template\
.\scripts\Enable-Sentries.ps1 -ModuleName "language" -WatchLogs 1
```
This will tail `/var/log/module` and treat the scribe as the **Language** module sidecar.

> Tip: mount a module-specific cold data volume to `/data/cold` for coldstore-agent to see it.
