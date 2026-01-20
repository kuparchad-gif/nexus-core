# Nexus Overlay Pack — 2025-09-02

**Install path:** `C:\Projects\Stacks\nexus-metatron` (run commands from this folder)

## What this adds
- `podman-compose.yaml` and `docker-compose.yaml` with three planes:
  - **base**: NATS (JetStream), Qdrant, Redis, Loki
  - **firmware**: UCE Router (FastAPI) + 7B drafter (llama.cpp server)
  - **inference**: 14B general model (llama.cpp server)
  - **service**: 35B/70B heavy runners (disabled by default)
- `scripts/` PowerShell helpers to bring it up/down and switch Podman machines.
- `bench/` harness to run golden prompts and compute speedup.
- `config/` minimal Loki config; `models/` mount point for GGUF files.

## Quick start (Podman preferred)
```powershell
cd C:\Projects\Stacks\nexus-metatron
# Copy this overlay over your tree, then:
.\scripts\Switch-PodmanMachine.ps1 -Name mgmt
.\scripts\Up-Nexus.ps1 -Profiles base,firmware,inference
.\scripts\Health-Check.ps1
```

Drop models into `models\`:
- `7b.gguf` for the drafter
- `14b.gguf` for the general inference

Then test:
```powershell
python .\bench\bench.py --url http://localhost:8007 --model /models/14b.gguf --out .\bench\baseline.json
# Make your optimizations → run again to produce chad.json
python .\bench\bench.py --url http://localhost:8007 --model /models/14b.gguf --out .\bench\chad.json
python .\bench\diff_report.py .\bench\baseline.json .\bench\chad.json
```

## Notes
- Compose **profiles** let you control which planes run: `base`, `firmware`, `inference`, `service`.
- Llama servers are **CPU-only** (set `-ngl 0`). Replace models/paths as needed.
- UCE Router chooses drafter vs inference by a simple context heuristic and exposes `/metrics`.

## References
- Hermes OS / gateway & lanes: see **Aethereal Hermes OS — Architecture Whitepaper (v1)**. 
- NIM lattice codec & Cerberus policy: see **NIM Lattice Codec — Whitepaper (Nexus v0.4)**.
- Hermes Firmware boot/seed/runbook: see **Hermes Firmware Process — Whitepaper (Nexus v0.4)**.
