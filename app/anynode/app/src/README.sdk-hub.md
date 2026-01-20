# Nexus SDK Hub — Cloud & Dev Tooling Pack

Single container that ships **all major SDKs/CLIs** for AWS, GCP, Azure, K8s, IaC, and Dev stacks,
with an **admin-gated API** so Viren (or you) can invoke checks and basic build steps safely.

**Service:** `sdk-hub` on **:8820** (configurable)

## Highlights
- Cloud SDKs: **AWS CLI v2**, **Google Cloud SDK**, **Azure CLI**
- K8s: **kubectl**, **helm**
- IaC: **terraform**, **terragrunt**, **packer**
- Dev: **Python 3.11 + pipx/poetry**, **Node.js LTS + pnpm**, **Go**, **Java 17 (JDK)**, **.NET 8 SDK**, **Rust (rustup)**
- Tooling: **git**, **git-lfs**, **gh**, **jq**, **yq**, **curl**, **unzip**, **zip**, **make**
- Messaging: **nats** CLI
- MLOps: **huggingface-cli**
- Stripe CLI present via dedicated service (already in your stack)

### Endpoints
- `GET /alive` — health
- `GET /versions` — prints versions of all SDKs
- `POST /exec` — run **allowlisted** commands only (admin token required)
- Optional NATS worker (subscribe `svc.build.request.>`) can be added later

### Files
```
.\compose.sdk-hub.yaml
.\wired\services\sdk_hub.env.example
.\cognikubes
exus_core\services\sdk_hub\Dockerfile
.\cognikubes
exus_core\services\sdk_hub\Start-SDKHub.ps1
.\cognikubes
exus_core\services\sdk_hubpp\main.py
.\modal\sdk_hub_modal.py
```
