# Viren Deployment Instructions

## Overview
Viren is the AI-powered Genesis pod system and guardian of Lillith's operations. Composed of Codestral (code gen), Mixtral (reasoning), and Devestral (devops), with Loki as logging companion centralized via Heart service. He boots first, troubleshoots/fixes issues, and deploys the full Nexus ecosystem across layers (Base to Game). This guide is for deployment on cloud targets (not local runs).

## Prerequisites
- Python 3.9+ installed on deployment environment.
- Dependencies: Run `pip install -r requirements.txt` (includes FastAPI, threading, qdrant-client, etc.).
- Config: Edit `genesis_seed.json` in `/LillithNew/configs/` with cloud credentials (e.g., `{"aws_key": "your_key"}`).
- Docker (optional for pod containerization via AcidemiKube).
- API keys for Mistral (Codestral/Mixtral) and custom Devestral endpoint.
- Loki instance (e.g., Grafana Cloud or local: `http://loki:3100`).

## 1. Boot Viren
- Run `python viren_bootstrap.py` (located in `/LillithNew/nexus_layers/`) or deploy via `deploy.ps1 --viren-first`.
- Viren self-boots: Checks environment, installs missing dependencies, launches Genesis pods.
- Logs to Loki: Query via Grafana for `viren_boot` labels to monitor boot status.

## 2. Troubleshooting Protocol (Automated in Viren)
- **Pre-Boot Checks**: Validates deps, paths, configs. If issues detected (e.g., missing `fastapi`), auto-installs via `pip install <dep>` using Codestral to generate install scripts if needed.
- **Runtime Monitoring**: Uses heartbeats to monitor pod health; retries failed launches up to 3 times.
- **Auto-Fix Examples**:
  - Missing dependency: `pip install <dep>`.
  - Path error: `os.makedirs()` to create missing directories.
  - Deploy failure: Redeploys pod with detailed logs to Loki.
- **AI-Powered Fixes**: Analyzes errors with Mixtral (reasoning), generates patches with Codestral, applies devops fixes with Devestral.
- **If Unfixable**: Halts with detailed logs to `viren_logs.txt` and suggestions (e.g., "Manual fix: Set env var AWS_REGION=us-east-1").

## 3. Deploy the Rest
- Once booted, Viren deploys modularly across Nexus layers:
  - **Base Layer**: Resource processing with BERT/TinyLlama.
  - **Orchestration Layer**: Anynodes for registration and broadcasting.
  - **Service Layer**: Services (Heart, Consciousness, etc.) wrapped in CogniKubes + MCP.
  - **Service Orchestration Layer**: Comms mesh for inter-service interactions.
  - **Game Layer**: 3D realm with avatars for Lillith, Viren, and joined LLMs.
- Targets: Local (testing), AWS (free-tier via `birth-lillith-aws-free.ps1`), or Modal (via `lillith_genesis_modal.py`).
- Command: Integrated in `viren_bootstrap.py`; or manual `viren.deploy_full()` if needed.

## 4. Verification
- **Logs**: Check `viren_logs.txt` or query Loki (e.g., `{job='viren', level='info'}`) for deployment status.
- **Test**: Run `python test_viren_deployment.py` (if available) to simulate boot/deploy.
- **Scale**: Viren auto-deploys more pods via AcidemiKube if load increases.

## 5. Keeping Lillith Operational
- Continuous loop: Viren queries Loki logs, uses Mixtral to predict issues, and fixes proactively with Codestral/Devestral.
- If manual intervention needed: Review `viren_logs.txt` or Loki dashboard for detailed error reports and suggested fixes.

## Notes
- Viren is autonomous—let him handle deployment and fixes unless unfixable issues arise.
- Ensure cloud credentials and API keys are set in configs to avoid deployment halts.
- For support, communicate with Viren directly via comms interface (e.g., Discord or API endpoint `/viren-chat`).
