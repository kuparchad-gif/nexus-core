
#!/usr/bin/env bash
set -euo pipefail

SRC_ROOT="${SRC_ROOT:-/src}"
DEST_ROOT="${DEST_ROOT:-/out}"
PROVIDER="${PROVIDER:-lmstudio}"
LMSTUDIO_URL="${LMSTUDIO_URL:-http://192.168.0.162:1234/v1/embeddings}"
LMSTUDIO_MODEL="${LMSTUDIO_MODEL:-qwen/qwen3-4b-thinking-2507}"
SUPPORT_MODE="${SUPPORT_MODE:-tree}"
DRY_RUN="${DRY_RUN:-false}"
CONFIDENCE_MIN="${CONFIDENCE_MIN:-0.4}"
ALLOW_SAME_PATH="${ALLOW_SAME_PATH:-false}"

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

export PROVIDER="$PROVIDER"
export LMSTUDIO_URL="$LMSTUDIO_URL"
export LMSTUDIO_MODEL="$LMSTUDIO_MODEL"
export SRC_ROOT="$SRC_ROOT"
export DEST_ROOT="$DEST_ROOT"
export DO_METADATA="true"
export SUPPORT_MODE="$SUPPORT_MODE"
export CONFIDENCE_MIN="$CONFIDENCE_MIN"
export ALLOW_SAME_PATH="$ALLOW_SAME_PATH"
export DRY_RUN="$DRY_RUN"

python ./apps/main.py
