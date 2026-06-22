#!/bin/bash
set -euo pipefail
ENV=${1:-production}
echo "🚀 Deploying NEXUS v6.0 to ${ENV}..."
wrangler deploy --env "$ENV"
echo "✅ Deployed to ${ENV}!"
