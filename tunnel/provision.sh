#!/bin/bash
# ============================================================================
# CLOUDFLARE TUNNEL PROVISIONING SCRIPT
# ============================================================================

set -euo pipefail

log() { echo "[TUNNEL] $1"; }
ok() { echo "[TUNNEL] ✅ $1"; }
err() { echo "[TUNNEL] ❌ $1"; }

log "Provisioning Cloudflare Tunnel..."

# Create tunnel
if ! cloudflared tunnel create nexus-edge-bridge 2>/dev/null; then
    log "Tunnel already exists, reusing..."
fi

# Get tunnel ID
TUNNEL_ID=$(cloudflared tunnel list | grep nexus-edge-bridge | awk '{print $1}')

if [[ -z "$TUNNEL_ID" ]]; then
    err "Failed to get tunnel ID"
    exit 1
fi

ok "Tunnel ID: $TUNNEL_ID"

# Generate credentials
mkdir -p ~/.cloudflared
cloudflared tunnel token --cred-file ~/.cloudflared/nexus-edge-bridge.json nexus-edge-bridge

# Configure tunnel
cat > tunnel/tunnel-config.yml << EOF
tunnel: nexus-edge-bridge
credentials-file: ~/.cloudflared/nexus-edge-bridge.json

ingress:
  - hostname: bridge.nexus-universal.internal
    service: http://localhost:8080
  - service: http_status:404
EOF

ok "Tunnel configured"

# Route traffic
cloudflared tunnel route dns nexus-edge-bridge bridge.nexus-universal.internal || true
cloudflared tunnel route ip nexus-edge-bridge || true

ok "Tunnel routing configured"

# Output status
log ""
log "Tunnel Status:"
cloudflared tunnel list | grep nexus-edge-bridge
log ""
log "To run the tunnel:"
log "  cloudflared tunnel run nexus-edge-bridge"
