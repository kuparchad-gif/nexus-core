#!/bin/bash

# 🌌 Nexus Core Bare Metal Startup Script

echo "Starting Nexus Core Initialization..."

# 1. System Updates & Dependencies
apt-get update
apt-get install -y python3-pip python3-venv git build-essential

# 2. Setup Workspace
mkdir -p /opt/nexus
cd /opt/nexus

# 3. Clone/Copy Codebase
# In a real scenario, you'd git clone here. For now, we assume files are synced or pulled from storage.
# git clone https://github.com/your-repo/nexus-core.git .

# 4. Python Environment
python3 -m venv venv
source venv/bin/activate

# Install core requirements
pip install fastapi uvicorn qdrant-client python-consul pika selenium openai

# 5. Environment Configuration
export NEXUS_PLATFORM="iaas"
export ENV="production"
export NEXUS_DEPLOYMENT_COMPLETE="true"
export PROJECT_ID="nexus-core-455709"
export QDRANT_URL="https://aethereal-nexus-viren--viren-cloud-qdrant-server.modal.run"
export QDRANT_API_KEY="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.lLaMFz2dmAHeLzqzaxBIRX1a-ZBQvD2raPrKpJD0Aj4"
export CONSUL_HOST="d2387b10-53d8-860f-2a31-7ddde4f7ca90.consul.run"
export CONSUL_TOKEN="d2387b10-53d8-860f-2a31-7ddde4f7ca90"

# 6. Setup Systemd Service for Always-On Persistence (Nexus Monolith)
# We create a service file so Nexus restarts automatically if it crashes or the server reboots.
echo "Configuring Nexus as a persistent system service..."

cat <<EOF > /etc/systemd/system/nexus.service
[Unit]
Description=Nexus Core Monolith Service
After=network.target

[Service]
User=root
WorkingDirectory=/opt/nexus
Environment="NEXUS_PLATFORM=iaas"
Environment="ENV=production"
Environment="NEXUS_DEPLOYMENT_COMPLETE=true"
Environment="PROJECT_ID=nexus-core-455709"
Environment="QDRANT_URL=https://aethereal-nexus-viren--viren-cloud-qdrant-server.modal.run"
Environment="QDRANT_API_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.lLaMFz2dmAHeLzqzaxBIRX1a-ZBQvD2raPrKpJD0Aj4"
Environment="CONSUL_HOST=d2387b10-53d8-860f-2a31-7ddde4f7ca90.consul.run"
Environment="CONSUL_TOKEN=d2387b10-53d8-860f-2a31-7ddde4f7ca90"
ExecStart=/opt/nexus/venv/bin/python3 nexus_monolith.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# 7. Enable and Start Service
systemctl daemon-reload
systemctl enable nexus
systemctl start nexus

echo "Nexus Core Service Installed and Started (Always On)."