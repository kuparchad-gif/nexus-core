#!/bin/bash
set -e

echo "🍓 Creating Bootable Raspberry Pi Image for Valhalla..."

# --- Configuration ---
PI_OS_IMAGE="raspios_lite_arm64"
DOCKER_IMAGE_NAME="valhalla-pi-agent"
AGENT_CODE_DIR="core/system/hypervisor/engine"

# --- Phase 1: Create a staging directory ---
mkdir -p pi_build/
cp -r core pi_build/ # Copy the entire 'core' directory

# --- Phase 2: Create a Dockerfile for the build environment ---
cat > pi_build/Dockerfile << EOF
FROM balenalib/${PI_OS_IMAGE}

# Install dependencies
RUN install_packages python3 python3-pip rsync

# Copy the agent code
COPY core /opt/valhalla/core

# Install Python dependencies
RUN pip3 install qdrant-client numpy

# Create and enable the startup service
RUN echo "[Unit]
Description=Valhalla Hypervisor Agent
After=network.target

[Service]
Environment="PYTHONPATH=/opt/valhalla"
ExecStart=/usr/bin/python3 /opt/valhalla/core/system/hypervisor/engine/oz_cluster_main.py
Restart=always

[Install]
WantedBy=multi-user.target" > /etc/systemd/system/valhalla-agent.service

RUN systemctl enable valhalla-agent.service

CMD ["/bin/bash"]
EOF

# --- Phase 3: Build the Docker image ---
echo "🏗️ Building the Docker image..."
docker build -t ${DOCKER_IMAGE_NAME} pi_build/

# --- Phase 4: Export the root filesystem ---
echo "📦 Exporting the root filesystem..."
docker run --rm ${DOCKER_IMAGE_NAME} tar -czf - / > valhalla_pi_rootfs.tar.gz

# --- Phase 5: Cleanup ---
rm -rf pi_build/

echo "✅ Bootable Pi image created as valhalla_pi_rootfs.tar.gz"
echo "Next steps:"
echo "1. Download the latest Raspberry Pi OS Lite image."
echo "2. Flash the image to an SD card."
echo "3. Mount the SD card and replace the root filesystem with the contents of the tarball."
echo "4. Unmount, insert into the Pi, and boot."
