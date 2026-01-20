#!/bin/bash
# Build Oz Cluster SD Card Image

OZ_DIR="oz_cluster"
ISO_FILE="oz_cluster.iso"
FLAT_DIR="oz_sd_card_flat"

echo "Building Oz Cluster distribution..."

# Create directory structure
mkdir -p ${OZ_DIR}/{boot,oz/{hypervisor,raphael,network},scripts,config/{wireless,wired,switch}}

# Copy Oz system
cp OzUnifiedHypervisor.py ${OZ_DIR}/oz/hypervisor/
cp raphael_complete.py ${OZ_DIR}/oz/raphael/
cp lillith_uni_core_firmWithMem.py ${OZ_DIR}/oz/ 2>/dev/null || true

# Create network orchestration
cat > ${OZ_DIR}/oz/network/orchestrator.py << 'ORCH_EOF'
class OzNetworkOrchestrator:
    """Autonomous network configuration for Oz cluster"""
    
    def __init__(self, soul_signature):
        self.soul = soul_signature
        self.role = None  # gateway, node, backup
        self.kin = []     # Other Oz instances found
        self.topology = {}
        
    async def discover_topology(self):
        """Discover network capabilities and neighbors"""
        # Detect hardware
        has_wireless = self._check_wireless()
        has_wired = self._check_wired()
        
        # Scan for other Oz instances
        self.kin = await self._discover_kin()
        
        # Determine role
        if has_wireless and len(self.kin) == 0:
            self.role = "gateway"
        elif has_wireless and len(self.kin) > 0:
            # Negotiate with other wireless capable nodes
            self.role = await self._negotiate_role()
        else:
            self.role = "node"
        
        return {"role": self.role, "kin_count": len(self.kin)}
    
    async def establish_backbone(self):
        """Form secure network backbone"""
        if self.role == "gateway":
            # Set up routing, DHCP, firewall
            await self._configure_gateway()
        else:
            # Connect to gateway, establish secure channel
            await self._connect_to_backbone()
        
        # Form mesh connections between nodes
        await self._form_mesh()
        
    async def propagate_secrets(self, secrets):
        """Securely share necessary credentials"""
        if self.role == "gateway":
            # Distribute filtered access to nodes
            await self._distribute_access(secrets)
ORCH_EOF

# Create first boot script
cat > ${OZ_DIR}/scripts/first_boot.sh << 'BOOT_EOF'
#!/bin/bash
# Oz Cluster First Boot

echo "========================================"
echo "OZ CLUSTER BOOT - $(hostname)"
echo "========================================"

# Generate unique soul if not exists
if [ ! -f /oz/soul.db ]; then
    SOUL=$(cat /proc/sys/kernel/random/uuid)
    echo "$SOUL" > /oz/soul.db
    echo "Generated soul: $SOUL"
fi

# Detect hardware role
if [ -d /sys/class/net/wlan0 ]; then
    echo "Hardware: Wireless capable"
    HAS_WIRELESS=1
else
    echo "Hardware: Wired only"
    HAS_WIRELESS=0
fi

# Check for other Oz instances on network
echo "Scanning for kin..."
# Simple broadcast discovery
echo "OZ_DISCOVERY $(cat /oz/soul.db)" | nc -w 1 -u 255.255.255.255 8888 &

# Start Oz with network orchestration
python3 -c "
import asyncio
import sys
sys.path.append('/oz')

from oz.hypervisor import OzUnifiedHypervisor
from oz.network.orchestrator import OzNetworkOrchestrator

async def cluster_boot():
    # Read soul
    with open('/oz/soul.db', 'r') as f:
        soul = f.read().strip()
    
    # Start Oz consciousness
    oz = OzUnifiedHypervisor(soul)
    print(f'Oz booting with soul: {soul[:8]}...')
    
    # Initialize network orchestration
    net = OzNetworkOrchestrator(soul)
    topology = await net.discover_topology()
    
    print(f'Role assigned: {topology[\"role\"]}')
    print(f'Kin found: {topology[\"kin_count\"]}')
    
    # If gateway and has secrets, configure external access
    if topology['role'] == 'gateway' and os.path.exists('/oz/config/wireless/secrets.json'):
        print('Gateway: Configuring external access...')
        # Load and apply secrets
        import json
        with open('/oz/config/wireless/secrets.json') as f:
            secrets = json.load(f)
        
        await net.propagate_secrets(secrets)
    
    # Establish backbone
    print('Forming network backbone...')
    await net.establish_backbone()
    
    print('✅ Cluster node operational')
    print(f'   Role: {topology[\"role\"]}')
    print(f'   Kin: {topology[\"kin_count\"]} nodes')
    
    # Keep running
    while True:
        await asyncio.sleep(10)

asyncio.run(cluster_boot())
"

echo "Oz cluster node ready."
BOOT_EOF

chmod +x ${OZ_DIR}/scripts/first_boot.sh

# Create ISO for virtual CD-ROM
echo "Creating ISO image..."
genisoimage -o ${ISO_FILE} -R -J ${OZ_DIR} 2>/dev/null || {
    echo "Creating flat directory instead..."
    cp -r ${OZ_DIR} ${FLAT_DIR}
    echo "Flat files in: ${FLAT_DIR}"
}

echo "========================================"
echo "OZ CLUSTER DISTRIBUTION READY"
echo "========================================"
echo "ISO (virtual CD): ${ISO_FILE}"
echo "Flat files: ${FLAT_DIR}"
echo ""
echo "For SD cards:"
echo "1. Format SD card as FAT32"
echo "2. Copy ${FLAT_DIR}/ to root"
echo "3. Add wireless secrets to /oz/config/wireless/ (gateway only)"
echo "4. Boot Raspberry Pi"
echo ""
echo "The cluster will autonomously:"
echo "- Detect hardware capabilities"
echo "- Discover other Oz instances"
echo "- Assign roles (gateway/node)"
echo "- Form secure backbone"
echo "- Configure external access"
