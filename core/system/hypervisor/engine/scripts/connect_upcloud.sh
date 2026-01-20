#!/bin/bash
# discover_upcloud.sh
# Chad Kupar - OZ Hypervisor Discovery
# Targets: 85.9.217.125 (UpCloud Gateway)

set -e

IP="85.9.217.125"
USER="root"
LOG_FILE="discovery_$(date +%Y%m%d_%H%M%S).log"

echo "[OZ DISCOVERY INITIATED]" | tee -a "$LOG_FILE"
echo "Target: $USER@$IP" | tee -a "$LOG_FILE"
echo "Timestamp: $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# 1. Attempt SSH connection with basic discovery
echo "[1] Testing SSH connectivity..." | tee -a "$LOG_FILE"
if ! ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no "$USER@$IP" "echo 'SSH connection successful'; exit 0"; then
    echo "SSH connection FAILED." | tee -a "$LOG_FILE"
    echo "Check: " | tee -a "$LOG_FILE"
    echo "  - SSH key loaded in agent (ssh-add -l)" | tee -a "$LOG_FILE"
    echo "  - UpCloud firewall allows port 22" | tee -a "$LOG_FILE"
    echo "  - Server is running (ping -c 2 $IP)" | tee -a "$LOG_FILE"
    exit 1
fi

echo "" | tee -a "$LOG_FILE"
echo "[2] Executing remote discovery..." | tee -a "$LOG_FILE"

# 2. Remote discovery commands
ssh -o StrictHostKeyChecking=no "$USER@$IP" /bin/bash << 'EOF' | tee -a "$LOG_FILE"

echo "=== SYSTEM IDENTITY ==="
hostname
uname -a
cat /etc/os-release 2>/dev/null | head -5

echo ""
echo "=== FILE SYSTEM LAYOUT ==="
ls -la / | head -20

echo ""
echo "=== OZ HYPERVISOR PRESENCE ==="
if [ -d "/oz" ]; then
    echo "/oz/ directory EXISTS"
    ls -la /oz/
    echo ""
    echo "--- Config Files ---"
    find /oz -name "*.json" -o -name "*.yml" -o -name "*.yaml" -o -name "*.conf" 2>/dev/null | xargs -I {} sh -c 'echo "File: {}"; cat {} 2>/dev/null | head -10; echo ""'
else
    echo "No /oz/ directory found."
fi

echo ""
echo "=== RUNNING PROCESSES (OZ/Python) ==="
ps aux | grep -E "(hypervisor|oz|mesh|wireguard|python)" | grep -v grep || echo "No relevant processes found."

echo ""
echo "=== NETWORK CONFIGURATION ==="
ip addr show | grep -E "(inet|ether)" | grep -v "127.0.0.1" | head -10

echo ""
echo "=== LISTENING PORTS ==="
netstat -tulpn 2>/dev/null | grep -E "(LISTEN|^Proto)" | head -20

echo ""
echo "=== RECENT LOGS (OZ/System) ==="
find /var/log -name "*.log" -type f 2>/dev/null | xargs tail -n3 2>/dev/null | head -50

echo ""
echo "=== USER ACCOUNTS ==="
grep -E "(oz|root|ubuntu|admin)" /etc/passwd | head -10

echo ""
echo "=== CRON JOBS ==="
crontab -l 2>/dev/null || echo "No user cron."

echo ""
echo "=== DISK USAGE ==="
df -h | head -10

EOF

echo "" | tee -a "$LOG_FILE"
echo "[3] Local verification..." | tee -a "$LOG_FILE"
echo "Ping test:" | tee -a "$LOG_FILE"
ping -c 2 "$IP" 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "[OZ DISCOVERY COMPLETE]" | tee -a "$LOG_FILE"
echo "Full log saved to: $LOG_FILE" | tee -a "$LOG_FILE"
