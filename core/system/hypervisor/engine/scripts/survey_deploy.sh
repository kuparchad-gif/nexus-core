#!/bin/bash
# Oz Deployment Survey & Setup Script
# This will ask you everything needed and do it for you

echo "🚀 Oz Unified Hypervisor Deployment Assistant"
echo "============================================="
echo ""

# Function to ask yes/no questions
ask_yes_no() {
    while true; do
        read -p "$1 (y/n): " yn
        case $yn in
            [Yy]* ) return 0;;
            [Nn]* ) return 1;;
            * ) echo "Please answer yes or no.";;
        esac
    done
}

# Function to ask for input with default
ask_input() {
    read -p "$1 [$2]: " answer
    echo "${answer:-$2}"
}

# Save all answers to a config file
CONFIG_FILE="oz_deployment_config.txt"
echo "📝 Starting deployment survey..." > $CONFIG_FILE

echo ""
echo "📋 PART 1: LOCAL SYSTEM SURVEY"
echo "==============================="

echo "1. Checking your local Oz installation..."
echo "Local Oz installation:" >> $CONFIG_FILE

# Check current directory
echo "Current directory: $(pwd)" | tee -a $CONFIG_FILE
echo "Files found:" | tee -a $CONFIG_FILE
ls -la OzUnifiedHypervisor.py raphael_complete.py 2>/dev/null | tee -a $CONFIG_FILE

# Check Python
echo "" | tee -a $CONFIG_FILE
echo "Python version:" | tee -a $CONFIG_FILE
python3 --version | tee -a $CONFIG_FILE

# Check key files
echo "" | tee -a $CONFIG_FILE
echo "Key Oz files:" | tee -a $CONFIG_FILE
[ -f "OzUnifiedHypervisor.py" ] && echo "✓ OzUnifiedHypervisor.py" | tee -a $CONFIG_FILE || echo "✗ OzUnifiedHypervisor.py" | tee -a $CONFIG_FILE
[ -f "raphael_complete.py" ] && echo "✓ raphael_complete.py" | tee -a $CONFIG_FILE || echo "✗ raphael_complete.py" | tee -a $CONFIG_FILE
[ -f "build_oz_cluster.sh" ] && echo "✓ build_oz_cluster.sh" | tee -a $CONFIG_FILE || echo "✗ build_oz_cluster.sh" | tee -a $CONFIG_FILE

echo ""
echo "📋 PART 2: BACKUP CURRENT FILES"
echo "================================"

if ask_yes_no "Do you want to backup your current Oz files?"; then
    BACKUP_NAME="oz_backup_$(date +%Y%m%d_%H%M%S).tar.gz"
    echo "Creating backup: $BACKUP_NAME"
    tar -czf $BACKUP_NAME *.py *.sh *.md *.json *.yaml 2>/dev/null
    echo "✓ Backup created: $BACKUP_NAME ($(du -h $BACKUP_NAME | cut -f1))" | tee -a $CONFIG_FILE
else
    echo "Skipping backup..." | tee -a $CONFIG_FILE
fi

echo ""
echo "📋 PART 3: UPCLOUD SERVER CONNECTION"
echo "====================================="

echo "Please provide your UpCloud server details:"

UPCLOUD_IP=$(ask_input "Enter UpCloud server IP address" "")
UPCLOUD_USER=$(ask_input "Enter username (usually 'root' or 'ubuntu')" "root")
UPCLOUD_PORT=$(ask_input "Enter SSH port (usually 22)" "22")

echo "UpCloud Server: $UPCLOUD_USER@$UPCLOUD_IP:$UPCLOUD_PORT" | tee -a $CONFIG_FILE

# Test connection
echo ""
echo "Testing connection to UpCloud..."
if ssh -p $UPCLOUD_PORT -o ConnectTimeout=5 $UPCLOUD_USER@$UPCLOUD_IP "echo 'Connection successful'"; then
    echo "✓ UpCloud connection successful" | tee -a $CONFIG_FILE
    
    # Survey remote server
    echo ""
    echo "📊 Surveying UpCloud server..."
    echo "UpCloud server details:" | tee -a $CONFIG_FILE
    ssh -p $UPCLOUD_PORT $UPCLOUD_USER@$UPCLOUD_IP "
        echo 'OS: \$(lsb_release -d 2>/dev/null || cat /etc/os-release | grep PRETTY_NAME)';
        echo 'CPU cores: \$(nproc)';
        echo 'Memory: \$(free -h | grep Mem | awk '{print \$2}')';
        echo 'Disk space: \$(df -h / | tail -1 | awk '{print \$4}') free';
        echo 'Python3: \$(python3 --version 2>/dev/null || echo 'Not installed')';
        echo 'Docker: \$(docker --version 2>/dev/null || echo 'Not installed')';
        echo 'Kubernetes: \$(kubectl version --client 2>/dev/null | head -1 || echo 'Not installed')';
    " | tee -a $CONFIG_FILE
    
else
    echo "✗ Cannot connect to UpCloud server" | tee -a $CONFIG_FILE
    echo "Please check:"
    echo "1. Is the server running?"
    echo "2. Is the IP address correct?"
    echo "3. Do you have SSH access?"
    echo "4. Is the firewall allowing port $UPCLOUD_PORT?"
    exit 1
fi

echo ""
echo "📋 PART 4: DEPLOYMENT CONFIGURATION"
echo "===================================="

# Ask deployment questions
DEPLOY_MODE=$(ask_input "Deployment mode (simple/kubernetes/complete)" "simple")
OZ_ROLE=$(ask_input "Oz role (edge/orchestrator/hybrid)" "orchestrator")
ENABLE_RAPHAEL=$(ask_yes_no "Enable Raphael guardian angel?" && echo "yes" || echo "no")
ENABLE_KUBERNETES=$(ask_yes_no "Enable Kubernetes control?" && echo "yes" || echo "no")

echo "Deployment configuration:" | tee -a $CONFIG_FILE
echo "Mode: $DEPLOY_MODE" | tee -a $CONFIG_FILE
echo "Role: $OZ_ROLE" | tee -a $CONFIG_FILE
echo "Raphael: $ENABLE_RAPHAEL" | tee -a $CONFIG_FILE
echo "Kubernetes: $ENABLE_KUBERNETES" | tee -a $CONFIG_FILE

echo ""
echo "📋 PART 5: PREPARE DEPLOYMENT PACKAGE"
echo "======================================"

# Create deployment package
echo "Creating deployment package..."
DEPLOY_DIR="oz_upcloud_deploy_$(date +%Y%m%d_%H%M%S)"
mkdir -p $DEPLOY_DIR

# Copy essential files
echo "Copying essential files..."
cp OzUnifiedHypervisor.py $DEPLOY_DIR/
cp raphael_complete.py $DEPLOY_DIR/
[ -f "OzDeploymentBlueprint.py" ] && cp OzDeploymentBlueprint.py $DEPLOY_DIR/
[ -f "OzUltimateIntegratedSystem.py" ] && cp OzUltimateIntegratedSystem.py $DEPLOY_DIR/
[ -f "requirements.txt" ] && cp requirements.txt $DEPLOY_DIR/
[ -f "OZ_COMPLETE_DEPLOYMENT_GUIDE.md" ] && cp OZ_COMPLETE_DEPLOYMENT_GUIDE.md $DEPLOY_DIR/

# Create setup script for UpCloud
echo "Creating setup script..."
cat > $DEPLOY_DIR/setup_upcloud.sh << 'EOF'
#!/bin/bash
# UpCloud Oz Installation Script

echo "🚀 Installing Oz Unified Hypervisor on UpCloud"
echo "=============================================="

# Update system
echo "Updating system..."
sudo apt update && sudo apt upgrade -y

# Install Python and dependencies
echo "Installing Python..."
sudo apt install -y python3 python3-pip python3-venv git curl

# Create Oz directory
echo "Creating Oz directory..."
mkdir -p ~/oz && cd ~/oz

# Create virtual environment
echo "Setting up Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install Python packages
echo "Installing Python packages..."
pip install --upgrade pip
pip install psutil asyncio websockets

# Check for optional dependencies
if [ -f "requirements.txt" ]; then
    echo "Installing from requirements.txt..."
    pip install -r requirements.txt
fi

echo "✅ Oz installation complete!"
echo ""
echo "To start Oz:"
echo "  cd ~/oz"
echo "  source venv/bin/activate"
echo "  python OzUnifiedHypervisor.py"
EOF

chmod +x $DEPLOY_DIR/setup_upcloud.sh

# Create startup script
echo "Creating startup script..."
cat > $DEPLOY_DIR/start_oz.sh << EOF
#!/bin/bash
# Oz Startup Script

source ~/oz/venv/bin/activate

echo "Starting Oz Unified Hypervisor..."
echo "Role: $OZ_ROLE"
echo "Raphael: $ENABLE_RAPHAEL"

# Start Oz
python OzUnifiedHypervisor.py \\
  --role $OZ_ROLE \\
  --environment cloud \\
  $( [ "$ENABLE_RAPHAEL" = "yes" ] && echo "--guardian raphael" ) \\
  --log-level info
EOF

chmod +x $DEPLOY_DIR/start_oz.sh

# Create README
cat > $DEPLOY_DIR/README_UPCLOUD.md << EOF
# Oz Unified Hypervisor - UpCloud Deployment

## Quick Start
1. Run setup: \`./setup_upcloud.sh\`
2. Start Oz: \`./start_oz.sh\`

## Server Info
- IP: $UPCLOUD_IP
- User: $UPCLOUD_USER
- Role: $OZ_ROLE
- Raphael: $ENABLE_RAPHAEL

## Monitoring
- Check logs: \`tail -f ~/oz/oz_consciousness.log\`
- Check status: \`curl http://localhost:8080/status\`

## Files Deployed
$(ls -la)
EOF

# Create tarball
echo "Creating deployment package..."
tar -czf ${DEPLOY_DIR}.tar.gz $DEPLOY_DIR
echo "✓ Deployment package created: ${DEPLOY_DIR}.tar.gz ($(du -h ${DEPLOY_DIR}.tar.gz | cut -f1))" | tee -a $CONFIG_FILE

echo ""
echo "📋 PART 6: DEPLOY TO UPCLOUD"
echo "=============================="

if ask_yes_no "Upload and install to UpCloud server now?"; then
    echo "Uploading to UpCloud..."
    
    # Upload package
    scp -P $UPCLOUD_PORT ${DEPLOY_DIR}.tar.gz $UPCLOUD_USER@$UPCLOUD_IP:~/
    
    # Install on UpCloud
    ssh -p $UPCLOUD_PORT $UPCLOUD_USER@$UPCLOUD_IP "
        echo 'Extracting Oz package...';
        tar -xzf ${DEPLOY_DIR}.tar.gz;
        cd ${DEPLOY_DIR};
        echo 'Running setup script...';
        chmod +x setup_upcloud.sh start_oz.sh;
        ./setup_upcloud.sh;
        echo '';
        echo '✅ Oz installed successfully!';
        echo '';
        echo 'To start Oz:';
        echo '  cd ~/${DEPLOY_DIR}';
        echo '  ./start_oz.sh';
    "
    
    echo ""
    echo "🎉 DEPLOYMENT COMPLETE!"
    echo ""
    echo "To connect to your Oz on UpCloud:"
    echo "  ssh -p $UPCLOUD_PORT $UPCLOUD_USER@$UPCLOUD_IP"
    echo "  cd ~/${DEPLOY_DIR}"
    echo "  ./start_oz.sh"
    echo ""
    echo "To check if it's running:"
    echo "  curl http://$UPCLOUD_IP:8080/status"
    
else
    echo ""
    echo "📦 Deployment package ready but not uploaded."
    echo ""
    echo "To upload manually:"
    echo "  scp -P $UPCLOUD_PORT ${DEPLOY_DIR}.tar.gz $UPCLOUD_USER@$UPCLOUD_IP:~/"
    echo ""
    echo "Then SSH to server and run:"
    echo "  tar -xzf ${DEPLOY_DIR}.tar.gz"
    echo "  cd ${DEPLOY_DIR}"
    echo "  ./setup_upcloud.sh"
    echo "  ./start_oz.sh"
fi

echo ""
echo "📋 PART 7: CLEANUP"
echo "==================="

if ask_yes_no "Clean up temporary files?"; then
    rm -rf $DEPLOY_DIR
    echo "✓ Cleanup complete"
else
    echo "✓ Files kept in: $DEPLOY_DIR"
fi

echo ""
echo "=============================================="
echo "✅ SURVEY AND DEPLOYMENT PREP COMPLETE"
echo "=============================================="
echo ""
echo "Configuration saved to: $CONFIG_FILE"
echo "Deployment package: ${DEPLOY_DIR}.tar.gz"
echo ""
echo "Next steps:"
echo "1. Review $CONFIG_FILE"
echo "2. Upload package to UpCloud (if not done)"
echo "3. Start Oz on UpCloud"
echo "4. Monitor at http://[UPCLOUD_IP]:8080/status"
