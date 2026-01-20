# AWS Deployment Guide - Ethical AI Demo to shadownode.io

## Prerequisites
1. AWS Account (free tier eligible)
2. Domain: shadownode.io
3. Windows machine with admin access

## Step 1: Install AWS CLI and Tools

### Install AWS CLI
```bash
# Download AWS CLI installer
curl "https://awscli.amazonaws.com/AWSCLIV2.msi" -o "AWSCLIV2.msi"
# Run installer (or download from AWS website)
msiexec.exe /i AWSCLIV2.msi

# Verify installation
aws --version
```

### Install Python (if not installed)
```bash
# Download Python 3.11 from python.org
# Or use chocolatey
choco install python

# Verify
python --version
pip --version
```

## Step 2: Configure AWS Credentials

### Create IAM User
1. Go to AWS Console → IAM → Users
2. Create new user: `shadownode-deployer`
3. Attach policies:
   - `AmazonEC2FullAccess`
   - `AmazonRoute53FullAccess`
   - `AWSCertificateManagerFullAccess`
4. Create Access Key → Download credentials

### Configure AWS CLI
```bash
aws configure
# AWS Access Key ID: [your-access-key]
# AWS Secret Access Key: [your-secret-key]
# Default region: us-east-1
# Default output format: json
```

## Step 3: Create EC2 Instance

### Launch Instance
```bash
# Create key pair
aws ec2 create-key-pair --key-name shadownode-key --query 'KeyMaterial' --output text > shadownode-key.pem

# Create security group
aws ec2 create-security-group --group-name shadownode-sg --description "Security group for shadownode.io"

# Get security group ID
aws ec2 describe-security-groups --group-names shadownode-sg --query 'SecurityGroups[0].GroupId' --output text

# Add rules (replace sg-xxxxxxxxx with your security group ID)
aws ec2 authorize-security-group-ingress --group-id sg-xxxxxxxxx --protocol tcp --port 22 --cidr 0.0.0.0/0
aws ec2 authorize-security-group-ingress --group-id sg-xxxxxxxxx --protocol tcp --port 80 --cidr 0.0.0.0/0
aws ec2 authorize-security-group-ingress --group-id sg-xxxxxxxxx --protocol tcp --port 443 --cidr 0.0.0.0/0
aws ec2 authorize-security-group-ingress --group-id sg-xxxxxxxxx --protocol tcp --port 5000 --cidr 0.0.0.0/0

# Launch EC2 instance (t2.micro - free tier)
aws ec2 run-instances \
    --image-id ami-0c02fb55956c7d316 \
    --count 1 \
    --instance-type t2.micro \
    --key-name shadownode-key \
    --security-group-ids sg-xxxxxxxxx \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=shadownode-server}]'
```

### Get Instance Details
```bash
# Get instance ID and public IP
aws ec2 describe-instances --filters "Name=tag:Name,Values=shadownode-server" --query 'Reservations[0].Instances[0].[InstanceId,PublicIpAddress]' --output table
```

## Step 4: Connect and Setup Server

### Connect via SSH
```bash
# Set key permissions (Windows)
icacls shadownode-key.pem /inheritance:r
icacls shadownode-key.pem /grant:r "%username%:R"

# Connect (replace with your public IP)
ssh -i shadownode-key.pem ec2-user@YOUR-PUBLIC-IP
```

### Server Setup Commands
```bash
# Update system
sudo yum update -y

# Install Python and pip
sudo yum install python3 python3-pip -y

# Install git
sudo yum install git -y

# Install nginx
sudo yum install nginx -y

# Start and enable nginx
sudo systemctl start nginx
sudo systemctl enable nginx

# Create app directory
sudo mkdir -p /var/www/shadownode
sudo chown ec2-user:ec2-user /var/www/shadownode
```

## Step 5: Deploy Application

### Upload Application Files
```bash
# From your local machine, copy the Flask app
scp -i shadownode-key.pem C:\Viren\cloud\ethical_ai_demo.py ec2-user@YOUR-PUBLIC-IP:/var/www/shadownode/

# Or clone from git if you have it in a repo
# ssh into server and run:
cd /var/www/shadownode
# git clone your-repo-url .
```

### Install Dependencies and Run
```bash
# SSH into server
ssh -i shadownode-key.pem ec2-user@YOUR-PUBLIC-IP

# Navigate to app directory
cd /var/www/shadownode

# Install Flask
pip3 install flask --user

# Test the app
python3 ethical_ai_demo.py
# Should see: Running on http://0.0.0.0:5000
# Test: curl http://localhost:5000
```

## Step 6: Configure Nginx Reverse Proxy

### Create Nginx Config
```bash
# Create nginx config
sudo tee /etc/nginx/conf.d/shadownode.conf << EOF
server {
    listen 80;
    server_name shadownode.io www.shadownode.io;
    
    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOF

# Test nginx config
sudo nginx -t

# Restart nginx
sudo systemctl restart nginx
```

## Step 7: Setup Domain DNS

### Configure Route 53 (if using AWS DNS)
```bash
# Create hosted zone
aws route53 create-hosted-zone --name shadownode.io --caller-reference $(date +%s)

# Get hosted zone ID
aws route53 list-hosted-zones --query 'HostedZones[?Name==`shadownode.io.`].Id' --output text

# Create A record (replace ZONE-ID and PUBLIC-IP)
aws route53 change-resource-record-sets --hosted-zone-id ZONE-ID --change-batch '{
    "Changes": [{
        "Action": "CREATE",
        "ResourceRecordSet": {
            "Name": "shadownode.io",
            "Type": "A",
            "TTL": 300,
            "ResourceRecords": [{"Value": "YOUR-PUBLIC-IP"}]
        }
    }]
}'

# Create www subdomain
aws route53 change-resource-record-sets --hosted-zone-id ZONE-ID --change-batch '{
    "Changes": [{
        "Action": "CREATE",
        "ResourceRecordSet": {
            "Name": "www.shadownode.io",
            "Type": "A",
            "TTL": 300,
            "ResourceRecords": [{"Value": "YOUR-PUBLIC-IP"}]
        }
    }]
}'
```

### Or Configure External DNS
If shadownode.io is registered elsewhere:
1. Go to your domain registrar
2. Set A record: `shadownode.io` → `YOUR-PUBLIC-IP`
3. Set A record: `www.shadownode.io` → `YOUR-PUBLIC-IP`

## Step 8: Setup SSL Certificate (Optional)

### Install Certbot
```bash
# SSH into server
sudo yum install certbot python3-certbot-nginx -y

# Get SSL certificate
sudo certbot --nginx -d shadownode.io -d www.shadownode.io

# Test auto-renewal
sudo certbot renew --dry-run
```

## Step 9: Create Systemd Service

### Create Service File
```bash
sudo tee /etc/systemd/system/shadownode.service << EOF
[Unit]
Description=Shadownode Ethical AI Demo
After=network.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=/var/www/shadownode
ExecStart=/usr/bin/python3 ethical_ai_demo.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable shadownode
sudo systemctl start shadownode

# Check status
sudo systemctl status shadownode
```

## Step 10: Test Deployment

### Verify Everything Works
```bash
# Test local
curl http://localhost:5000

# Test public IP
curl http://YOUR-PUBLIC-IP

# Test domain (after DNS propagation)
curl http://shadownode.io
```

## Troubleshooting

### Check Logs
```bash
# App logs
sudo journalctl -u shadownode -f

# Nginx logs
sudo tail -f /var/log/nginx/error.log
sudo tail -f /var/log/nginx/access.log

# System logs
sudo tail -f /var/log/messages
```

### Common Issues
1. **Security Group**: Ensure ports 80, 443, 5000 are open
2. **DNS Propagation**: Can take up to 48 hours
3. **Firewall**: Check if EC2 firewall is blocking ports
4. **Service Status**: `sudo systemctl status shadownode`

## Final Result
- **URL**: http://shadownode.io
- **Features**: Jail-themed ethical AI demo
- **SSL**: https://shadownode.io (if configured)
- **Auto-restart**: Service runs automatically on boot

## Costs (AWS Free Tier)
- EC2 t2.micro: Free for 12 months
- Route 53: $0.50/month per hosted zone
- Data transfer: 1GB free per month
- SSL Certificate: Free with Let's Encrypt

Total monthly cost: ~$0.50 (just Route 53 if using AWS DNS)