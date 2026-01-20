# Cloud Viren Setup Instructions

This document provides instructions for setting up Weaviate and Binary Technologies in a cloud environment.

## Prerequisites

- A cloud VM with at least 4GB RAM and 2 CPUs
- Docker and Docker Compose installed
- Python 3.8+ installed

## Installation Steps

### 1. Install Docker (if not already installed)

```bash
# For Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y docker.io docker-compose

# Start and enable Docker
sudo systemctl start docker
sudo systemctl enable docker
```

### 2. Upload the Viren Cloud Package

Upload the `virencloud.zip` file to your cloud server and extract it:

```bash
unzip virencloud.zip -d /opt/viren-cloud
cd /opt/viren-cloud
```

### 3. Configure Weaviate

The Docker Compose file for Weaviate is included in the package. You may need to modify it based on your cloud environment:

```bash
# Edit the Docker Compose file if needed
nano data/docker-compose.yml

# Start Weaviate
cd data
docker-compose up -d
```

### 4. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 5. Initialize Binary Protocol

```bash
python scripts/install_binary_protocol.py --cloud
```

### 6. Configure Cloud-Specific Settings

Edit the cloud connection configuration:

```bash
nano cloud/cloud_connection.py
```

Update the following settings:
- API endpoints
- Authentication credentials
- Storage paths

### 7. Start Cloud Viren

```bash
python cloud/modal_deployment.py
```

## Verification

To verify that Cloud Viren is running correctly:

```bash
python scripts/viren_diagnostic.py --cloud
```

## Connecting Desktop Viren to Cloud Viren

1. On your desktop system, edit the file `config/colony_config.json` to include the cloud endpoint:

```json
{
  "cloud_endpoints": [
    {
      "url": "https://your-cloud-server-address/api",
      "api_key": "your-api-key"
    }
  ]
}
```

2. Restart Desktop Viren to connect to the cloud instance.

## Troubleshooting

If you encounter issues:

1. Check Docker container status:
   ```bash
   docker ps
   ```

2. View Weaviate logs:
   ```bash
   docker logs weaviate
   ```

3. Check Viren logs:
   ```bash
   cat logs/binary_protocol/cloud_viren.log
   ```