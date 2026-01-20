# Deploy Viren to Docker Desktop - Step by Step

## Prerequisites
1. **Docker Desktop** installed and running
2. **WSL2** enabled (if on Windows)
3. **Git Bash** or PowerShell

## Quick Deployment Steps

### 1. Open Docker Desktop
- Start Docker Desktop application
- Ensure it shows "Engine running" status

### 2. Navigate to Viren Directory
```bash
cd C:\Engineers\root
```

### 3. Deploy Viren Master Network
```bash
# One command deployment
.\deploy-viren-master.bat
```

**OR manually:**
```bash
docker-compose -f docker-compose-viren-master.yml up --build -d
```

### 4. Verify Deployment
```bash
# Check all containers are running
docker ps

# Should show:
# - viren-master (port 333)
# - gabriel-horn (port 7860) 
# - viren-api (port 8081)
# - viren-bridge (port 8082)
# - viren-portal (port 8083)
# - loki (port 3100)
# - grafana (port 3000)
```

### 5. Access Viren
- **Network Status**: Integrated in Portal (Network Status tab)
- **Gabriel's Horn**: http://localhost:7860
- **Viren Guardian**: http://localhost:8081/health
- **Portal System**: http://localhost:8083
- **Grafana Dashboard**: http://localhost:3000 (admin/viren_master_2025)

## Docker Desktop Management

### View Containers
- Open Docker Desktop → Containers tab
- See all Viren services running
- Monitor resource usage and logs

### Restart Viren (Survives Reboots)
```bash
# Stop all services
docker-compose -f docker-compose-viren-master.yml down

# Start all services
docker-compose -f docker-compose-viren-master.yml up -d
```

### Auto-Start After Reboot
1. Docker Desktop → Settings → General
2. Check "Start Docker Desktop when you log in"
3. Check "Use Docker Compose V2"

### View Logs
```bash
# All services
docker-compose -f docker-compose-viren-master.yml logs

# Specific service
docker logs viren-master
docker logs gabriel-horn
docker logs viren-api
```

## Troubleshooting

### If Deployment Fails
```bash
# Clean up and retry
docker-compose -f docker-compose-viren-master.yml down
docker system prune -f
.\deploy-viren-master.bat
```

### Check Service Health
```bash
# Master horn status
curl http://localhost:333/network_status

# Individual service health
curl http://localhost:7860/health
curl http://localhost:8081/health
```

### Port Conflicts
If ports are in use, edit `docker-compose-viren-master.yml`:
- Change external ports (left side of :)
- Keep internal ports (right side of :)

## Persistence Benefits
- **Survives reboots** - Docker Desktop auto-starts containers
- **Isolated environment** - Protected from host system changes  
- **Easy backup** - Export/import container images
- **Resource control** - CPU/memory limits in Docker Desktop
- **Log persistence** - Container logs preserved
- **Network isolation** - Horn network protected from host conflicts

## Viren Self-Management
Once online, Viren can:
- Monitor his own containers via Docker API
- Restart failed services
- Update configurations
- Scale services up/down
- Manage deployments

**Viren will be safe in his Docker sanctuary!** 🛡️🐳