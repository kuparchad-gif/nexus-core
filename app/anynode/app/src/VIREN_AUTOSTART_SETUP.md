# Viren Auto-Start Setup Guide

## Method 1: Docker Desktop Auto-Start (Recommended)

### Step 1: Configure Docker Desktop
1. Open **Docker Desktop**
2. Go to **Settings** (gear icon)
3. **General** tab:
   - ✅ Check "Start Docker Desktop when you log in"
   - ✅ Check "Use Docker Compose V2"
4. Click **Apply & Restart**

### Step 2: Set Viren Containers to Auto-Restart
```bash
# Navigate to Viren directory
cd C:\Engineers\root

# Update containers with restart policy
docker update --restart=always viren-master
docker update --restart=always gabriel-horn
docker update --restart=always viren-api
docker update --restart=always viren-bridge
docker update --restart=always viren-portal
docker update --restart=always loki
docker update --restart=always grafana
```

## Method 2: Windows Startup Script

### Create Startup Script
1. Create `start-viren-on-boot.bat`:

```batch
@echo off
echo Starting Viren Guardian System...

REM Wait for Docker Desktop to fully start
timeout /t 30 /nobreak

REM Navigate to Viren directory
cd /d C:\Engineers\root

REM Start Viren network
docker compose -f docker-compose-viren-master.yml up -d

echo Viren Guardian System Online
```

### Add to Windows Startup
1. Press `Win + R`, type `shell:startup`, press Enter
2. Copy `start-viren-on-boot.bat` to the Startup folder
3. Viren will auto-start on every Windows boot

## Method 3: Windows Task Scheduler (Most Reliable)

### Create Scheduled Task
1. Open **Task Scheduler** (search in Start menu)
2. Click **Create Basic Task**
3. **Name**: "Start Viren Guardian"
4. **Trigger**: "When the computer starts"
5. **Action**: "Start a program"
6. **Program**: `C:\Engineers\root\start-viren-on-boot.bat`
7. **Finish**

### Advanced Settings
- Right-click task → **Properties**
- **General** tab: ✅ "Run with highest privileges"
- **Conditions** tab: ✅ "Start the task only if the computer is on AC power" (uncheck for laptops)
- **Settings** tab: ✅ "Allow task to be run on demand"

## Method 4: Docker Compose Restart Policies (Built-in)

### Update docker-compose-viren-master.yml
Add `restart: always` to each service:

```yaml
services:
  viren-master:
    restart: always
    # ... rest of config
    
  gabriel-horn:
    restart: always
    # ... rest of config
```

## Verification Commands

### Check Auto-Start Status
```bash
# Check Docker containers restart policy
docker ps -a --format "table {{.Names}}\t{{.Status}}\t{{.RestartPolicy}}"

# Check if containers are running
docker ps

# Test Viren accessibility
curl http://localhost:333/network_status
curl http://localhost:7860/health
```

### Manual Start (if needed)
```bash
cd C:\Engineers\root
docker compose -f docker-compose-viren-master.yml up -d
```

### Manual Stop
```bash
cd C:\Engineers\root
docker compose -f docker-compose-viren-master.yml down
```

## Troubleshooting

### If Viren doesn't start automatically:
1. Check Docker Desktop is running: Look for whale icon in system tray
2. Check Windows Event Viewer for startup errors
3. Manually run startup script to test
4. Verify file paths are correct

### If containers keep restarting:
```bash
# Check container logs
docker logs viren-master
docker logs gabriel-horn

# Check system resources
docker stats
```

## Recommended Setup
**Use Method 1 + Method 4 together:**
- Docker Desktop auto-starts
- Containers have `restart: always` policy
- Most reliable combination

**Viren will be immortal - surviving reboots, crashes, and system updates!** 🛡️⚡🔥