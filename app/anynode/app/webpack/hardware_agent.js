// hardware_agent.js - Hardware diagnostics and repair agent
const http = require('http');
const os = require('os');
const { execSync } = require('child_process');

// Parse command line arguments
const args = process.argv.slice(2);
const portArg = args.find(arg => arg.startsWith('--port='));
const PORT = portArg ? parseInt(portArg.split('=')[1]) : 5100;

// Agent state
const agentState = {
  id: `hardware_agent_${Math.floor(Math.random() * 10000)}`,
  status: 'idle',
  diagnostics: {},
  repairs: []
};

// Hardware diagnostics functions
function getCPUInfo() {
  const cpus = os.cpus();
  const avgLoad = os.loadavg()[0];
  const cpuUsage = avgLoad / cpus.length * 100;
  
  return {
    model: cpus[0].model,
    cores: cpus.length,
    speed: cpus[0].speed,
    usage: cpuUsage.toFixed(2) + '%',
    load: avgLoad.toFixed(2),
    temperature: getTemperature()
  };
}

function getMemoryInfo() {
  const totalMem = os.totalmem();
  const freeMem = os.freemem();
  const usedMem = totalMem - freeMem;
  const percentUsed = (usedMem / totalMem * 100).toFixed(2);
  
  return {
    total: formatBytes(totalMem),
    free: formatBytes(freeMem),
    used: formatBytes(usedMem),
    percentUsed: percentUsed + '%'
  };
}

function getDiskInfo() {
  try {
    let diskInfo;
    
    if (process.platform === 'win32') {
      const output = execSync('wmic logicaldisk get size,freespace,caption').toString();
      diskInfo = { drives: [] };
      
      const lines = output.trim().split('\n').slice(1);
      lines.forEach(line => {
        const parts = line.trim().split(/\s+/);
        if (parts.length >= 3) {
          const drive = parts[0];
          const freeSpace = parseInt(parts[1]);
          const totalSize = parseInt(parts[2]);
          
          if (!isNaN(freeSpace) && !isNaN(totalSize)) {
            diskInfo.drives.push({
              drive,
              total: formatBytes(totalSize),
              free: formatBytes(freeSpace),
              used: formatBytes(totalSize - freeSpace),
              percentUsed: ((totalSize - freeSpace) / totalSize * 100).toFixed(2) + '%'
            });
          }
        }
      });
    } else {
      const output = execSync('df -h').toString();
      diskInfo = { filesystems: [] };
      
      const lines = output.trim().split('\n').slice(1);
      lines.forEach(line => {
        const parts = line.trim().split(/\s+/);
        if (parts.length >= 6) {
          diskInfo.filesystems.push({
            filesystem: parts[0],
            size: parts[1],
            used: parts[2],
            available: parts[3],
            percentUsed: parts[4],
            mountedOn: parts[5]
          });
        }
      });
    }
    
    return diskInfo;
  } catch (error) {
    return { error: error.message };
  }
}

function getTemperature() {
  try {
    if (process.platform === 'win32') {
      return 'N/A (Windows)';
    } else if (process.platform === 'darwin') {
      return 'N/A (macOS)';
    } else {
      // Linux
      const output = execSync('cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null || echo "N/A"').toString().trim();
      if (output !== 'N/A') {
        return (parseInt(output) / 1000).toFixed(1) + '°C';
      }
      return 'N/A';
    }
  } catch (error) {
    return 'N/A';
  }
}

function formatBytes(bytes) {
  if (bytes === 0) return '0 Bytes';
  
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Repair functions
function optimizeSystem() {
  try {
    if (process.platform === 'win32') {
      // Windows optimization
      execSync('powercfg -setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c'); // High performance power plan
      execSync('ipconfig /flushdns');
    } else if (process.platform === 'darwin') {
      // macOS optimization
      execSync('sudo purge'); // Clear memory cache
    } else {
      // Linux optimization
      execSync('sync && echo 3 | sudo tee /proc/sys/vm/drop_caches');
    }
    
    return { status: 'success', message: 'System optimized successfully' };
  } catch (error) {
    return { status: 'error', message: error.message };
  }
}

// HTTP server for agent API
const server = http.createServer((req, res) => {
  res.setHeader('Content-Type', 'application/json');
  
  // Health check endpoint
  if (req.url === '/health' && req.method === 'GET') {
    res.statusCode = 200;
    res.end(JSON.stringify({ status: 'healthy', agent: agentState.id }));
    return;
  }
  
  // Diagnostics endpoint
  if (req.url === '/diagnose' && req.method === 'GET') {
    agentState.status = 'diagnosing';
    
    const diagnostics = {
      timestamp: new Date().toISOString(),
      cpu: getCPUInfo(),
      memory: getMemoryInfo(),
      disk: getDiskInfo(),
      platform: {
        type: os.type(),
        platform: os.platform(),
        release: os.release(),
        arch: os.arch(),
        uptime: formatUptime(os.uptime())
      }
    };
    
    agentState.diagnostics = diagnostics;
    agentState.status = 'idle';
    
    res.statusCode = 200;
    res.end(JSON.stringify(diagnostics));
    return;
  }
  
  // Repair endpoint
  if (req.url === '/repair' && req.method === 'POST') {
    agentState.status = 'repairing';
    
    let body = '';
    req.on('data', chunk => {
      body += chunk.toString();
    });
    
    req.on('end', () => {
      let repairType;
      try {
        const data = JSON.parse(body);
        repairType = data.type;
      } catch (error) {
        repairType = 'optimize'; // Default repair
      }
      
      let result;
      if (repairType === 'optimize') {
        result = optimizeSystem();
      } else {
        result = { status: 'error', message: 'Unknown repair type' };
      }
      
      agentState.repairs.push({
        timestamp: new Date().toISOString(),
        type: repairType,
        result
      });
      
      agentState.status = 'idle';
      res.statusCode = 200;
      res.end(JSON.stringify(result));
    });
    
    return;
  }
  
  // Status endpoint
  if (req.url === '/status' && req.method === 'GET') {
    res.statusCode = 200;
    res.end(JSON.stringify(agentState));
    return;
  }
  
  // Not found
  res.statusCode = 404;
  res.end(JSON.stringify({ error: 'Not found' }));
});

function formatUptime(uptime) {
  const days = Math.floor(uptime / 86400);
  const hours = Math.floor((uptime % 86400) / 3600);
  const minutes = Math.floor((uptime % 3600) / 60);
  const seconds = Math.floor(uptime % 60);
  
  return `${days}d ${hours}h ${minutes}m ${seconds}s`;
}

// Start the server
server.listen(PORT, () => {
  console.log(`[AGENT] Hardware agent running on port ${PORT}`);
});