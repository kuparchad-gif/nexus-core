// network_agent.js - Network diagnostics and repair agent
const http = require('http');
const { execSync } = require('child_process');
const os = require('os');
const dns = require('dns');

// Parse command line arguments
const args = process.argv.slice(2);
const portArg = args.find(arg => arg.startsWith('--port='));
const PORT = portArg ? parseInt(portArg.split('=')[1]) : 5101;

// Agent state
const agentState = {
  id: `network_agent_${Math.floor(Math.random() * 10000)}`,
  status: 'idle',
  diagnostics: {},
  repairs: []
};

// Network diagnostics functions
function getNetworkInterfaces() {
  const interfaces = os.networkInterfaces();
  const result = {};
  
  for (const [name, netInterface] of Object.entries(interfaces)) {
    result[name] = netInterface.map(iface => ({
      address: iface.address,
      netmask: iface.netmask,
      family: iface.family,
      mac: iface.mac,
      internal: iface.internal,
      cidr: iface.cidr
    }));
  }
  
  return result;
}

function pingHost(host = '8.8.8.8') {
  try {
    const command = process.platform === 'win32' 
      ? `ping -n 4 ${host}` 
      : `ping -c 4 ${host}`;
    
    const output = execSync(command).toString();
    
    // Extract ping statistics
    let avgTime;
    let packetLoss;
    
    if (process.platform === 'win32') {
      const avgMatch = output.match(/Average = (\d+)ms/);
      avgTime = avgMatch ? avgMatch[1] + ' ms' : 'N/A';
      
      const lossMatch = output.match(/(\d+)% loss/);
      packetLoss = lossMatch ? lossMatch[1] + '%' : 'N/A';
    } else {
      const avgMatch = output.match(/min\/avg\/max\/mdev = [^\/]+\/([^\/]+)/);
      avgTime = avgMatch ? avgMatch[1] + ' ms' : 'N/A';
      
      const lossMatch = output.match(/(\d+)% packet loss/);
      packetLoss = lossMatch ? lossMatch[1] + '%' : 'N/A';
    }
    
    return {
      host,
      success: true,
      avgTime,
      packetLoss,
      raw: output
    };
  } catch (error) {
    return {
      host,
      success: false,
      error: error.message
    };
  }
}

function checkDNS(domain = 'google.com') {
  return new Promise((resolve) => {
    const startTime = Date.now();
    
    dns.lookup(domain, (err, address) => {
      const responseTime = Date.now() - startTime;
      
      if (err) {
        resolve({
          domain,
          success: false,
          error: err.message
        });
      } else {
        resolve({
          domain,
          success: true,
          address,
          responseTime: responseTime + ' ms'
        });
      }
    });
  });
}

function traceroute(host = '8.8.8.8') {
  try {
    const command = process.platform === 'win32' 
      ? `tracert ${host}` 
      : `traceroute -m 15 ${host}`;
    
    const output = execSync(command).toString();
    
    return {
      host,
      success: true,
      output
    };
  } catch (error) {
    return {
      host,
      success: false,
      error: error.message
    };
  }
}

// Repair functions
function repairNetworkConnection() {
  try {
    if (process.platform === 'win32') {
      // Windows network repair
      execSync('ipconfig /release');
      execSync('ipconfig /renew');
      execSync('ipconfig /flushdns');
      execSync('netsh winsock reset');
    } else if (process.platform === 'darwin') {
      // macOS network repair
      execSync('sudo ifconfig en0 down && sudo ifconfig en0 up');
      execSync('sudo killall -HUP mDNSResponder');
    } else {
      // Linux network repair
      execSync('sudo systemctl restart NetworkManager');
      execSync('sudo systemctl restart systemd-resolved');
    }
    
    return { status: 'success', message: 'Network connection repaired successfully' };
  } catch (error) {
    return { status: 'error', message: error.message };
  }
}

// HTTP server for agent API
const server = http.createServer(async (req, res) => {
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
    
    const interfaces = getNetworkInterfaces();
    const pingResult = pingHost();
    const dnsResult = await checkDNS();
    
    const diagnostics = {
      timestamp: new Date().toISOString(),
      interfaces,
      connectivity: {
        ping: pingResult,
        dns: dnsResult
      }
    };
    
    agentState.diagnostics = diagnostics;
    agentState.status = 'idle';
    
    res.statusCode = 200;
    res.end(JSON.stringify(diagnostics));
    return;
  }
  
  // Traceroute endpoint
  if (req.url.startsWith('/traceroute') && req.method === 'GET') {
    agentState.status = 'tracing';
    
    const urlParts = req.url.split('?');
    let host = '8.8.8.8';
    
    if (urlParts.length > 1) {
      const params = new URLSearchParams(urlParts[1]);
      if (params.has('host')) {
        host = params.get('host');
      }
    }
    
    const result = traceroute(host);
    agentState.status = 'idle';
    
    res.statusCode = 200;
    res.end(JSON.stringify(result));
    return;
  }
  
  // Repair endpoint
  if (req.url === '/repair' && req.method === 'POST') {
    agentState.status = 'repairing';
    
    const result = repairNetworkConnection();
    
    agentState.repairs.push({
      timestamp: new Date().toISOString(),
      type: 'network_repair',
      result
    });
    
    agentState.status = 'idle';
    res.statusCode = 200;
    res.end(JSON.stringify(result));
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

// Start the server
server.listen(PORT, () => {
  console.log(`[AGENT] Network agent running on port ${PORT}`);
});