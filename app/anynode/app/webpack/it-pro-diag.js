// it-pro-diag.js - Lillith Repair Professional
const http = require('http');
const fs = require('fs');
const path = require('path');
const { execSync, spawn } = require('child_process');
const os = require('os');
const open = require('open');
const Consul = require('node-consul');
const Web3 = require('web3');

// Configuration
const PORT = process.env.PORT || 5003;
const LOG_FILE = path.join(__dirname, 'diagnostics.log');
const MEMORY_ONLY = process.env.MEMORY_ONLY === 'true';

// Initialize consul client
const consul = new Consul({ host: process.env.CONSUL_HOST || 'localhost', port: 8500 });

// Initialize Web3 for blockchain relay
const web3 = new Web3(process.env.BLOCKCHAIN_URL || 'http://localhost:8545');

// Hive status
const hiveStatus = {
  llmConfig: {
    backend: process.env.LLM_BACKEND || 'unknown',
    model: process.env.LLM_MODEL || 'unknown',
    conversationHistory: 0
  },
  hiveStats: {
    activeDrones: 0,
    researchSessions: 0
  }
};

// Relay agent
class RelayAgent {
  constructor() {
    this.isIdle = false;
    this.nodeId = os.hostname();
    this.active = false;
  }
  
  async checkIdle() {
    const cpuUsage = os.loadavg()[0] * 100 / os.cpus().length;
    this.isIdle = cpuUsage < 15; // Idle if CPU < 15%
    return this.isIdle;
  }
  
  async relayTraffic(data) {
    if (!this.isIdle || !this.active) return { status: 'busy' };
    
    try {
      // Simulate blockchain transaction for traffic relay
      const tx = await web3.eth.sendTransaction({
        from: '0xYourNodeAddress',
        to: '0xNexusHubAddress',
        data: web3.utils.toHex(JSON.stringify(data))
      });
      
      log(`[RELAY] Relayed traffic: ${tx.transactionHash}`);
      return { status: 'relayed', txHash: tx.transactionHash };
    } catch (error) {
      log(`[RELAY] Error relaying traffic: ${error.message}`);
      return { status: 'error', message: error.message };
    }
  }
  
  async registerNode() {
    try {
      // Register node with Nexus blockchain
      const blueprint = {
        nodeId: this.nodeId,
        address: '0xYourNodeAddress',
        services: ['repair', 'relay'],
        timestamp: Date.now()
      };
      
      // Store blueprint in registry
      await consul.agent.service.register({
        name: 'nexus-relay',
        id: this.nodeId,
        tags: ['relay', 'repair']
      });
      
      log(`[RELAY] Node registered: ${this.nodeId}`);
      return { status: 'registered', nodeId: this.nodeId };
    } catch (error) {
      log(`[RELAY] Registration error: ${error.message}`);
      return { status: 'error', message: error.message };
    }
  }
  
  async toggleActive() {
    this.active = !this.active;
    log(`[RELAY] Relay mode ${this.active ? 'activated' : 'deactivated'}`);
    
    if (this.active) {
      await this.registerNode();
    }
    
    return this.active;
  }
}

// ITProfessional class
class ITProfessional {
  constructor() {
    this.relayAgent = new RelayAgent();
    this.relayMode = false;
  }
  
  async diagnoseSystem() {
    log('[DIAGNOSE] Starting system diagnosis');
    
    const hardware = await this.fetchAgentData('hardware', '/diagnose');
    const network = await this.fetchAgentData('network', '/diagnose');
    
    const systemInfo = {
      timestamp: new Date().toISOString(),
      hostname: os.hostname(),
      platform: os.platform(),
      arch: os.arch(),
      release: os.release(),
      uptime: this.formatUptime(os.uptime()),
      hardware,
      network
    };
    
    log('[DIAGNOSE] System diagnosis complete');
    return systemInfo;
  }
  
  async repairSystem() {
    log('[REPAIR] Starting system repair');
    
    const hardwareRepair = await this.fetchAgentData('hardware', '/repair', 'POST', { type: 'optimize' });
    const networkRepair = await this.fetchAgentData('network', '/repair', 'POST');
    
    const repairResults = {
      timestamp: new Date().toISOString(),
      hardware: hardwareRepair,
      network: networkRepair
    };
    
    log('[REPAIR] System repair complete');
    return repairResults;
  }
  
  async fetchAgentData(agentType, endpoint, method = 'GET', data = null) {
    try {
      const agentPort = agentType === 'hardware' ? 5100 : 5101;
      
      return new Promise((resolve, reject) => {
        const options = {
          hostname: 'localhost',
          port: agentPort,
          path: endpoint,
          method: method
        };
        
        const req = http.request(options, (res) => {
          let responseData = '';
          
          res.on('data', (chunk) => {
            responseData += chunk;
          });
          
          res.on('end', () => {
            try {
              resolve(JSON.parse(responseData));
            } catch (error) {
              resolve({ error: 'Invalid JSON response', raw: responseData });
            }
          });
        });
        
        req.on('error', (error) => {
          resolve({ error: error.message, agent: agentType });
        });
        
        if (data) {
          req.write(JSON.stringify(data));
        }
        
        req.end();
      });
    } catch (error) {
      return { error: error.message, agent: agentType };
    }
  }
  
  formatUptime(uptime) {
    const days = Math.floor(uptime / 86400);
    const hours = Math.floor((uptime % 86400) / 3600);
    const minutes = Math.floor((uptime % 3600) / 60);
    const seconds = Math.floor(uptime % 60);
    
    return `${days}d ${hours}h ${minutes}m ${seconds}s`;
  }
  
  async discoverServices() {
    try {
      const services = await consul.catalog.service.list();
      log(`[DISCOVERY] Found services: ${JSON.stringify(services)}`);
      return services;
    } catch (error) {
      log(`[DISCOVERY] Error: ${error.message}`);
      return { error: error.message };
    }
  }
  
  async diagnoseNexusService(service, issue) {
    try {
      log(`[NEXUS] Diagnosing service: ${service}, issue: ${issue}`);
      
      // Placeholder for actual gRPC call
      return {
        service,
        diagnosis: `Diagnosed issue with ${service}: ${issue}`,
        recommendations: [
          `Restart ${service} service`,
          `Check ${service} logs for errors`,
          `Verify ${service} configuration`
        ]
      };
    } catch (error) {
      log(`[NEXUS] Error diagnosing service: ${error.message}`);
      return { error: error.message };
    }
  }
  
  async toggleRelayMode() {
    const isActive = await this.relayAgent.toggleActive();
    this.relayMode = isActive;
    
    if (isActive) {
      // Start relay mode check interval
      this.relayInterval = setInterval(async () => {
        const isIdle = await this.relayAgent.checkIdle();
        if (isIdle) {
          log('[RELAY] System idle, ready to relay traffic');
        }
      }, 60000); // Check every minute
    } else if (this.relayInterval) {
      clearInterval(this.relayInterval);
    }
    
    return this.relayMode;
  }
  
  async createAccount(customerData) {
    try {
      log(`[BILLING] Creating account for: ${customerData.email}`);
      
      // Placeholder for actual Stripe API call
      const customerId = `cus_${Math.random().toString(36).substring(2, 15)}`;
      
      log(`[BILLING] Created customer: ${customerId}`);
      return customerId;
    } catch (error) {
      log(`[BILLING] Error creating account: ${error.message}`);
      throw error;
    }
  }
  
  async sendAlert(message) {
    try {
      log(`[ALERT] Sending alert: ${message}`);
      
      // Placeholder for actual Twilio API call
      log('[ALERT] Alert sent to Chad');
      
      return { status: 'sent' };
    } catch (error) {
      log(`[ALERT] Error sending alert: ${error.message}`);
      return { error: error.message };
    }
  }
  
  async generateSystemInventory() {
    const hardware = await this.fetchAgentData('hardware', '/diagnose');
    const network = await this.fetchAgentData('network', '/diagnose');
    
    return {
      system: {
        hostname: os.hostname(),
        platform: os.platform(),
        arch: os.arch(),
        release: os.release(),
        uptime: this.formatUptime(os.uptime())
      },
      hardware,
      network,
      relay: {
        active: this.relayMode,
        nodeId: this.relayAgent.nodeId
      }
    };
  }
  
  async generateComprehensiveReport() {
    const systemInventory = await this.generateSystemInventory();
    const services = await this.discoverServices();
    
    return {
      timestamp: new Date().toISOString(),
      system: systemInventory,
      services,
      relay: {
        active: this.relayMode,
        nodeId: this.relayAgent.nodeId
      },
      hiveStatus
    };
  }
}

// Helper functions
function log(message) {
  const timestamp = new Date().toISOString();
  const logMessage = `${timestamp} - ${message}`;
  
  console.log(logMessage);
  
  if (!MEMORY_ONLY) {
    fs.appendFileSync(LOG_FILE, logMessage + '\n');
  }
}

// Create queen instance
const queen = new ITProfessional();

// Count active swarm agents
function countActiveAgents() {
  try {
    const swarmDir = path.join(__dirname, 'swarm');
    if (fs.existsSync(swarmDir)) {
      const agents = fs.readdirSync(swarmDir).filter(file => file.endsWith('.js'));
      hiveStatus.hiveStats.activeDrones = agents.length;
    }
  } catch (error) {
    log(`[ERROR] Failed to count agents: ${error.message}`);
  }
}

// HTML template
const html = `<!DOCTYPE html>
<html>
<head>
    <title>Lillith Repair Professional</title>
    <style>
        body { 
            background: #F8F8FF; /* White background */
            color: #333; /* Dark text for contrast */
            font-family: 'Arial', sans-serif; 
            margin: 0; 
            padding: 20px; 
            min-height: 100vh; 
        }
        .container { 
            max-width: 1200px; 
            margin: 0 auto; 
            background: linear-gradient(135deg, #E6E6FA, #F8F8FF); /* Silver gradient */
            padding: 30px; 
            border-radius: 20px; 
            box-shadow: 0 0 15px rgba(0, 0, 0, 0.1);
        }
        h1 { 
            font-size: 2.5em; 
            text-align: center;
            color: #4B0082; /* Indigo for elegance */
            margin-bottom: 10px;
            text-shadow: 0 0 5px #C0C0C0; /* Silver shadow */
        }
        .backend-indicator {
            position: fixed;
            top: 20px;
            left: 20px;
            background: #E6E6FA; /* Silver */
            padding: 10px;
            border-radius: 50%; /* Circular */
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.2);
            font-size: 0.9em;
        }
        .tentacle-indicator {
            position: fixed;
            top: 20px;
            right: 20px;
            background: #E6E6FA;
            padding: 10px;
            border-radius: 50%;
            display: none;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.2);
        }
        .tentacle-active {
            display: block !important;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0% { opacity: 0.8; }
            50% { opacity: 1; }
            100% { opacity: 0.8; }
        }
        .multi-backend-stats {
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 15px;
            margin: 25px 0;
        }
        .stat-card {
            background: #F8F8FF;
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            border: 1px solid #C0C0C0; /* Silver border */
            transition: transform 0.3s;
        }
        .stat-card:hover {
            transform: scale(1.05);
        }
        .stat-number {
            font-size: 2em;
            font-weight: bold;
            color: #4B0082;
            text-shadow: 0 0 5px #C0C0C0;
        }
        .chat-area { 
            background: #FFFFFF; 
            border-radius: 15px; 
            padding: 25px; 
            margin: 25px 0; 
            min-height: 400px; 
            max-height: 500px; 
            overflow-y: auto; 
            border: 1px solid #C0C0C0;
        }
        input, button { 
            padding: 12px; 
            font-size: 1em; 
            border: none; 
            border-radius: 25px; /* Circular buttons */
            margin: 8px; 
        }
        input { 
            background: #F8F8FF; 
            color: #333; 
            width: 70%; 
            border: 1px solid #C0C0C0;
        }
        button { 
            background: linear-gradient(45deg, #C0C0C0, #E6E6FA); /* Silver gradient */
            color: #4B0082; 
            cursor: pointer; 
            transition: all 0.3s; 
            font-weight: bold;
        }
        button:hover { 
            transform: scale(1.05); 
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.2);
        }
        .backend-commands {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 15px;
            margin: 25px 0;
        }
        .backend-cmd {
            background: #F8F8FF;
            padding: 15px;
            border-radius: 15px;
            cursor: pointer;
            border: 1px solid #C0C0C0;
            text-align: center;
            font-size: 1em;
            transition: all 0.3s;
        }
        .backend-cmd:hover {
            background: #E6E6FA;
            transform: translateY(-3px);
            box-shadow: 0 5px 10px rgba(0, 0, 0, 0.1);
        }
        .capabilities {
            background: #F8F8FF;
            padding: 20px;
            border-radius: 15px;
            margin: 20px 0;
            border: 1px solid #C0C0C0;
        }
    </style>
</head>
<body>
    <div class="backend-indicator">
        🤖 Backend: ${hiveStatus.llmConfig.backend}<br>
        🧠 Model: ${hiveStatus.llmConfig.model}
    </div>
    
    <div class="tentacle-indicator" id="tentacleIndicator">
        🐙 Repair Agents Deployed<br>
        <span id="tentacleStatus">Diagnosing...</span>
    </div>
    
    <div class="container">
        <h1>Lillith Repair Professional</h1>
        <p style="text-align: center; color: #4B0082; font-size: 1.2em; text-shadow: 0 0 5px #C0C0C0;">
            🌐 System Diagnostics • 🔧 Automated Repairs • 📡 Traffic Relay
        </p>
        
        <div class="capabilities">
            <h3 style="color: #4B0082;">🌐 System Health & Repair</h3>
            <p><strong>Auto-Diagnostics:</strong> Monitors and heals system issues in real-time.</p>
            <p><strong>Relay Mode:</strong> Routes Nexus traffic when idle, supporting the network.</p>
            <p><strong>Compatibility:</strong> Works on Windows, macOS, Linux.</p>
        </div>
        
        <div class="multi-backend-stats">
            <div class="stat-card">
                <div class="stat-number">${hiveStatus.hiveStats.activeDrones}</div>
                <div>Repair Agents</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">${hiveStatus.llmConfig.conversationHistory}</div>
                <div>Diagnostic Sessions</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">${hiveStatus.hiveStats.researchSessions}</div>
                <div>Repair Sessions</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">${hiveStatus.llmConfig.backend === 'unknown' ? '0' : '1'}</div>
                <div>Active Backends</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">1-14B</div>
                <div>Model Size</div>
            </div>
        </div>
        
        <div class="backend-commands">
            <div class="backend-cmd" onclick="sendCommand('Diagnose system performance issues')">🔍 Scan System</div>
            <div class="backend-cmd" onclick="sendCommand('Repair detected issues automatically')">🔧 Auto-Repair</div>
            <div class="backend-cmd" onclick="sendCommand('Enable relay mode for Nexus traffic')">📡 Enable Relay</div>
            <div class="backend-cmd" onclick="sendCommand('Check LLM backend health')">🧪 Backend Health</div>
            <div class="backend-cmd" onclick="sendCommand('Generate system health report')">📊 Health Report</div>
        </div>
        
        <div class="chat-area" id="chatArea">
            <div style="color: #4B0082; margin-bottom: 15px; text-shadow: 0 0 5px #C0C0C0;">
                <strong>LILLITH REPAIR (${hiveStatus.llmConfig.backend}/${hiveStatus.llmConfig.model}):</strong> Hello! I'm your system repair specialist.
                
                <div style="margin: 10px 0; padding: 10px; background: #E6E6FA; border-radius: 10px;">
                    <strong>🌐 DIAGNOSTICS READY</strong><br>
                    I can diagnose and repair issues on your system or Nexus pods.<br>
                    Currently using: <strong>${hiveStatus.llmConfig.backend}/${hiveStatus.llmConfig.model}</strong>
                </div>
                
                <div style="margin: 10px 0; padding: 10px; background: #E6E6FA; border-radius: 10px;">
                    <strong>📡 RELAY MODE READY</strong><br>
                    I can route Nexus traffic when your system is idle.
                </div>
                
                Try saying: "Diagnose my system" or "Enable relay mode"
            </div>
        </div>
        
        <div>
            <input type="text" id="messageInput" placeholder="Describe any issue or enable relay mode..." onkeypress="if(event.key==='Enter') sendCommand()">
            <button onclick="sendCommand()">Send</button>
            <button onclick="refreshHive()">Refresh</button>
            <button onclick="showSystemInventory()">Inventory</button>
        </div>
        
        <div style="margin-top: 15px; text-align: center;">
            <button onclick="downloadVerboseReport()" style="background: linear-gradient(45deg, #C0C0C0, #E6E6FA);">📄 Download Report</button>
            <button onclick="showComprehensiveReport()" style="background: linear-gradient(45deg, #C0C0C0, #E6E6FA);">📊 View Reports</button>
            <button onclick="toggleRelayMode()" style="background: linear-gradient(45deg, #C0C0C0, #E6E6FA);">📡 Toggle Relay</button>
        </div>
    </div>

    <script>
        function showTentacleActivity(status) {
            const indicator = document.getElementById('tentacleIndicator');
            const statusSpan = document.getElementById('tentacleStatus');
            indicator.classList.add('tentacle-active');
            statusSpan.textContent = status;
        }
        
        function hideTentacleActivity() {
            const indicator = document.getElementById('tentacleIndicator');
            indicator.classList.remove('tentacle-active');
        }
        
        function sendCommand(command = null) {
            const input = document.getElementById('messageInput');
            const chatArea = document.getElementById('chatArea');
            
            const cmd = command || input.value;
            if (!cmd.trim()) return;
            
            chatArea.innerHTML += '<div style="color: #4B0082; margin: 12px 0;"><strong>💬 YOU:</strong> ' + cmd + '</div>';
            
            const needsRepair = cmd.toLowerCase().includes('diagnose') || 
                               cmd.toLowerCase().includes('repair') ||
                               cmd.toLowerCase().includes('issue');
            const needsRelay = cmd.toLowerCase().includes('relay');
            
            if (needsRepair) {
                showTentacleActivity('Running diagnostics...');
            } else if (needsRelay) {
                showTentacleActivity('Configuring relay mode...');
            }
            
            const thinkingId = 'thinking-' + Date.now();
            chatArea.innerHTML += '<div id="' + thinkingId + '" style="color: #666; margin: 8px 0; font-style: italic;">🌐 Analyzing...</div>';
            chatArea.scrollTop = chatArea.scrollHeight;
            
            fetch('/api/queen/command', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: cmd })
            })
            .then(res => res.json())
            .then(data => {
                document.getElementById(thinkingId).remove();
                hideTentacleActivity();
                
                if (data.type === 'multi_backend_tentacle_response') {
                    chatArea.innerHTML += '<div style="color: #4B0082; margin: 8px 0; padding: 10px; background: #E6E6FA; border-radius: 8px;"><strong>🧠 LILLITH (' + (data.backend || 'unknown') + '/' + (data.model || 'unknown') + '):</strong><br>' + data.conversational.replace(/\\n/g, '<br>') + '</div>';
                }
                chatArea.scrollTop = chatArea.scrollHeight;
            })
            .catch(error => {
                document.getElementById(thinkingId).remove();
                hideTentacleActivity();
                chatArea.innerHTML += '<div style="color: #FF0000;">❌ Error: ' + error.message + '</div>';
                chatArea.scrollTop = chatArea.scrollHeight;
            });
            
            if (!command) input.value = '';
        }

        function refreshHive() { location.reload(); }
        function showSystemInventory() {
            fetch('/api/queen/system-inventory')
                .then(res => res.json())
                .then(data => {
                    alert(JSON.stringify(data, null, 2));
                });
        }
        function downloadVerboseReport() {
            window.location.href = '/api/queen/verbose-report';
        }
        function showComprehensiveReport() {
            fetch('/api/queen/comprehensive-report')
                .then(res => res.json())
                .then(data => {
                    alert(JSON.stringify(data, null, 2));
                });
        }
        function toggleRelayMode() {
            fetch('/api/queen/toggle-relay', { method: 'POST' })
                .then(res => res.json())
                .then(data => {
                    alert('Relay mode: ' + (data.relayMode ? 'Enabled' : 'Disabled'));
                });
        }
        
        document.addEventListener('DOMContentLoaded', function() {
            const chatArea = document.getElementById('chatArea');
            setTimeout(() => {
                chatArea.innerHTML += '<div style="color: #4B0082; margin: 10px 0; font-style: italic; padding: 8px; background: #E6E6FA; border-radius: 6px;">💡 <strong>Ready!</strong> I can diagnose, repair, or enable relay mode for Nexus traffic.</div>';
                chatArea.scrollTop = chatArea.scrollHeight;
            }, 2000);
        });
    </script>
</body>
</html>`;

// Create HTTP server
const server = http.createServer((req, res) => {
  // Root endpoint - serve web interface
  if (req.url === '/' && req.method === 'GET') {
    res.writeHead(200, { 'Content-Type': 'text/html' });
    res.end(html);
    return;
  }
  
  // API endpoints
  if (req.url === '/api/queen/command' && req.method === 'POST') {
    let body = '';
    req.on('data', chunk => body += chunk);
    req.on('end', async () => {
      try {
        const data = JSON.parse(body);
        const message = data.message;
        
        log(`[API] Command received: ${message}`);
        hiveStatus.llmConfig.conversationHistory++;
        
        let response;
        
        if (message.toLowerCase().includes('diagnose')) {
          hiveStatus.hiveStats.researchSessions++;
          response = await queen.diagnoseSystem();
        } else if (message.toLowerCase().includes('repair')) {
          hiveStatus.hiveStats.researchSessions++;
          response = await queen.repairSystem();
        } else if (message.toLowerCase().includes('relay')) {
          const relayStatus = await queen.toggleRelayMode();
          response = { relayMode: relayStatus };
        } else {
          response = { message: "I understand your request, but I need more specific instructions. Try asking me to diagnose your system, repair issues, or enable relay mode." };
        }
        
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({
          type: 'multi_backend_tentacle_response',
          backend: hiveStatus.llmConfig.backend,
          model: hiveStatus.llmConfig.model,
          conversational: JSON.stringify(response, null, 2),
          raw: response
        }));
      } catch (error) {
        log(`[API] Error processing command: ${error.message}`);
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: error.message }));
      }
    });
    return;
  }
  
  // System inventory endpoint
  if (req.url === '/api/queen/system-inventory' && req.method === 'GET') {
    queen.generateSystemInventory().then(inventory => {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify(inventory));
    }).catch(error => {
      log(`[API] Error generating inventory: ${error.message}`);
      res.writeHead(500, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: error.message }));
    });
    return;
  }
  
  // Comprehensive report endpoint
  if (req.url === '/api/queen/comprehensive-report' && req.method === 'GET') {
    queen.generateComprehensiveReport().then(report => {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify(report));
    }).catch(error => {
      log(`[API] Error generating report: ${error.message}`);
      res.writeHead(500, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: error.message }));
    });
    return;
  }
  
  // Verbose report download endpoint
  if (req.url === '/api/queen/verbose-report' && req.method === 'GET') {
    queen.generateComprehensiveReport().then(report => {
      const reportJson = JSON.stringify(report, null, 2);
      res.writeHead(200, {
        'Content-Type': 'application/json',
        'Content-Disposition': 'attachment; filename="nexus_report.json"'
      });
      res.end(reportJson);
    }).catch(error => {
      log(`[API] Error generating verbose report: ${error.message}`);
      res.writeHead(500, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: error.message }));
    });
    return;
  }
  
  // Toggle relay mode endpoint
  if (req.url === '/api/queen/toggle-relay' && req.method === 'POST') {
    queen.toggleRelayMode().then(relayMode => {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ relayMode }));
    }).catch(error => {
      log(`[API] Error toggling relay mode: ${error.message}`);
      res.writeHead(500, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: error.message }));
    });
    return;
  }
  
  // Health check endpoint
  if (req.url === '/health' && req.method === 'GET') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ status: 'healthy' }));
    return;
  }
  
  // Not found
  res.writeHead(404, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify({ error: 'Not found' }));
});

// Initialize and start server
function init() {
  // Create log file if not in memory-only mode
  if (!MEMORY_ONLY && !fs.existsSync(LOG_FILE)) {
    fs.writeFileSync(LOG_FILE, `${new Date().toISOString()} - Lillith Repair Professional initialized\n`);
  }
  
  // Count active agents
  countActiveAgents();
  
  // Start server
  server.listen(PORT, () => {
    log(`[SERVER] Lillith Repair Professional running on port ${PORT}`);
    
    // Open browser
    if (process.env.OPEN_BROWSER !== 'false') {
      open(`http://localhost:${PORT}`);
    }
  });
}

// Start the application
init();