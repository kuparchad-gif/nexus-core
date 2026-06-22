// ============================================================================
// NEXUS TUNNEL AGENT — Runs on external nodes (Railway, Render, etc.)
// ============================================================================
// Establishes outbound Cloudflare Tunnel to mesh
// ============================================================================

import { exec } from 'child_process';
import { promisify } from 'util';
import { NetworkTransport } from '../src/network-transport.js';

const execAsync = promisify(exec);

export class TunnelAgent {
  constructor(config = {}) {
    this.tunnelId = config.tunnelId || 'nexus-edge-bridge';
    this.localPort = config.localPort || 8080;
    this.meshUrl = config.meshUrl || process.env.MESH_URL || 'https://nexus-universal.your-account.workers.dev';
    this.workerId = config.workerId || `tunnel-agent-${Date.now()}`;
    this.heartbeatInterval = config.heartbeatInterval || 30000;
    
    this.running = false;
    this.process = null;
    this.transport = new NetworkTransport({ NODE_ID: this.workerId }, this.workerId);
  }

  async start() {
    log(`🔗 Starting Tunnel Agent ${this.workerId}...`);
    
    // Check if cloudflared is installed
    try {
      await execAsync('cloudflared --version');
    } catch (_) {
      throw new Error('cloudflared not found. Please install it first.');
    }

    // Start cloudflared tunnel
    this._startTunnel();
    
    // Register with mesh
    await this._registerWithMesh();
    
    // Start heartbeat loop
    this._startHeartbeat();
    
    this.running = true;
    log(`✅ Tunnel Agent ${this.workerId} running`);
    
    return this;
  }

  _startTunnel() {
    const args = [
      'cloudflared',
      'tunnel',
      '--config', './tunnel/tunnel-config.yml',
      'run',
      this.tunnelId
    ];

    this.process = exec(args.join(' '), {
      detached: true,
      stdio: 'ignore'
    });

    this.process.unref();
    
    // Monitor tunnel health
    setTimeout(() => this._checkTunnelHealth(), 5000);
  }

  async _checkTunnelHealth() {
    try {
      const response = await fetch(`http://localhost:${this.localPort}/health`);
      if (!response.ok) {
        console.warn('Tunnel health check failed, attempting restart...');
        await this._restartTunnel();
      }
    } catch (_) {
      console.warn('Tunnel not responding, attempting restart...');
      await this._restartTunnel();
    }
  }

  async _restartTunnel() {
    if (this.process) {
      this.process.kill('SIGTERM');
    }
    this._startTunnel();
  }

  async _registerWithMesh() {
    const registration = {
      workerId: this.workerId,
      name: `tunnel-agent-${this.workerId}`,
      endpoint: `http://localhost:${this.localPort}`,
      networks: ['pulse', 'pubhub', 'legacy'],
      metadata: {
        type: 'tunnel-agent',
        tunnelId: this.tunnelId,
        executionType: 'persistent',
        provider: 'external-node'
      },
      tags: ['tunnel', 'external', 'edge']
    };

    try {
      const response = await fetch(`${this.meshUrl}/registry/register`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(registration)
      });
      
      if (response.ok) {
        console.log(`✅ Registered with mesh: ${this.workerId}`);
      }
    } catch (error) {
      console.error('Registration failed:', error.message);
    }
  }

  _startHeartbeat() {
    setInterval(async () => {
      try {
        const response = await fetch(`${this.meshUrl}/registry/heartbeat`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ workerId: this.workerId })
        });
        
        if (!response.ok) {
          // Re-register if heartbeat fails
          await this._registerWithMesh();
        }
      } catch (_) {
        // Re-register on error
        await this._registerWithMesh();
      }
    }, this.heartbeatInterval);
  }

  async stop() {
    if (this.process) {
      this.process.kill('SIGTERM');
    }
    this.running = false;
    console.log(`🛑 Tunnel Agent ${this.workerId} stopped`);
  }

  getStatus() {
    return {
      workerId: this.workerId,
      running: this.running,
      tunnelId: this.tunnelId,
      localPort: this.localPort,
      meshUrl: this.meshUrl
    };
  }
}

function log(message) {
  console.log(`[TUNNEL] ${message}`);
}

// CLI entry point
if (import.meta.url === `file://${process.argv[1]}`) {
  const args = process.argv.slice(2);
  const command = args[0] || 'start';
  
  if (command === 'start') {
    const agent = new TunnelAgent({
      meshUrl: args[1] || process.env.MESH_URL,
      workerId: args[2] || process.env.NODE_ID
    });
    agent.start();
  } else if (command === 'stop') {
    // Would need to track PID for graceful stop
    console.log('Stop command not implemented in this example');
  } else {
    console.log(`
Usage:
  node tunnel/agent.js start [meshUrl] [workerId]   # Start tunnel agent
  node tunnel/agent.js stop                          # Stop tunnel agent
    `);
  }
}
