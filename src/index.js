// ============================================================================
// NEXUS UNIVERSAL WORKER v6.0 — COMPLETE UNIFIED IMPLEMENTATION
// ============================================================================
// Includes: Network Intelligence, Service Bridges, Tunnel Agent Support
// Blockchain, IoT Discovery, Edge Intelligence
// ============================================================================

import { WorkerRegistry, MeshCoordinator, IoTDiscoveryDO } from './registry.js';
import { NexusKV } from './kv-manager.js';
import { NetworkTransport } from './network-transport.js';
import { ServiceBridgeRouter } from '../integrations/service-bridge.js';
import { MeshNetwork } from './mesh.js';
import { BlockchainRegistry } from '../blockchain/chain-registry.js';
import { IoTDiscovery } from '../iot/device-discovery.js';
import { EdgeIntelligence } from '../iot/edge-intelligence.js';
import {
  NETWORK_TYPES,
  DEFAULT_NETWORK,
  negotiateNetwork,
  negotiateCapAwareNetwork,
  getNetworkEndpoint,
  SERVICE_INTEGRATIONS
} from '../network/network-config.js';

export { WorkerRegistry, MeshCoordinator, IoTDiscoveryDO } from './registry.js';

// ============================================================================
// MAIN FETCH HANDLER
// ============================================================================

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    const path = url.pathname;
    const method = request.method;

    const kv = new NexusKV(env);
    const transport = new NetworkTransport(env, env.NODE_ID);
    const router = new ServiceBridgeRouter(env);
    const mesh = new MeshNetwork(env);
    const blockchain = new BlockchainRegistry(env);
    const iot = new IoTDiscovery(env);
    const edge = new EdgeIntelligence(env, iot);

    // ===== CORS =====
    if (method === 'OPTIONS') {
      return new Response(null, {
        headers: {
          'Access-Control-Allow-Origin': '*',
          'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
          'Access-Control-Allow-Headers': 'Content-Type, Authorization, X-Network-Type'
        }
      });
    }

    // ===== HEALTH =====
    if (path === '/health') {
      return json({
        status: 'ok',
        node: env.NODE_ID,
        env: env.ENV,
        network: DEFAULT_NETWORK,
        networks: Object.keys(NETWORK_TYPES),
        services: Object.keys(SERVICE_INTEGRATIONS || {}),
        blockchain: {
          solana: !!env.SOLANA_RPC,
          ethereum: !!env.ETH_RPC
        },
        iot: {
          enabled: iot.discoveryInterval !== null,
          devices: iot.discoveredDevices.size
        },
        version: '6.0.0',
        timestamp: Date.now()
      });
    }

    // ===== NETWORK STATUS =====
    if (path === '/network/status') {
      return json({
        node: env.NODE_ID,
        networks: NETWORK_TYPES,
        active: transport.getStatus(),
        default: DEFAULT_NETWORK,
        supported: Object.keys(NETWORK_TYPES)
      });
    }

    // ===== NETWORK NEGOTIATE =====
    if (path === '/network/negotiate' && method === 'POST') {
      const { capabilities, preferred, peer } = await request.json().catch(() => ({}));
      const chosen = peer 
        ? negotiateCapAwareNetwork(peer, capabilities || Object.keys(NETWORK_TYPES), preferred)
        : negotiateNetwork(capabilities || Object.keys(NETWORK_TYPES), preferred);
      
      return json({
        chosen,
        network: NETWORK_TYPES[chosen.toUpperCase()],
        timestamp: Date.now()
      });
    }

    // ===== MESH TOPOLOGY =====
    if (path === '/mesh/topology') {
      const topology = await mesh.getTopology();
      return json(topology);
    }

    if (path === '/mesh/health') {
      const health = await mesh.healthCheck();
      return json(health);
    }

    if (path === '/mesh/heal' && method === 'POST') {
      const result = await mesh.autoHeal();
      return json(result);
    }

    if (path === '/mesh/broadcast' && method === 'POST') {
      const { message, ttl = 3 } = await request.json().catch(() => ({}));
      if (!message) return json({ error: 'message required' }, 400);
      const result = await mesh.broadcast(message, ttl);
      return json(result);
    }

    // ===== BLOCKCHAIN ENDPOINTS =====
    if (path === '/blockchain/register' && method === 'POST') {
      const { deviceId, publicKey, capabilities, location } = await request.json().catch(() => ({}));
      if (!deviceId || !publicKey) return json({ error: 'deviceId and publicKey required' }, 400);
      
      const result = await blockchain.registerDeviceOnChain({
        deviceId,
        publicKey,
        capabilities: capabilities || [],
        location: location || 'unknown'
      });
      return json(result);
    }

    if (path === '/blockchain/verify') {
      const deviceId = url.searchParams.get('deviceId');
      if (!deviceId) return json({ error: 'deviceId required' }, 400);
      
      const result = await blockchain.verifyDeviceOnChain(deviceId);
      return json(result);
    }

    if (path === '/blockchain/reputation' && method === 'POST') {
      const { deviceId, action, score } = await request.json().catch(() => ({}));
      if (!deviceId) return json({ error: 'deviceId required' }, 400);
      
      const result = await blockchain.updateDeviceReputation(deviceId, action || 'unknown', score || 0.1);
      return json(result);
    }

    // ===== IoT ENDPOINTS =====
    if (path === '/iot/discover' && method === 'POST') {
      if (!iot.discoveryInterval) {
        await iot.startDiscovery();
      }
      return json({ 
        status: 'scanning',
        devices: iot.getDiscoveredDevices().length,
        active: iot.getActiveDevices().length,
        timestamp: Date.now()
      });
    }

    if (path === '/iot/devices') {
      const active = url.searchParams.get('active') === 'true';
      const devices = active ? iot.getActiveDevices() : iot.getDiscoveredDevices();
      return json({
        devices,
        count: devices.length,
        timestamp: Date.now()
      });
    }

    if (path === '/iot/device' && method === 'GET') {
      const deviceId = url.searchParams.get('deviceId');
      if (!deviceId) return json({ error: 'deviceId required' }, 400);
      
      const device = await iot.getDeviceInfo(deviceId);
      if (!device) return json({ error: 'Device not found' }, 404);
      return json(device);
    }

    if (path === '/iot/ingest' && method === 'POST') {
      const { deviceId, sensorData } = await request.json().catch(() => ({}));
      if (!deviceId || !sensorData) return json({ error: 'deviceId and sensorData required' }, 400);
      
      const result = await edge.ingestSensorData(deviceId, sensorData);
      return json(result);
    }

    // ===== EDGE INTELLIGENCE ENDPOINTS =====
    if (path === '/edge/federated' && method === 'POST') {
      const { modelId, deviceIds } = await request.json().catch(() => ({}));
      if (!modelId || !deviceIds) return json({ error: 'modelId and deviceIds required' }, 400);
      
      const result = await edge.federatedLearningRound(deviceIds, modelId);
      return json(result);
    }

    if (path === '/edge/fusion' && method === 'POST') {
      const { deviceIds, sensorTypes } = await request.json().catch(() => ({}));
      if (!deviceIds || !sensorTypes) return json({ error: 'deviceIds and sensorTypes required' }, 400);
      
      const result = await edge.sensorFusion(deviceIds, sensorTypes);
      return json(result);
    }

    if (path === '/edge/anomaly' && method === 'POST') {
      const { deviceId } = await request.json().catch(() => ({}));
      if (!deviceId) return json({ error: 'deviceId required' }, 400);
      
      const result = await edge.detectAnomalies(deviceId);
      return json(result);
    }

    if (path === '/edge/predict' && method === 'POST') {
      const { deviceId } = await request.json().catch(() => ({}));
      if (!deviceId) return json({ error: 'deviceId required' }, 400);
      
      const result = await edge.predictDeviceHealth(deviceId);
      return json(result);
    }

    // ===== PULSE ENDPOINT (WebSocket) =====
    if (path === '/pulse') {
      return handlePulseConnection(request, env, kv, transport);
    }

    // ===== PUBHUB ENDPOINT (SSE) =====
    if (path === '/pubhub/events') {
      return handlePubHubEvents(request, env, kv, transport);
    }

    // ===== LEGACY ENDPOINT (REST) =====
    if (path === '/legacy') {
      return handleLegacyRequest(request, env, kv, transport);
    }

    // ===== SERVICE BRIDGE ENDPOINTS =====
    if (path === '/bridge/deploy' && method === 'POST') {
      const { code, platforms = ['railway', 'vercel', 'netlify', 'render'] } = await request.json().catch(() => ({}));
      if (!code) return json({ error: 'code required' }, 400);
      
      const results = await router.deployToAll(code, platforms);
      return json({
        deployed: results,
        timestamp: Date.now(),
        note: 'Deployed to selected platforms'
      });
    }

    if (path === '/bridge/service' && method === 'POST') {
      const { service, endpoint, method: serviceMethod = 'GET', data = {} } = await request.json().catch(() => ({}));
      if (!service || !endpoint) return json({ error: 'service and endpoint required' }, 400);
      
      const result = await router.callService(service, endpoint, serviceMethod, data);
      return json(result);
    }

    if (path === '/bridge/github/workflow' && method === 'POST') {
      const { workflowId, inputs = {} } = await request.json().catch(() => ({}));
      if (!workflowId) return json({ error: 'workflowId required' }, 400);
      
      const result = await router.triggerGitHubWorkflow(workflowId, inputs);
      return json(result);
    }

    if (path === '/bridge/status') {
      return json({
        services: router.getStatus(),
        timestamp: Date.now()
      });
    }

    // ===== TUNNEL AGENT ENDPOINTS =====
    if (path === '/tunnel/register' && method === 'POST') {
      const data = await request.json().catch(() => ({}));
      const { workerId, endpoint, metadata = {} } = data;
      
      if (!workerId || !endpoint) {
        return json({ error: 'workerId and endpoint required' }, 400);
      }

      const registration = {
        id: workerId,
        name: `tunnel-${workerId}`,
        endpoint,
        networks: ['pulse', 'pubhub', 'legacy'],
        metadata: {
          ...metadata,
          type: 'tunnel-agent',
          registeredAt: Date.now(),
          registeredBy: env.NODE_ID
        },
        tags: ['tunnel', 'external'],
        lastHeartbeat: Date.now(),
        status: 'active'
      };

      await kv.registerWorker(registration);
      
      // Also register in DO for consistency
      const registryId = env.WORKER_REGISTRY.idFromName('global');
      const registry = env.WORKER_REGISTRY.get(registryId);
      await registry.fetch(new Request('http://internal/register', {
        method: 'POST',
        body: JSON.stringify(registration)
      }));

      return json({ success: true, registered: registration });
    }

    // ===== REGISTRY ENDPOINTS =====
    if (path === '/registry/discover') {
      return handleDiscovery(request, env, kv, transport);
    }

    if (path === '/registry/register') {
      return handleRegistration(request, env, kv, transport);
    }

    if (path === '/registry/heartbeat') {
      return handleHeartbeat(request, env, kv);
    }

    // ===== HYPERCORE PROXY =====
    if (path.startsWith('/hypercore/')) {
      return handleHypercoreProxy(request, env, kv, transport);
    }

    // ===== DASHBOARD =====
    if (path === '/' || path === '/dashboard') {
      return new Response(renderDashboard(env), {
        headers: { 'Content-Type': 'text/html' }
      });
    }

    // ===== DEFAULT: Mesh Coordination =====
    const id = env.MESH_COORDINATOR.idFromName('global');
    const coordinator = env.MESH_COORDINATOR.get(id);
    return coordinator.fetch(request);
  },

  // =========================================================================
  // CRON: Network cleanup & service sync
  // =========================================================================
  async scheduled(event, env, ctx) {
    const kv = new NexusKV(env);
    const transport = new NetworkTransport(env, env.NODE_ID);
    const router = new ServiceBridgeRouter(env);
    const mesh = new MeshNetwork(env);
    const iot = new IoTDiscovery(env);
    const edge = new EdgeIntelligence(env, iot);

    // Cleanup stale network connections
    transport.cleanup();

    // Broadcast network status
    await transport.publish('network:health', {
      node: env.NODE_ID,
      networks: Object.keys(NETWORK_TYPES),
      active: transport.getStatus(),
      timestamp: Date.now()
    });

    // Service bridge health check
    const serviceStatus = router.getStatus();
    await transport.publish('services:status', {
      node: env.NODE_ID,
      services: serviceStatus,
      timestamp: Date.now()
    });

    // Mesh health check and auto-heal
    const health = await mesh.healthCheck();
    if (health.healthy < health.total * 0.8) {
      await mesh.autoHeal();
    }

    // Registry cleanup
    const registryId = env.WORKER_REGISTRY.idFromName('global');
    const registry = env.WORKER_REGISTRY.get(registryId);
    await registry.fetch(new Request('http://internal/cleanup', {
      method: 'POST',
      body: JSON.stringify({ cron: event.cron })
    }));

    // IoT scan if enabled
    if (env.IOT_ENABLED !== 'false') {
      if (!iot.discoveryInterval) {
        await iot.startDiscovery();
      }
    }

    // Federated learning round
    if (env.FEDERATED_LEARNING_ENABLED !== 'false') {
      const activeDevices = iot.getActiveDevices();
      const deviceIds = activeDevices.map(d => d.id).slice(0, 10);
      if (deviceIds.length > 2) {
        await edge.federatedLearningRound(deviceIds, `model_${Date.now()}`);
      }
    }

    console.log(`[CRON] Network sync complete at ${new Date().toISOString()}`);
  }
};

// ============================================================================
// HANDLER FUNCTIONS
// ============================================================================

async function handlePulseConnection(request, env, kv, transport) {
  const upgradeHeader = request.headers.get('Upgrade');
  if (upgradeHeader !== 'websocket') {
    return new Response('WebSocket upgrade required', { status: 426 });
  }

  const [client, server] = Object.values(new WebSocketPair());
  server.accept();

  const workerId = env.NODE_ID;
  server.send(JSON.stringify({
    type: 'handshake_ack',
    workerId,
    network: 'pulse',
    version: '6.0.0',
    timestamp: Date.now()
  }));

  // Handle messages
  server.addEventListener('message', async (event) => {
    try {
      const data = JSON.parse(event.data);
      switch (data.type) {
        case 'ping':
          server.send(JSON.stringify({ type: 'pong', timestamp: Date.now() }));
          break;
        case 'message':
          await transport.publish(data.topic || 'pulse:direct', data.payload, {
            from: data.from || 'unknown'
          });
          break;
        case 'register':
          // Register a tunnel agent via WebSocket
          const registration = {
            workerId: data.workerId,
            endpoint: data.endpoint,
            metadata: data.metadata || {},
            networks: data.networks || ['pulse', 'pubhub', 'legacy']
          };
          await kv.registerWorker(registration);
          server.send(JSON.stringify({
            type: 'registered',
            workerId: data.workerId,
            timestamp: Date.now()
          }));
          break;
        default:
          break;
      }
    } catch (_) {
      server.send(JSON.stringify({ error: 'Invalid message format' }));
    }
  });

  // Heartbeat
  const heartbeat = setInterval(() => {
    try {
      server.send(JSON.stringify({
        type: 'heartbeat',
        timestamp: Date.now()
      }));
    } catch (_) {
      clearInterval(heartbeat);
    }
  }, 5000);

  server.addEventListener('close', () => {
    clearInterval(heartbeat);
  });

  return new Response(null, {
    status: 101,
    webSocket: client,
    headers: { 'X-Network-Type': 'pulse' }
  });
}

async function handlePubHubEvents(request, env, kv, transport) {
  const stream = new ReadableStream({
    start(controller) {
      const unsubscribe = transport.subscribe('*', (data) => {
        controller.enqueue(`data: ${JSON.stringify(data)}\n\n`);
      });

      controller.enqueue(`event: connected\ndata: ${JSON.stringify({
        workerId: env.NODE_ID,
        timestamp: Date.now()
      })}\n\n`);

      const heartbeat = setInterval(() => {
        controller.enqueue(`event: heartbeat\ndata: ${JSON.stringify({
          timestamp: Date.now()
        })}\n\n`);
      }, 15000);

      request.signal.addEventListener('abort', () => {
        clearInterval(heartbeat);
        unsubscribe();
        controller.close();
      });
    }
  });

  return new Response(stream, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive',
      'X-Network-Type': 'pubhub'
    }
  });
}

async function handleLegacyRequest(request, env, kv, transport) {
  const url = new URL(request.url);
  const method = request.method;

  if (method === 'GET' && url.pathname === '/legacy/status') {
    return json({
      mode: 'legacy',
      node: env.NODE_ID,
      networks: Object.keys(NETWORK_TYPES),
      active: transport.getStatus()
    });
  }

  if (method === 'POST' && url.pathname === '/legacy/message') {
    const body = await request.json().catch(() => ({}));
    const { topic, data } = body;

    if (topic) {
      await transport.publish(topic, data);
      return json({ success: true, delivered: true });
    }

    return json({ error: 'topic required' }, 400);
  }

  return json({
    legacy: true,
    message: 'Legacy endpoint active',
    available: ['GET /legacy/status', 'POST /legacy/message']
  });
}

async function handleDiscovery(request, env, kv, transport) {
  const { keyword = env.DISCOVERY_KEYWORD || 'universal', networkType } = await request.json().catch(() => ({}));

  const allWorkers = await kv.getAllWorkers();
  let matching = allWorkers.filter(w =>
    w.name?.toLowerCase().includes(keyword.toLowerCase()) ||
    w.id?.toLowerCase().includes(keyword.toLowerCase()) ||
    w.tags?.some(t => t.toLowerCase().includes(keyword.toLowerCase()))
  );

  // Filter by network type if specified
  if (networkType) {
    matching = matching.filter(w => (w.networks || []).includes(networkType));
  }

  const now = Date.now();
  const healthy = matching.filter(w => (now - (w.lastHeartbeat || 0)) < 120000);

  const enhanced = healthy.map(w => ({
    ...w,
    networks: w.networks || ['pulse', 'pubhub', 'legacy'],
    preferred: w.preferredNetwork || DEFAULT_NETWORK,
    endpoint: getNetworkEndpoint(w, w.preferredNetwork || DEFAULT_NETWORK)
  }));

  return json({
    keyword,
    networkType: networkType || 'all',
    found: enhanced.length,
    workers: enhanced,
    timestamp: now
  });
}

async function handleRegistration(request, env, kv, transport) {
  const body = await request.json().catch(() => ({}));
  const {
    workerId,
    name,
    endpoint,
    tags = [],
    metadata = {},
    networks = ['pulse', 'pubhub', 'legacy'],
    preferredNetwork = DEFAULT_NETWORK
  } = body;

  if (!workerId || !endpoint) {
    return json({ error: 'workerId and endpoint required' }, 400);
  }

  const registration = {
    id: workerId,
    name: name || workerId,
    endpoint,
    networks,
    preferredNetwork,
    tags: [...tags, 'auto-registered', `network:${preferredNetwork}`],
    metadata: {
      ...metadata,
      registeredAt: Date.now(),
      registeredBy: env.NODE_ID,
      env: env.ENV,
      networkTypes: networks
    },
    lastHeartbeat: Date.now(),
    status: 'active'
  };

  await kv.registerWorker(registration);

  // Register in Durable Object
  const registryId = env.WORKER_REGISTRY.idFromName('global');
  const registry = env.WORKER_REGISTRY.get(registryId);
  await registry.fetch(new Request('http://internal/register', {
    method: 'POST',
    body: JSON.stringify(registration)
  }));

  // Broadcast
  await transport.publish('network:registration', {
    workerId,
    networks,
    timestamp: Date.now()
  });

  return json({ success: true, registered: registration });
}

async function handleHeartbeat(request, env, kv) {
  const { workerId } = await request.json().catch(() => ({}));
  if (!workerId) return json({ error: 'workerId required' }, 400);

  const worker = await kv.getWorker(workerId);
  if (!worker) return json({ error: 'Worker not registered' }, 404);

  worker.lastHeartbeat = Date.now();
  worker.status = 'active';
  await kv.registerWorker(worker);

  return json({ success: true, acknowledged: true });
}

async function handleHypercoreProxy(request, env, kv, transport) {
  const url = new URL(request.url);
  const targetWorkerId = url.pathname.split('/')[2];

  if (!targetWorkerId) {
    return json({ error: 'Specify worker ID in path: /hypercore/{workerId}/...' }, 400);
  }

  const worker = await kv.getWorker(targetWorkerId);
  if (!worker || worker.status !== 'active') {
    return json({ error: 'Worker not found or inactive', workerId: targetWorkerId }, 404);
  }

  // Determine best network
  const preferred = worker.preferredNetwork || 'legacy';
  const targetUrl = new URL(url.pathname.replace(`/hypercore/${targetWorkerId}`, ''), worker.endpoint);
  targetUrl.search = url.search;

  const proxyReq = new Request(targetUrl, {
    method: request.method,
    headers: request.headers,
    body: request.body
  });

  proxyReq.headers.set('X-Nexus-Gateway', env.NODE_ID);
  proxyReq.headers.set('X-Nexus-Proxy-To', targetWorkerId);
  proxyReq.headers.set('X-Network-Type', preferred);

  return fetch(proxyReq);
}

// ============================================================================
// DASHBOARD RENDERER
// ============================================================================

function renderDashboard(env) {
  return `<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>NEXUS v6.0 — Universal Mesh</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { background: #0a0a12; color: #d0d0f0; font-family: monospace; padding: 20px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
    .card { background: #141424; border: 1px solid #2a2a48; padding: 18px; border-radius: 8px; }
    .value { font-size: 1.8rem; color: #00ffcc; }
    .label { color: #88aadd; font-size: 0.8rem; }
    button { background: #3a3af8; color: white; border: none; padding: 10px 16px; border-radius: 4px; cursor: pointer; font-family: monospace; }
    button:hover { background: #5050ff; }
    .badge { display: inline-block; padding: 4px 10px; border-radius: 12px; font-size: 0.7rem; margin: 2px; }
    .badge.pulse { background: #ff0044; }
    .badge.pubhub { background: #ff8800; }
    .badge.legacy { background: #4488ff; }
    .badge.tcpip { background: #8844ff; }
    .badge.iot { background: #00aa66; }
    .badge.blockchain { background: #ff6600; }
    pre { background: #0a0a12; padding: 12px; border-radius: 4px; border: 1px solid #1a1a2e; overflow: auto; font-size: 0.7rem; max-height: 300px; }
  </style>
</head>
<body>
  <h1>🌀 NEXUS v6.0 — Complete Universal Mesh</h1>
  <p style="color: #88aadd; margin-bottom: 20px;">Network Intelligence • Service Bridges • Blockchain • IoT • Edge Intelligence</p>

  <div class="grid" id="statusGrid">
    <div class="card">
      <div class="label">Node</div>
      <div class="value" id="node">${env.NODE_ID || 'unknown'}</div>
    </div>
    <div class="card">
      <div class="label">Environment</div>
      <div class="value" id="env">${env.ENV || 'production'}</div>
    </div>
    <div class="card">
      <div class="label">Default Network</div>
      <div class="value" id="network">${DEFAULT_NETWORK}</div>
    </div>
  </div>

  <div class="grid" id="metricsGrid">
    <div class="card">
      <div class="label">Workers</div>
      <div class="value" id="workers">...</div>
    </div>
    <div class="card">
      <div class="label">Active Networks</div>
      <div class="value" id="networks">...</div>
    </div>
    <div class="card">
      <div class="label">Service Bridges</div>
      <div class="value" id="bridges">...</div>
    </div>
    <div class="card">
      <div class="label">IoT Devices</div>
      <div class="value" id="iot">...</div>
    </div>
  </div>

  <div class="grid">
    <div class="card">
      <h3>🔗 Network Types</h3>
      <div>
        <span class="badge pulse">PULSE</span>
        <span class="badge pubhub">PUBHUB</span>
        <span class="badge legacy">LEGACY</span>
        <span class="badge tcpip">TCP/IP</span>
      </div>
      <p style="color:#88aadd;font-size:0.8rem;margin-top:8px;">
        ${Object.keys(NETWORK_TYPES).join(' • ')}
      </p>
    </div>
    <div class="card">
      <h3>📦 Service Integrations</h3>
      <div style="display:flex;flex-wrap:wrap;gap:4px;margin-top:8px;">
        ${Object.keys(SERVICE_INTEGRATIONS || {}).map(s => `<span class="badge" style="background:#222244;">${s}</span>`).join('')}
      </div>
    </div>
    <div class="card">
      <h3>⛓️ Blockchain</h3>
      <div>
        <span class="badge blockchain">Solana</span>
        <span class="badge blockchain">Ethereum</span>
        <span class="badge blockchain">IPFS</span>
      </div>
    </div>
    <div class="card">
      <h3>📡 IoT Protocols</h3>
      <div>
        <span class="badge iot">mDNS</span>
        <span class="badge iot">BLE</span>
        <span class="badge iot">LoRaWAN</span>
        <span class="badge iot">MQTT</span>
        <span class="badge iot">Zigbee</span>
      </div>
    </div>
    <div class="card">
      <h3>⚡ Quick Actions</h3>
      <button onclick="refreshStatus()">🔄 Refresh</button>
      <button onclick="deployAll()">🚀 Deploy All</button>
      <button onclick="checkHealth()">💚 Health Check</button>
      <button onclick="discoverIoT()">📡 Discover IoT</button>
    </div>
  </div>

  <div class="card">
    <h3>📋 Detailed Status</h3>
    <pre id="details">Loading...</pre>
  </div>

  <script>
    async function refreshStatus() {
      try {
        const res = await fetch('/health');
        const data = await res.json();
        document.getElementById('details').textContent = JSON.stringify(data, null, 2);
        document.getElementById('workers').textContent = data.workers || '?';
        document.getElementById('networks').textContent = data.networks?.length || '?';
        document.getElementById('iot').textContent = data.iot?.devices || '?';
        
        const bridgeRes = await fetch('/bridge/status');
        const bridgeData = await bridgeRes.json();
        document.getElementById('bridges').textContent = Object.keys(bridgeData.services || {}).length;
      } catch(e) {
        document.getElementById('details').textContent = 'Error: ' + e.message;
      }
    }

    async function deployAll() {
      const code = 'export default { fetch() { return new Response("Hello from NEXUS v6.0!") } }';
      const res = await fetch('/bridge/deploy', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code, platforms: ['railway', 'vercel', 'netlify'] })
      });
      const data = await res.json();
      document.getElementById('details').textContent = JSON.stringify(data, null, 2);
    }

    async function checkHealth() {
      const res = await fetch('/network/status');
      const data = await res.json();
      document.getElementById('details').textContent = JSON.stringify(data, null, 2);
    }

    async function discoverIoT() {
      const res = await fetch('/iot/discover', { method: 'POST' });
      const data = await res.json();
      document.getElementById('details').textContent = JSON.stringify(data, null, 2);
    }

    refreshStatus();
    setInterval(refreshStatus, 10000);
  </script>
</body>
</html>`;
}

function json(data, status = 200) {
  return new Response(JSON.stringify(data, null, 2), {
    status,
    headers: { 'Content-Type': 'application/json' }
  });
}
