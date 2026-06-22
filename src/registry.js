// ============================================================================
// DURABLE OBJECTS — Registry & Mesh Coordinator & IoT Discovery
// ============================================================================

export class WorkerRegistry {
  constructor(state, env) {
    this.state = state;
    this.env = env;
  }

  async fetch(request) {
    const url = new URL(request.url);

    if (url.pathname === '/register') {
      const data = await request.json();
      await this.state.storage.put(`worker:${data.id}`, data);
      await this._rebuildIndex();
      return json({ success: true });
    }

    if (url.pathname === '/discover') {
      const { networkType } = await request.json().catch(() => ({}));
      const workers = await this.state.storage.get('index') || [];
      const filtered = networkType 
        ? workers.filter(w => (w.networks || []).includes(networkType))
        : workers;
      return json({ workers: filtered, count: filtered.length });
    }

    if (url.pathname === '/cleanup') {
      const workers = await this.state.storage.list({ prefix: 'worker:' });
      const now = Date.now();
      let cleaned = 0;

      for (const [key, worker] of workers) {
        if ((now - (worker.lastHeartbeat || 0)) > 300000) {
          await this.state.storage.delete(key);
          cleaned++;
        }
      }

      if (cleaned > 0) await this._rebuildIndex();
      return json({ cleaned, timestamp: now });
    }

    if (url.pathname === '/list') {
      const all = await this.state.storage.get('index') || [];
      return json({ workers: all });
    }

    return new Response('Not found', { status: 404 });
  }

  async _rebuildIndex() {
    const workers = await this.state.storage.list({ prefix: 'worker:' });
    const all = Array.from(workers.values());
    await this.state.storage.put('index', all);
  }
}

export class MeshCoordinator {
  constructor(state, env) {
    this.state = state;
    this.env = env;
  }

  async fetch(request) {
    const url = new URL(request.url);

    if (url.pathname === '/elect') {
      const current = await this.state.storage.get('leader');
      const term = (current?.term || 0) + 1;
      const leader = { 
        workerId: this.env.NODE_ID, 
        electedAt: Date.now(), 
        term,
        network: DEFAULT_NETWORK || 'pulse'
      };
      await this.state.storage.put('leader', leader);
      return json({ leader, previous: current });
    }

    if (url.pathname === '/leader') {
      const leader = await this.state.storage.get('leader');
      return json({ leader: leader || null });
    }

    if (url.pathname === '/topology') {
      const kv = new (await import('./kv-manager.js')).NexusKV(this.env);
      const workers = await kv.getAllWorkers();
      const leader = await this.state.storage.get('leader');
      
      const byNetwork = {};
      for (const w of workers) {
        for (const net of (w.networks || ['legacy'])) {
          if (!byNetwork[net]) byNetwork[net] = [];
          byNetwork[net].push(w.id);
        }
      }

      return json({
        node: this.env.NODE_ID,
        leader,
        total: workers.length,
        byNetwork,
        timestamp: Date.now()
      });
    }

    return json({ message: 'MeshCoordinator active', node: this.env.NODE_ID });
  }
}

export class IoTDiscoveryDO {
  constructor(state, env) {
    this.state = state;
    this.env = env;
    this.discoveredDevices = new Map();
  }

  async fetch(request) {
    const url = new URL(request.url);

    if (url.pathname === '/discover') {
      const devices = await this.state.storage.get('devices') || [];
      return json({ devices, count: devices.length });
    }

    if (url.pathname === '/register' && request.method === 'POST') {
      const data = await request.json();
      const devices = await this.state.storage.get('devices') || [];
      devices.push(data);
      await this.state.storage.put('devices', devices);
      return json({ success: true });
    }

    if (url.pathname === '/status') {
      const devices = await this.state.storage.get('devices') || [];
      return json({
        total: devices.length,
        active: devices.filter(d => (Date.now() - d.lastSeen) < 300000).length,
        timestamp: Date.now()
      });
    }

    return new Response('Not found', { status: 404 });
  }
}

function json(data, status = 200) {
  return new Response(JSON.stringify(data, null, 2), {
    status,
    headers: { 'Content-Type': 'application/json' }
  });
}
