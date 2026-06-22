// ============================================================================
// UNIFIED KV MANAGER — One KV to rule them all
// ============================================================================

export class NexusKV {
  constructor(env) {
    if (!env?.NEXUS_KV) {
      throw new Error("NexusKV: NEXUS_KV binding not found");
    }
    this.kv = env.NEXUS_KV;
    this.localCache = new Map();
  }

  async registerWorker(data) {
    const id = data.id || data.workerId;
    if (!id) throw new Error("registerWorker: missing id");
    await this._set(`registry:workers:${id}`, data);
    await this._updateWorkerList();
    return true;
  }

  async getWorker(workerId) {
    return this._get(`registry:workers:${workerId}`);
  }

  async getAllWorkers() {
    return this._get('registry:workers:all') ?? [];
  }

  async _updateWorkerList() {
    const list = await this.kv.list({ prefix: 'registry:workers:' });
    const workers = [];
    for (const key of list.keys) {
      if (key.name === 'registry:workers:all') continue;
      const data = await this.kv.get(key.name);
      if (!data) continue;
      try { workers.push(JSON.parse(data)); } catch (_) {}
    }
    workers.sort((a, b) => (b.lastHeartbeat ?? 0) - (a.lastHeartbeat ?? 0));
    await this._set('registry:workers:all', workers);
    return workers;
  }

  async removeWorker(workerId) {
    await this.kv.delete(`registry:workers:${workerId}`);
    this.localCache.delete(`registry:workers:${workerId}`);
    return this._updateWorkerList();
  }

  async getConfig(key) { return this._get(`config:${key}`); }
  async setConfig(key, value) { return this._set(`config:${key}`, value); }

  async incrementMetric(name, amount = 1) {
    const current = (await this._get(`metrics:${name}`)) ?? 0;
    await this._set(`metrics:${name}`, current + amount);
    return current + amount;
  }

  async _get(key) {
    try {
      const data = await this.kv.get(key);
      return data ? JSON.parse(data) : null;
    } catch (_) { return null; }
  }

  async _set(key, data, ttl = 0) {
    try {
      const opts = ttl > 0 ? { expirationTtl: ttl } : {};
      await this.kv.put(key, JSON.stringify(data), opts);
      this.localCache.delete(key);
      return true;
    } catch (_) { return false; }
  }

  async delete(key) {
    try {
      await this.kv.delete(key);
      this.localCache.delete(key);
      return true;
    } catch (_) { return false; }
  }

  async list(prefix) {
    try { return (await this.kv.list({ prefix })).keys; }
    catch (_) { return []; }
  }
}
