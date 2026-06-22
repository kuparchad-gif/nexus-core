// ============================================================================
// ADVANCED MESH OPERATIONS
// ============================================================================

import { NexusKV } from './kv-manager.js';
import { NetworkTransport } from './network-transport.js';

export class MeshNetwork {
    constructor(env) {
        this.env = env;
        this.kv = new NexusKV(env);
        this.transport = new NetworkTransport(env, env.NODE_ID);
        this.localCache = new Map();
        this.routingTable = new Map();
    }

    async findPath(sourceId, targetId) {
        const visited = new Set();
        const queue = [[sourceId, []]];
        
        while (queue.length > 0) {
            const [current, path] = queue.shift();
            
            if (current === targetId) {
                return { path: [...path, current], success: true };
            }
            
            if (visited.has(current)) continue;
            visited.add(current);
            
            const neighbors = await this.getNeighbors(current);
            for (const neighbor of neighbors) {
                if (!visited.has(neighbor)) {
                    queue.push([neighbor, [...path, current]]);
                }
            }
        }
        
        return { path: [], success: false };
    }

    async getNeighbors(workerId) {
        const worker = await this.kv.getWorker(workerId);
        if (!worker) return [];
        
        const allWorkers = await this.kv.getAllWorkers();
        const neighbors = allWorkers
            .filter(w => w.id !== workerId)
            .filter(w => this.canReach(worker, w))
            .map(w => w.id);
        
        return neighbors;
    }

    canReach(a, b) {
        const networksA = a.networks || ['legacy'];
        const networksB = b.networks || ['legacy'];
        return networksA.some(n => networksB.includes(n));
    }

    async getTopology() {
        const workers = await this.kv.getAllWorkers();
        const edges = [];
        
        for (const worker of workers) {
            const neighbors = await this.getNeighbors(worker.id);
            for (const neighbor of neighbors) {
                edges.push([worker.id, neighbor]);
            }
        }
        
        return {
            nodes: workers.map(w => ({ id: w.id, metadata: w.metadata })),
            edges: edges,
            timestamp: Date.now()
        };
    }

    async broadcast(message, ttl = 3) {
        const seen = new Set();
        const queue = [{ id: this.env.NODE_ID, ttl }];
        const results = [];
        
        while (queue.length > 0) {
            const { id, ttl: currentTTL } = queue.shift();
            
            if (seen.has(id) || currentTTL <= 0) continue;
            seen.add(id);
            
            const response = await this._sendToWorker(id, message);
            results.push({ workerId: id, response });
            
            const neighbors = await this.getNeighbors(id);
            for (const neighbor of neighbors) {
                if (!seen.has(neighbor)) {
                    queue.push({ id: neighbor, ttl: currentTTL - 1 });
                }
            }
        }
        
        return { 
            results,
            reached: results.length,
            seen: seen.size 
        };
    }

    async _sendToWorker(workerId, message) {
        try {
            const worker = await this.kv.getWorker(workerId);
            if (!worker) return { error: 'Worker not found' };
            
            const response = await fetch(`${worker.endpoint}/ask`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    question: typeof message === 'string' ? message : JSON.stringify(message) 
                })
            });
            
            if (response.ok) {
                return await response.json();
            }
            return { error: `HTTP ${response.status}` };
        } catch (error) {
            return { error: error.message };
        }
    }

    async healthCheck() {
        const workers = await this.kv.getAllWorkers();
        const results = [];
        
        for (const worker of workers) {
            try {
                const response = await fetch(`${worker.endpoint}/health`);
                const status = response.ok ? 'healthy' : 'unhealthy';
                results.push({ 
                    workerId: worker.id, 
                    status, 
                    statusCode: response.status 
                });
            } catch (_) {
                results.push({ 
                    workerId: worker.id, 
                    status: 'down', 
                    error: 'Connection failed' 
                });
            }
        }
        
        return {
            timestamp: Date.now(),
            total: results.length,
            healthy: results.filter(r => r.status === 'healthy').length,
            results
        };
    }

    async autoHeal() {
        const health = await this.healthCheck();
        const unhealthy = health.results.filter(r => r.status !== 'healthy');
        
        const actions = [];
        for (const worker of unhealthy) {
            try {
                const data = await this.kv.getWorker(worker.workerId);
                if (data) {
                    await this.kv.registerWorker({
                        ...data,
                        lastHeartbeat: Date.now(),
                        status: 'recovering'
                    });
                    actions.push({
                        workerId: worker.workerId,
                        action: 're-registered',
                        status: 'recovering'
                    });
                }
            } catch (_) {
                actions.push({
                    workerId: worker.workerId,
                    action: 'failed',
                    error: 'Could not recover'
                });
            }
        }
        
        return {
            healed: actions.length,
            actions,
            timestamp: Date.now()
        };
    }
}
