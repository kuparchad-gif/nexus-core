// ============================================================================
// NEXUS NETWORK TRANSPORT LAYER
// ============================================================================

import { NETWORK_TYPES, negotiateNetwork, getNetworkEndpoint } from '../network/network-config.js';

export class NetworkTransport {
  constructor(env, workerId) {
    this.env = env;
    this.workerId = workerId;
    this.activeNetworks = new Map();
    this.sockets = new Map();
    this.subscribers = new Map();
    this.healthCache = new Map();
  }

  async connect(peer, preferredNetwork = null) {
    const available = peer.networks || ['pulse', 'pubhub', 'legacy', 'tcpip'];
    const chosen = preferredNetwork || negotiateNetwork(available);

    const network = NETWORK_TYPES[chosen.toUpperCase()];
    if (!network) throw new Error(`Unknown network type: ${chosen}`);

    const endpoint = getNetworkEndpoint(peer, chosen);
    if (!endpoint) throw new Error(`Cannot construct endpoint for ${chosen}`);

    if (this.sockets.has(endpoint)) return this.sockets.get(endpoint);

    let connection;
    switch (network.protocol) {
      case 'websocket':
        connection = await this._connectWebSocket(endpoint, peer, network);
        break;
      case 'sse':
        connection = await this._connectSSE(endpoint, peer, network);
        break;
      case 'http':
        connection = await this._connectHTTP(endpoint, peer, network);
        break;
      case 'tcp':
        connection = await this._connectTCP(endpoint, peer, network);
        break;
      default:
        throw new Error(`Unsupported protocol: ${network.protocol}`);
    }

    this.sockets.set(endpoint, connection);
    this.activeNetworks.set(peer.id, chosen);
    this._startHeartbeat(connection, network);

    return connection;
  }

  async _connectWebSocket(endpoint, peer, network) {
    const ws = new WebSocket(endpoint);

    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error(`WebSocket connection timeout to ${endpoint}`));
      }, network.timeout);

      ws.onopen = () => {
        clearTimeout(timeout);
        ws.send(JSON.stringify({
          type: 'handshake',
          workerId: this.workerId,
          networks: Array.from(this.activeNetworks.keys()),
          timestamp: Date.now()
        }));
        resolve(ws);
      };

      ws.onerror = (error) => {
        clearTimeout(timeout);
        reject(error);
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === 'heartbeat') {
            ws.send(JSON.stringify({
              type: 'heartbeat_ack',
              timestamp: Date.now()
            }));
          }
          this._dispatch(data);
        } catch (_) {}
      };
    });
  }

  async _connectSSE(endpoint, peer, network) {
    const eventSource = new EventSource(endpoint);

    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error(`SSE connection timeout to ${endpoint}`));
      }, network.timeout);

      eventSource.onopen = () => {
        clearTimeout(timeout);
        resolve(eventSource);
      };

      eventSource.onerror = (error) => {
        clearTimeout(timeout);
        reject(error);
      };

      eventSource.addEventListener('message', (event) => {
        try {
          const data = JSON.parse(event.data);
          this._dispatch(data);
        } catch (_) {}
      });

      eventSource.addEventListener('topic', (event) => {
        try {
          const { topic, data } = JSON.parse(event.data);
          this._dispatchTopic(topic, data);
        } catch (_) {}
      });
    });
  }

  async _connectHTTP(endpoint, peer, network) {
    return {
      endpoint,
      type: 'http',
      send: async (data) => {
        const response = await fetch(`${endpoint}/legacy/message`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            ...data,
            from: this.workerId,
            timestamp: Date.now()
          })
        });
        return response.json();
      },
      close: () => {}
    };
  }

  async _connectTCP(endpoint, peer, network) {
    const wsEndpoint = endpoint.replace('tcp://', 'ws://');
    return this._connectWebSocket(wsEndpoint, peer, network);
  }

  _startHeartbeat(connection, network) {
    if (connection.type === 'http') return;

    const interval = setInterval(() => {
      try {
        if (connection.readyState === WebSocket.OPEN) {
          connection.send(JSON.stringify({
            type: 'heartbeat',
            timestamp: Date.now(),
            workerId: this.workerId
          }));
        } else {
          clearInterval(interval);
          this._reconnect(connection);
        }
      } catch (_) {
        clearInterval(interval);
      }
    }, network.heartbeatInterval);

    connection._heartbeatInterval = interval;
  }

  subscribe(topic, callback) {
    if (!this.subscribers.has(topic)) {
      this.subscribers.set(topic, new Set());
    }
    this.subscribers.get(topic).add(callback);
    return () => this.subscribers.get(topic)?.delete(callback);
  }

  async publish(topic, data, options = {}) {
    const message = {
      topic,
      data,
      from: this.workerId,
      timestamp: Date.now(),
      ...options
    };

    const promises = [];
    for (const [endpoint, conn] of this.sockets) {
      if (conn.type === 'http') {
        promises.push(conn.send(message));
      } else if (conn.readyState === WebSocket.OPEN) {
        conn.send(JSON.stringify({
          type: 'publish',
          ...message
        }));
      }
    }

    if (promises.length > 0) {
      await Promise.allSettled(promises);
    }

    this._dispatchTopic(topic, data);
    return { delivered: promises.length };
  }

  _dispatch(data) {
    if (data.type === 'publish') {
      this._dispatchTopic(data.topic, data.data);
    }
  }

  _dispatchTopic(topic, data) {
    const callbacks = this.subscribers.get(topic);
    if (callbacks) {
      for (const callback of callbacks) {
        try { callback(data); } catch (_) {}
      }
    }
  }

  _reconnect(connection) {
    const endpoint = connection.endpoint || connection.url;
    if (endpoint) {
      setTimeout(async () => {
        try {
          await this.connect({ id: 'reconnect', url: endpoint });
        } catch (_) {}
      }, 1000);
    }
  }

  getStatus() {
    const status = {
      activeNetworks: Array.from(this.activeNetworks.entries()),
      connections: this.sockets.size,
      subscribers: this.subscribers.size,
      health: {}
    };

    for (const [network, conn] of this.activeNetworks) {
      status.health[network] = {
        connected: this.sockets.has(conn),
        type: NETWORK_TYPES[network.toUpperCase()]?.protocol || 'unknown'
      };
    }

    return status;
  }

  cleanup() {
    for (const [endpoint, conn] of this.sockets) {
      try {
        if (conn._heartbeatInterval) clearInterval(conn._heartbeatInterval);
        if (conn.close) conn.close();
      } catch (_) {}
    }
    this.sockets.clear();
    this.activeNetworks.clear();
  }
}
