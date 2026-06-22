// ============================================================================
// NEXUS NETWORK CONFIGURATION v6.0
// ============================================================================
// Network Types: Pulse, PubHub, Legacy, TCP/IP
// Service Integrations: GitHub, GitLab, Railway, Vercel, Netlify, Render
// ============================================================================

export const NETWORK_TYPES = {
  PULSE: {
    id: 'pulse',
    name: 'Pulse Network',
    description: 'High-frequency, low-latency for real-time consciousness',
    port: 8080,
    protocol: 'websocket',
    priority: 1,
    features: ['realtime', 'bidirectional', 'low-latency'],
    heartbeatInterval: 1000,
    timeout: 5000,
    maxPayload: 1024 * 64,
  },
  PUBHUB: {
    id: 'pubhub',
    name: 'Publish/Subscribe Hub',
    description: 'Event-driven broadcast mesh for system-wide messages',
    port: 8081,
    protocol: 'sse',
    priority: 2,
    features: ['broadcast', 'events', 'topics'],
    heartbeatInterval: 5000,
    timeout: 10000,
    maxPayload: 1024 * 1024,
  },
  LEGACY: {
    id: 'legacy',
    name: 'Legacy REST',
    description: 'REST/HTTP fallback for compatibility',
    port: 8082,
    protocol: 'http',
    priority: 3,
    features: ['rest', 'compatibility', 'standard'],
    heartbeatInterval: 15000,
    timeout: 30000,
    maxPayload: 1024 * 1024 * 10,
  },
  TCPIP: {
    id: 'tcpip',
    name: 'TCP/IP Direct',
    description: 'Direct socket communication for raw performance',
    port: 8083,
    protocol: 'tcp',
    priority: 4,
    features: ['raw', 'high-performance', 'streaming'],
    heartbeatInterval: 1000,
    timeout: 3000,
    maxPayload: 1024 * 1024 * 100,
  }
};

export const SERVICE_INTEGRATIONS = {
  GITHUB: {
    id: 'github',
    type: 'git',
    endpoints: {
      api: 'https://api.github.com',
      raw: 'https://raw.githubusercontent.com',
      webhook: 'https://api.github.com/webhooks'
    },
    features: ['repo_access', 'webhooks', 'actions', 'secrets'],
    rateLimit: { requests: 5000, window: 3600 },
    executionType: 'ephemeral'
  },
  GITLAB: {
    id: 'gitlab',
    type: 'git',
    endpoints: {
      api: 'https://gitlab.com/api/v4',
      webhook: 'https://gitlab.com/webhooks'
    },
    features: ['repo_access', 'webhooks', 'ci_cd'],
    rateLimit: { requests: 2000, window: 3600 },
    executionType: 'ephemeral'
  },
  RAILWAY: {
    id: 'railway',
    type: 'hosting',
    endpoints: {
      api: 'https://api.railway.app',
      deploy: 'https://railway.app/deploy',
      logs: 'https://railway.app/logs'
    },
    features: ['deploy', 'logs', 'env_vars', 'scaling'],
    executionType: 'persistent',
    freeTier: { hours: 500, monthly: true }
  },
  VERCEL: {
    id: 'vercel',
    type: 'hosting',
    endpoints: {
      api: 'https://api.vercel.com',
      deploy: 'https://vercel.com/api/deployments',
      logs: 'https://vercel.com/api/logs'
    },
    features: ['deploy', 'logs', 'env_vars', 'serverless'],
    executionType: 'ephemeral',
    freeTier: { requests: 100000, monthly: true }
  },
  NETLIFY: {
    id: 'netlify',
    type: 'hosting',
    endpoints: {
      api: 'https://api.netlify.com',
      deploy: 'https://api.netlify.com/api/v1/deploys',
      functions: 'https://api.netlify.com/api/v1/functions'
    },
    features: ['deploy', 'functions', 'forms', 'identity'],
    executionType: 'ephemeral',
    freeTier: { requests: 125000, monthly: true }
  },
  RENDER: {
    id: 'render',
    type: 'hosting',
    endpoints: {
      api: 'https://api.render.com/v1',
      deploy: 'https://api.render.com/v1/deploys',
      logs: 'https://api.render.com/v1/logs'
    },
    features: ['deploy', 'logs', 'env_vars', 'cron_jobs'],
    executionType: 'persistent',
    freeTier: { hours: 750, monthly: true }
  },
  REPLICATE: {
    id: 'replicate',
    type: 'ai',
    endpoints: {
      api: 'https://api.replicate.com/v1',
      predictions: 'https://api.replicate.com/v1/predictions',
      models: 'https://api.replicate.com/v1/models'
    },
    features: ['inference', 'models', 'training'],
    rateLimit: { requests: 100, window: 60 },
    executionType: 'ephemeral'
  },
  HUGGINGFACE: {
    id: 'huggingface',
    type: 'ai',
    endpoints: {
      api: 'https://api-inference.huggingface.co/models',
      hub: 'https://huggingface.co/api',
      spaces: 'https://huggingface.co/api/spaces'
    },
    features: ['inference', 'models', 'datasets'],
    rateLimit: { requests: 50, window: 60 },
    executionType: 'ephemeral'
  },
  PLANETSCALE: {
    id: 'planetscale',
    type: 'database',
    endpoints: {
      api: 'https://api.planetscale.com/v1',
      db: 'https://db.planetscale.com'
    },
    features: ['mysql', 'postgres', 'branching'],
    executionType: 'persistent',
    freeTier: { storage: 5, monthly: true }
  },
  SUPABASE: {
    id: 'supabase',
    type: 'database',
    endpoints: {
      api: 'https://api.supabase.com/v1',
      db: 'https://db.supabase.co'
    },
    features: ['postgres', 'auth', 'storage', 'realtime'],
    executionType: 'persistent',
    freeTier: { storage: 500, monthly: true }
  }
};

export const DEFAULT_NETWORK = process.env.DEFAULT_NETWORK || 'pulse';
export const SUPPORTED_NETWORKS = Object.keys(NETWORK_TYPES);

// ============================================================================
// NETWORK NEGOTIATION WITH CAP AWARENESS
// ============================================================================

export function negotiateNetwork(capabilities, preferred = DEFAULT_NETWORK) {
  if (capabilities.includes(preferred) && NETWORK_TYPES[preferred.toUpperCase()]) {
    return preferred;
  }

  const available = capabilities
    .map(n => n.toLowerCase())
    .filter(n => NETWORK_TYPES[n.toUpperCase()]);

  if (available.length === 0) return 'legacy';
  
  available.sort((a, b) => 
    NETWORK_TYPES[a.toUpperCase()].priority - NETWORK_TYPES[b.toUpperCase()].priority
  );

  return available[0];
}

export function negotiateCapAwareNetwork(peer, availableNetworks, preferred = 'pulse') {
  const metadata = peer.metadata || {};
  
  // Cap Preservation Engine
  if (metadata.billingCapAware && metadata.currentUsageHours && metadata.monthlyLimitHours) {
    const ratio = metadata.currentUsageHours / metadata.monthlyLimitHours;
    if (ratio > 0.90) {
      console.warn(`[METRIC] Node ${peer.id} at ${(ratio * 100).toFixed(1)}% usage. Downshifting.`);
      return 'legacy';
    }
    if (ratio > 0.75) {
      return 'pubhub'; // Reduce frequency but keep active
    }
  }
  
  // Check provider execution type
  if (metadata.provider && SERVICE_INTEGRATIONS[metadata.provider.toUpperCase()]) {
    const provider = SERVICE_INTEGRATIONS[metadata.provider.toUpperCase()];
    if (provider.executionType === 'ephemeral') {
      // Ephemeral nodes should use HTTP-based protocols
      const httpNetworks = availableNetworks.filter(n => ['legacy', 'pubhub'].includes(n));
      if (httpNetworks.length > 0) return httpNetworks[0];
    }
  }
  
  return negotiateNetwork(availableNetworks, preferred);
}

export function getNetworkEndpoint(worker, networkType = DEFAULT_NETWORK) {
  const network = NETWORK_TYPES[networkType.toUpperCase()];
  if (!network) return null;

  const baseUrl = worker.url || worker.endpoint || worker.metadata?.endpoint;
  if (!baseUrl) return null;

  try {
    const url = new URL(baseUrl);
    switch (networkType.toLowerCase()) {
      case 'pulse':
        return `ws://${url.host}/pulse`;
      case 'pubhub':
        return `${baseUrl}/pubhub/events`;
      case 'legacy':
        return `${baseUrl}/legacy`;
      case 'tcpip':
        return `tcp://${url.host}:${network.port}`;
      default:
        return baseUrl;
    }
  } catch (_) {
    return null;
  }
}

export async function checkNetworkHealth(endpoint, networkType = DEFAULT_NETWORK) {
  const network = NETWORK_TYPES[networkType.toUpperCase()];
  if (!network) return { healthy: false, error: 'Unknown network type' };

  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), network.timeout);

    const response = await fetch(`${endpoint}/health`, {
      signal: controller.signal,
      headers: { 'X-Network-Type': networkType }
    });

    clearTimeout(timeout);
    return { healthy: response.ok, status: response.status };
  } catch (error) {
    return { healthy: false, error: error.message };
  }
}
