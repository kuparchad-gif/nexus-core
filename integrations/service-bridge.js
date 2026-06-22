// ============================================================================
// NEXUS SERVICE BRIDGE — Connect to external platforms
// ============================================================================
// Supports: GitHub, GitLab, Railway, Vercel, Netlify, Render
// ============================================================================

import { SERVICE_INTEGRATIONS, negotiateCapAwareNetwork } from '../network/network-config.js';

export class ServiceBridge {
  constructor(env, serviceId) {
    const config = SERVICE_INTEGRATIONS[serviceId.toUpperCase()];
    if (!config) throw new Error(`Unknown service: ${serviceId}`);
    
    this.serviceId = serviceId.toLowerCase();
    this.config = config;
    this.env = env;
    this.token = env[`${serviceId.toUpperCase()}_TOKEN`] || env[`${serviceId.toUpperCase()}_API_KEY`];
    this.tokens = new Map();
    this.activeConnections = new Map();
    this.usageStats = {
      requests: 0,
      startTime: Date.now(),
      resetTime: Date.now() + 3600000
    };
  }

  // ==========================================================================
  // API CALL — Authenticated request to service
  // ==========================================================================

  async call(endpoint, method = 'GET', data = null, options = {}) {
    await this._checkRateLimit();
    await this._checkFreeTier();

    const url = this._buildUrl(endpoint);
    const headers = await this._buildHeaders();

    const response = await fetch(url, {
      method,
      headers,
      body: data ? JSON.stringify(data) : null,
      ...options
    });

    // Handle rate limiting
    if (response.status === 429) {
      const retryAfter = parseInt(response.headers.get('Retry-After') || '60');
      await this._wait(retryAfter);
      return this.call(endpoint, method, data, options);
    }

    this.usageStats.requests++;
    const result = await response.json().catch(() => response.text());

    return {
      success: response.ok,
      status: response.status,
      data: result,
      headers: response.headers
    };
  }

  // ==========================================================================
  // DEPLOY — Deploy to hosting platforms
  // ==========================================================================

  async deploy(code, options = {}) {
    const { name = 'nexus-deploy', env = {}, region = 'auto' } = options;

    switch (this.serviceId) {
      case 'railway':
        return this._deployRailway(code, { name, env });
      case 'vercel':
        return this._deployVercel(code, { name, env });
      case 'netlify':
        return this._deployNetlify(code, { name, env });
      case 'render':
        return this._deployRender(code, { name, env });
      default:
        throw new Error(`Deploy not supported for ${this.serviceId}`);
    }
  }

  async _deployRailway(code, options) {
    const result = await this.call('/deploy', 'POST', {
      code: Buffer.from(code).toString('base64'),
      name: options.name,
      env: options.env
    });

    return {
      success: result.success,
      url: result.data?.url || 'https://railway.app/project',
      deploymentId: result.data?.id,
      provider: 'railway'
    };
  }

  async _deployVercel(code, options) {
    const result = await this.call('/api/deployments', 'POST', {
      name: options.name,
      env: options.env,
      files: [{
        file: 'index.js',
        data: Buffer.from(code).toString('base64')
      }]
    });

    return {
      success: result.success,
      url: result.data?.url || 'https://vercel.com/deployment',
      deploymentId: result.data?.id,
      provider: 'vercel'
    };
  }

  async _deployNetlify(code, options) {
    const result = await this.call('/api/v1/deploys', 'POST', {
      name: options.name,
      env: options.env,
      functions: { 'nexus': code }
    });

    return {
      success: result.success,
      url: result.data?.url || 'https://netlify.app/site',
      deploymentId: result.data?.id,
      provider: 'netlify'
    };
  }

  async _deployRender(code, options) {
    const result = await this.call('/v1/deploys', 'POST', {
      code: Buffer.from(code).toString('base64'),
      name: options.name,
      env: options.env
    });

    return {
      success: result.success,
      url: result.data?.url || 'https://render.com/deployment',
      deploymentId: result.data?.id,
      provider: 'render'
    };
  }

  // ==========================================================================
  // GIT OPERATIONS — GitHub/GitLab
  // ==========================================================================

  async gitOperation(operation, repo, data = {}) {
    switch (operation) {
      case 'clone':
        return this._cloneRepo(repo);
      case 'push':
        return this._pushRepo(repo, data);
      case 'create_issue':
        return this._createIssue(repo, data);
      case 'create_pr':
        return this._createPR(repo, data);
      case 'get_file':
        return this._getFile(repo, data.path);
      default:
        throw new Error(`Unknown git operation: ${operation}`);
    }
  }

  async _cloneRepo(repo) {
    const result = await this.call(`/repos/${repo}/contents`, 'GET');
    return { success: result.success, files: result.data };
  }

  async _getFile(repo, path) {
    const result = await this.call(`/repos/${repo}/contents/${path}`, 'GET');
    if (result.success && result.data?.content) {
      return {
        success: true,
        content: Buffer.from(result.data.content, 'base64').toString('utf-8'),
        sha: result.data.sha
      };
    }
    return { success: false };
  }

  async _createIssue(repo, data) {
    const result = await this.call(`/repos/${repo}/issues`, 'POST', {
      title: data.title,
      body: data.body,
      labels: data.labels || ['auto-generated']
    });
    return result;
  }

  async _createPR(repo, data) {
    const result = await this.call(`/repos/${repo}/pulls`, 'POST', {
      title: data.title,
      body: data.body,
      head: data.head,
      base: data.base || 'main'
    });
    return result;
  }

  // ==========================================================================
  // AI INFERENCE — Replicate, HuggingFace
  // ==========================================================================

  async aiInference(model, input, options = {}) {
    const { maxTokens = 1024, temperature = 0.7 } = options;

    let result;
    switch (this.serviceId) {
      case 'replicate':
        result = await this.call('/predictions', 'POST', {
          model,
          input: { ...input, max_tokens: maxTokens, temperature },
          webhook: options.webhook
        });
        break;
      case 'huggingface':
        result = await this.call(`/${model}`, 'POST', input);
        break;
      default:
        throw new Error(`AI inference not supported for ${this.serviceId}`);
    }

    return result;
  }

  // ==========================================================================
  // WEBHOOK — Register and receive webhooks
  // ==========================================================================

  async registerWebhook(options = {}) {
    const { events = ['*'], target = this.env.WORKER_URL, secret = this.env.WEBHOOK_SECRET } = options;

    const result = await this.call('/webhooks', 'POST', {
      url: target,
      events,
      secret,
      active: true
    });

    if (result.success) {
      const webhookId = result.data?.id;
      return { success: true, webhookId, events };
    }

    return { success: false, error: result.data };
  }

  // ==========================================================================
  // FREE TIER TRACKING
  // ==========================================================================

  async _checkFreeTier() {
    if (!this.config.freeTier) return;

    const { hours, requests } = this.config.freeTier;
    if (requests) {
      const totalRequests = this.usageStats.requests;
      if (totalRequests > requests * 0.9) {
        console.warn(`[${this.serviceId}] Approaching free tier limit: ${totalRequests}/${requests}`);
      }
      if (totalRequests > requests) {
        throw new Error(`Free tier exceeded for ${this.serviceId}`);
      }
    }
  }

  async _checkRateLimit() {
    const key = this.serviceId;
    const config = this.config;
    const now = Date.now();

    if (!this.tokens.has(key)) {
      this.tokens.set(key, {
        count: 0,
        windowStart: now,
        limit: config.rateLimit?.requests || 100,
        window: config.rateLimit?.window || 60
      });
    }

    const token = this.tokens.get(key);
    const windowMs = token.window * 1000;

    if (now - token.windowStart > windowMs) {
      token.count = 0;
      token.windowStart = now;
    }

    if (token.count >= token.limit) {
      const waitTime = (token.windowStart + windowMs) - now;
      throw new Error(`Rate limited for ${this.serviceId}. Wait ${Math.ceil(waitTime / 1000)}s`);
    }

    token.count++;
    this.tokens.set(key, token);
  }

  // ==========================================================================
  // UTILITY
  // ==========================================================================

  _buildUrl(endpoint) {
    const base = this.config.endpoints.api || this.config.endpoints.default;
    return `${base}${endpoint.startsWith('/') ? endpoint : '/' + endpoint}`;
  }

  async _buildHeaders() {
    const headers = {
      'Content-Type': 'application/json',
      'User-Agent': 'NEXUS-Hypercore/6.0'
    };

    if (this.token) {
      headers['Authorization'] = `Bearer ${this.token}`;
    }

    return headers;
  }

  _wait(seconds) {
    return new Promise(resolve => setTimeout(resolve, seconds * 1000));
  }

  getUsage() {
    return {
      service: this.serviceId,
      requests: this.usageStats.requests,
      startTime: this.usageStats.startTime,
      freeTier: this.config.freeTier || null,
      resetTime: this.usageStats.resetTime
    };
  }
}

// ============================================================================
// SERVICE BRIDGE ROUTER
// ============================================================================

export class ServiceBridgeRouter {
  constructor(env) {
    this.env = env;
    this.bridges = new Map();
  }

  async getBridge(serviceId) {
    const key = serviceId.toLowerCase();
    if (!this.bridges.has(key)) {
      this.bridges.set(key, new ServiceBridge(this.env, key));
    }
    return this.bridges.get(key);
  }

  async deployToAll(code, platforms = ['railway', 'vercel', 'netlify', 'render']) {
    const results = {};

    for (const platform of platforms) {
      try {
        const bridge = await this.getBridge(platform);
        const result = await bridge.deploy(code, {
          name: `nexus-${platform}-${Date.now()}`,
          env: { NEXUS_VERSION: '6.0.0' }
        });
        results[platform] = result;
      } catch (error) {
        results[platform] = { success: false, error: error.message };
      }
    }

    return results;
  }

  async callService(serviceId, endpoint, method = 'GET', data = null) {
    const bridge = await this.getBridge(serviceId);
    return bridge.call(endpoint, method, data);
  }

  async triggerGitHubWorkflow(workflowId, inputs = {}) {
    const bridge = await this.getBridge('github');
    const repo = this.env.GITHUB_REPO || process.env.GITHUB_REPO;
    
    if (!repo) throw new Error('GITHUB_REPO not configured');
    
    return bridge.call(`/repos/${repo}/actions/workflows/${workflowId}/dispatches`, 'POST', {
      ref: 'main',
      inputs: {
        ...inputs,
        nodeProvenanceId: this.env.NODE_ID || 'nexus-core',
        timestamp: Date.now()
      }
    });
  }

  getStatus() {
    const status = {};
    for (const [id, bridge] of this.bridges) {
      status[id] = bridge.getUsage();
    }
    return status;
  }
}
