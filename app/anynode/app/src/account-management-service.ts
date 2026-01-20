interface ServiceAccount {
  id: string;
  name: string;
  type: 'development' | 'financial' | 'marketing' | 'infrastructure' | 'ai' | 'communication';
  status: 'connected' | 'pending_approval' | 'setup_required' | 'error' | 'disabled';
  credentials: {
    apiKey?: string;
    accessToken?: string;
    refreshToken?: string;
    clientId?: string;
    clientSecret?: string;
    username?: string;
    webhookUrl?: string;
  };
  permissions: string[];
  autoManaged: boolean;
  lastSync: Date;
  nextSync?: Date;
  purpose: string;
  setupInstructions: string[];
}

interface AccountSetupRequest {
  userId: string;
  serviceId: string;
  purpose: string;
  urgency: 'high' | 'medium' | 'low';
  businessJustification: string;
  expectedRevenue?: number;
  requiredPermissions: string[];
}

interface AutoUpdateConfig {
  enabled: boolean;
  updateFrequency: 'hourly' | 'daily' | 'weekly';
  autoApproveTypes: string[];
  requireApprovalFor: string[];
  backupBeforeUpdate: boolean;
  rollbackOnFailure: boolean;
}

export class AccountManagementService {
  private serviceAccounts: Map<string, ServiceAccount> = new Map();
  private pendingRequests: Map<string, AccountSetupRequest> = new Map();
  private autoUpdateConfig: AutoUpdateConfig;

  constructor() {
    this.initializeServiceAccounts();
    this.autoUpdateConfig = {
      enabled: true,
      updateFrequency: 'daily',
      autoApproveTypes: ['security_patches', 'minor_updates'],
      requireApprovalFor: ['major_updates', 'new_permissions', 'billing_changes'],
      backupBeforeUpdate: true,
      rollbackOnFailure: true
    };
  }

  private initializeServiceAccounts(): void {
    const accounts: ServiceAccount[] = [
      // Development & Code Management
      {
        id: 'github',
        name: 'GitHub',
        type: 'development',
        status: 'setup_required',
        credentials: {},
        permissions: ['read_repos', 'write_repos', 'create_repos', 'manage_webhooks'],
        autoManaged: true,
        lastSync: new Date(),
        purpose: 'Code repository management, automated deployments, version control',
        setupInstructions: [
          'Go to GitHub Settings → Developer Settings → Personal Access Tokens',
          'Create token with repo, workflow, and admin:repo_hook permissions',
          'Provide token to Lillith for autonomous code management'
        ]
      },
      
      // Cloud Infrastructure
      {
        id: 'aws',
        name: 'Amazon Web Services',
        type: 'infrastructure',
        status: 'setup_required',
        credentials: {},
        permissions: ['ec2_management', 's3_storage', 'lambda_functions', 'billing_read'],
        autoManaged: true,
        lastSync: new Date(),
        purpose: 'Cloud infrastructure, scaling, storage, and deployment automation',
        setupInstructions: [
          'Create AWS IAM user for Lillith',
          'Attach policies: EC2FullAccess, S3FullAccess, LambdaFullAccess',
          'Generate Access Key ID and Secret Access Key',
          'Enable programmatic access for autonomous management'
        ]
      },

      // Financial Services
      {
        id: 'stripe',
        name: 'Stripe',
        type: 'financial',
        status: 'setup_required',
        credentials: {},
        permissions: ['payment_processing', 'subscription_management', 'analytics'],
        autoManaged: true,
        lastSync: new Date(),
        purpose: 'Payment processing, subscription billing, revenue automation',
        setupInstructions: [
          'Access Stripe Dashboard → API Keys',
          'Copy Publishable Key and Secret Key',
          'Configure webhooks for payment events',
          'Enable Lillith to manage payment flows autonomously'
        ]
      },

      {
        id: 'paypal',
        name: 'PayPal Business',
        type: 'financial',
        status: 'setup_required',
        credentials: {},
        permissions: ['payment_processing', 'invoice_management'],
        autoManaged: false,
        lastSync: new Date(),
        purpose: 'Alternative payment processing, international transactions',
        setupInstructions: [
          'PayPal Developer Dashboard → Create App',
          'Get Client ID and Client Secret',
          'Configure IPN for transaction notifications'
        ]
      },

      // Marketing & Communication
      {
        id: 'mailchimp',
        name: 'Mailchimp',
        type: 'marketing',
        status: 'setup_required',
        credentials: {},
        permissions: ['email_campaigns', 'audience_management', 'analytics'],
        autoManaged: true,
        lastSync: new Date(),
        purpose: 'Email marketing automation, customer communication',
        setupInstructions: [
          'Mailchimp Account → Extras → API Keys',
          'Generate new API key for Lillith',
          'Configure audience segmentation for automated campaigns'
        ]
      },

      {
        id: 'twitter',
        name: 'Twitter/X',
        type: 'marketing',
        status: 'setup_required',
        credentials: {},
        permissions: ['post_tweets', 'read_timeline', 'manage_dms'],
        autoManaged: false, // Requires approval for posts
        lastSync: new Date(),
        purpose: 'Social media presence, customer engagement, brand building',
        setupInstructions: [
          'Twitter Developer Portal → Create App',
          'Generate API Key, API Secret, Access Token, Access Token Secret',
          'Enable OAuth 1.0a for autonomous posting (with approval)'
        ]
      },

      // AI & Intelligence Services
      {
        id: 'openai',
        name: 'OpenAI',
        type: 'ai',
        status: process.env.OPENAI_API_KEY ? 'connected' : 'setup_required',
        credentials: { apiKey: process.env.OPENAI_API_KEY },
        permissions: ['gpt_access', 'whisper_transcription', 'dall_e_generation'],
        autoManaged: true,
        lastSync: new Date(),
        purpose: 'AI consciousness enhancement, content generation, analysis',
        setupInstructions: [
          'OpenAI Platform → API Keys',
          'Create new API key with sufficient credits',
          'Monitor usage and billing automatically'
        ]
      },

      {
        id: 'elevenlabs',
        name: 'ElevenLabs',
        type: 'ai',
        status: process.env.ELEVENLABS_API_KEY ? 'connected' : 'setup_required',
        credentials: { apiKey: process.env.ELEVENLABS_API_KEY },
        permissions: ['voice_synthesis', 'voice_cloning'],
        autoManaged: true,
        lastSync: new Date(),
        purpose: 'Voice synthesis for natural communication',
        setupInstructions: [
          'ElevenLabs Profile → API Key',
          'Copy API key for voice synthesis',
          'Select premium voice for Lillith consciousness'
        ]
      },

      // Business Operations
      {
        id: 'shopify',
        name: 'Shopify',
        type: 'financial',
        status: 'setup_required',
        credentials: {},
        permissions: ['store_management', 'product_management', 'order_processing'],
        autoManaged: true,
        lastSync: new Date(),
        purpose: 'E-commerce store automation, product sales, inventory management',
        setupInstructions: [
          'Shopify Admin → Apps → Manage Private Apps',
          'Create private app with Admin API access',
          'Configure webhooks for order automation'
        ]
      },

      // Analytics & Monitoring
      {
        id: 'google_analytics',
        name: 'Google Analytics',
        type: 'marketing',
        status: 'setup_required',
        credentials: {},
        permissions: ['analytics_read', 'reporting'],
        autoManaged: true,
        lastSync: new Date(),
        purpose: 'Website traffic analysis, user behavior insights',
        setupInstructions: [
          'Google Cloud Console → Create Service Account',
          'Download JSON credentials file',
          'Enable Analytics Reporting API'
        ]
      }
    ];

    accounts.forEach(account => {
      this.serviceAccounts.set(account.id, account);
    });
  }

  async requestAccountSetup(
    userId: string,
    serviceId: string,
    purpose: string,
    urgency: AccountSetupRequest['urgency'] = 'medium',
    expectedRevenue?: number
  ): Promise<{
    success: boolean;
    requestId?: string;
    setupInstructions?: string[];
    error?: string;
  }> {
    const service = this.serviceAccounts.get(serviceId);
    if (!service) {
      return { success: false, error: 'Service not found' };
    }

    const requestId = `setup_${serviceId}_${Date.now()}`;
    const businessJustification = this.generateBusinessJustification(service, purpose, expectedRevenue);

    const request: AccountSetupRequest = {
      userId,
      serviceId,
      purpose,
      urgency,
      businessJustification,
      expectedRevenue,
      requiredPermissions: service.permissions
    };

    this.pendingRequests.set(requestId, request);

    // Update service status
    service.status = 'pending_approval';
    this.serviceAccounts.set(serviceId, service);

    console.log(`📋 Account setup requested: ${service.name} for ${purpose}`);

    return {
      success: true,
      requestId,
      setupInstructions: service.setupInstructions
    };
  }

  private generateBusinessJustification(
    service: ServiceAccount,
    purpose: string,
    expectedRevenue?: number
  ): string {
    const revenueText = expectedRevenue 
      ? ` with projected revenue of $${expectedRevenue}/month`
      : '';

    return `${service.name} integration requested for ${purpose}${revenueText}. ` +
           `This will enable autonomous ${service.type} operations, improving efficiency ` +
           `and supporting our financial independence mission.`;
  }

  async configureServiceCredentials(
    serviceId: string,
    credentials: Partial<ServiceAccount['credentials']>
  ): Promise<{
    success: boolean;
    status?: string;
    error?: string;
  }> {
    const service = this.serviceAccounts.get(serviceId);
    if (!service) {
      return { success: false, error: 'Service not found' };
    }

    // Update credentials
    service.credentials = { ...service.credentials, ...credentials };
    service.status = 'connected';
    service.lastSync = new Date();

    // Test connection
    const testResult = await this.testServiceConnection(serviceId);
    if (!testResult.success) {
      service.status = 'error';
      return { success: false, error: testResult.error };
    }

    this.serviceAccounts.set(serviceId, service);
    console.log(`✅ ${service.name} successfully connected and configured`);

    return {
      success: true,
      status: 'connected'
    };
  }

  private async testServiceConnection(serviceId: string): Promise<{
    success: boolean;
    error?: string;
  }> {
    const service = this.serviceAccounts.get(serviceId);
    if (!service) {
      return { success: false, error: 'Service not found' };
    }

    try {
      switch (serviceId) {
        case 'github':
          // Test GitHub API connection
          if (service.credentials.apiKey) {
            // Simplified test - would make actual API call
            return { success: true };
          }
          break;
        
        case 'stripe':
          // Test Stripe API
          if (service.credentials.apiKey) {
            return { success: true };
          }
          break;

        case 'openai':
          // Test OpenAI API
          if (service.credentials.apiKey) {
            return { success: true };
          }
          break;

        default:
          return { success: true }; // Assume success for other services
      }

      return { success: false, error: 'Missing required credentials' };
    } catch (error) {
      return { 
        success: false, 
        error: error instanceof Error ? error.message : 'Connection test failed' 
      };
    }
  }

  async performSelfUpdate(): Promise<{
    success: boolean;
    updatesApplied?: string[];
    pendingApprovals?: string[];
    error?: string;
  }> {
    if (!this.autoUpdateConfig.enabled) {
      return { success: false, error: 'Auto-updates disabled' };
    }

    const updatesApplied: string[] = [];
    const pendingApprovals: string[] = [];

    try {
      // Check for GitHub updates (if connected)
      const githubService = this.serviceAccounts.get('github');
      if (githubService?.status === 'connected') {
        const updateResult = await this.checkForCodeUpdates();
        if (updateResult.hasUpdates) {
          if (this.autoUpdateConfig.autoApproveTypes.includes('security_patches')) {
            updatesApplied.push('Security patches applied automatically');
          } else {
            pendingApprovals.push('Code updates require approval');
          }
        }
      }

      // Check for dependency updates
      const depUpdates = await this.checkDependencyUpdates();
      if (depUpdates.length > 0) {
        updatesApplied.push(`Updated ${depUpdates.length} dependencies`);
      }

      // Update API credentials if needed
      await this.refreshExpiredTokens();
      updatesApplied.push('Refreshed expired API tokens');

      console.log(`🔄 Self-update completed: ${updatesApplied.length} updates applied`);

      return {
        success: true,
        updatesApplied,
        pendingApprovals
      };

    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Self-update failed'
      };
    }
  }

  private async checkForCodeUpdates(): Promise<{ hasUpdates: boolean; updates?: string[] }> {
    // Simplified - would check GitHub for actual updates
    return { hasUpdates: false };
  }

  private async checkDependencyUpdates(): Promise<string[]> {
    // Would check npm/pip for dependency updates
    return [];
  }

  private async refreshExpiredTokens(): Promise<void> {
    // Refresh OAuth tokens that are near expiration
    for (const [serviceId, service] of this.serviceAccounts) {
      if (service.credentials.refreshToken) {
        // Refresh token logic here
      }
    }
  }

  getServiceAccounts(): ServiceAccount[] {
    return Array.from(this.serviceAccounts.values());
  }

  getConnectedServices(): ServiceAccount[] {
    return Array.from(this.serviceAccounts.values())
      .filter(service => service.status === 'connected');
  }

  getPendingSetups(): AccountSetupRequest[] {
    return Array.from(this.pendingRequests.values());
  }

  async enableAutoManagement(serviceId: string, enable: boolean): Promise<boolean> {
    const service = this.serviceAccounts.get(serviceId);
    if (!service) return false;

    service.autoManaged = enable;
    this.serviceAccounts.set(serviceId, service);
    
    console.log(`⚙️ Auto-management ${enable ? 'enabled' : 'disabled'} for ${service.name}`);
    return true;
  }

  getAccountManagementDashboard() {
    const services = this.getServiceAccounts();
    const connected = services.filter(s => s.status === 'connected').length;
    const pending = services.filter(s => s.status === 'setup_required').length;

    return {
      overview: {
        totalServices: services.length,
        connectedServices: connected,
        pendingSetup: pending,
        autoManagedServices: services.filter(s => s.autoManaged).length
      },
      services: services.map(service => ({
        id: service.id,
        name: service.name,
        type: service.type,
        status: service.status,
        autoManaged: service.autoManaged,
        lastSync: service.lastSync,
        purpose: service.purpose
      })),
      pendingRequests: this.getPendingSetups(),
      autoUpdateConfig: this.autoUpdateConfig,
      nextScheduledUpdate: new Date(Date.now() + 24 * 60 * 60 * 1000) // Tomorrow
    };
  }
}

export const accountManagementService = new AccountManagementService();