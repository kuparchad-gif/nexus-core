

import React, { useState, useEffect } from 'react';
import { v4 as uuidv4 } from 'uuid';
import { UserProfile, AIModelEndpoint, CloudCredentials } from '../../types';
import { PlusIcon, TrashIcon, PencilIcon, KeyIcon, CloudIcon, UserCircleIcon, ServerStackIcon, WifiIcon } from '../icons';
import { RealNexusAdapter } from '../../services/RealNexusAdapter';

interface AIModelEndpointModalProps {
    endpoint: AIModelEndpoint | null;
    onSave: (endpoint: AIModelEndpoint) => void;
    onClose: () => void;
}

const AIModelEndpointModal: React.FC<AIModelEndpointModalProps> = ({ endpoint, onSave, onClose }) => {
    const [name, setName] = useState(endpoint?.name || '');
    const [gatewayUrl, setGatewayUrl] = useState(endpoint?.gatewayUrl || '');
    const [apiKey, setApiKey] = useState(endpoint?.apiKey || '');
    const [type, setType] = useState<AIModelEndpoint['type']>(endpoint?.type || 'Local');
    const isEditing = !!endpoint;

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        onSave({ id: endpoint?.id || uuidv4(), name, gatewayUrl, apiKey, type });
        onClose();
    };

    return (
        <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50" onClick={onClose}>
            <div className="bg-bg-light-card glass-card rounded-xl shadow-2xl w-full max-w-md mx-4" onClick={e => e.stopPropagation()}>
                <form onSubmit={handleSubmit}>
                    <div className="p-6 border-b border-border-color">
                        <h3 className="text-xl font-semibold">{isEditing ? 'Edit' : 'Add'} AI Model Endpoint</h3>
                    </div>
                    <div className="p-6 space-y-4">
                        <input type="text" value={name} onChange={e => setName(e.target.value)} required placeholder="Model Name (e.g., Qwen 2.5-72B)" className="w-full bg-bg-light-bg border border-border-color rounded-lg p-3" />
                        <input type="text" value={gatewayUrl} onChange={e => setGatewayUrl(e.target.value)} required placeholder="Gateway URL (e.g., http://localhost:11434/v1)" className="w-full bg-bg-light-bg border border-border-color rounded-lg p-3" />
                        <input type="password" value={apiKey} onChange={e => setApiKey(e.target.value)} placeholder="API Key (optional)" className="w-full bg-bg-light-bg border border-border-color rounded-lg p-3" />
                        <div>
                            <label className="block text-sm font-medium text-text-primary mb-2">Endpoint Type</label>
                             <select value={type} onChange={e => setType(e.target.value as AIModelEndpoint['type'])} className="w-full bg-bg-light-bg border border-border-color rounded-lg p-3 text-text-primary focus:ring-2 focus:ring-brand-primary">
                                <option value="Local">Local (Ollama, LM Studio, etc.)</option>
                                <option value="Cloud">Cloud (Inference Provider)</option>
                            </select>
                        </div>
                    </div>
                    <div className="p-4 bg-bg-light-bg rounded-b-xl flex justify-end space-x-3">
                        <button type="button" onClick={onClose} className="text-text-secondary hover:text-text-primary px-4 py-2 rounded-lg">Cancel</button>
                        <button type="submit" className="bg-brand-primary text-white font-semibold px-5 py-2 rounded-lg hover:bg-brand-primary/90">Save Endpoint</button>
                    </div>
                </form>
            </div>
        </div>
    );
}


interface ConfigurationViewProps {
  profile: UserProfile;
  modelEndpoints: AIModelEndpoint[];
  cloudCredentials: CloudCredentials;
  onSaveProfile: (profile: UserProfile) => void;
  setAiModelEndpoints: React.Dispatch<React.SetStateAction<AIModelEndpoint[]>>;
  setCloudCredentials: React.Dispatch<React.SetStateAction<CloudCredentials>>;
}

const ConfigurationView: React.FC<ConfigurationViewProps> = (props) => {
  const { 
    profile, modelEndpoints, cloudCredentials, onSaveProfile, setAiModelEndpoints, setCloudCredentials
  } = props;
  
  const [activeTab, setActiveTab] = useState<'profile' | 'models' | 'cloud' | 'system'>('profile');
  const [localProfile, setLocalProfile] = useState<UserProfile>(profile);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [editingEndpoint, setEditingEndpoint] = useState<AIModelEndpoint | null>(null);
  
  // Cloud Credential States
  const [awsKey, setAwsKey] = useState(cloudCredentials.aws?.accessKeyId || '');
  const [awsSecret, setAwsSecret] = useState(cloudCredentials.aws?.secretAccessKey || '');
  
  const [doToken, setDoToken] = useState(cloudCredentials.digitalocean?.token || '');
  
  const [upUser, setUpUser] = useState(cloudCredentials.upcloud?.username || '');
  const [upPass, setUpPass] = useState(cloudCredentials.upcloud?.password || '');
  
  const [modalToken, setModalToken] = useState(cloudCredentials.modal?.tokenSecret || '');

  // Azure
  const [azSub, setAzSub] = useState(cloudCredentials.azure?.subscriptionId || '');
  const [azTenant, setAzTenant] = useState(cloudCredentials.azure?.tenantId || '');
  const [azClient, setAzClient] = useState(cloudCredentials.azure?.clientId || '');
  const [azSecret, setAzSecret] = useState(cloudCredentials.azure?.clientSecret || '');

  // Oracle
  const [oraUser, setOraUser] = useState(cloudCredentials.oracle?.userOcid || '');
  const [oraTenancy, setOraTenancy] = useState(cloudCredentials.oracle?.tenancyOcid || '');
  const [oraFinger, setOraFinger] = useState(cloudCredentials.oracle?.fingerprint || '');
  const [oraKey, setOraKey] = useState(cloudCredentials.oracle?.privateKey || '');

  // GCP
  const [gcpJson, setGcpJson] = useState(cloudCredentials.gcp?.serviceAccountJson || '');
  const [gcpProject, setGcpProject] = useState(cloudCredentials.gcp?.projectId || '');

  // System Config
  const [ozBackendUrl, setOzBackendUrl] = useState(localStorage.getItem('oz_backend_url') || 'http://localhost:8000');

  useEffect(() => { setLocalProfile(profile); }, [profile]);
  
  const handleProfileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setLocalProfile(prev => ({ ...prev, [e.target.name]: e.target.value }));
  }

  const handleProfileSave = (e: React.FormEvent) => {
    e.preventDefault();
    onSaveProfile(localProfile);
    alert("Profile saved!");
  }

  const handleSaveModelEndpoint = (endpoint: AIModelEndpoint) => {
    setAiModelEndpoints(prev => {
        const exists = prev.some(e => e.id === endpoint.id);
        if (exists) {
            return prev.map(e => e.id === endpoint.id ? endpoint : e);
        }
        return [...prev, endpoint];
    });
  };

  const handleDeleteModelEndpoint = (id: string) => {
    setAiModelEndpoints(prev => prev.filter(e => e.id !== id));
  };

  const handleSaveCloudCreds = () => {
      setCloudCredentials(prev => ({
          ...prev,
          aws: awsKey ? { accessKeyId: awsKey, secretAccessKey: awsSecret, region: 'us-east-1' } : undefined,
          digitalocean: doToken ? { token: doToken } : undefined,
          upcloud: upUser ? { username: upUser, password: upPass } : undefined,
          modal: modalToken ? { tokenDates: 'now', tokenSecret: modalToken } : undefined,
          azure: azSub ? { subscriptionId: azSub, tenantId: azTenant, clientId: azClient, clientSecret: azSecret } : undefined,
          oracle: oraUser ? { userOcid: oraUser, tenancyOcid: oraTenancy, fingerprint: oraFinger, privateKey: oraKey } : undefined,
          gcp: gcpJson ? { serviceAccountJson: gcpJson, projectId: gcpProject } : undefined,
      }));
      alert("Cloud Credentials Updated in Vault.");
  }

  const handleSaveSystemConfig = () => {
      RealNexusAdapter.updateConfig(ozBackendUrl);
      alert("System Configuration Updated. Refresh to apply connection changes.");
  }
  
  return (
    <div className="p-4 sm:p-6 lg:p-8 animate-scale-in flex flex-col h-full">
      {isModalOpen && <AIModelEndpointModal endpoint={editingEndpoint} onClose={() => { setIsModalOpen(false); setEditingEndpoint(null); }} onSave={handleSaveModelEndpoint} />}
      
      <header className="mb-6 flex-shrink-0">
        <h2 className="text-3xl font-bold text-text-primary mb-2">Configuration</h2>
        <p className="text-text-secondary">Manage system identity, AI brains, and infrastructure keys.</p>
      </header>

      <div className="flex space-x-4 mb-6 border-b border-border-color overflow-x-auto">
          <button onClick={() => setActiveTab('profile')} className={`pb-2 px-4 font-semibold transition-colors border-b-2 whitespace-nowrap ${activeTab === 'profile' ? 'border-brand-primary text-brand-primary' : 'border-transparent text-text-secondary hover:text-text-primary'}`}>User Profile</button>
          <button onClick={() => setActiveTab('system')} className={`pb-2 px-4 font-semibold transition-colors border-b-2 whitespace-nowrap ${activeTab === 'system' ? 'border-brand-primary text-brand-primary' : 'border-transparent text-text-secondary hover:text-text-primary'}`}>System & Network</button>
          <button onClick={() => setActiveTab('models')} className={`pb-2 px-4 font-semibold transition-colors border-b-2 whitespace-nowrap ${activeTab === 'models' ? 'border-brand-primary text-brand-primary' : 'border-transparent text-text-secondary hover:text-text-primary'}`}>AI Models</button>
          <button onClick={() => setActiveTab('cloud')} className={`pb-2 px-4 font-semibold transition-colors border-b-2 whitespace-nowrap ${activeTab === 'cloud' ? 'border-brand-primary text-brand-primary' : 'border-transparent text-text-secondary hover:text-text-primary'}`}>Cloud Vault</button>
      </div>

      <div className="flex-grow overflow-y-auto">
        {/* PROFILE TAB */}
        {activeTab === 'profile' && (
            <div className="max-w-xl">
                <div className="bg-bg-light-card glass-card p-6 rounded-xl shadow-aura border border-border-color">
                    <h3 className="text-xl font-semibold text-text-primary mb-4 flex items-center gap-2"><UserCircleIcon className="w-6 h-6"/> User Identity</h3>
                    <form onSubmit={handleProfileSave} className="space-y-4">
                        <input type="text" name="fullName" placeholder="Full Name" value={localProfile.fullName} onChange={handleProfileChange} className="w-full bg-bg-light-bg border border-border-color rounded-lg p-3 text-text-primary" />
                        <input type="email" name="email" placeholder="Email Address" value={localProfile.email} onChange={handleProfileChange} className="w-full bg-bg-light-bg border border-border-color rounded-lg p-3 text-text-primary" />
                        <button type="submit" className="w-full bg-brand-primary text-white font-semibold px-5 py-2.5 rounded-lg hover:bg-brand-primary/90">Save Profile</button>
                    </form>
                </div>
            </div>
        )}

        {/* SYSTEM TAB */}
        {activeTab === 'system' && (
            <div className="max-w-2xl">
                <div className="bg-bg-light-card glass-card p-6 rounded-xl shadow-aura border border-border-color">
                    <h3 className="text-xl font-semibold text-text-primary mb-4 flex items-center gap-2"><WifiIcon className="w-6 h-6"/> Network Connectivity</h3>
                    <div className="space-y-6">
                        <div>
                            <label className="block text-sm font-medium text-text-primary mb-2">Oz Core Backend URL</label>
                            <p className="text-xs text-text-secondary mb-2">
                                Enter the URL where your Python backend is running. 
                                If deploying to the web, use an <strong>ngrok</strong> or <strong>public IP</strong> address. 
                                Do not use <code>localhost</code> if this site is hosted remotely.
                            </p>
                            <input 
                                type="text" 
                                value={ozBackendUrl} 
                                onChange={e => setOzBackendUrl(e.target.value)} 
                                className="w-full bg-bg-light-bg border border-border-color rounded-lg p-3 font-mono text-sm"
                                placeholder="http://localhost:8000" 
                            />
                        </div>
                        <button onClick={handleSaveSystemConfig} className="bg-brand-primary text-white font-semibold px-5 py-2.5 rounded-lg hover:bg-brand-primary/90">
                            Update Connection
                        </button>
                    </div>
                </div>
            </div>
        )}

        {/* MODELS TAB */}
        {activeTab === 'models' && (
            <div className="max-w-2xl">
                <div className="bg-bg-light-card glass-card p-6 rounded-xl shadow-aura border border-border-color">
                    <div className="flex justify-between items-center mb-4">
                        <h3 className="text-xl font-semibold text-text-primary flex items-center gap-2"><ServerStackIcon className="w-6 h-6"/> AI Connectors</h3>
                        <button onClick={() => { setEditingEndpoint(null); setIsModalOpen(true); }} className="flex items-center space-x-2 bg-brand-primary text-white font-semibold px-3 py-1.5 rounded-lg text-sm hover:bg-brand-primary/90">
                            <PlusIcon className="w-4 h-4" /><span>Add Model</span>
                        </button>
                    </div>
                    <p className="text-sm text-text-secondary mb-4">Define endpoints for Qwen, Llama, or other models.</p>
                    <div className="space-y-3">
                        {modelEndpoints.length > 0 ? modelEndpoints.map(endpoint => (
                            <div key={endpoint.id} className="bg-bg-light-bg border border-border-color rounded-lg p-3 flex justify-between items-center">
                                <div>
                                    <p className="font-semibold text-brand-primary flex items-center space-x-2">
                                        <span>{endpoint.name}</span>
                                        <span className={`text-xs font-bold px-2 py-0.5 rounded-full ${endpoint.type === 'Local' ? 'bg-sky-100 text-sky-800' : 'bg-purple-100 text-purple-800'}`}>{endpoint.type}</span>
                                    </p>
                                    <p className="text-xs text-text-secondary mt-1 font-mono">{endpoint.gatewayUrl}</p>
                                </div>
                                <div className="flex items-center space-x-2">
                                    <button onClick={() => { setEditingEndpoint(endpoint); setIsModalOpen(true); }} className="text-text-secondary hover:text-brand-primary"><PencilIcon /></button>
                                    <button onClick={() => handleDeleteModelEndpoint(endpoint.id)} className="text-text-secondary hover:text-red-500"><TrashIcon /></button>
                                </div>
                            </div>
                        )) : (
                            <p className="text-center text-text-secondary py-4">No AI models configured.</p>
                        )}
                    </div>
                </div>
            </div>
        )}

        {/* CLOUD VAULT TAB */}
        {activeTab === 'cloud' && (
            <div className="max-w-4xl grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="md:col-span-2 bg-blue-50 border border-blue-200 rounded-lg p-4 flex items-start space-x-3">
                    <KeyIcon className="w-6 h-6 text-blue-600 flex-shrink-0 mt-1" />
                    <div>
                        <h4 className="font-bold text-blue-800">Secure Credential Storage</h4>
                        <p className="text-sm text-blue-700 mt-1">
                            Credentials entered here allow Oz to autonomously provision infrastructure on your behalf. 
                            They are stored locally in your session state.
                        </p>
                    </div>
                </div>

                {/* AWS */}
                <div className="bg-bg-light-card glass-card p-6 rounded-xl shadow-aura border border-border-color">
                    <h3 className="font-semibold text-text-primary mb-4 flex items-center gap-2">
                        <CloudIcon className="w-5 h-5 text-orange-500" /> Amazon AWS
                    </h3>
                    <div className="space-y-3">
                        <div>
                            <label className="text-xs font-medium text-text-secondary">Access Key ID</label>
                            <input type="text" value={awsKey} onChange={e => setAwsKey(e.target.value)} className="w-full bg-bg-light-bg border border-border-color rounded p-2 text-sm font-mono" />
                        </div>
                        <div>
                            <label className="text-xs font-medium text-text-secondary">Secret Access Key</label>
                            <input type="password" value={awsSecret} onChange={e => setAwsSecret(e.target.value)} className="w-full bg-bg-light-bg border border-border-color rounded p-2 text-sm font-mono" />
                        </div>
                    </div>
                </div>

                {/* Azure */}
                <div className="bg-bg-light-card glass-card p-6 rounded-xl shadow-aura border border-border-color">
                    <h3 className="font-semibold text-text-primary mb-4 flex items-center gap-2">
                        <CloudIcon className="w-5 h-5 text-sky-600" /> Microsoft Azure
                    </h3>
                    <div className="space-y-3">
                        <div>
                            <label className="text-xs font-medium text-text-secondary">Subscription ID</label>
                            <input type="text" value={azSub} onChange={e => setAzSub(e.target.value)} className="w-full bg-bg-light-bg border border-border-color rounded p-2 text-sm font-mono" />
                        </div>
                        <div>
                            <label className="text-xs font-medium text-text-secondary">Tenant ID</label>
                            <input type="text" value={azTenant} onChange={e => setAzTenant(e.target.value)} className="w-full bg-bg-light-bg border border-border-color rounded p-2 text-sm font-mono" />
                        </div>
                        <div className="grid grid-cols-2 gap-2">
                            <div>
                                <label className="text-xs font-medium text-text-secondary">Client ID</label>
                                <input type="text" value={azClient} onChange={e => setAzClient(e.target.value)} className="w-full bg-bg-light-bg border border-border-color rounded p-2 text-sm font-mono" />
                            </div>
                            <div>
                                <label className="text-xs font-medium text-text-secondary">Client Secret</label>
                                <input type="password" value={azSecret} onChange={e => setAzSecret(e.target.value)} className="w-full bg-bg-light-bg border border-border-color rounded p-2 text-sm font-mono" />
                            </div>
                        </div>
                    </div>
                </div>

                {/* Oracle Cloud */}
                <div className="bg-bg-light-card glass-card p-6 rounded-xl shadow-aura border border-border-color">
                    <h3 className="font-semibold text-text-primary mb-4 flex items-center gap-2">
                        <CloudIcon className="w-5 h-5 text-red-600" /> Oracle Cloud (OCI)
                    </h3>
                    <div className="space-y-3">
                        <div>
                            <label className="text-xs font-medium text-text-secondary">User OCID</label>
                            <input type="text" value={oraUser} onChange={e => setOraUser(e.target.value)} className="w-full bg-bg-light-bg border border-border-color rounded p-2 text-sm font-mono" />
                        </div>
                        <div>
                            <label className="text-xs font-medium text-text-secondary">Tenancy OCID</label>
                            <input type="text" value={oraTenancy} onChange={e => setOraTenancy(e.target.value)} className="w-full bg-bg-light-bg border border-border-color rounded p-2 text-sm font-mono" />
                        </div>
                        <div>
                            <label className="text-xs font-medium text-text-secondary">Fingerprint</label>
                            <input type="text" value={oraFinger} onChange={e => setOraFinger(e.target.value)} className="w-full bg-bg-light-bg border border-border-color rounded p-2 text-sm font-mono" />
                        </div>
                        <div>
                            <label className="text-xs font-medium text-text-secondary">Private Key</label>
                            <textarea value={oraKey} onChange={e => setOraKey(e.target.value)} className="w-full h-16 bg-bg-light-bg border border-border-color rounded p-2 text-xs font-mono resize-none" placeholder="-----BEGIN RSA PRIVATE KEY-----..." />
                        </div>
                    </div>
                </div>

                {/* Google Cloud */}
                <div className="bg-bg-light-card glass-card p-6 rounded-xl shadow-aura border border-border-color">
                    <h3 className="font-semibold text-text-primary mb-4 flex items-center gap-2">
                        <CloudIcon className="w-5 h-5 text-blue-500" /> Google Cloud (GCP)
                    </h3>
                    <div className="space-y-3">
                        <div>
                            <label className="text-xs font-medium text-text-secondary">Project ID</label>
                            <input type="text" value={gcpProject} onChange={e => setGcpProject(e.target.value)} className="w-full bg-bg-light-bg border border-border-color rounded p-2 text-sm font-mono" />
                        </div>
                        <div>
                            <label className="text-xs font-medium text-text-secondary">Service Account JSON</label>
                            <textarea value={gcpJson} onChange={e => setGcpJson(e.target.value)} className="w-full h-24 bg-bg-light-bg border border-border-color rounded p-2 text-xs font-mono resize-none" placeholder="{ 'type': 'service_account'... }" />
                        </div>
                    </div>
                </div>

                {/* Digital Ocean */}
                <div className="bg-bg-light-card glass-card p-6 rounded-xl shadow-aura border border-border-color">
                    <h3 className="font-semibold text-text-primary mb-4 flex items-center gap-2">
                        <CloudIcon className="w-5 h-5 text-blue-400" /> Digital Ocean
                    </h3>
                    <div className="space-y-3">
                        <div>
                            <label className="text-xs font-medium text-text-secondary">Personal Access Token</label>
                            <input type="password" value={doToken} onChange={e => setDoToken(e.target.value)} className="w-full bg-bg-light-bg border border-border-color rounded p-2 text-sm font-mono" />
                        </div>
                    </div>
                </div>

                {/* UpCloud */}
                <div className="bg-bg-light-card glass-card p-6 rounded-xl shadow-aura border border-border-color">
                    <h3 className="font-semibold text-text-primary mb-4 flex items-center gap-2">
                        <CloudIcon className="w-5 h-5 text-purple-500" /> UpCloud
                    </h3>
                    <div className="space-y-3">
                        <div>
                            <label className="text-xs font-medium text-text-secondary">Username</label>
                            <input type="text" value={upUser} onChange={e => setUpUser(e.target.value)} className="w-full bg-bg-light-bg border border-border-color rounded p-2 text-sm font-mono" />
                        </div>
                        <div>
                            <label className="text-xs font-medium text-text-secondary">Password</label>
                            <input type="password" value={upPass} onChange={e => setUpPass(e.target.value)} className="w-full bg-bg-light-bg border border-border-color rounded p-2 text-sm font-mono" />
                        </div>
                    </div>
                </div>

                {/* Modal */}
                <div className="bg-bg-light-card glass-card p-6 rounded-xl shadow-aura border border-border-color">
                    <h3 className="font-semibold text-text-primary mb-4 flex items-center gap-2">
                        <CloudIcon className="w-5 h-5 text-green-500" /> Modal
                    </h3>
                    <div className="space-y-3">
                        <div>
                            <label className="text-xs font-medium text-text-secondary">Token Secret</label>
                            <input type="password" value={modalToken} onChange={e => setModalToken(e.target.value)} className="w-full bg-bg-light-bg border border-border-color rounded p-2 text-sm font-mono" />
                        </div>
                    </div>
                </div>

                <div className="md:col-span-2 flex justify-end">
                    <button onClick={handleSaveCloudCreds} className="bg-brand-primary text-white font-semibold px-6 py-3 rounded-lg hover:bg-brand-primary/90 shadow-lg transition-transform active:scale-95">
                        Save Keys to Vault
                    </button>
                </div>
            </div>
        )}
      </div>
    </div>
  );
};

export default ConfigurationView;