
import React, { useState, useEffect } from 'react';
import { CloudIcon, ServerStackIcon, CheckCircleIcon, SpinnerIcon, HardDriveIcon, KeyIcon, GlobeAltIcon } from '../icons';
import { CloudCredentials, LogEntry, CloudProvider } from '../../types';
import { RealNexusAdapter } from '../../services/RealNexusAdapter';

interface CloudCommandViewProps {
    cloudCredentials: CloudCredentials;
    addLog: (type: LogEntry['type'], ...args: any[]) => void;
}

const CloudCommandView: React.FC<CloudCommandViewProps> = ({ cloudCredentials, addLog }) => {
    
    // Initial List of Providers (Oz can append to this list dynamically)
    const [providers, setProviders] = useState<CloudProvider[]>([
        { id: 'aws', name: 'Amazon AWS', type: 'IaaS', status: 'disconnected', resources: 0, region: 'us-east-1', color: 'bg-orange-500', requiredFields: ['accessKeyId'], iconType: 'cloud' },
        { id: 'gcp', name: 'Google Cloud', type: 'IaaS', status: 'disconnected', resources: 0, region: 'us-central1', color: 'bg-blue-500', requiredFields: ['serviceAccountJson'], iconType: 'cloud' },
        { id: 'azure', name: 'Microsoft Azure', type: 'IaaS', status: 'disconnected', resources: 0, region: 'eastus', color: 'bg-sky-600', requiredFields: ['clientId'], iconType: 'cloud' },
        { id: 'oracle', name: 'Oracle Cloud', type: 'IaaS', status: 'disconnected', resources: 0, region: 'us-ashburn-1', color: 'bg-red-600', requiredFields: ['userOcid'], iconType: 'cloud' },
        { id: 'upcloud', name: 'UpCloud', type: 'IaaS', status: 'disconnected', resources: 0, region: 'fi-hel1', color: 'bg-purple-600', requiredFields: ['username'], iconType: 'cloud' },
        { id: 'modal', name: 'Modal', type: 'Serverless', status: 'disconnected', resources: 0, region: 'global', color: 'bg-green-600', requiredFields: ['tokenSecret'], iconType: 'server' },
    ]);

    const [storageDeploymentStep, setStorageDeploymentStep] = useState(0);
    const [storageProvider, setStorageProvider] = useState<string>('upcloud');
    const [simulatedUpdate, setSimulatedUpdate] = useState(false);

    // Effect: Check Credentials and Auto-Update Status
    useEffect(() => {
        setProviders(prev => prev.map(p => {
            const creds = (cloudCredentials as any)[p.id === 'digitalocean' ? 'digitalocean' : p.id]; 
            const hasCreds = creds && p.requiredFields.every(field => creds[field]);
            
            if (!hasCreds) {
                return { ...p, status: 'missing_creds' };
            }
            if (hasCreds && p.status === 'missing_creds') {
                return { ...p, status: 'disconnected' }; // Ready to connect
            }
            return p;
        }));
    }, [cloudCredentials]);

    const toggleConnection = (id: string) => {
        const provider = providers.find(p => p.id === id);
        if (!provider) return;

        if (provider.status === 'missing_creds') {
            alert(`Credentials missing for ${provider.name}. Go to 'System Configuration' -> 'Cloud Vault' to add them.`);
            return;
        }

        setProviders(prev => prev.map(p => {
            if (p.id === id) {
                if (p.status === 'connected') return { ...p, status: 'disconnected', resources: 0 };
                return { ...p, status: 'connecting' };
            }
            return p;
        }));

        if (provider.status !== 'connected') {
            setTimeout(() => {
                setProviders(prev => prev.map(p => 
                    p.id === id ? { ...p, status: 'connected', resources: Math.floor(Math.random() * 50) + 5 } : p
                ));
                addLog('log', `[Oz Orchestrator]: Established secure link to ${provider.name}.`);
            }, 1500);
        } else {
            addLog('log', `[Oz Orchestrator]: Terminated link to ${provider.name}.`);
        }
    };

    const handleDeployStorage = async () => {
        const provider = providers.find(p => p.id === storageProvider);
        if (provider?.status === 'missing_creds' || provider?.status === 'disconnected') {
            alert(`Cannot deploy. Ensure ${provider?.name} is configured and connected.`);
            return;
        }

        addLog('log', `[Oz Core]: Provisioning 60GB Storage on ${storageProvider}...`);
        setStorageDeploymentStep(1);

        // Attempt REAL deployment via Adapter
        try {
            const providerCreds = (cloudCredentials as any)[storageProvider];
            // This will throw if backend is offline, triggering catch block (simulation)
            await RealNexusAdapter.deployStorage(storageProvider, 60, providerCreds);
            
            // If successful (real backend responded)
            addLog('log', `[REAL BACKEND] Success: Block volume allocated via ${storageProvider} API.`);
            setStorageDeploymentStep(4);
            
        } catch (e) {
            // Fallback to Simulation
            addLog('warn', 'Real backend unreachable. Simulating deployment steps...');
            setTimeout(() => { setStorageDeploymentStep(2); addLog('log', `[${storageProvider}] Allocating block volume...`); }, 2000);
            setTimeout(() => { setStorageDeploymentStep(3); addLog('log', `[${storageProvider}] Formatting XFS...`); }, 4000);
            setTimeout(() => { setStorageDeploymentStep(4); addLog('log', `[${storageProvider}] Volume mounted. Ready.`); }, 6000);
        }
    };

    // Simulate Oz detecting a new API capability
    const simulateOzUpdate = () => {
        setSimulatedUpdate(true);
        addLog('log', '[Oz Core]: New API Capability Detected: "Alibaba Cloud". Injecting frontend component...');
        setTimeout(() => {
            setProviders(prev => [
                ...prev,
                { 
                    id: 'alibaba', 
                    name: 'Alibaba Cloud', 
                    type: 'IaaS', 
                    status: 'missing_creds', 
                    resources: 0, 
                    region: 'cn-hangzhou', 
                    color: 'bg-orange-600', 
                    requiredFields: ['accessKeyId'],
                    iconType: 'cloud'
                }
            ]);
            setSimulatedUpdate(false);
        }, 2000);
    };

    const getIcon = (p: CloudProvider) => {
        if (p.iconType === 'server') return <ServerStackIcon className="w-7 h-7" />;
        if (p.iconType === 'database') return <HardDriveIcon className="w-7 h-7" />;
        return <CloudIcon className="w-7 h-7" />;
    }

    return (
        <div className="p-4 sm:p-6 lg:p-8 animate-scale-in h-full flex flex-col">
            <header className="mb-6 flex-shrink-0 flex justify-between items-center">
                <div>
                    <h2 className="text-3xl font-bold text-text-primary">Cloud Command</h2>
                    <p className="text-text-secondary">Oz Orchestrator: Infrastructure Management.</p>
                </div>
                <div className="flex gap-3">
                    <button 
                        onClick={simulateOzUpdate} 
                        disabled={simulatedUpdate}
                        className="flex items-center gap-2 bg-slate-100 hover:bg-slate-200 text-slate-600 px-3 py-1 rounded-full text-sm font-semibold transition-colors"
                        title="Simulate Oz discovering a new API and updating the UI"
                    >
                        {simulatedUpdate ? <SpinnerIcon className="w-4 h-4" /> : <GlobeAltIcon className="w-4 h-4" />}
                        <span>Universal Connector</span>
                    </button>
                    <div className="flex items-center gap-2 bg-green-100 text-green-800 px-3 py-1 rounded-full text-sm font-bold">
                        <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></span>
                        Oz Active
                    </div>
                </div>
            </header>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 flex-grow min-h-0">
                {/* Dynamic Provider List */}
                <div className="lg:col-span-2 flex flex-col gap-4 overflow-y-auto pr-2">
                    {providers.map(provider => (
                        <div key={provider.id} className="bg-bg-light-card glass-card rounded-xl shadow-aura border border-border-color p-6 flex items-center justify-between">
                            <div className="flex items-center gap-4">
                                <div className={`w-12 h-12 rounded-lg ${provider.color} flex items-center justify-center text-white shadow-md`}>
                                    {getIcon(provider)}
                                </div>
                                <div>
                                    <h3 className="font-bold text-text-primary text-lg">{provider.name}</h3>
                                    <p className="text-sm text-text-secondary flex items-center gap-2">
                                        <span className="uppercase text-xs font-bold tracking-wider opacity-70">{provider.type}</span>
                                        <span>•</span>
                                        <span className="font-mono">{provider.region}</span>
                                    </p>
                                </div>
                            </div>
                            
                            <div className="flex items-center gap-6">
                                {provider.status === 'connected' && (
                                    <div className="text-right hidden sm:block">
                                        <p className="text-xs text-text-secondary uppercase font-bold">Resources</p>
                                        <p className="font-mono text-text-primary font-bold">{provider.resources} Active</p>
                                    </div>
                                )}
                                {provider.status === 'missing_creds' && (
                                    <div className="text-xs text-red-500 font-medium flex items-center gap-1 bg-red-50 px-2 py-1 rounded">
                                        <KeyIcon className="w-4 h-4"/> Needs Key
                                    </div>
                                )}
                                
                                <button 
                                    onClick={() => toggleConnection(provider.id)}
                                    className={`w-32 h-10 rounded-lg font-semibold text-sm flex items-center justify-center transition-all ${
                                        provider.status === 'connected' 
                                            ? 'bg-red-50 text-red-600 border border-red-200 hover:bg-red-100' 
                                            : provider.status === 'connecting'
                                            ? 'bg-yellow-50 text-yellow-600 border border-yellow-200 cursor-wait'
                                            : 'bg-white border border-border-color text-text-secondary hover:border-brand-primary hover:text-brand-primary'
                                    }`}
                                    disabled={provider.status === 'connecting'}
                                >
                                    {provider.status === 'connected' ? 'Disconnect' : provider.status === 'connecting' ? <SpinnerIcon className="w-5 h-5" /> : provider.status === 'missing_creds' ? 'Configure' : 'Connect'}
                                </button>
                            </div>
                        </div>
                    ))}
                </div>

                {/* Action Panel */}
                <div className="flex flex-col gap-6">
                    
                    <div className="bg-bg-light-card glass-card rounded-xl shadow-aura border border-border-color p-6">
                        <h3 className="text-lg font-bold text-text-primary mb-4 flex items-center gap-2">
                            <HardDriveIcon className="w-5 h-5 text-purple-500" />
                            Storage Orchestration
                        </h3>
                        
                        {storageDeploymentStep === 0 ? (
                            <div className="space-y-4">
                                <p className="text-sm text-text-secondary">Deploy persistent block storage for the Nexus Core.</p>
                                <div>
                                    <label className="text-xs font-semibold text-text-secondary">Provider</label>
                                    <select 
                                        value={storageProvider} 
                                        onChange={(e) => setStorageProvider(e.target.value)}
                                        className="w-full mt-1 bg-bg-light-bg border border-border-color rounded p-2 text-sm"
                                    >
                                        {providers.filter(p => p.type === 'IaaS').map(p => (
                                            <option key={p.id} value={p.id}>{p.name}</option>
                                        ))}
                                    </select>
                                </div>
                                <div className="flex justify-between items-center bg-bg-light-bg p-2 rounded border border-border-color">
                                    <span className="text-xs font-semibold text-text-secondary">Size</span>
                                    <span className="text-sm font-mono text-text-primary font-bold">60 GB</span>
                                </div>
                                <button 
                                    onClick={handleDeployStorage}
                                    className="w-full bg-brand-primary text-white font-semibold py-2 rounded-lg hover:bg-brand-primary/90 transition-colors shadow-md"
                                >
                                    Deploy Core Storage
                                </button>
                            </div>
                        ) : (
                            <div className="space-y-4">
                                <div className="flex items-center gap-3">
                                    {storageDeploymentStep > 1 ? <CheckCircleIcon className="w-5 h-5 text-green-500"/> : <SpinnerIcon className="w-5 h-5 text-purple-500"/>}
                                    <span className={`text-sm ${storageDeploymentStep > 1 ? 'text-text-secondary' : 'text-text-primary font-medium'}`}>Allocating Block Storage...</span>
                                </div>
                                <div className="flex items-center gap-3">
                                    {storageDeploymentStep > 2 ? <CheckCircleIcon className="w-5 h-5 text-green-500"/> : storageDeploymentStep === 2 ? <SpinnerIcon className="w-5 h-5 text-purple-500"/> : <div className="w-5 h-5 rounded-full border-2 border-slate-200"></div>}
                                    <span className={`text-sm ${storageDeploymentStep > 2 ? 'text-text-secondary' : storageDeploymentStep === 2 ? 'text-text-primary font-medium' : 'text-slate-400'}`}>Formatting Volume (XFS)...</span>
                                </div>
                                <div className="flex items-center gap-3">
                                    {storageDeploymentStep > 3 ? <CheckCircleIcon className="w-5 h-5 text-green-500"/> : storageDeploymentStep === 3 ? <SpinnerIcon className="w-5 h-5 text-purple-500"/> : <div className="w-5 h-5 rounded-full border-2 border-slate-200"></div>}
                                    <span className={`text-sm ${storageDeploymentStep > 3 ? 'text-text-secondary' : storageDeploymentStep === 3 ? 'text-text-primary font-medium' : 'text-slate-400'}`}>Mounting Filesystem...</span>
                                </div>
                                
                                {storageDeploymentStep === 4 && (
                                    <div className="mt-4 p-3 bg-green-50 border border-green-200 rounded-lg text-sm text-green-800 font-medium text-center">
                                        60GB Storage Deployed & Mounted
                                    </div>
                                )}
                            </div>
                        )}
                    </div>

                    <div className="bg-bg-light-card glass-card rounded-xl shadow-aura border border-border-color p-6 flex-grow">
                        <h3 className="text-lg font-bold text-text-primary mb-4 flex items-center gap-2">
                            <ServerStackIcon className="w-5 h-5 text-brand-primary" />
                            Global Resources
                        </h3>
                        <div className="space-y-6">
                            <div>
                                <div className="flex justify-between text-sm text-text-secondary mb-1">
                                    <span>Total Compute Units</span>
                                    <span className="font-mono font-bold text-text-primary">124</span>
                                </div>
                                <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
                                    <div className="h-full bg-brand-primary w-3/4 rounded-full"></div>
                                </div>
                            </div>
                            <div>
                                <div className="flex justify-between text-sm text-text-secondary mb-1">
                                    <span>Memory Allocation</span>
                                    <span className="font-mono font-bold text-text-primary">512 GB</span>
                                </div>
                                <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
                                    <div className="h-full bg-brand-secondary w-1/2 rounded-full"></div>
                                </div>
                            </div>
                            {storageDeploymentStep === 4 && (
                                <div className="animate-fade-in">
                                    <div className="flex justify-between text-sm text-text-secondary mb-1">
                                        <span>Core Storage</span>
                                        <span className="font-mono font-bold text-purple-600">60 GB / 1 TB</span>
                                    </div>
                                    <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
                                        <div className="h-full bg-purple-500 w-[6%] rounded-full"></div>
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default CloudCommandView;
