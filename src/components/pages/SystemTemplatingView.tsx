import React, { useState, useEffect } from 'react';
import { AILogoIcon, CheckCircleIcon, SpinnerIcon, CodeBracketSquareIcon } from '../icons';

const simulationLogs = [
    { agent: 'Viren', text: 'Probing target backend at https://api.newapp.com...' },
    { agent: 'Viren', text: 'Discovered 5 REST API endpoints.' },
    { agent: 'Viraa', text: 'Connecting to Qdrant DB at vector.aethereal.io...' },
    { agent: 'Viraa', text: 'Authentication successful. Namespace \'app_metrics\' found.' },
    { agent: 'Lilith', text: 'Accessing web... Researching best practices for NodeJS app management.' },
    { agent: 'Lilith', text: 'Identified key metrics: API latency, error rate, user sessions.' },
    { agent: 'Lilith', text: 'Planning UI: Dashboard with charts, endpoint status list, live log viewer.' },
    { agent: 'Viren', text: 'Acknowledged plan. Preparing to build components.' },
];

const generatedComponents = ['LatencyChart.tsx', 'EndpointStatus.tsx', 'LogViewer.tsx', 'UserSessionMap.tsx'];

const SystemTemplatingView: React.FC = () => {
    const [step, setStep] = useState(0);
    const [logs, setLogs] = useState<string[]>([]);
    const [isDeploying, setIsDeploying] = useState(false);

    useEffect(() => {
        if (step === 2 && logs.length < simulationLogs.length) {
            const timer = setTimeout(() => {
                const logEntry = `[${simulationLogs[logs.length].agent}] ${simulationLogs[logs.length].text}`;
                setLogs(prev => [...prev, logEntry]);
            }, 600);
            return () => clearTimeout(timer);
        }
        if (step === 2 && logs.length === simulationLogs.length) {
            setTimeout(() => setStep(3), 1000);
        }
        if (step === 3) {
             setTimeout(() => setStep(4), 3000);
        }
    }, [step, logs]);

    const handleDeploy = () => {
        setIsDeploying(true);
        setTimeout(() => {
            setStep(1);
            setTimeout(() => setStep(2), 1500);
        }, 2000);
    };

    const getStepClass = (stepNumber: number) => {
        if (step > stepNumber) return 'completed';
        if (step === stepNumber) return 'active';
        return 'pending';
    };

    return (
        <div className="p-4 sm:p-6 lg:p-8 animate-scale-in">
            <header className="mb-6">
                <h2 className="text-3xl font-bold text-text-primary">System Templating</h2>
                <p className="text-text-secondary">Deploy a new Nexus instance to autonomously create a management UI for a target application.</p>
            </header>

            <div className="bg-bg-light-card glass-card rounded-xl shadow-aura border border-border-color p-8">
                {step === 0 && (
                    <div className="text-center animate-fade-in">
                        <h3 className="text-xl font-semibold mb-4">Ready to Deploy New Instance</h3>
                        <p className="text-text-secondary mb-6 max-w-xl mx-auto">This process will simulate an agent spawning, analyzing a target environment, and generating a custom UI. The entire workflow is autonomous.</p>
                        <button onClick={handleDeploy} disabled={isDeploying} className="bg-brand-primary text-white font-semibold px-6 py-3 rounded-lg hover:bg-brand-primary/90 transition-colors shadow-aura flex items-center justify-center mx-auto disabled:bg-slate-400">
                            {isDeploying ? <SpinnerIcon className="w-5 h-5 mr-2" /> : <AILogoIcon className="w-5 h-5 mr-2" />}
                            {isDeploying ? 'Deploying...' : 'Spawn New Nexus Instance'}
                        </button>
                    </div>
                )}

                {step > 0 && (
                    <div className="flex space-x-8">
                        {/* Stepper */}
                        <div className="w-1/4">
                            <ul className="space-y-4">
                                {['Deploy', 'Analyze', 'Generate', 'Finalize'].map((label, index) => (
                                    <li key={label} className={`flex items-center space-x-3 ${getStepClass(index + 1)}`}>
                                        <div className="step-icon">
                                            {step > index + 1 ? <CheckCircleIcon className="w-5 h-5" /> : <SpinnerIcon className="w-5 h-5 animate-spin" />}
                                        </div>
                                        <span className="font-semibold">{label}</span>
                                    </li>
                                ))}
                            </ul>
                        </div>
                        {/* Content */}
                        <div className="w-3/4 pl-8 border-l border-border-color min-h-[400px]">
                            {step === 1 && <div className="animate-fade-in"><h3 className="text-xl font-semibold">Deploying Agent...</h3><p className="text-text-secondary">Spawning a new templated instance of the Nexus Core.</p></div>}
                            {step === 2 && (
                                <div className="animate-fade-in">
                                    <h3 className="text-xl font-semibold mb-4">Analyzing Environment...</h3>
                                    <div className="bg-shell-bg text-shell-text font-mono text-xs rounded-lg p-4 h-64 overflow-y-auto">
                                        {logs.map((log, i) => <p key={i} className="whitespace-pre-wrap animate-fade-in">{log}</p>)}
                                    </div>
                                </div>
                            )}
                            {step === 3 && (
                                <div className="animate-fade-in">
                                    <h3 className="text-xl font-semibold mb-4">Generating UI Components...</h3>
                                    <div className="space-y-2">
                                        {generatedComponents.map(comp => (
                                            <div key={comp} className="bg-bg-light-bg p-3 rounded-lg flex items-center space-x-3">
                                                <CodeBracketSquareIcon className="w-5 h-5 text-brand-primary" />
                                                <span className="font-mono text-sm">{comp}</span>
                                                <div className="flex-grow h-2 bg-slate-300 rounded-full overflow-hidden">
                                                    <div className="bg-brand-secondary h-full animate-progress"></div>
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}
                            {step === 4 && (
                                <div className="text-center animate-fade-in">
                                    <CheckCircleIcon className="w-16 h-16 text-green-500 mx-auto mb-4" />
                                    <h3 className="text-2xl font-bold">Deployment Complete</h3>
                                    <p className="text-text-secondary mt-2">[Viraa] Archived deployment logs and generated components.</p>
                                    <p className="text-text-secondary">[System] Connection to Master Web Router established.</p>
                                    <button className="mt-6 bg-green-600 text-white font-semibold px-6 py-3 rounded-lg hover:bg-green-700 transition-colors">
                                        View New Application Dashboard
                                    </button>
                                </div>
                            )}
                        </div>
                    </div>
                )}
            </div>
             <style jsx>{`
                .step-icon {
                    width: 2rem; height: 2rem;
                    display: flex; align-items: center; justify-content: center;
                    border-radius: 50%;
                    background-color: var(--bg-light-bg);
                    color: var(--text-secondary);
                }
                .active .step-icon { background-color: var(--brand-primary); color: white; }
                .completed .step-icon { background-color: #10b981; color: white; }
                .completed .step-icon .animate-spin { display: none; }
                .pending .step-icon .animate-spin { display: none; }
                .pending .step-icon::before { content: '●'; font-size: 2rem; line-height: 1; }
                
                .active span { color: var(--brand-primary); }
                .completed span { color: var(--text-primary); }
                
                @keyframes progress {
                  from { width: 0%; }
                  to { width: 100%; }
                }
                .animate-progress {
                  animation: progress 2s ease-out forwards;
                }
            `}</style>
        </div>
    );
};

export default SystemTemplatingView;