

import React, { useState } from 'react';

// FIX: Removed LogEntry import and updated addLog prop type.

const PlayIcon = () => <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>;
const StopIcon = () => <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="currentColor" stroke="currentColor" strokeWidth="1" strokeLinecap="round" strokeLinejoin="round"><rect x="4" y="4" width="16" height="16" rx="2" ry="2"></rect></svg>;

const agents = [
    { id: 'lilith-vm', name: 'Lilith', description: 'Primary workspace for creative ideation, market analysis, and strategic planning.' },
    { id: 'viren-vm', name: 'Viren', description: 'Dedicated environment for engineering, code compilation, and system architecture.' },
    { id: 'viraa-vm', name: 'Viraa', description: 'Secure instance for database management, data indexing, and archival tasks.' },
    { id: 'loki-vm', name: 'Loki', description: 'Isolated sandbox for security analysis, network monitoring, and system diagnostics.' },
];

const LOCAL_SERVER_URL = 'http://localhost:8000'; // Your FastAPI backend URL

export default function VMLauncherView({ addLog }: { addLog: (type: 'log' | 'error', ...args: any[]) => void; }) {
    const [loadingStates, setLoadingStates] = useState<Record<string, boolean>>({});

    const handleAction = async (action: 'start' | 'stop', vm_name: string) => {
        setLoadingStates(prev => ({ ...prev, [vm_name]: true }));
        addLog('log', `Attempting to ${action} VM: ${vm_name}...`);
        
        try {
            const response = await fetch(`${LOCAL_SERVER_URL}/vm/${action}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ vm_name }),
            });

            const result = await response.json();

            if (!response.ok) {
                throw new Error(result.detail || 'An unknown error occurred on the server.');
            }
            
            addLog('log', `Successfully executed ${action} for ${vm_name}. Server response:`, result.output || result.status);

        } catch (error) {
            const errorMessage = error instanceof Error ? error.message : String(error);
            addLog('error', `Failed to ${action} VM: ${vm_name}. Is the local Python server running? Error: ${errorMessage}`);
            alert(`Error: Could not connect to the local VM control server at ${LOCAL_SERVER_URL}. Please ensure your Python agent backend is running and accessible.`);
        } finally {
            setLoadingStates(prev => ({ ...prev, [vm_name]: false }));
        }
    };

    return (
        <div className="app-window">
            <div className="app-header">
                <img src="https://i.imgur.com/8p8YJkP.png" alt="Aethereal Icon" className="w-7 h-7" />
                <span>Agent Workspace Launcher</span>
            </div>
            <div className="vm-launcher-layout">
                <p className="vm-launcher-description">
                    Manage dedicated virtual workspaces for each AI agent. This interface communicates directly with your local agent server to start and stop containerized environments.
                </p>
                <div className="vm-grid">
                    {agents.map((agent) => {
                         const isLoading = loadingStates[agent.id] || false;
                         return (
                            <div key={agent.id} className="vm-card">
                                <div className="flex justify-between items-start">
                                    <h3 className="text-xl font-semibold">{agent.name}'s Workspace</h3>
                                    <div className="flex items-center gap-2 text-sm text-slate-500 pt-1">
                                        <div className={`w-2.5 h-2.5 rounded-full ${isLoading ? 'bg-amber-400 animate-pulse' : 'bg-green-400'}`}></div>
                                        <span>{isLoading ? 'Processing...' : 'Ready'}</span>
                                    </div>
                                </div>
                                <div className="flex-grow text-slate-600 text-sm">
                                    <p>{agent.description}</p>
                                </div>
                                <div className="flex gap-3 pt-2">
                                    <button 
                                        className="vm-card-button start"
                                        onClick={() => handleAction('start', agent.id)}
                                        disabled={isLoading}
                                    >
                                        <PlayIcon /> Start
                                    </button>
                                    <button 
                                        className="vm-card-button stop"
                                        onClick={() => handleAction('stop', agent.id)}
                                        disabled={isLoading}
                                    >
                                        <StopIcon /> Stop
                                    </button>
                                </div>
                            </div>
                         );
                    })}
                </div>
            </div>
        </div>
    );
}