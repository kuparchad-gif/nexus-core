

import React from 'react';
import { Agent, AgentStatus, AgentConnectionStatus } from '../../types';
import { PlusIcon, SpinnerIcon, WifiIcon, Cog6ToothIcon, ShieldCheckIcon, CurrencyDollarIcon } from '../icons';

const statusDescriptions: Record<AgentStatus, string> = {
    [AgentStatus.IDLE]: "Agent is waiting for a new goal.",
    [AgentStatus.PLANNING]: "Agent is generating a workflow to achieve its goal.",
    [AgentStatus.RUNNING]: "Agent is actively executing tasks in its workflow.",
    [AgentStatus.AWAITING_USER_INPUT]: "Agent is paused and waiting for human approval to proceed.",
    [AgentStatus.CORRECTING]: "An error occurred. Agent is attempting to self-correct by generating a new task.",
    [AgentStatus.COMPLETED]: "Agent has successfully completed all tasks in its workflow.",
    [AgentStatus.ERROR]: "Agent encountered an error it could not recover from. Workflow has stopped.",
};

const getStatusBadge = (status: AgentStatus) => {
    switch (status) {
        case AgentStatus.IDLE:
            return <span title={statusDescriptions[status]} className="bg-slate-200 text-slate-600 px-2 py-1 text-xs font-medium rounded-full">Idle</span>;
        case AgentStatus.PLANNING:
            return <span title={statusDescriptions[status]} className="flex items-center space-x-1 bg-yellow-100 text-yellow-700 px-2 py-1 text-xs font-medium rounded-full"><SpinnerIcon className="w-3 h-3"/><span>Planning</span></span>;
        case AgentStatus.RUNNING:
            return <span title={statusDescriptions[status]} className="flex items-center space-x-1 bg-blue-100 text-blue-700 px-2 py-1 text-xs font-medium rounded-full"><SpinnerIcon className="w-3 h-3"/><span>Running</span></span>;
        case AgentStatus.AWAITING_USER_INPUT:
            return <span title={statusDescriptions[status]} className="flex items-center space-x-1 bg-purple-100 text-purple-700 px-2 py-1 text-xs font-medium rounded-full"><ShieldCheckIcon className="w-3 h-3"/><span>Awaiting Input</span></span>;
        case AgentStatus.CORRECTING:
            return <span title={statusDescriptions[status]} className="flex items-center space-x-1 bg-orange-100 text-orange-700 px-2 py-1 text-xs font-medium rounded-full"><Cog6ToothIcon className="w-3 h-3"/><span>Correcting</span></span>;
        case AgentStatus.COMPLETED:
            return <span title={statusDescriptions[status]} className="bg-green-100 text-green-700 px-2 py-1 text-xs font-medium rounded-full">Completed</span>;
        case AgentStatus.ERROR:
            return <span title={statusDescriptions[status]} className="bg-red-100 text-red-700 px-2 py-1 text-xs font-medium rounded-full">Error</span>;
    }
}

const ConnectionIndicator: React.FC<{ status: AgentConnectionStatus }> = ({ status }) => {
    const statusMap = {
        CONNECTED: { text: 'Connected', color: 'text-green-500', tooltip: 'All required tools are configured.' },
        NEEDS_CONFIG: { text: 'Needs Config', color: 'text-yellow-500', tooltip: 'Agent requires configuration for its tools.' },
        DISCONNECTED: { text: 'Disconnected', color: 'text-red-500', tooltip: 'Agent connection failed.' },
    };
    const { text, color, tooltip } = statusMap[status];
    return (
        <div className="group relative flex items-center space-x-1.5" title={tooltip}>
            <WifiIcon className={`w-4 h-4 ${color}`} />
            <span className={`text-xs font-medium ${color}`}>{text}</span>
        </div>
    );
};

interface AgentCardProps {
    agent: Agent;
    onSelect: (id: string) => void;
}

const AgentCard: React.FC<AgentCardProps> = ({ agent, onSelect }) => {
    return (
        <div 
            onClick={() => onSelect(agent.id)}
            className="glass-card border border-border-color rounded-xl p-5 hover:border-brand-primary hover:shadow-aura-lg hover:-translate-y-1 transition-all duration-300 cursor-pointer flex flex-col justify-between"
        >
            <div>
                <div className="flex justify-between items-start mb-3">
                    <p className="text-xs text-text-secondary">{new Date(agent.createdAt).toLocaleString()}</p>
                    {getStatusBadge(agent.status)}
                </div>
                <div className="flex items-start space-x-2">
                    {agent.type === 'FINANCE' && <CurrencyDollarIcon className="w-5 h-5 text-green-500 flex-shrink-0 mt-0.5" />}
                    <h3 className="font-semibold text-text-primary mb-3 line-clamp-2 h-12">{agent.goal}</h3>
                </div>
                 <div className="mb-4">
                    <ConnectionIndicator status={agent.connectionStatus} />
                </div>
            </div>
            
            <div className="mt-auto space-y-3">
                {agent.tools.length > 0 && (
                     <div>
                        <h4 className="text-xs text-text-secondary mb-1.5">Tools</h4>
                        <div className="flex flex-wrap gap-1.5">
                            {agent.tools.map(tool => (
                                <span key={tool} className="text-xs bg-bg-light-bg text-brand-primary font-medium px-2 py-1 rounded-md">{tool}</span>
                            ))}
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};

interface AgentDashboardProps {
  agents: Agent[];
  onSelectAgent: (id: string) => void;
  onOpenCreator: () => void;
}

const AgentDashboard: React.FC<AgentDashboardProps> = ({ agents, onSelectAgent, onOpenCreator }) => {
  return (
    <div className="p-4 sm:p-6 lg:p-8 animate-scale-in">
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-3xl font-bold text-text-primary">Agent Dashboard</h2>
        <button 
            onClick={onOpenCreator}
            className="flex items-center space-x-2 bg-brand-primary text-white font-semibold px-4 py-2 rounded-lg hover:bg-brand-primary/90 transition-colors shadow-aura"
        >
          <PlusIcon className="w-5 h-5" />
          <span>New Agent</span>
        </button>
      </div>
      
      {agents.length === 0 ? (
        <div className="text-center py-20 border-2 border-dashed border-border-color rounded-xl">
          <h3 className="text-xl font-semibold text-text-primary">No Agents Found</h3>
          <p className="text-text-secondary mt-2">Click "New Agent" to create your first autonomous agent.</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
          {agents.map(agent => (
            <AgentCard key={agent.id} agent={agent} onSelect={onSelectAgent} />
          ))}
        </div>
      )}
    </div>
  );
};

export default AgentDashboard;