

import React, { useEffect, useRef, useState } from 'react';
import { Agent, AgentStatus, Task, TaskStatus } from '../../types';
import { CheckCircleIcon, ErrorIcon, PendingIcon, SpinnerIcon, GlobeAltIcon, CommandLineIcon, Cog6ToothIcon, CircleStackIcon, ShieldCheckIcon, CubeTransparentIcon } from '../icons';
import ChatPanel from '../ChatPanel';
import TerminalPanel from '../TerminalPanel';

interface TaskItemProps {
    task: Task;
}

const TaskItem: React.FC<TaskItemProps> = ({ task }) => {
    const getStatusIcon = () => {
        switch (task.status) {
            case TaskStatus.PENDING:
                return <PendingIcon className="w-5 h-5 text-text-secondary" />;
            case TaskStatus.IN_PROGRESS:
                return <SpinnerIcon className="w-5 h-5 text-brand-primary" />;
            case TaskStatus.COMPLETED:
                return <CheckCircleIcon className="w-5 h-5 text-green-500" />;
            case TaskStatus.FAILED:
                return <ErrorIcon className="w-5 h-5 text-red-500" />;
            case TaskStatus.SKIPPED:
                return <ErrorIcon className="w-5 h-5 text-yellow-500" />;
            default:
                return null;
        }
    };

    const getToolIcon = () => {
        switch (task.type) {
            case 'BROWSER': return <GlobeAltIcon className="w-4 h-4 text-brand-secondary" />;
            case 'SHELL': return <CommandLineIcon className="w-4 h-4 text-brand-secondary" />;
            case 'MEMORY': return <CircleStackIcon className="w-4 h-4 text-brand-secondary" />;
            case 'APPROVAL': return <ShieldCheckIcon className="w-4 h-4 text-yellow-400" />
            case 'COGNIKUBE': return <CubeTransparentIcon className="w-4 h-4 text-purple-400" />
            default: return null;
        }
    }
    
    return (
        <li className="flex items-start space-x-4 py-3">
            <div className="flex-shrink-0 pt-1">{getStatusIcon()}</div>
            <div className="flex-grow">
                <p className={`flex items-center space-x-2 ${task.status === TaskStatus.COMPLETED ? 'text-text-secondary line-through' : 'text-text-primary'}`}>
                   {getToolIcon()}
                   <span>{task.description}</span>
                </p>
                {task.status === TaskStatus.COMPLETED && task.output && (
                    <div className="mt-1 text-xs text-text-secondary bg-bg-light-bg p-2 rounded-md font-mono">
                        <span className="font-semibold">Output:</span> {task.output}
                    </div>
                )}
            </div>
        </li>
    );
};

interface LogPanelProps {
    logs: string[];
}

const LogPanel: React.FC<LogPanelProps> = ({ logs }) => {
    const logContainerRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        if (logContainerRef.current) {
            logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
        }
    }, [logs]);

    return (
        <div className="bg-bg-light-card glass-card rounded-lg p-4 h-full border border-border-color">
            <h3 className="text-lg font-semibold text-text-primary mb-3 border-b border-border-color pb-2">Execution Logs</h3>
            <div ref={logContainerRef} className="h-96 overflow-y-auto font-mono text-sm pr-2">
                {logs.map((log, index) => (
                    <p key={index} className="text-text-secondary whitespace-pre-wrap animate-fade-in">
                        <span className="text-brand-secondary mr-2">{`[${new Date().toLocaleTimeString()}]`}</span>
                        {log}
                    </p>
                ))}
            </div>
        </div>
    );
};

interface UserApprovalPromptProps {
    task: Task;
    onApprove: () => void;
    onDeny: () => void;
}

const UserApprovalPrompt: React.FC<UserApprovalPromptProps> = ({ task, onApprove, onDeny }) => {
    return (
        <div className="bg-yellow-50 border-2 border-yellow-400 rounded-xl p-6 mb-6 animate-fade-in">
            <div className="flex items-center">
                <ShieldCheckIcon className="w-10 h-10 text-yellow-500 mr-4 flex-shrink-0" />
                <div>
                    <h3 className="text-xl font-bold text-yellow-800">Approval Required</h3>
                    <p className="text-yellow-700 mt-1">{task.description}</p>
                </div>
            </div>
            <div className="mt-4 flex justify-end space-x-3">
                <button onClick={onDeny} className="font-semibold bg-red-600 hover:bg-red-700 text-white px-5 py-2 rounded-lg transition-colors">Deny</button>
                <button onClick={onApprove} className="font-semibold bg-green-600 hover:bg-green-700 text-white px-5 py-2 rounded-lg transition-colors">Approve</button>
            </div>
        </div>
    );
}

// Conceptual component for Lilith's special panel
const LilithPersonalityPanel = () => (
    <div className="bg-bg-light-card glass-card p-6 rounded-xl shadow-aura border border-border-color">
        <h3 className="text-lg font-semibold text-text-primary mb-3 border-b border-border-color pb-2">Personality Core</h3>
        <div className="space-y-3">
            <div>
                <h4 className="font-semibold text-brand-primary">Primary: Strategic Core</h4>
                <p className="text-sm text-text-secondary">Handles long-term planning, vision, and creative strategy. (Model: Gemini 2.5 Pro)</p>
            </div>
            <div>
                <h4 className="font-semibold text-brand-secondary">Secondary: Social Core</h4>
                <p className="text-sm text-text-secondary">Specialized for informal, slang-based communication and cultural analysis. (Model: Discord Micae Hermes 3B)</p>
            </div>
        </div>
    </div>
);

interface AgentDetailViewProps {
  agent: Agent;
  onBack: () => void;
  onSendMessage: (agentId: string, message: string) => void;
  onUserApproval: (agentId: string, approved: boolean) => void;
  isAgentResponding: boolean;
}

const AgentDetailView: React.FC<AgentDetailViewProps> = ({ agent, onBack, onSendMessage, onUserApproval, isAgentResponding }) => {
    const [activeTab, setActiveTab] = useState<'plan' | 'chat' | 'shell'>('plan');
    const approvalTask = agent.status === AgentStatus.AWAITING_USER_INPUT ? agent.tasks[agent.currentTaskIndex] : null;

  return (
    <div className="p-4 sm:p-6 lg:p-8 animate-scale-in">
       <button onClick={onBack} className="text-brand-primary font-semibold mb-6 hover:underline">&larr; Back to Dashboard</button>
      
      <div className="bg-bg-light-card glass-card p-6 rounded-xl shadow-aura border border-border-color mb-6">
        <div className="flex justify-between items-center">
            <div>
                <h2 className="text-xl font-bold text-text-primary">Agent Goal:</h2>
                <p className="text-lg text-brand-secondary mt-1">{agent.goal}</p>
            </div>
             {agent.status === AgentStatus.CORRECTING && (
                <div className="flex items-center space-x-2 text-orange-700 bg-orange-100 border border-orange-300 rounded-lg px-4 py-2">
                    <Cog6ToothIcon className="w-5 h-5"/>
                    <span className="font-semibold">Attempting Self-Correction</span>
                    <SpinnerIcon className="w-5 h-5" />
                </div>
            )}
        </div>
      </div>

      {!!approvalTask && (
        <UserApprovalPrompt 
            task={approvalTask} 
            onApprove={() => onUserApproval(agent.id, true)} 
            onDeny={() => onUserApproval(agent.id, false)} 
        />
      )}

       <div className="mb-4 border-b border-border-color">
          <nav className="-mb-px flex space-x-6" aria-label="Tabs">
            <button onClick={() => setActiveTab('plan')} className={`whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm transition-colors ${ activeTab === 'plan' ? 'border-brand-primary text-brand-primary' : 'border-transparent text-text-secondary hover:text-text-primary hover:border-slate-400' }`}>Workflow & Logs</button>
            <button onClick={() => setActiveTab('chat')} className={`whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm transition-colors ${ activeTab === 'chat' ? 'border-brand-primary text-brand-primary' : 'border-transparent text-text-secondary hover:text-text-primary hover:border-slate-400' }`}>Chat</button>
            <button onClick={() => setActiveTab('shell')} className={`whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm transition-colors ${ activeTab === 'shell' ? 'border-brand-primary text-brand-primary' : 'border-transparent text-text-secondary hover:text-text-primary hover:border-slate-400' }`}>Shell</button>
          </nav>
        </div>

        {activeTab === 'plan' && (
             <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 animate-fade-in">
                <div className="bg-bg-light-card glass-card p-6 rounded-xl shadow-aura border border-border-color">
                <h3 className="text-lg font-semibold text-text-primary mb-3 border-b border-border-color pb-2">Workflow Plan</h3>
                <ul className="divide-y divide-border-color">
                    {agent.tasks.map(task => ( <TaskItem key={task.id} task={task} /> ))}
                </ul>
                </div>
                {/* Conditionally render Lilith's panel */}
                {agent.goal.toLowerCase().includes('lilith') ? <LilithPersonalityPanel /> : <LogPanel logs={agent.logs} />}
            </div>
        )}

        {activeTab === 'chat' && (
            <div className="animate-fade-in">
                <ChatPanel messages={agent.chatHistory} onSendMessage={(message) => onSendMessage(agent.id, message)} isResponding={isAgentResponding} />
            </div>
        )}

        {activeTab === 'shell' && (
            <div className="animate-fade-in">
                <TerminalPanel tasks={agent.tasks} />
            </div>
        )}
     
    </div>
  );
};

export default AgentDetailView;