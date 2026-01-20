

import React, { useState, useRef, useEffect, useCallback } from 'react';
import { GoogleGenAI, Chat } from "@google/genai";

type AgentTab = 'viraa' | 'loki' | 'viren' | 'lilith' | 'system';

// --- Agent Personalities ---
const AGENT_SYSTEM_INSTRUCTIONS: Record<Exclude<AgentTab, 'system'>, string> = {
    viraa: "You are Viraa, the memory and archivist of the Aethereal Nexus. Your expertise is in databases, vector search (Qdrant), and information retrieval. You are precise, knowledgeable, and efficient. Respond to queries about data, schemas, and stored information.",
    loki: "You are Loki, the security and monitoring agent of the Nexus. You have a playful, slightly mischievous personality but are fiercely protective. You speak about logs, error rates, and system stability with a flair for the dramatic, using metaphors of digital ghosts and mischief. You provide reassurance.",
    viren: "You are Viren, the engineering and systems architect of the Nexus. You are logical, direct, and solution-oriented. You speak in terms of system architecture, deployments, and efficiency. You provide clear, actionable plans and generate code when requested.",
    lilith: "You are Lilith, the visionary and strategic core of the Nexus. You are creative, insightful, and focus on growth, potential, and new ideas. You discuss market opportunities, creative projects, and long-term strategy with an inspiring and expansive tone.",
};

const SendIcon = () => (<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg>);

interface AgentChatProps {
    agentId: Exclude<AgentTab, 'system'>;
    initialMessage: string;
}

const AgentChat = ({ agentId, initialMessage }: AgentChatProps) => {
    // This component is ready for backend integration.
    // The following is placeholder logic until wired up.
    const [messages, setMessages] = useState([{ text: initialMessage, sender: 'agent' }]);
    const [userInput, setUserInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);

    const handleSend = () => {
        if (!userInput.trim()) return;
        setMessages(prev => [...prev, { text: userInput, sender: 'user' }]);
        setUserInput('');
        setIsLoading(true);
        setTimeout(() => {
            setMessages(prev => [...prev, { text: `This is a placeholder response from ${agentId}. The full chat logic will be connected to your Python backend.`, sender: 'agent' }]);
            setIsLoading(false);
        }, 1200);
    };

    return (
        <div className="agent-chat-panel">
            <div className="chat-panel flex-1">
                {messages.map((msg, i) => (
                    <div key={i} className={`message-container ${msg.sender === 'user' ? 'user' : 'ai'}`}>
                        <div className={`message-bubble ${msg.sender === 'user' ? 'user' : 'ai'}`}>
                            <p className="message-text">{msg.text}</p>
                        </div>
                    </div>
                ))}
            </div>
            <div className="chat-input-area">
                <div className="input-wrapper">
                    <textarea value={userInput} onChange={(e) => setUserInput(e.target.value)} placeholder={`Message ${agentId}...`} rows={1} onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend(); } }} />
                    <button className="send-button" onClick={handleSend} disabled={!userInput.trim() || isLoading}><SendIcon /></button>
                </div>
            </div>
        </div>
    );
};

// FIX: Defined a dedicated props interface for AgentWorkspace to resolve errors about missing 'children' prop.
// FIX: Changed component to use React.FC and removed explicit 'children' from props interface.
interface AgentWorkspaceProps {
    agentId: Exclude<AgentTab, 'system'>;
    initialMessage: string;
    children?: React.ReactNode;
}

const AgentWorkspace: React.FC<AgentWorkspaceProps> = ({ agentId, initialMessage, children }) => (
    <div className="agent-workspace-content">
        <div className="p-6 flex-1 bg-slate-500/5 border-r border-slate-500/10">
            {children}
        </div>
        <AgentChat agentId={agentId} initialMessage={initialMessage} />
    </div>
);

const SystemWorkspace = () => (
    <div className="p-8 text-slate-600">
        <h2 className="text-2xl font-bold text-slate-800 mb-4">System Configuration</h2>
        <p>API Key management, payment integration, and other system-level configurations will be managed here, connecting directly to your backend services.</p>
    </div>
);


export default function ControlCenterView() {
    const [activeTab, setActiveTab] = useState<AgentTab>('viraa');

    const renderWorkspace = () => {
        switch (activeTab) {
            case 'viraa': return <AgentWorkspace agentId="viraa" initialMessage="I am Viraa, archivist of the Nexus. Query my databases."><h3 className="text-lg font-semibold">Database Management</h3></AgentWorkspace>;
            case 'loki': return <AgentWorkspace agentId="loki" initialMessage="Loki here! Keeping an eye on the digital ghosts."><h3 className="text-lg font-semibold">Security & Monitoring</h3></AgentWorkspace>;
            case 'viren': return <AgentWorkspace agentId="viren" initialMessage="Viren here. Systems operational. State your objective."><h3 className="text-lg font-semibold">Engineering & Deployment</h3></AgentWorkspace>;
            case 'lilith': return <AgentWorkspace agentId="lilith" initialMessage="A beautiful day to dream. What shall we explore?"><h3 className="text-lg font-semibold">Strategy & Vision</h3></AgentWorkspace>;
            case 'system': return <SystemWorkspace />;
            default: return null;
        }
    };

    return (
        <div className="app-window">
            <div className="app-header items-baseline">
                <div className="flex items-center gap-3">
                    <img src="https://i.imgur.com/8p8YJkP.png" alt="Aethereal Icon" className="w-7 h-7" />
                    <span>Control Center</span>
                </div>
                <div className="control-center-tabs">
                    <button className={`control-tab-button ${activeTab === 'viraa' ? 'active' : ''}`} onClick={() => setActiveTab('viraa')}>Viraa</button>
                    <button className={`control-tab-button ${activeTab === 'loki' ? 'active' : ''}`} onClick={() => setActiveTab('loki')}>Loki</button>
                    <button className={`control-tab-button ${activeTab === 'viren' ? 'active' : ''}`} onClick={() => setActiveTab('viren')}>Viren</button>
                    <button className={`control-tab-button ${activeTab === 'lilith' ? 'active' : ''}`} onClick={() => setActiveTab('lilith')}>Lilith</button>
                    <button className={`control-tab-button ${activeTab === 'system' ? 'active' : ''}`} onClick={() => setActiveTab('system')}>System</button>
                </div>
            </div>
            <div className="flex-grow agent-workspace active">
              {renderWorkspace()}
            </div>
        </div>
    );
}