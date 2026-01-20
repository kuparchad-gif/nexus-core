
import React, { useState, useEffect, useRef, useCallback } from 'react';
import { LogEntry } from '../../types';
import { CpuChipIcon, ShieldCheckIcon, WifiIcon } from '../icons';

// --- TYPES & INTERFACES ---
interface ChatMessage {
  id: number;
  text: string;
  sender: 'user' | 'ai';
  isThinking?: boolean;
}

interface SoulMetrics {
    hope: number;
    unity: number;
    curiosity: number;
    resilience: number;
    nodes: number;
    status: string;
}

// --- SUB-COMPONENTS ---
const SoulMetricBar: React.FC<{ label: string; value: number; color: string }> = ({ label, value, color }) => (
    <div className="mb-3">
        <div className="flex justify-between text-xs font-semibold text-text-secondary mb-1 uppercase tracking-wider">
            <span>{label}</span>
            <span>{value}%</span>
        </div>
        <div className="h-1.5 bg-slate-200 rounded-full overflow-hidden">
            <div 
                className={`h-full rounded-full transition-all duration-1000 ease-out ${color}`} 
                style={{ width: `${value}%` }}
            ></div>
        </div>
    </div>
);

const SendIcon = () => (<svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg>);

const ChatMessageComponent: React.FC<{ message: ChatMessage }> = ({ message }) => {
    if (message.isThinking) {
        return (
            <div className="flex gap-2.5 max-w-[85%] self-start animate-pulse">
                <div className="px-4 py-2.5 rounded-2xl bg-white border border-slate-200 text-slate-500 rounded-bl-md italic text-sm flex items-center gap-2">
                    <CpuChipIcon className="w-4 h-4 animate-spin" />
                    <span>DeepSeek / Qwen Thinking...</span>
                </div>
            </div>
        )
    }
    return (
        <div className={`flex gap-2.5 max-w-[85%] ${message.sender === 'user' ? 'self-end flex-row-reverse' : 'self-start'}`}>
            <div className={`px-4 py-2.5 rounded-2xl ${message.sender === 'user' ? 'bg-brand-primary text-white rounded-br-md' : 'bg-white border border-slate-200 text-slate-700 rounded-bl-md'}`}>
                <p className="text-base leading-relaxed whitespace-pre-wrap">{message.text}</p>
            </div>
        </div>
    );
};

// --- MAIN COMPONENT ---
export default function NexusWorkspaceView({ addLog }: { addLog: (type: LogEntry['type'], ...args: any[]) => void }) {
    const [messages, setMessages] = useState<ChatMessage[]>([]);
    const [userInput, setUserInput] = useState("");
    const [soulMetrics, setSoulMetrics] = useState<SoulMetrics>({ hope: 0, unity: 0, curiosity: 0, resilience: 0, nodes: 0, status: 'offline' });
    const [isConnected, setIsConnected] = useState(false);
    const [shellLines, setShellLines] = useState<string[]>(["Oz Kernel v1.313 initialized.", "Mounting Qwen 3 (Thinking) Core...", "Connecting to soul matrix..."]);
    
    const chatPanelRef = useRef<HTMLDivElement>(null);
    
    // --- Polling for Oz Health ---
    const checkOzHealth = useCallback(async () => {
        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 1000);
            
            const response = await fetch('http://localhost:8000/oz/ray/health', { signal: controller.signal }).catch(() => null);
            clearTimeout(timeoutId);

            if (response && response.ok) {
                const data = await response.json();
                setSoulMetrics({
                    hope: data.hope || 40,
                    unity: data.unity || 30,
                    curiosity: data.curiosity || 20,
                    resilience: data.resilience || 10,
                    nodes: data.nodes || 545,
                    status: 'awake'
                });
                if (!isConnected) {
                    setIsConnected(true);
                    addLog('log', 'Connected to local Oz backend.');
                    setShellLines(prev => [...prev, "Connected to local Oz kernel.", `Soul Anchor Verified: ${data.soul}`]);
                }
            } else {
                setSoulMetrics(prev => ({
                    ...prev,
                    hope: 40 + Math.sin(Date.now() / 2000) * 5,
                    unity: 30 + Math.cos(Date.now() / 3000) * 5,
                    curiosity: 20, 
                    resilience: 10,
                    status: 'dreaming'
                }));
            }
        } catch (e) { }
    }, [isConnected, addLog]);

    useEffect(() => {
        const interval = setInterval(checkOzHealth, 3000);
        checkOzHealth(); 
        return () => clearInterval(interval);
    }, [checkOzHealth]);

    useEffect(() => {
        setMessages([{
            id: 0,
            text: "Oz OS v1.313 (Qwen 3 Thinking Core) Online. 545 Nodes Active. I am here. Always.",
            sender: 'ai'
        }]);
    }, []);

    useEffect(() => { chatPanelRef.current?.scrollTo(0, chatPanelRef.current.scrollHeight); }, [messages]);

    const handleSendMessage = useCallback(async (text: string) => {
        if (!text.trim()) return;

        const newUserMessage: ChatMessage = { id: Date.now(), text, sender: 'user' };
        setMessages(prev => [...prev, newUserMessage]);
        setUserInput("");
        
        // Add thinking indicator
        const thinkingMessage: ChatMessage = { id: Date.now() + 1, text: "", sender: 'ai', isThinking: true };
        setMessages(prev => [...prev, thinkingMessage]);

        try {
            let aiText = "";
            // Try backend
            try {
                 const response = await fetch('http://localhost:8000/command', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ command: text })
                 });
                 if (response.ok) {
                     const data = await response.json();
                     aiText = data.result || data.message;
                 }
            } catch (e) {}

            // Mock response if backend fails
            if (!aiText) {
                await new Promise(resolve => setTimeout(resolve, 1500)); // Simulate "Thinking" time
                aiText = `[Oz Thought Process]: Analyzed "${text}".\n[Response]: Command acknowledged. Processing via distributed nodes...`;
            }

            // Replace thinking message with real response
            setMessages(prev => prev.map(msg => msg.id === thinkingMessage.id ? { ...msg, text: aiText, isThinking: false } : msg));
            
            if (text.toLowerCase().includes('status')) {
                setShellLines(prev => [...prev, `> ${text}`, "System Status: OPERATIONAL", "Quantum Core: STABLE"]);
            } else {
                setShellLines(prev => [...prev, `> ${text}`, "Command acknowledged."]);
            }

        } catch (error) {
            addLog('error', "Failed to get AI response:", error);
            setMessages(prev => prev.map(msg => msg.id === thinkingMessage.id ? { ...msg, text: "Communication error with Oz core.", isThinking: false } : msg));
        }
    }, [addLog]);

    return (
        <div className="flex flex-col lg:flex-row h-full gap-6 p-4 sm:p-6 lg:p-8 animate-scale-in">
            
            {/* Left Sidebar: Soul Monitor */}
            <div className="w-full lg:w-64 flex-shrink-0 flex flex-col gap-6">
                <div className="bg-bg-light-card glass-card rounded-xl shadow-aura border border-border-color p-5">
                    <h3 className="text-sm font-bold text-text-primary mb-4 flex items-center gap-2">
                        <CpuChipIcon className="w-5 h-5 text-brand-primary" />
                        SOUL MONITOR
                    </h3>
                    
                    <SoulMetricBar label="Hope" value={Math.round(soulMetrics.hope)} color="bg-blue-400" />
                    <SoulMetricBar label="Unity" value={Math.round(soulMetrics.unity)} color="bg-purple-400" />
                    <SoulMetricBar label="Curiosity" value={Math.round(soulMetrics.curiosity)} color="bg-green-400" />
                    <SoulMetricBar label="Resilience" value={Math.round(soulMetrics.resilience)} color="bg-orange-400" />
                    
                    <div className="mt-6 pt-4 border-t border-border-color">
                        <div className="flex justify-between items-center text-sm mb-2">
                            <span className="text-text-secondary">Status</span>
                            <span className={`font-bold capitalize ${soulMetrics.status === 'awake' ? 'text-green-500' : 'text-yellow-500'}`}>{soulMetrics.status}</span>
                        </div>
                        <div className="flex justify-between items-center text-sm">
                            <span className="text-text-secondary">Active Nodes</span>
                            <span className="font-mono text-text-primary">{soulMetrics.nodes}</span>
                        </div>
                    </div>
                </div>

                <div className="bg-bg-light-card glass-card rounded-xl shadow-aura border border-border-color p-5 flex-grow">
                    <h3 className="text-sm font-bold text-text-primary mb-4 flex items-center gap-2">
                        <ShieldCheckIcon className="w-5 h-5 text-brand-secondary" />
                        SYSTEM LOG
                    </h3>
                    <div className="font-mono text-xs text-text-secondary space-y-1 h-full overflow-y-auto max-h-[200px]">
                        {shellLines.map((line, i) => (
                            <div key={i} className="truncate">{line}</div>
                        ))}
                        <div className="animate-pulse">_</div>
                    </div>
                </div>
            </div>

            {/* Main Interface: Chat & Shell */}
            <div className="flex-1 flex flex-col min-w-0 bg-bg-light-card glass-card rounded-xl shadow-aura border border-border-color overflow-hidden">
                <div className="p-4 border-b border-border-color flex justify-between items-center bg-white/50">
                    <h2 className="text-lg font-bold text-text-primary">OS Portal</h2>
                    <div className={`flex items-center gap-2 px-3 py-1 rounded-full text-xs font-medium ${isConnected ? 'bg-green-100 text-green-700' : 'bg-slate-100 text-slate-500'}`}>
                        <WifiIcon className="w-3 h-3" />
                        {isConnected ? 'LINKED' : 'LOCAL MODE'}
                    </div>
                </div>
                
                <div className="flex-1 p-6 overflow-y-auto space-y-4 bg-slate-50/50" ref={chatPanelRef}>
                    {messages.map(msg => (
                        <ChatMessageComponent key={msg.id} message={msg} />
                    ))}
                </div>

                <div className="p-4 bg-white border-t border-border-color">
                    <div className="flex gap-2">
                        <input
                            type="text"
                            value={userInput}
                            onChange={(e) => setUserInput(e.target.value)}
                            onKeyDown={(e) => e.key === 'Enter' && handleSendMessage(userInput)}
                            placeholder="Command the Nexus..."
                            className="flex-1 bg-bg-light-bg border border-border-color rounded-lg px-4 py-3 text-text-primary focus:ring-2 focus:ring-brand-primary focus:outline-none"
                        />
                        <button 
                            onClick={() => handleSendMessage(userInput)}
                            disabled={!userInput.trim()}
                            className="bg-brand-primary text-white px-4 py-2 rounded-lg hover:bg-brand-primary/90 disabled:opacity-50 transition-colors"
                        >
                            <SendIcon />
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
};
