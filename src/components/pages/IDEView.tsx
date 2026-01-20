

import React, { useState, useEffect, useRef, useCallback } from 'react';
import { FolderIcon, FileIcon, CheckCircleIcon, CommandLineIcon, ChatBubbleLeftRightIcon, PaperAirplaneIcon, SpinnerIcon, AILogoIcon } from '../icons';
import { RealNexusAdapter } from '../../services/RealNexusAdapter';

// --- TYPES ---
interface FileNode {
  id: string;
  name: string;
  type: 'file' | 'folder';
  content?: string;
  children?: FileNode[];
}

interface IDEMessage {
    id: string;
    sender: 'user' | 'ai';
    text: string;
}

// --- MOCK DATA (Fallback) ---
const createInitialFileTree = (): FileNode => ({
  id: 'root', name: 'aethereal-project', type: 'folder', children: [
    { id: '1', name: 'main_agent_loop.py', type: 'file', content: '# Main agent process for Aethereal Nexus\n\nclass Agent:\n    def __init__(self):\n        print("Agent Initialized")\n\n    def run(self):\n        while True:\n            # Main loop logic here\n            pass' },
    { id: '2', name: 'README.md', type: 'file', content: '# Aethereal Project\n\nThis project contains the core logic for the autonomous agents.' },
    { id: '3', name: 'modules', type: 'folder', children: [
      { id: '4', name: 'memory_manager.py', type: 'file', content: '# Handles Qdrant vector database integration\n\nclass MemoryManager:\n    def connect(self):\n        print("Connected to Qdrant")' },
      { id: '5', name: 'tools', type: 'folder', children: [
        { id: '6', name: 'web_browser.py', type: 'file', content: '# Web browsing tool implementation' }
      ]}
    ]},
  ],
});

// --- HELPER FUNCTIONS ---
const findFile = (node: FileNode, fileId: string): FileNode | null => {
  if (node.id === fileId && node.type === 'file') return node;
  if (node.children) {
    for (const child of node.children) {
      const found = findFile(child, fileId);
      if (found) return found;
    }
  }
  return null;
};

const updateFileContent = (node: FileNode, fileId: string, newContent: string): FileNode => {
    if (node.id === fileId && node.type === 'file') {
        return { ...node, content: newContent };
    }
    if (node.children) {
        return { ...node, children: node.children.map(child => updateFileContent(child, fileId, newContent)) };
    }
    return node;
};

const highlightSyntax = (text: string) => {
    const keywords = ['ls', 'cd', 'git', 'npm', 'node', 'python', 'pip', 'docker', 'podman', 'echo', 'viren'];
    const keywordRegex = new RegExp(`\\b(${keywords.join('|')})\\b`, 'g');
    
    return text
        .replace(keywordRegex, '<span class="shell-keyword">$1</span>')
        .replace(/(--?\w+)/g, '<span class="shell-flag">$1</span>')
        .replace(/([a-zA-Z0-9-_\/]+\.[a-zA-Z]{2,})/g, '<span class="shell-path">$1</span>')
        .replace(/(Error|Failed|fatal|not found)/gi, '<span class="shell-error">$1</span>');
};

// --- SUB-COMPONENTS ---
const FileExplorerItem: React.FC<{ node: FileNode; level: number; onSelect: (id: string) => void; activeId: string | null; }> = ({ node, level, onSelect, activeId }) => (
    <>
        <div
            className={`flex items-center space-x-2 p-1.5 rounded-md cursor-pointer hover:bg-slate-200 ${activeId === node.id ? 'bg-brand-primary/10 text-brand-primary' : ''}`}
            style={{ paddingLeft: `${12 + level * 20}px` }}
            onClick={() => node.type === 'file' && onSelect(node.id)}
        >
            {node.type === 'folder' ? <FolderIcon /> : <FileIcon />}
            <span className="text-sm truncate">{node.name}</span>
        </div>
        {node.children && node.children.map(child => <FileExplorerItem key={child.id} node={child} level={level + 1} onSelect={onSelect} activeId={activeId} />)}
    </>
);

const GuestCursor: React.FC = () => {
    const [position, setPosition] = useState({ top: 20, left: 40 });
    useEffect(() => {
        const interval = setInterval(() => {
            setPosition({
                top: Math.random() * 80 + 10,
                left: Math.random() * 60 + 20
            });
        }, 3000);
        return () => clearInterval(interval);
    }, []);
    return <div className="guest-cursor" style={{ top: `${position.top}%`, left: `${position.left}%` }} title="Viren (Guest)"></div>
}

const IDEChatPanel: React.FC<{ messages: IDEMessage[]; onSendMessage: (text: string) => void; isThinking: boolean }> = ({ messages, onSendMessage, isThinking }) => {
    const [input, setInput] = useState('');
    const messagesEndRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages, isThinking]);

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        if (input.trim()) {
            onSendMessage(input.trim());
            setInput('');
        }
    };

    return (
        <div className="flex flex-col h-full bg-bg-light-bg">
            <div className="flex-grow overflow-y-auto p-4 space-y-4">
                {messages.map(msg => (
                    <div key={msg.id} className={`flex gap-3 ${msg.sender === 'user' ? 'flex-row-reverse' : ''}`}>
                        <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${msg.sender === 'ai' ? 'bg-brand-primary text-white' : 'bg-slate-300 text-slate-600'}`}>
                            {msg.sender === 'ai' ? <AILogoIcon className="w-5 h-5" /> : <span className="font-bold text-xs">YOU</span>}
                        </div>
                        <div className={`max-w-[80%] p-3 rounded-lg text-sm ${msg.sender === 'ai' ? 'bg-white border border-border-color text-text-primary' : 'bg-brand-primary text-white'}`}>
                            <p className="whitespace-pre-wrap">{msg.text}</p>
                        </div>
                    </div>
                ))}
                {isThinking && (
                    <div className="flex gap-3">
                        <div className="w-8 h-8 rounded-full bg-brand-primary text-white flex items-center justify-center flex-shrink-0">
                            <AILogoIcon className="w-5 h-5" />
                        </div>
                        <div className="bg-white border border-border-color text-text-secondary px-4 py-2 rounded-lg flex items-center gap-2 text-sm">
                            <SpinnerIcon className="w-4 h-4" />
                            <span>Analyzing code...</span>
                        </div>
                    </div>
                )}
                <div ref={messagesEndRef} />
            </div>
            <form onSubmit={handleSubmit} className="p-3 border-t border-border-color bg-white">
                <div className="relative">
                    <input 
                        type="text" 
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        placeholder="Ask Oz about your code..."
                        className="w-full bg-bg-light-bg border border-border-color rounded-lg pl-4 pr-10 py-2 text-sm focus:ring-2 focus:ring-brand-primary focus:border-brand-primary outline-none"
                    />
                    <button 
                        type="submit"
                        disabled={!input.trim() || isThinking}
                        className="absolute right-2 top-1/2 -translate-y-1/2 text-brand-primary hover:text-brand-secondary disabled:text-slate-300 transition-colors"
                    >
                        <PaperAirplaneIcon className="w-5 h-5" />
                    </button>
                </div>
            </form>
        </div>
    );
};

// --- MAIN COMPONENT ---
const IDEView: React.FC = () => {
  const [fileTree, setFileTree] = useState<FileNode>(createInitialFileTree());
  const [activeFileId, setActiveFileId] = useState<string | null>('1');
  const [isLiveMode, setIsLiveMode] = useState(false);
  
  // Terminal State
  const [terminalHistory, setTerminalHistory] = useState<{ type: 'command' | 'output', content: string }[]>([]);
  const [terminalInput, setTerminalInput] = useState('');
  const terminalEndRef = useRef<HTMLDivElement>(null);
  const terminalInputRef = useRef<HTMLInputElement>(null);
  const wsRef = useRef<any>(null);

  // Chat State
  const [bottomTab, setBottomTab] = useState<'terminal' | 'chat'>('terminal');
  const [chatMessages, setChatMessages] = useState<IDEMessage[]>([{ id: 'init', sender: 'ai', text: 'I am ready to assist with your engineering tasks. Open a file and I can help you write or debug code.' }]);
  const [isChatThinking, setIsChatThinking] = useState(false);

  const [isSaved, setIsSaved] = useState(true);
  const [isSaving, setIsSaving] = useState(false);
  const activeFile = activeFileId ? findFile(fileTree, activeFileId) : null;

  // --- INITIALIZATION ---
  useEffect(() => {
      const init = async () => {
          const health = await RealNexusAdapter.checkHealth();
          setIsLiveMode(health);
          if (health) {
              // Load Real Files
              try {
                  const realFiles = await RealNexusAdapter.fs.listFiles();
                  setFileTree(realFiles); 
              } catch (e) {
                  console.warn("Failed to load real files, using fallback");
              }
              
              // Connect Real Terminal
              wsRef.current = RealNexusAdapter.connectTerminal(
                  (data) => setTerminalHistory(prev => [...prev, { type: 'output', content: data }]),
                  () => setTerminalHistory(prev => [...prev, { type: 'output', content: '<span class="text-green-400">Connected to Oz Shell.</span>' }]),
                  () => setTerminalHistory(prev => [...prev, { type: 'output', content: '<span class="text-yellow-400">Shell Disconnected.</span>' }])
              );
          } else {
              // Fallback Simulation
              setTerminalHistory([
                  { type: 'output', content: '<span class="text-green-400">Viren Engineering Agent Online. All systems nominal.</span>' },
                  { type: 'output', content: 'Ready for commands...' },
              ]);
              // Load from local storage if available
              const savedTree = localStorage.getItem('ideFileTree');
              if (savedTree) setFileTree(JSON.parse(savedTree));
          }
      };
      init();

      return () => {
          if (wsRef.current) wsRef.current.close();
      };
  }, []);

  useEffect(() => {
      if (!isLiveMode && activeFileId) localStorage.setItem('ideActiveFileId', activeFileId);
  }, [activeFileId, isLiveMode]);
  
  useEffect(() => {
    if (bottomTab === 'terminal') {
        terminalEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [terminalHistory, bottomTab]);

  const handleContentChange = (newContent: string) => {
    setIsSaved(false);
    if(activeFileId) {
        setFileTree(prevTree => updateFileContent(prevTree, activeFileId, newContent));
    }
  };
  
  const handleManualSave = async () => {
      setIsSaving(true);
      if (isLiveMode && activeFile) {
          try {
              await RealNexusAdapter.fs.writeFile(activeFile.name, activeFile.content || '');
              setIsSaved(true);
          } catch (e) {
              alert("Failed to save file to backend.");
          }
      } else {
          // Local save
          localStorage.setItem('ideFileTree', JSON.stringify(fileTree));
          setTimeout(() => setIsSaved(true), 500);
      }
      setIsSaving(false);
  };
  
  const handleTerminalSubmit = (e: React.FormEvent) => {
      e.preventDefault();
      const command = terminalInput.trim();
      if (!command) return;
      
      const newHistory = [...terminalHistory, { type: 'command' as const, content: command }];
      setTerminalHistory(newHistory);
      setTerminalInput('');
      
      if (isLiveMode && wsRef.current) {
          wsRef.current.send(command);
      } else {
          // Simulation Mode with "Streaming" Effect
          if (command === 'clear') {
              setTerminalHistory([]);
              return;
          }
          
          let output = `Command not found: ${command}`;
          if (command.startsWith('ls')) output = 'main_agent_loop.py  README.md  modules/';
          else if (command.startsWith('echo')) output = command.substring(5);
          else if (command.startsWith('viren')) output = '<span class="text-green-400">Viren is online and ready.</span>';

          // Simulate streaming by appending characters
          const id = Date.now();
          let i = 0;
          const streamInterval = setInterval(() => {
              if (i < output.length) {
                  setTerminalHistory(prev => {
                      const last = prev[prev.length - 1];
                      // Check if last entry corresponds to this command response to append, else add new
                      if (last && last.type === 'output' && !last.content.includes(output)) {
                           // simple logic for demo: just append full output at end for cleaner code in this snippet
                           return prev; 
                      }
                      return [...prev]; // placeholder
                  });
                  i++;
              } else {
                  clearInterval(streamInterval);
                  setTerminalHistory(prev => [...prev, { type: 'output', content: output }]);
                  terminalEndRef.current?.scrollIntoView({ behavior: 'smooth' });
              }
          }, 10);
          // Actually just push it for stability in this demo
          setTerminalHistory(prev => [...prev, { type: 'output', content: output }]);
      }
      terminalInputRef.current?.focus();
  }

  const handleChatSubmit = (text: string) => {
      setChatMessages(prev => [...prev, { id: Date.now().toString(), sender: 'user', text }]);
      setIsChatThinking(true);
      
      // Simulate AI response aware of context
      setTimeout(() => {
          let responseText = "I can help with that.";
          if (activeFile) {
              responseText = `I see you are working on '${activeFile.name}'. Based on the current code structure, I suggest refactoring the main loop to handle exceptions more gracefully.`;
          } else {
              responseText = "Please select a file so I can provide specific assistance.";
          }
          
          setChatMessages(prev => [...prev, { id: (Date.now() + 1).toString(), sender: 'ai', text: responseText }]);
          setIsChatThinking(false);
      }, 1500);
  };

  return (
    <div className="p-4 sm:p-6 lg:p-8 animate-scale-in h-full flex flex-col">
      <header className="mb-6 flex-shrink-0 flex justify-between items-end">
        <div>
            <h2 className="text-3xl font-bold text-text-primary">Engineering Suite</h2>
            <p className="text-text-secondary">Collaborative IDE with Oz Copilot integration. {isLiveMode && <span className="text-green-500 font-bold text-xs ml-2">[LIVE LINK ACTIVE]</span>}</p>
        </div>
        <div className="flex items-center space-x-3">
            <span className={`text-xs font-medium transition-colors ${isSaved ? 'text-green-600' : 'text-yellow-600'}`}>
                {isSaved ? 'All changes saved' : 'Unsaved changes...'}
            </span>
            <button 
                onClick={handleManualSave}
                disabled={isSaving}
                className={`px-4 py-2 rounded-lg text-sm font-semibold transition-colors flex items-center gap-2 ${isSaved ? 'bg-slate-100 text-slate-500' : 'bg-brand-primary text-white hover:bg-brand-primary/90'}`}
            >
                {isSaving ? <SpinnerIcon className="w-4 h-4" /> : isSaved ? <CheckCircleIcon className="w-4 h-4" /> : null}
                {isSaving ? 'Saving...' : 'Save File'}
            </button>
        </div>
      </header>
      <div className="flex-grow grid grid-cols-1 lg:grid-cols-4 gap-6 min-h-0">
        <div className="lg:col-span-1 bg-bg-light-card glass-card rounded-xl shadow-aura border border-border-color flex flex-col h-full">
            <h3 className="text-base font-semibold text-text-primary p-3 border-b border-border-color flex-shrink-0">File Explorer</h3>
            <div className="p-2 overflow-y-auto">
                <FileExplorerItem node={fileTree} level={0} activeId={activeFileId} onSelect={setActiveFileId} />
            </div>
        </div>
        <div className="lg:col-span-3 flex flex-col gap-6 h-full min-h-0">
            <div className="flex-grow bg-white rounded-xl shadow-aura border border-border-color overflow-hidden flex flex-col relative">
                <div className="bg-bg-light-bg border-b border-border-color px-4 py-2 text-sm text-text-secondary font-mono flex-shrink-0 flex justify-between items-center">
                    <span>// Editing: {activeFile?.name || 'No file selected'}</span>
                    {activeFile && <span className="text-xs opacity-50">UTF-8</span>}
                </div>
                <div className="flex-grow relative p-4 font-mono text-sm overflow-y-auto bg-[#1e1e1e] text-white">
                    <GuestCursor />
                    {activeFile ? (
                        <textarea
                            key={activeFile.id}
                            value={activeFile.content}
                            onChange={(e) => handleContentChange(e.target.value)}
                            className="w-full h-full bg-transparent border-0 outline-none resize-none absolute inset-0 p-4 leading-relaxed"
                            spellCheck="false"
                        />
                    ) : (
                        <p className="text-text-secondary">Select a file to begin editing.</p>
                    )}
                </div>
            </div>
            <div className="h-64 bg-white rounded-xl shadow-aura border border-border-color flex flex-col font-mono text-sm overflow-hidden">
                 <div className="flex border-b border-border-color bg-bg-light-bg">
                     <button 
                        onClick={() => setBottomTab('terminal')}
                        className={`px-4 py-2 text-xs font-semibold flex items-center gap-2 ${bottomTab === 'terminal' ? 'bg-white border-t-2 border-brand-primary text-brand-primary' : 'text-text-secondary hover:bg-slate-100'}`}
                     >
                         <CommandLineIcon className="w-4 h-4" /> Terminal {isLiveMode && <span className="w-2 h-2 bg-green-500 rounded-full"></span>}
                     </button>
                     <button 
                        onClick={() => setBottomTab('chat')}
                        className={`px-4 py-2 text-xs font-semibold flex items-center gap-2 ${bottomTab === 'chat' ? 'bg-white border-t-2 border-brand-primary text-brand-primary' : 'text-text-secondary hover:bg-slate-100'}`}
                     >
                         <ChatBubbleLeftRightIcon className="w-4 h-4" /> Oz Assistant
                     </button>
                 </div>
                 
                 {bottomTab === 'terminal' ? (
                     <div className="p-3 overflow-y-auto flex-grow bg-shell-bg text-shell-text" onClick={() => terminalInputRef.current?.focus()}>
                         {terminalHistory.map((line, i) => (
                            <div key={i}>
                             {line.type === 'command' ? (
                                <p><span className="text-green-400">nexus@host</span>:<span className="text-blue-400">~</span>$ {line.content}</p>
                             ) : (
                                <p dangerouslySetInnerHTML={{ __html: highlightSyntax(line.content) }} />
                             )}
                            </div>
                         ))}
                         <form onSubmit={handleTerminalSubmit} className="flex">
                            <label htmlFor="terminal-input" className="flex items-center flex-1">
                                <span className="text-green-400">nexus@host</span>:<span className="text-blue-400">~</span>$&nbsp;
                                <input
                                    id="terminal-input"
                                    ref={terminalInputRef}
                                    type="text"
                                    value={terminalInput}
                                    onChange={(e) => setTerminalInput(e.target.value)}
                                    className="bg-transparent border-0 outline-none w-full"
                                    spellCheck="false"
                                    autoFocus
                                />
                            </label>
                         </form>
                         <div ref={terminalEndRef} />
                     </div>
                 ) : (
                     <IDEMessagePanel messages={chatMessages} onSendMessage={handleChatSubmit} isThinking={isChatThinking} />
                 )}
            </div>
        </div>
      </div>
    </div>
  );
};

// Helper wrapper for the sub-component to avoid definition hoisting issues
const IDEMessagePanel = IDEChatPanel;

export default IDEView;