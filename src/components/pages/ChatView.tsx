

import React, { useState, useEffect, useRef } from 'react';
import { v4 as uuidv4 } from 'uuid';
import { Agent, AIModelEndpoint, ChatMessage, ToolCall, ToolResult } from '../../types';
import { AILogoIcon, PaperAirplaneIcon, SpinnerIcon, CodeBracketIcon, MicIcon, SpeakerWaveIcon } from '../icons';

// A component to render the result of a tool call
const ToolResultDisplay: React.FC<{ result: ToolResult }> = ({ result }) => {
    if (result.toolName === 'ide' || result.toolName === 'email_tool') {
        return (
             <div className="border-t border-border-color mt-3 pt-3">
                <p className="text-xs font-semibold text-text-secondary mb-2">{result.output as string}</p>
            </div>
        )
    }
    // Default display for other tools
    return <pre className="text-xs bg-bg-light mt-2 p-2 rounded-md whitespace-pre-wrap">{JSON.stringify(result.output, null, 2)}</pre>;
};


interface ChatViewProps {
    agents: Agent[];
    modelEndpoints: AIModelEndpoint[];
}

const ChatView: React.FC<ChatViewProps> = ({ agents, modelEndpoints }) => {
    const [messages, setMessages] = useState<ChatMessage[]>([
        { id: uuidv4(), sender: 'AGENT', message: 'Welcome to the Nexus Chat Hub. I can delegate tasks to agents and tools. Try `/ide open main.py` or `/tools create_email_account --provider gmail` to start.', timestamp: new Date().toISOString() }
    ]);
    const [input, setInput] = useState('');
    const [isResponding, setIsResponding] = useState(false);
    const [isListening, setIsListening] = useState(false);
    const [selectedTarget, setSelectedTarget] = useState('Nexus-Orchestrator');
    
    const messagesEndRef = useRef<HTMLDivElement>(null);
    const recognitionRef = useRef<any>(null);
    const inputRef = useRef<HTMLInputElement>(null);


    // --- Text-to-Speech ---
    const speak = (text: string) => {
        window.speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance(text);
        window.speechSynthesis.speak(utterance);
    };

    // --- Voice-to-Text ---
    useEffect(() => {
        const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
        if (SpeechRecognition) {
            recognitionRef.current = new SpeechRecognition();
            recognitionRef.current.continuous = false;
            recognitionRef.current.interimResults = false;
            recognitionRef.current.lang = 'en-US';
            
            recognitionRef.current.onresult = (event: any) => {
                const transcript = event.results[0][0].transcript;
                setInput(transcript);
                setIsListening(false);
            };
            recognitionRef.current.onerror = (event: any) => {
                console.error("Speech recognition error:", event.error);
                setIsListening(false);
            };
            recognitionRef.current.onend = () => {
                setIsListening(false);
            };
        } else {
            console.warn("Speech Recognition not supported by this browser.");
        }
    }, []);

    const toggleListening = () => {
        if (!recognitionRef.current) return;
        if (isListening) {
            recognitionRef.current.stop();
        } else {
            recognitionRef.current.start();
        }
        setIsListening(!isListening);
    };


    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(scrollToBottom, [messages, isResponding]);
    
    // Keep focus on input
    useEffect(() => {
        if (!isResponding && inputRef.current) {
            inputRef.current.focus();
        }
    }, [isResponding, messages]);
    
    const handleToolCall = (command: string, args: string[]) => {
        const toolCall: ToolCall = { toolName: command, args: { params: args.join(' ') } };
        const toolCallMessage: ChatMessage = {
            id: uuidv4(),
            sender: 'AGENT',
            message: `Calling tool: \`${command}\`...`,
            timestamp: new Date().toISOString(),
            toolCall: toolCall,
        };
        setMessages(prev => [...prev, toolCallMessage]);
        
        // Simulate tool execution and result
        setTimeout(() => {
            let toolResult: ToolResult;
            if (command === 'ide') {
                toolResult = { toolName: 'ide', output: `Success: The IDE has been instructed to open ${args.join(' ')}.` };
            } else if (command.includes('email')) {
                 toolResult = { toolName: 'email_tool', output: `Success: An email account is being provisioned via the respective tool.`};
            } else {
                 toolResult = { toolName: command, output: 'Tool executed successfully (mock).' };
            }

            const toolResultMessage: ChatMessage = {
                id: uuidv4(),
                sender: 'AGENT',
                message: `Tool \`${command}\` finished execution.`,
                timestamp: new Date().toISOString(),
                toolResult: toolResult,
            };
            speak(toolResultMessage.message);
            setMessages(prev => [...prev, toolResultMessage]);
            setIsResponding(false);
        }, 2000);
    }

    const handleSend = (e: React.FormEvent) => {
        e.preventDefault();
        const trimmedInput = input.trim();
        if (trimmedInput) {
            const userMessage: ChatMessage = {
                id: uuidv4(),
                sender: 'USER',
                message: trimmedInput,
                timestamp: new Date().toISOString()
            };
            setMessages(prev => [...prev, userMessage]);
            setInput('');
            setIsResponding(true);

            // Immediate refocus to ensure "attention remains"
            setTimeout(() => inputRef.current?.focus(), 10);

            if (trimmedInput.startsWith('/')) {
                const parts = trimmedInput.slice(1).split(' ');
                const command = parts[0];
                const args = parts.slice(1);
                handleToolCall(command, args);
                return;
            }

            // Simulate normal response
            setTimeout(() => {
                const responseText = `Message for [${selectedTarget}] received. Acknowledged: "${userMessage.message}"`;
                const agentResponse: ChatMessage = {
                    id: uuidv4(),
                    sender: 'AGENT',
                    message: responseText,
                    timestamp: new Date().toISOString()
                };
                speak(responseText);
                setMessages(prev => [...prev, agentResponse]);
                setIsResponding(false);
            }, 1200);
        }
    };
    
    const conversationTargets = [
        { id: 'Nexus-Orchestrator', name: 'Nexus Orchestrator' },
        ...agents.map(a => ({ id: a.id, name: `Agent: ${a.goal.substring(0, 30)}...`})),
        ...modelEndpoints.map(m => ({ id: m.id, name: `Model: ${m.name}`}))
    ]

    return (
        <div className="p-4 sm:p-6 lg:p-8 animate-scale-in h-full flex flex-col">
            <div className="bg-bg-light-card glass-card rounded-xl shadow-aura border border-border-color flex flex-col h-full">
                 <div className="p-4 border-b border-border-color flex-shrink-0">
                    <div className="flex items-center space-x-3">
                         <label htmlFor="chat-target" className="text-sm font-medium text-text-secondary">Target:</label>
                         <select 
                            id="chat-target"
                            value={selectedTarget}
                            onChange={e => setSelectedTarget(e.target.value)}
                            className="w-full max-w-sm bg-bg-light-bg border border-border-color rounded-lg p-2 text-text-primary focus:ring-2 focus:ring-brand-primary"
                        >
                            {conversationTargets.map(target => (
                                <option key={target.id} value={target.id}>{target.name}</option>
                            ))}
                        </select>
                    </div>
                </div>
                <div className="flex-grow p-6 overflow-y-auto">
                    <div className="space-y-6">
                        {messages.map((msg) => (
                            <div key={msg.id} className={`flex items-end gap-3 ${msg.sender === 'USER' ? 'justify-end' : ''}`}>
                                {msg.sender === 'AGENT' && (
                                    <div className="w-8 h-8 rounded-full bg-bg-light-bg flex items-center justify-center flex-shrink-0">
                                        <AILogoIcon className="w-5 h-5 text-brand-primary" />
                                    </div>
                                )}
                                <div className={`max-w-xl group relative p-3 rounded-2xl ${
                                    msg.sender === 'USER' 
                                    ? 'bg-user-message-bg text-white rounded-br-none' 
                                    : 'bg-ai-message-bg text-text-primary rounded-bl-none'
                                }`}>
                                    <p className="text-sm whitespace-pre-wrap">{msg.message}</p>
                                    {!!msg.toolCall && (
                                        <div className="border-t border-border-color mt-3 pt-3">
                                            <div className="flex items-center space-x-2 text-sm text-text-secondary">
                                                {msg.toolCall.toolName === 'ide' && <CodeBracketIcon className="w-4 h-4" />}
                                                <span>Executing <span className="font-semibold">{msg.toolCall.toolName}</span>...</span>
                                            </div>
                                        </div>
                                    )}
                                     {msg.toolResult && <ToolResultDisplay result={msg.toolResult} />}
                                     {msg.sender === 'AGENT' && (
                                        <button onClick={() => speak(msg.message)} className="absolute top-1/2 -right-2.5 -translate-y-1/2 translate-x-full opacity-0 group-hover:opacity-50 transition-opacity p-1.5 rounded-full hover:bg-slate-200/80" aria-label="Read message aloud">
                                            <SpeakerWaveIcon className="w-4 h-4" />
                                        </button>
                                     )}
                                </div>
                            </div>
                        ))}
                        {isResponding && (
                            <div className="flex items-end gap-3">
                                <div className="w-8 h-8 rounded-full bg-bg-light-bg flex items-center justify-center flex-shrink-0">
                                    <AILogoIcon className="w-5 h-5 text-brand-primary" />
                                </div>
                                <div className="bg-ai-message-bg text-text-primary rounded-2xl rounded-bl-none p-3 inline-flex items-center space-x-2">
                                    <SpinnerIcon className="w-4 h-4" />
                                    <span className="text-sm text-text-secondary">Nexus is processing...</span>
                                </div>
                            </div>
                        )}
                        <div ref={messagesEndRef} />
                    </div>
                </div>
                <div className="p-4 border-t border-border-color flex-shrink-0">
                    <form onSubmit={handleSend} className="relative">
                        <input
                            ref={inputRef}
                            type="text"
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            placeholder={isListening ? "Listening..." : `Message ${selectedTarget} or use /tool...`}
                            className="w-full bg-bg-light-bg border border-border-color rounded-full py-3 pl-14 pr-14 text-text-primary placeholder-text-secondary focus:ring-2 focus:ring-brand-primary focus:border-brand-primary transition"
                            // Removed disabled={isResponding} to fix focus stealing
                        />
                        <button 
                            type="button" 
                            onClick={toggleListening}
                            className={`absolute top-1/2 -translate-y-1/2 left-2 w-10 h-10 rounded-full flex items-center justify-center transition-colors ${
                                isListening ? 'bg-red-500 text-white' : 'text-text-secondary hover:bg-slate-200'
                            }`}
                            style={{ animation: isListening ? 'pulse-mic 2s infinite' : 'none' }}
                            aria-label="Use microphone"
                        >
                            <MicIcon className="w-5 h-5" />
                        </button>
                         <button 
                            type="submit" 
                            disabled={!input.trim()}
                            className="absolute top-1/2 -translate-y-1/2 right-2 w-10 h-10 bg-brand-primary text-white rounded-full flex items-center justify-center hover:bg-brand-primary/90 disabled:bg-slate-400 disabled:cursor-not-allowed transition-colors"
                            aria-label="Send message"
                        >
                            <PaperAirplaneIcon className="w-5 h-5" />
                        </button>
                    </form>
                </div>
            </div>
        </div>
    );
};

export default ChatView;