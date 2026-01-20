

import React, { useState, useEffect, useRef } from 'react';
import { ChatMessage } from '../types';
import { AILogoIcon, PaperAirplaneIcon, SpinnerIcon } from './icons';

interface ChatPanelProps {
    messages: ChatMessage[];
    onSendMessage: (message: string) => void;
    isResponding: boolean;
    title?: string;
}

const ChatPanel: React.FC<ChatPanelProps> = ({ messages, onSendMessage, isResponding, title }) => {
    const [input, setInput] = useState('');
    const messagesEndRef = useRef<HTMLDivElement>(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(scrollToBottom, [messages, isResponding]);

    const handleSend = (e: React.FormEvent) => {
        e.preventDefault();
        if (input.trim()) {
            onSendMessage(input.trim());
            setInput('');
        }
    };

    return (
        <div className="bg-bg-light-card glass-card rounded-xl shadow-aura border border-border-color flex flex-col h-[70vh]">
            {title && (
                 <div className="p-4 border-b border-border-color flex-shrink-0">
                    <h3 className="text-lg font-semibold text-text-primary text-center">{title}</h3>
                </div>
            )}
            <div className="flex-grow p-6 overflow-y-auto">
                <div className="space-y-6">
                    {messages.map((msg) => (
                        <div key={msg.id} className={`flex items-end gap-3 ${msg.sender === 'USER' ? 'justify-end' : ''}`}>
                            {msg.sender === 'AGENT' && (
                                <div className="w-8 h-8 rounded-full bg-bg-light-bg flex items-center justify-center flex-shrink-0">
                                    <AILogoIcon className="w-5 h-5 text-brand-primary" />
                                </div>
                            )}
                            <div className={`max-w-xl p-3 rounded-2xl ${
                                msg.sender === 'USER' 
                                ? 'bg-brand-primary text-white rounded-br-none' 
                                : 'bg-bg-light-bg text-text-primary rounded-bl-none'
                            }`}>
                                <p className="text-sm whitespace-pre-wrap">{msg.message}</p>
                            </div>
                        </div>
                    ))}
                    {isResponding && (
                        <div className="flex items-end gap-3">
                             <div className="w-8 h-8 rounded-full bg-bg-light-bg flex items-center justify-center flex-shrink-0">
                                <AILogoIcon className="w-5 h-5 text-brand-primary" />
                            </div>
                            <div className="bg-bg-light-bg text-text-primary rounded-2xl rounded-bl-none p-3 inline-flex items-center space-x-2">
                                <SpinnerIcon className="w-4 h-4" />
                                <span className="text-sm text-text-secondary">Nexus is thinking...</span>
                            </div>
                        </div>
                    )}
                    <div ref={messagesEndRef} />
                </div>
            </div>
            <div className="p-4 border-t border-border-color flex-shrink-0">
                <form onSubmit={handleSend} className="relative">
                    <input
                        type="text"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        placeholder="Interact with the agent..."
                        className="w-full bg-bg-light-bg border border-border-color rounded-full py-3 pl-4 pr-14 text-text-primary placeholder-text-secondary focus:ring-2 focus:ring-brand-primary focus:border-brand-primary transition"
                        disabled={isResponding}
                    />
                     <button 
                        type="submit" 
                        disabled={!input.trim() || isResponding}
                        className="absolute top-1/2 -translate-y-1/2 right-2 w-10 h-10 bg-brand-primary text-white rounded-full flex items-center justify-center hover:bg-brand-primary/90 disabled:bg-slate-400 disabled:cursor-not-allowed transition-colors"
                        aria-label="Send message"
                    >
                        <PaperAirplaneIcon className="w-5 h-5" />
                    </button>
                </form>
            </div>
        </div>
    );
};

export default ChatPanel;