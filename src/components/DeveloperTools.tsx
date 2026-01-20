import React from 'react';
import { LogEntry } from '../types';

interface DeveloperToolsProps {
  isOpen: boolean;
  onToggle: () => void;
  logs: LogEntry[];
}

export default function DeveloperTools({ isOpen, onToggle, logs }: DeveloperToolsProps) {
  const getLogColor = (type: LogEntry['type']) => {
    switch (type) {
      case 'error': return 'text-red-500';
      case 'warn': return 'text-yellow-500';
      default: return 'text-text-secondary';
    }
  };

  return (
    <div className={`fixed bottom-0 left-0 right-0 bg-bg-light-card border-t border-border-color z-50 shadow-2xl transition-transform duration-300 ease-in-out ${isOpen ? 'translate-y-0' : 'translate-y-[calc(100%-2.5rem)]'}`}>
      <div className="flex justify-between items-center p-2 cursor-pointer bg-bg-light-bg hover:bg-slate-200" onClick={onToggle}>
        <span className="font-semibold text-sm text-text-primary px-2">Developer Tools</span>
        <span className={`transform transition-transform duration-200 px-2 ${isOpen ? 'rotate-180' : ''}`}>▲</span>
      </div>
      <div className="h-64 overflow-y-auto p-4 font-mono text-xs">
        {logs.map((log, i) => (
          <div key={i} className={`py-1 border-b border-border-color ${getLogColor(log.type)}`}>
            <strong className="text-slate-400 mr-2">[{log.timestamp}]</strong>
            <span className="font-medium mr-2">[{log.type.toUpperCase()}]:</span>
            <span className="whitespace-pre-wrap">{log.message}</span>
          </div>
        ))}
        <div />
      </div>
    </div>
  );
}
