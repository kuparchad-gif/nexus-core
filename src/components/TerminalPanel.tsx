

import React, { useEffect, useRef } from 'react';
import { Task, TaskStatus } from '../types';
import { ShieldCheckIcon } from './icons';

interface TerminalPanelProps {
    tasks: Task[];
}

const TerminalPanel: React.FC<TerminalPanelProps> = ({ tasks }) => {
    const terminalEndRef = useRef<HTMLDivElement>(null);
    const shellTasks = tasks.filter(t => t.type === 'SHELL');

    useEffect(() => {
        terminalEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [shellTasks]);

    return (
        <div className="bg-black rounded-xl shadow-lg border border-border-color h-[70vh] font-mono text-sm text-white p-4 flex flex-col">
            <div className="flex-shrink-0 flex items-center justify-between mb-4 border-b border-gray-700 pb-2">
                <div className="flex items-center space-x-2">
                    <div className="w-3 h-3 rounded-full bg-red-500"></div>
                    <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
                    <div className="w-3 h-3 rounded-full bg-green-500"></div>
                    <p className="text-gray-400 text-xs pl-4">Agent Shell</p>
                </div>
                 <div className="flex items-center space-x-2 text-green-400 text-xs">
                    <ShieldCheckIcon className="w-4 h-4" />
                    <span>Sandboxed</span>
                </div>
            </div>
            <div className="flex-grow overflow-y-auto pr-2">
                {shellTasks.length === 0 && (
                     <p className="text-gray-500">
                        <span className="text-green-400">nexus@host</span>:<span className="text-blue-400">~</span>$ No shell commands have been executed yet.
                    </p>
                )}
                {shellTasks.map(task => (
                    <div key={task.id} className="mb-2">
                        <p>
                            <span className="text-green-400">nexus@host</span>:<span className="text-blue-400">~</span>$ {task.params?.command || task.description}
                        </p>
                        {task.status === TaskStatus.COMPLETED && task.output && (
                            <p className="text-gray-300 whitespace-pre-wrap">{task.output}</p>
                        )}
                         {task.status === TaskStatus.FAILED && (
                            <p className="text-red-400 whitespace-pre-wrap">Error: Command failed.</p>
                        )}
                    </div>
                ))}
                <div className="animate-pulse">
                    <span className="text-green-400">nexus@host</span>:<span className="text-blue-400">~</span>$ <span className="bg-white w-2 h-4 inline-block"></span>
                </div>
                <div ref={terminalEndRef} />
            </div>
        </div>
    );
};

export default TerminalPanel;