/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/

import React, { useState } from 'react';
import { VirtualEnvironment, AppMode } from '../../types';
import { HardDrive, Plus, Play, Square, Code, BrainCircuit } from 'lucide-react';

const initialEnvs: VirtualEnvironment[] = [
    { id: 'env-1', name: 'Node.js Dev Server', status: 'Running', baseImage: 'Ubuntu 22.04', installedApps: ['vscode'] },
    { id: 'env-2', name: 'Graphics Workstation', status: 'Stopped', baseImage: 'GPU Optimized', installedApps: ['ai_toolbench'] },
    { id: 'env-3', name: 'Data Science Lab', status: 'Stopped', baseImage: 'Ubuntu 22.04', installedApps: [] },
];

const EnvironmentCard: React.FC<{ env: VirtualEnvironment }> = ({ env }) => {
    const statusColor = env.status === 'Running' ? 'text-green-500' : 'text-slate-500';
    return (
        <div className="glass-panel p-4 rounded-xl flex flex-col justify-between">
            <div>
                <div className="flex justify-between items-start">
                    <h3 className="font-bold text-slate-800">{env.name}</h3>
                    <div className={`flex items-center gap-1.5 text-xs font-semibold ${statusColor}`}>
                        <div className={`w-2 h-2 rounded-full ${env.status === 'Running' ? 'bg-green-500' : 'bg-slate-500'}`}></div>
                        {env.status}
                    </div>
                </div>
                <p className="text-sm text-slate-600">{env.baseImage}</p>
            </div>
            <div className="mt-4 flex justify-between items-end">
                <div className="text-sm text-slate-700">
                    <p className="font-semibold">Installed Apps:</p>
                    <div className="flex items-center gap-2 mt-1">
                        {env.installedApps.map(app => (
                            <div key={app} title={app} className="p-1.5 bg-black/5 rounded">
                                {app === 'vscode' ? <Code size={16} /> : <BrainCircuit size={16} />}
                            </div>
                        ))}
                        {env.installedApps.length === 0 && <p className="text-xs text-slate-500">None</p>}
                    </div>
                </div>
                <div className="flex items-center gap-2">
                    <button className="p-2 bg-slate-200/80 hover:bg-slate-300 rounded-md text-slate-700 transition-colors" title="Stop Environment">
                        <Square size={16} />
                    </button>
                    <button className="p-2 bg-indigo-600 hover:bg-indigo-500 rounded-md text-white transition-colors" title="Start Environment">
                        <Play size={16} />
                    </button>
                </div>
            </div>
        </div>
    );
}

const EnvironmentsWorkspace: React.FC = () => {
    const [environments, setEnvironments] = useState(initialEnvs);

    return (
        <div className="w-full h-full p-4 flex flex-col">
            <div className="flex-shrink-0 flex justify-between items-center mb-4">
                <div className="flex items-center gap-3">
                    <HardDrive size={32} className="text-slate-700" />
                    <div>
                        <h1 className="text-2xl font-bold text-slate-800">Environments</h1>
                        <p className="text-slate-600">Manage your virtual development and creative environments.</p>
                    </div>
                </div>
                <button className="flex items-center gap-2 px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-500 transition-colors font-semibold">
                    <Plus size={18} /> Provision New
                </button>
            </div>
            <div className="flex-grow overflow-y-auto pr-2">
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {environments.map(env => <EnvironmentCard key={env.id} env={env} />)}
                </div>
            </div>
        </div>
    );
};

export default EnvironmentsWorkspace;