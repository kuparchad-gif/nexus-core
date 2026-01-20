/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/

import React, { useState } from 'react';
import { AppMode, QueryResult } from '../../types';
import { ChevronsUp, ChevronsDown, Briefcase, Database, Play } from 'lucide-react';
import { useAppStore } from '../../services/store';

const EmailClientView: React.FC = () => (
    <div className="w-full h-full flex p-4 gap-4">
        <div className="w-1/4 bg-black/5 rounded-lg p-2 space-y-2">
            <div className="p-2 bg-indigo-600/20 text-indigo-700 rounded font-semibold">Inbox (3)</div>
            <div className="p-2 hover:bg-black/5 rounded">Sent</div>
            <div className="p-2 hover:bg-black/5 rounded">Drafts</div>
        </div>
        <div className="w-3/4 bg-black/5 rounded-lg p-4">
            <h3 className="font-semibold border-b border-black/10 pb-2">Welcome to Aethereal Mail</h3>
        </div>
    </div>
);

const DocumentEditorView: React.FC = () => (
    <div className="w-full h-full p-4">
        <div className="w-full h-full bg-white rounded-lg shadow-inner p-8">
            <div className="h-4 w-3/4 bg-gray-200 rounded mb-4"></div>
            <div className="h-3 w-full bg-gray-200 rounded mb-2"></div>
            <div className="h-3 w-5/6 bg-gray-200 rounded mb-6"></div>
            <div className="h-3 w-full bg-gray-200 rounded mb-2"></div>
            <div className="h-3 w-full bg-gray-200 rounded mb-2"></div>
            <div className="h-3 w-1/2 bg-gray-200 rounded"></div>
        </div>
    </div>
);


const DatabaseClientView: React.FC = () => {
    const [dbType, setDbType] = useState('PostgreSQL');
    const [connection, setConnection] = useState({ host: 'localhost', port: '5432', user: 'admin', password: '', dbname: 'nexus_db' });
    const [query, setQuery] = useState('SELECT * FROM users LIMIT 10;');
    const [results, setResults] = useState<QueryResult | null>(null);
    const [isLoading, setIsLoading] = useState(false);

    const handleConnectionChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        setConnection({ ...connection, [e.target.name]: e.target.value });
    };

    const handleRunQuery = () => {
        setIsLoading(true);
        setResults(null);
        // Simulate API call
        setTimeout(() => {
            setResults({
                headers: ['id', 'username', 'email', 'created_at'],
                rows: [
                    [1, 'aethereal_user', 'user@nexus.ai', '2024-01-01 12:00:00'],
                    [2, 'admin_bot', 'bot@nexus.ai', '2024-01-02 14:30:00'],
                    [3, 'creative_mind', 'creator@nexus.ai', '2024-02-10 18:45:00'],
                    [4, 'dev_ops_guru', 'ops@nexus.ai', '2024-02-11 09:15:00'],
                ]
            });
            setIsLoading(false);
        }, 1000);
    };

    const inputClasses = "w-full p-1.5 bg-white/30 border border-black/10 rounded-md focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 focus:bg-white/60 transition-all duration-200 ease-in-out";

    return (
        <div className="w-full h-full flex flex-col p-4 gap-4 text-sm">
            {/* Connection & Query */}
            <div className="h-1/2 flex gap-4">
                {/* Connection Details */}
                <div className="w-1/3 flex flex-col glass-panel rounded-xl p-4 space-y-3">
                    <h3 className="font-semibold text-slate-800 flex items-center gap-2 border-b border-white/30 pb-2"><Database size={16} /> Connection</h3>
                    <div>
                        <label className="block font-medium text-slate-700 mb-1">Database Type</label>
                        <select value={dbType} onChange={(e) => setDbType(e.target.value)} className={inputClasses}>
                            <option>PostgreSQL</option>
                            <option>MySQL</option>
                            <option>MongoDB</option>
                        </select>
                    </div>
                    {['host', 'port', 'user', 'password', 'dbname'].map(field => (
                        <div key={field}>
                            <label className="block font-medium text-slate-700 mb-1 capitalize">{field}</label>
                            <input
                                type={field === 'password' ? 'password' : 'text'}
                                name={field}
                                value={connection[field as keyof typeof connection]}
                                onChange={handleConnectionChange}
                                className={inputClasses}
                            />
                        </div>
                    ))}
                </div>
                {/* Query Editor */}
                <div className="w-2/3 flex flex-col glass-panel rounded-xl p-4">
                    <div className="flex items-center justify-between mb-2">
                        <h3 className="font-semibold text-slate-800 flex items-center gap-2">SQL Query</h3>
                        <button onClick={handleRunQuery} disabled={isLoading} className="flex items-center gap-2 px-3 py-1 bg-indigo-600 text-white rounded-md text-sm hover:bg-indigo-500 disabled:bg-slate-400">
                            <Play size={14} /> Run Query
                        </button>
                    </div>
                    <textarea
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        className="w-full flex-grow p-2 border rounded-lg text-xs bg-white/20 border-white/30 text-slate-900 focus:ring-2 focus:ring-indigo-500"
                        style={{ fontFamily: "'Fira Code', monospace" }}
                        spellCheck="false"
                    />
                </div>
            </div>
            {/* Results */}
            <div className="h-1/2 flex flex-col glass-panel rounded-xl p-4">
                 <h3 className="font-semibold text-slate-800 mb-2">Results</h3>
                 <div className="flex-grow overflow-auto rounded-lg border border-white/30 bg-white/20">
                    {isLoading ? <div className="p-4 flex items-center justify-center h-full text-slate-600">Running query...</div> : !results ? <div className="p-4 text-slate-500 flex items-center justify-center h-full">Query results will appear here.</div> : (
                        <div className="animate-fade-in">
                        <table className="w-full text-sm text-left">
                            <thead className="bg-black/5 sticky top-0 backdrop-blur-sm">
                                <tr>{results.headers.map(h => <th key={h} className="p-2 font-semibold capitalize">{h}</th>)}</tr>
                            </thead>
                            <tbody>
                                {results.rows.map((row, i) => <tr key={i} className="border-b border-white/30 last:border-b-0 hover:bg-black/5 transition-colors">{row.map((cell, j) => <td key={j} className="p-2 whitespace-nowrap">{cell}</td>)}</tr>)}
                            </tbody>
                        </table>
                        </div>
                    )}
                 </div>
            </div>
        </div>
    );
};

const AdminWorkspace: React.FC = () => {
  const { appMode, setAppMode, toggleToolsDrawer, isToolsDrawerOpen } = useAppStore();

  const renderAppView = () => {
    switch(appMode) {
        case 'email':
            return <EmailClientView />;
        case 'documents':
            return <DocumentEditorView />;
        case 'database':
            return <DatabaseClientView />;
        default:
            return (
                <div className="flex flex-col items-center justify-center h-full text-slate-600 text-center p-8">
                    <Briefcase size={64} className="mb-4 text-slate-500" />
                    <h1 className="text-3xl font-bold text-slate-800">Administrative Workspace</h1>
                    <p className="mt-2 max-w-md">
                        Launch an application like Email, Documents, or the Database client from the Tools Drawer below.
                    </p>
                </div>
            );
    }
  };

  return (
    <div className="w-full h-full p-4 relative">
      <div className="w-full h-full glass-panel rounded-2xl flex flex-col items-center justify-center overflow-hidden">
        {renderAppView()}
      </div>

       <button 
          onClick={toggleToolsDrawer}
          className="absolute bottom-8 left-1/2 -translate-x-1/2 z-20 flex items-center gap-2 px-4 py-2 glass-panel rounded-full text-sm font-semibold text-slate-800 hover:bg-white/50 hover:glow-border transition-all"
        >
          {isToolsDrawerOpen ? <ChevronsDown size={16}/> : <ChevronsUp size={16}/>}
          <span>Launch App</span>
        </button>
    </div>
  );
};

export default AdminWorkspace;