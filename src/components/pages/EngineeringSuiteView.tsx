

import React, { useState } from 'react';

// --- ICONS ---
const FolderIcon = () => <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"></path></svg>;
const FileIcon = () => <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"></path><polyline points="13 2 13 9 20 9"></polyline></svg>;

// --- TYPES ---
interface FileNode { id: string; name: string; type: 'file' | 'folder'; content?: string; children?: FileNode[]; }

// --- MOCK DATA ---
const initialFileTree: FileNode = {
    id: 'root', name: 'aethereal-project', type: 'folder', children: [
        { id: '1', name: 'main_agent_loop.py', type: 'file', content: '# Main agent process' },
        { id: '2', name: 'README.md', type: 'file', content: '# Aethereal Project' },
        { id: '3', name: 'modules', type: 'folder', children: [
            { id: '4', name: 'memory_manager.py', type: 'file', content: '# Handles Qdrant integration' }
        ]},
    ],
};

interface FileExplorerItemProps {
    node: FileNode; level: number; onSelect: (id: string) => void; activeId: string | null;
}

// FIX: Changed component to React.FC to correctly type component props and resolve 'key' property error.
const FileExplorerItem: React.FC<FileExplorerItemProps> = ({ node, level, onSelect, activeId }) => (
    <>
        <div className={`file-item ${activeId === node.id ? 'active' : ''}`} style={{ paddingLeft: `${12 + level * 20}px` }} onClick={() => node.type === 'file' && onSelect(node.id)}>
            {node.type === 'folder' ? <FolderIcon /> : <FileIcon />}
            <span className="file-item-name">{node.name}</span>
        </div>
        {node.children && node.children.map(child => <FileExplorerItem key={child.id} node={child} level={level + 1} onSelect={onSelect} activeId={activeId} />)}
    </>
);

// --- MAIN COMPONENT ---
// FIX: Removed LogEntry import and updated addLog prop type.
export default function EngineeringSuiteView({ addLog }: { addLog: (type: 'log' | 'error', ...args: any[]) => void; }) {
    const [activeFileId, setActiveFileId] = useState<string | null>('1');

    return (
        <div className="app-window">
            <div className="app-header">
                <img src="https://i.imgur.com/8p8YJkP.png" alt="Aethereal Icon" className="w-7 h-7" />
                <span>Engineering Suite</span>
            </div>
            <div className="engineering-suite-layout">
                <div className="file-explorer ide-panel">
                    <div className="ide-panel-header">File Explorer</div>
                    <div className="file-list p-2">
                        <FileExplorerItem node={initialFileTree} level={0} activeId={activeFileId} onSelect={setActiveFileId} />
                    </div>
                </div>

                <div className="code-editor-container ide-panel !p-0">
                     <iframe 
                        src="https://vscode.dev/" 
                        className="w-full h-full border-0"
                        title="VSCode Editor"
                        sandbox="allow-scripts allow-same-origin allow-forms allow-popups allow-top-navigation allow-modals"
                    ></iframe>
                </div>

                <div className="terminal-container ide-panel">
                    <div className="ide-panel-header">Terminal</div>
                    <div className="shell">
                        <p>$ viren --status</p>
                        <p className="text-green-400">Viren Engineering Agent Online. All systems nominal.</p>
                        <p>$ ls -l modules/</p>
                        <p>memory_manager.py</p>
                        <p>$</p>
                    </div>
                </div>
            </div>
        </div>
    );
}