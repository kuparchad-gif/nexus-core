

import React, { useState } from 'react';
import { v4 as uuidv4 } from 'uuid';
import { Tool } from '../../types';
import { PlusIcon, PencilIcon, TrashIcon, WrenchScrewdriverIcon } from '../icons';

interface ToolEditorProps {
    tool: Tool | null;
    onSave: (tool: Tool) => void;
    onClose: () => void;
}

const ToolEditor: React.FC<ToolEditorProps> = ({ tool, onSave, onClose }) => {
    const [name, setName] = useState(tool?.name || '');
    const [description, setDescription] = useState(tool?.description || '');
    const [tags, setTags] = useState(tool?.tags?.join(', ') || '');
    const [inputSchema, setInputSchema] = useState(JSON.stringify(tool?.input_schema || {}, null, 2));
    const isEditing = !!tool;

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        try {
            const parsedSchema = JSON.parse(inputSchema);
            const toolData: Tool = {
                id: tool?.id || uuidv4(),
                name,
                description,
                input_schema: parsedSchema,
                tags: tags.split(',').map(t => t.trim()).filter(Boolean),
            };
            onSave(toolData);
            onClose();
        } catch (error) {
            alert('Invalid JSON in Input Schema.');
        }
    };

    return (
        <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50" onClick={onClose}>
            <div className="bg-bg-light-card glass-card rounded-xl shadow-2xl w-full max-w-2xl mx-4" onClick={e => e.stopPropagation()}>
                <form onSubmit={handleSubmit}>
                    <div className="p-6 border-b border-border-color">
                        <h3 className="text-xl font-semibold">{isEditing ? 'Edit Tool' : 'Create New Tool'}</h3>
                        <p className="text-text-secondary text-sm mt-1">Define a new capability for your agents.</p>
                    </div>
                    <div className="p-6 space-y-4 max-h-[60vh] overflow-y-auto">
                        <input type="text" value={name} onChange={e => setName(e.target.value)} required placeholder="Tool Name (e.g., WebBrowser)" className="w-full bg-bg-light-bg border border-border-color rounded-lg p-3" />
                        <textarea value={description} onChange={e => setDescription(e.target.value)} required placeholder="Description of what this tool does." className="w-full h-24 bg-bg-light-bg border border-border-color rounded-lg p-3 resize-none" />
                        <input type="text" value={tags} onChange={e => setTags(e.target.value)} placeholder="Tags (comma-separated, e.g., web, system)" className="w-full bg-bg-light-bg border border-border-color rounded-lg p-3" />
                        <div>
                           <label className="block text-sm font-medium text-text-primary mb-2">Input Schema (JSON)</label>
                           <textarea value={inputSchema} onChange={e => setInputSchema(e.target.value)} required className="w-full h-40 bg-bg-light-bg border border-border-color rounded-lg p-3 font-mono text-sm resize-y" />
                        </div>
                    </div>
                    <div className="p-4 bg-bg-light-bg rounded-b-xl flex justify-end space-x-3">
                        <button type="button" onClick={onClose} className="text-text-secondary hover:text-text-primary px-4 py-2 rounded-lg">Cancel</button>
                        <button type="submit" className="bg-brand-primary text-white font-semibold px-5 py-2 rounded-lg hover:bg-brand-primary/90">Save Tool</button>
                    </div>
                </form>
            </div>
        </div>
    );
};


interface ToolRegistryViewProps {
    tools: Tool[];
    setTools: React.Dispatch<React.SetStateAction<Tool[]>>;
}

const ToolRegistryView: React.FC<ToolRegistryViewProps> = ({ tools, setTools }) => {
    const [isEditorOpen, setIsEditorOpen] = useState(false);
    const [editingTool, setEditingTool] = useState<Tool | null>(null);

    const handleSaveTool = (tool: Tool) => {
        setTools(prev => {
            const exists = prev.some(t => t.id === tool.id);
            if (exists) {
                return prev.map(t => t.id === tool.id ? tool : t);
            }
            return [...prev, tool];
        });
    };

    const handleDeleteTool = (id: string) => {
        if (window.confirm('Are you sure you want to delete this tool?')) {
            setTools(prev => prev.filter(t => t.id !== id));
        }
    };

    const openEditor = (tool: Tool | null) => {
        setEditingTool(tool);
        setIsEditorOpen(true);
    };

    return (
        <div className="p-4 sm:p-6 lg:p-8 animate-scale-in">
            {isEditorOpen && <ToolEditor tool={editingTool} onClose={() => setIsEditorOpen(false)} onSave={handleSaveTool} />}
            <div className="flex justify-between items-center mb-6">
                <div>
                    <h2 className="text-3xl font-bold text-text-primary">Tool Registry</h2>
                    <p className="text-text-secondary mt-1">Manage the custom tools and capabilities available to your agents.</p>
                </div>
                <button onClick={() => openEditor(null)} className="flex items-center space-x-2 bg-brand-primary text-white font-semibold px-4 py-2 rounded-lg hover:bg-brand-primary/90 transition-colors">
                    <PlusIcon className="w-5 h-5" />
                    <span>New Tool</span>
                </button>
            </div>

            <div className="bg-bg-light-card glass-card border border-border-color rounded-xl shadow-aura">
                {tools.length === 0 ? (
                    <div className="text-center p-12">
                         <WrenchScrewdriverIcon className="w-12 h-12 text-text-secondary mx-auto mb-4" />
                         <h3 className="text-xl font-semibold text-text-primary">No Tools Defined</h3>
                         <p className="text-text-secondary mt-2">Click "New Tool" to add a custom capability for your agents.</p>
                    </div>
                ) : (
                    <div className="overflow-x-auto">
                        <table className="w-full text-left">
                            <thead className="border-b border-border-color">
                                <tr>
                                    <th className="p-4 text-sm font-semibold text-text-secondary">Name</th>
                                    <th className="p-4 text-sm font-semibold text-text-secondary">Description</th>
                                    <th className="p-4 text-sm font-semibold text-text-secondary">Tags</th>
                                    <th className="p-4 text-sm font-semibold text-text-secondary"></th>
                                </tr>
                            </thead>
                            <tbody>
                                {tools.map(tool => (
                                    <tr key={tool.id} className="border-b border-border-color last:border-b-0 hover:bg-slate-50">
                                        <td className="p-4 font-medium text-brand-primary">{tool.name}</td>
                                        <td className="p-4 text-sm text-text-secondary max-w-md truncate">{tool.description}</td>
                                        <td className="p-4">
                                            <div className="flex flex-wrap gap-1.5">
                                                {tool.tags?.map(tag => (
                                                    <span key={tag} className="text-xs bg-slate-200 text-text-secondary px-2 py-1 rounded-md">{tag}</span>
                                                ))}
                                            </div>
                                        </td>
                                        <td className="p-4">
                                            <div className="flex items-center space-x-3 justify-end">
                                                <button onClick={() => openEditor(tool)} className="text-text-secondary hover:text-brand-primary"><PencilIcon /></button>
                                                <button onClick={() => handleDeleteTool(tool.id)} className="text-text-secondary hover:text-red-500"><TrashIcon /></button>
                                            </div>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                )}
            </div>
        </div>
    );
};

export default ToolRegistryView;