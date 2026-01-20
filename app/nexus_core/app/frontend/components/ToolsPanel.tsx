/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/

import React from 'react';
import { AppMode, WorkspaceMode } from '../types';
import { 
    SlidersHorizontal, Scissors, Box, Waves, Mic, Image as ImageIcon, Music, Volume2, 
    UserCircle, Package, Layers, Code, Database, BarChart2, Terminal, Folder, Mail, FileText, Sheet, 
    Palette, Play, Settings, Cpu, BrainCircuit, Gamepad2, DatabaseZap, Bug, Save
} from 'lucide-react';

interface ToolsPanelProps {
  workspace: WorkspaceMode;
  app: AppMode;
}

const ToolHeader: React.FC<{ icon: React.ReactNode, title: string, subtitle: string }> = ({ icon, title, subtitle }) => (
    <div className="flex items-center gap-4 mb-6">
        <div className="flex-shrink-0 w-12 h-12 bg-indigo-600/10 text-indigo-600 rounded-lg flex items-center justify-center">
            {icon}
        </div>
        <div>
            <h3 className="text-xl font-bold text-slate-800">{title}</h3>
            <p className="text-sm text-slate-600">{subtitle}</p>
        </div>
    </div>
);

const ToolSection: React.FC<{ title: string, children: React.ReactNode }> = ({ title, children }) => (
    <div className="mb-6">
        <h4 className="text-sm font-bold text-slate-700 mb-3 border-b border-black/10 pb-2">{title}</h4>
        <div className="space-y-3">
            {children}
        </div>
    </div>
);

const ButtonTool: React.FC<{ label: string, icon?: React.ReactNode }> = ({ label, icon }) => (
     <button className="w-full flex items-center justify-center gap-2 text-sm py-2 px-3 bg-black/5 text-slate-700 rounded-md hover:bg-black/10 hover:text-slate-900 transition-colors">{icon}{label}</button>
);


const ToolsPanel: React.FC<ToolsPanelProps> = ({ workspace, app }) => {
    const renderTools = () => {
        // Workspace-level tools can be defined here, but for now we focus on app-level
        switch (app) {
            // Creative
            case 'image':
            case 'avatar':
                return <> <ToolHeader icon={<Palette size={24}/>} title="Image Generation" subtitle="Refine and edit visuals" /> <ToolSection title="Actions"><ButtonTool label="Upscale Image" /><ButtonTool label="Generate Variations" /></ToolSection> </>;
            case 'image_to_animation':
                return <> <ToolHeader icon={<Scissors size={24}/>} title="Animation" subtitle="Timeline and effects" /> <ToolSection title="Editing"><ButtonTool label="Trim Video" /><ButtonTool label="Add Effects" /></ToolSection> </>;
            case 'image_to_3d':
                return <> <ToolHeader icon={<Box size={24}/>} title="3D Model" subtitle="Materials and lighting" /> <ToolSection title="Material"><ButtonTool label="Change Texture" /><ButtonTool label="Adjust Lighting" /></ToolSection> </>;
            case 'music':
                 return <> <ToolHeader icon={<Music size={24}/>} title="Music Composition" subtitle="Tempo and instruments" /> <ToolSection title="Controls"><ButtonTool label="Change Tempo" /><ButtonTool label="Switch Instrument" /></ToolSection> </>;
            case 'scene':
                return <> <ToolHeader icon={<Layers size={24}/>} title="Scene" subtitle="Interactivity and export" /> <ToolSection title="Interactivity"><ButtonTool label="Export Scene" /><ButtonTool label="Add Physics" /></ToolSection> </>;
            
            // Coding Workspace Apps
            case 'code':
                return <> <ToolHeader icon={<Code size={24}/>} title="Code Studio" subtitle="Execution and debugging" /> <ToolSection title="Execution"><ButtonTool label="Run Code" icon={<Play size={14}/>} /><ButtonTool label="Debug" icon={<Bug size={14}/>} /></ToolSection> </>;

            // Admin Workspace Apps
            case 'email':
                return <> <ToolHeader icon={<Mail size={24}/>} title="Email Client" subtitle="Compose and manage" /> <ToolSection title="Actions"><ButtonTool label="New Message" /><ButtonTool label="Refresh Inbox" /></ToolSection> </>;
            case 'documents':
                 return <> <ToolHeader icon={<FileText size={24}/>} title="Document Editor" subtitle="Formatting and styles" /> <ToolSection title="Formatting"><ButtonTool label="Apply Heading" /><ButtonTool label="Save Document" icon={<Save size={14}/>}/></ToolSection> </>;
            case 'database':
                return <> <ToolHeader icon={<Database size={24}/>} title="Database Client" subtitle="Connect and query" /> <ToolSection title="Connection"><ButtonTool label="Connect" icon={<DatabaseZap size={14}/>}/><ButtonTool label="Run Query" icon={<Play size={14}/>}/></ToolSection> </>;

            // Operations Workspace Apps
            case 'metrics':
                 return <> <ToolHeader icon={<BarChart2 size={24}/>} title="Metrics" subtitle="Timeframes and sources" /> <ToolSection title="Controls"><ButtonTool label="Refresh" /><ButtonTool label="Change Timeframe" /></ToolSection> </>;
            
            // Game Workspace Apps
            case 'game_asset':
                return <> <ToolHeader icon={<Gamepad2 size={24}/>} title="Game Engine" subtitle="Scene and object properties" /> <ToolSection title="Scene"><ButtonTool label="Play Scene" icon={<Play size={14}/>}/><ButtonTool label="Add Object" /></ToolSection> </>;

            default:
                 return (
                    <div className="flex flex-col items-center justify-center h-full text-slate-500 p-4 text-center">
                        <SlidersHorizontal size={40} className="mb-4" />
                        <h3 className="font-semibold text-slate-700">Contextual Tools</h3>
                        <p>Select an application to see its tools.</p>
                    </div>
                );
        }
    };

    return (
        <div className="w-full h-full p-6 animate-fade-in">
            {renderTools()}
        </div>
    );
};

export default ToolsPanel;