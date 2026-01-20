/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/

import React from 'react';
import { SimpleGrid, Button, Title, Text, Group } from '@mantine/core';
import { 
    Scissors, Box, Waves, Mic, Image as ImageIcon, Music, Volume2, 
    UserCircle, Package, Layers, Code, Database, BarChart2, Terminal, Folder, Mail, FileText, Sheet, 
    Palette, Play, Settings, Cpu, BrainCircuit, Gamepad2, DatabaseZap, Bug, Save,
    SlidersHorizontal, Download, Share2, Copy
} from 'lucide-react';
import { useAppStore } from '../services/store';

const ToolHeader: React.FC<{ icon: React.ReactNode, title: string, subtitle: string }> = ({ icon, title, subtitle }) => (
    <Group mb="xl">
        <div className="flex-shrink-0 w-12 h-12 bg-primary-light text-primary rounded-lg flex items-center justify-center" style={{backgroundColor: 'color-mix(in srgb, var(--primary-color) 15%, transparent)', color: 'var(--primary-color)'}}>
            {icon}
        </div>
        <div>
            <Title order={4}>{title}</Title>
            <Text size="sm" c="dimmed">{subtitle}</Text>
        </div>
    </Group>
);

const ToolSection: React.FC<{ title: string, children: React.ReactNode }> = ({ title, children }) => (
    <div className="mb-6">
        <Text size="sm" fw={700} mb="sm" style={{borderBottom: '1px solid var(--border-color)', paddingBottom: '0.5rem'}}>{title}</Text>
        <SimpleGrid cols={1} spacing="sm">
            {children}
        </SimpleGrid>
    </div>
);

const ToolsDrawer: React.FC = () => {
    const { workspaceMode, appMode } = useAppStore();

    const renderTools = () => {
        if (workspaceMode === 'creative') {
            switch (appMode) {
                case 'image':
                case 'avatar':
                    return <> <ToolHeader icon={<Palette size={24}/>} title="Image Tools" subtitle="Refine and edit visuals" /> <ToolSection title="Actions"><Button variant="light">Upscale Image</Button><Button variant="light">Generate Variations</Button><Button leftSection={<Download size={14} />} variant="light">Download</Button></ToolSection> </>;
                case 'image_to_animation':
                    return <> <ToolHeader icon={<Scissors size={24}/>} title="Animation Tools" subtitle="Timeline and effects" /> <ToolSection title="Editing"><Button variant="light">Trim Video</Button><Button variant="light">Add Effects</Button></ToolSection> </>;
                case 'image_to_3d':
                    return <> <ToolHeader icon={<Box size={24}/>} title="3D Model Tools" subtitle="Materials and lighting" /> <ToolSection title="Material"><Button variant="light">Change Texture</Button><Button variant="light">Adjust Lighting</Button></ToolSection> </>;
                case 'music':
                     return <> <ToolHeader icon={<Music size={24}/>} title="Music Tools" subtitle="Tempo and instruments" /> <ToolSection title="Controls"><Button variant="light">Change Tempo</Button><Button variant="light">Switch Instrument</Button></ToolSection> </>;
                case 'scene':
                    return <> <ToolHeader icon={<Layers size={24}/>} title="p5.js Scene Tools" subtitle="Interactivity and export" /> <ToolSection title="Actions"><Button leftSection={<Download size={14} />} variant="light">Export as HTML</Button><Button leftSection={<Copy size={14} />} variant="light">Copy Code</Button></ToolSection> </>;
                default:
                    return <> <ToolHeader icon={<SlidersHorizontal size={24}/>} title="Creative Tools" subtitle="General creative actions" /> <ToolSection title="General"><Button variant="light">Batch Generate</Button><Button variant="light">Export All</Button></ToolSection> </>;
            }
        }
        
        if (workspaceMode === 'operations') {
            return <> <ToolHeader icon={<Cpu size={24}/>} title="Operations" subtitle="System health and diagnostics" /> <ToolSection title="Actions"><Button variant="light">Run Diagnostics</Button><Button variant="light">Clear Caches</Button></ToolSection> </>;
        }
        
        return (
            <div className="flex flex-col items-center justify-center h-full text-slate-500 p-4 text-center">
                <SlidersHorizontal size={40} className="mb-4" />
                <Title order={5}>Contextual Tools</Title>
                <Text size="sm">No tools available for this workspace.</Text>
            </div>
        );
    };

    return (
        <div className="w-full h-full p-4 animate-fade-in">
            {renderTools()}
        </div>
    );
};

export default ToolsDrawer;
