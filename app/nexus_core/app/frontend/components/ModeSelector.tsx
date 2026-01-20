/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/

import React from 'react';
import { 
    Layers, Box, Film, Waves, Mic, Package, Image as ImageIcon, Volume2, UserCircle, Music,
    Code, Database, BarChart2, Terminal, Folder, Mail, FileText, Sheet, BrainCircuit, MessageSquare 
} from 'lucide-react';
import { AppMode } from '../types';
import { SimpleGrid, Button, Tooltip } from '@mantine/core';
import { useAppStore } from '../services/store';

interface AppSelectorProps {
  availableApps: AppMode[];
}

const appIcons: Record<AppMode, { name: string; icon: React.ReactNode }> = {
  // Creative
  scene: { name: 'Scene', icon: <Layers size={18} /> },
  image: { name: 'Image', icon: <ImageIcon size={18} /> },
  avatar: { name: 'Avatar', icon: <UserCircle size={18} /> },
  image_to_3d: { name: '3D Model', icon: <Box size={18} /> },
  image_to_animation: { name: 'Animate', icon: <Film size={18} /> },
  music: { name: 'Music', icon: <Music size={18} /> },
  sound_fx: { name: 'Sound FX', icon: <Volume2 size={18} /> },
  audio_enhance: { name: 'Audio', icon: <Waves size={18} /> },
  voiceover: { name: 'Voiceover', icon: <Mic size={18} /> },
  game_asset: { name: 'Game Asset', icon: <Package size={18} /> },
  ai_toolbench: { name: 'AI Toolbench', icon: <BrainCircuit size={18} /> },
  // Dev & Ops
  code: { name: 'Code', icon: <Code size={18} /> },
  database: { name: 'Database', icon: <Database size={18} /> },
  metrics: { name: 'Metrics', icon: <BarChart2 size={18} /> },
  shell: { name: 'Shell', icon: <Terminal size={18} /> },
  vscode: { name: 'VS Code', icon: <Code size={18} /> },
  // Professional
  storage: { name: 'Storage', icon: <Folder size={18} /> },
  email: { name: 'Email', icon: <Mail size={18} /> },
  documents: { name: 'Docs', icon: <FileText size={18} /> },
  spreadsheets: { name: 'Sheets', icon: <Sheet size={18} /> },
  // System
  chat: { name: 'Chat', icon: <MessageSquare size={18} /> },
};

const AppSelector: React.FC<AppSelectorProps> = ({ availableApps }) => {
  const { appMode, setAppMode } = useAppStore();
  
  return (
    <div>
        <SimpleGrid cols={5} spacing="sm">
            {availableApps.map((appId) => {
                const app = appIcons[appId];
                if (!app) return null;
                return (
                    <Tooltip key={appId} label={app.name} position="bottom">
                        <Button
                            variant={appMode === appId ? 'filled' : 'light'}
                            onClick={() => setAppMode(appId)}
                            style={{ height: 64, flexDirection: 'column', padding: '0.5rem' }}
                        >
                           {app.icon}
                           <span className="text-xs text-center leading-tight truncate w-full mt-1">{app.name}</span>
                        </Button>
                    </Tooltip>
                );
            })}
        </SimpleGrid>
    </div>
  );
};

export default AppSelector;
