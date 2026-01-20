/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/

import React from 'react';
import { Box, Layers, SlidersHorizontal, Play, Camera } from 'lucide-react';
import { Paper, Title, List, ThemeIcon, Text } from '@mantine/core';
import Gauge from '../Gauge';

const GameWorkspace: React.FC = () => {
  return (
    <div className="w-full h-full p-4 flex gap-4 text-slate-800">
      {/* Hierarchy Panel */}
      <Paper withBorder p="md" className="w-1/5 glass-panel">
        <Title order={4} className="flex items-center justify-between mb-4">Hierarchy <Layers size={20} /></Title>
        <List spacing="xs" size="sm">
            <List.Item>Scene</List.Item>
            <List.Item ml="md">Directional Light</List.Item>
            <List.Item ml="md" icon={<Camera size={14} />}>Main Camera</List.Item>
            <List.Item ml="md" icon={<Box size={14} />}>Player Object</List.Item>
            <List.Item ml="md">Terrain</List.Item>
        </List>
      </Paper>

      {/* Scene Viewer */}
      <div className="w-3/5 flex flex-col gap-4">
        <div className="h-full glass-panel rounded-2xl p-2 flex flex-col">
           <div className="flex-shrink-0 flex items-center justify-center p-1 space-x-2 bg-black/5 rounded-md mb-2">
                <button className="p-1.5 bg-indigo-600 text-white rounded-md hover:bg-indigo-500 transition-colors"><Play size={16}/></button>
                <p className="text-sm font-semibold text-slate-700">Game View</p>
           </div>
           <div className="flex-grow bg-gradient-to-br from-sky-400 to-blue-600 rounded-lg flex items-center justify-center text-white relative overflow-hidden">
                <div className="absolute w-full h-1/3 bg-green-500 bottom-0"></div>
                <Box size={64} className="opacity-50 z-10"/>
                <div className="absolute top-10 right-10 w-16 h-16 bg-yellow-300 rounded-full opacity-70"></div>
           </div>
        </div>
      </div>

      {/* Inspector Panel */}
      <Paper withBorder p="md" className="w-1/5 glass-panel overflow-y-auto">
        <Title order={4} className="flex items-center justify-between mb-4">Inspector <SlidersHorizontal size={20} /></Title>
         <div className="space-y-3 text-sm">
            <Paper withBorder p="xs"><Text fw={500}>Transform</Text><Text size="xs">Position: X:0, Y:1, Z:0</Text></Paper>
            <Paper withBorder p="xs"><Text fw={500}>Mesh Renderer</Text><Text size="xs">Mesh: Capsule</Text></Paper>
            <Paper withBorder p="xs"><Text fw={500}>Script: PlayerController</Text><Text size="xs">Speed: 5</Text></Paper>
            <div className="pt-4">
                <Gauge value={88.4} label="Engine Performance" color="var(--accent-color)" />
            </div>
        </div>
      </Paper>
    </div>
  );
};

export default GameWorkspace;
