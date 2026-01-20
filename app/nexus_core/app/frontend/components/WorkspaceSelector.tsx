/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/

import React from 'react';
import { Palette, Code, Briefcase, BarChart2, Gamepad2, HardDrive } from 'lucide-react';
import { WorkspaceMode } from '../types';
import { SegmentedControl, Group, Text } from '@mantine/core';
import { useAppStore } from '../services/store';

const workspaces: { value: WorkspaceMode; label: string; icon: React.ReactNode }[] = [
  { value: 'creative', label: 'Creative', icon: <Palette size={20} /> },
  { value: 'coding', label: 'Coding', icon: <Code size={20} /> },
  { value: 'admin', label: 'Admin', icon: <Briefcase size={20} /> },
  { value: 'operations', label: 'Operations', icon: <BarChart2 size={20} /> },
  { value: 'game', label: 'Game', icon: <Gamepad2 size={20} /> },
  { value: 'environments', label: 'Environments', icon: <HardDrive size={20} /> },
];

const WorkspaceSelector: React.FC = () => {
  const { workspaceMode, setWorkspaceMode } = useAppStore();

  const data = workspaces.map(workspace => ({
    value: workspace.value,
    label: (
      <Group gap="xs" justify="center" style={{padding: '0 0.5rem'}}>
        {workspace.icon}
        <Text size="sm" className="hidden lg:inline">{workspace.label}</Text>
      </Group>
    ),
  }));
  
  return (
    <SegmentedControl
        value={workspaceMode}
        onChange={(value) => setWorkspaceMode(value as WorkspaceMode)}
        data={data}
        radius="lg"
        size="md"
    />
  );
};

export default WorkspaceSelector;
