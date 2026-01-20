/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/

import React, { useState, useEffect } from 'react';
import { Settings as SettingsType, AppMode, Theme } from '../types';
import { getSettings, saveSettings } from '../services/settingsService';
import { Modal, Tabs, TextInput, ColorInput, Select, Slider, Switch, Grid, Text, Title, Group } from '@mantine/core';
import { Server, Palette, SlidersHorizontal, Fan } from 'lucide-react';
import Knob from './Knob';

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSettingsChange: (newSettings: SettingsType) => void;
}

const availableFonts = [
    { value: 'Inter, sans-serif', label: 'Inter' },
    { value: 'Poppins, sans-serif', label: 'Poppins' },
    { value: 'Roboto Mono, monospace', label: 'Roboto Mono' },
];

const availableMonospaceFonts = [
    { value: 'Fira Code, monospace', label: 'Fira Code' },
    { value: 'Source Code Pro, monospace', label: 'Source Code Pro' },
    { value: 'Roboto Mono, monospace', label: 'Roboto Mono' },
];

const SettingsModal: React.FC<SettingsModalProps> = ({ isOpen, onClose, onSettingsChange }) => {
  const [settings, setSettings] = useState<SettingsType>(getSettings);
  
  useEffect(() => {
    if (isOpen) {
        setSettings(getSettings());
    }
  }, [isOpen]);
  
  const handleSave = () => {
    saveSettings(settings);
    onSettingsChange(settings);
    onClose();
  };

  const handleApiChange = (e: React.ChangeEvent<HTMLInputElement>, key?: AppMode) => {
    const { name, value } = e.target;
    if (key) {
      setSettings(s => ({ ...s, api: { ...s.api, endpoints: { ...s.api.endpoints, [key]: value } } }));
    } else {
      setSettings(s => ({ ...s, api: { ...s.api, [name]: value } }));
    }
  };

  const handleThemeChange = (key: keyof Theme, value: any) => {
    setSettings(s => ({ ...s, ui: { ...s.ui, [key]: value } }));
  };

  const handleAppChange = (key: string, value: any) => {
    setSettings(s => ({ ...s, app: { ...s.app, [key]: value } }));
  };

  return (
    <Modal opened={isOpen} onClose={handleSave} title={<Title order={3}>Settings</Title>} size="xl" centered>
        <Tabs defaultValue="ui">
            <Tabs.List>
                <Tabs.Tab value="ui" leftSection={<Palette size={16} />}>Theme Editor</Tabs.Tab>
                <Tabs.Tab value="api" leftSection={<Server size={16} />}>API Endpoints</Tabs.Tab>
                <Tabs.Tab value="app" leftSection={<SlidersHorizontal size={16} />}>Application</Tabs.Tab>
            </Tabs.List>

            <Tabs.Panel value="ui" pt="md">
                <Grid>
                    <Grid.Col span={6}>
                        <ColorInput label="Primary Color" value={settings.ui.primaryColor} onChange={(c) => handleThemeChange('primaryColor', c)} />
                        <ColorInput mt="sm" label="Accent Color" value={settings.ui.accentColor} onChange={(c) => handleThemeChange('accentColor', c)} />
                        <ColorInput mt="sm" label="Background Color" value={settings.ui.backgroundColor} onChange={(c) => handleThemeChange('backgroundColor', c)} />
                        <ColorInput mt="sm" label="Text Color" value={settings.ui.textColor} onChange={(c) => handleThemeChange('textColor', c)} />
                        <ColorInput mt="sm" label="Border Color" value={settings.ui.borderColor} onChange={(c) => handleThemeChange('borderColor', c)} />
                    </Grid.Col>
                    <Grid.Col span={6}>
                        <Select label="Main Font" data={availableFonts} value={settings.ui.fontFamily} onChange={(f) => handleThemeChange('fontFamily', f)} />
                        <Select mt="sm" label="Monospace Font" data={availableMonospaceFonts} value={settings.ui.fontFamilyMonospace} onChange={(f) => handleThemeChange('fontFamilyMonospace', f)} />
                        <Text size="sm" fw={500} mt="sm">Border Radius: {settings.ui.radius.toFixed(2)}rem</Text>
                        <Slider value={settings.ui.radius} onChange={(r) => handleThemeChange('radius', r)} min={0} max={2} step={0.1} />
                        <Text size="sm" fw={500} mt="sm">Shadow Intensity: {settings.ui.shadow.toFixed(2)}</Text>
                        <Slider value={settings.ui.shadow} onChange={(s) => handleThemeChange('shadow', s)} min={0} max={0.2} step={0.01} />
                        <Group mt="md" justify="space-between">
                            <Text size="sm" fw={500}>Glass Opacity</Text>
                            <Knob value={settings.ui.opacity} onChange={(o) => handleThemeChange('opacity', o)} min={0.1} max={1} step={0.05} />
                        </Group>
                    </Grid.Col>
                </Grid>
            </Tabs.Panel>
            
            <Tabs.Panel value="api" pt="md">
                <TextInput label="Base URL" name="baseUrl" value={settings.api.baseUrl} onChange={handleApiChange} />
                <Title order={5} mt="lg" mb="sm">Endpoints</Title>
                <div style={{maxHeight: 300, overflowY: 'auto'}}>
                 {Object.keys(settings.api.endpoints).map(key => (
                    <TextInput key={key} label={key} value={settings.api.endpoints[key as AppMode]} onChange={(e) => handleApiChange(e, key as AppMode)} mt="xs" styles={{ label: { textTransform: 'capitalize'}}} />
                 ))}
                </div>
            </Tabs.Panel>

            <Tabs.Panel value="app" pt="md">
                <Select label="Default Creative Mode" value={settings.app.defaultCreativeMode} onChange={(v) => handleAppChange('defaultCreativeMode', v)} data={Object.keys(getSettings().api.endpoints).filter(k => k !== 'chat')} />
                <Switch mt="md" label="Enable Asset Requirement Message for 'Scene' Mode" checked={settings.app.sceneAssetValidation} onChange={(e) => handleAppChange('sceneAssetValidation', e.currentTarget.checked)} />
                <Switch mt="md" label="Fun Mode" description="Adds a little flair to the UI." checked={settings.app.funMode} onChange={(e) => handleAppChange('funMode', e.currentTarget.checked)} onLabel={<Fan size={14}/>} offLabel={<Fan size={14}/>} />
            </Tabs.Panel>
        </Tabs>
    </Modal>
  );
};

export default SettingsModal;
