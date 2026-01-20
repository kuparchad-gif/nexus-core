/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/

import React, { useState, useMemo, useEffect, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { PanelGroup, Panel, PanelResizeHandle } from 'react-resizable-panels';
import { UploadCloud, Bot, User, Send, ChevronsUp, ChevronsDown, RotateCcw, Grid, X, Info, MoveHorizontal, LockKeyhole, LockKeyholeOpen } from 'lucide-react';
import { ActionIcon, Button, Textarea, Tooltip, Paper, Text, Group, Box, ScrollArea, Center } from '@mantine/core';
import { notifications } from '@mantine/notifications';

import RightPanel from '../RightPanel';
import ModelViewer from '../ModelViewer';
import P5Viewer from '../P5Viewer';
import NexusIdleAnimation from '../NexusIdleAnimation';
import AppSelector from '../ModeSelector';
import AssetPreview from '../AssetPreview';
import Soundboard from '../Soundboard';
import VirtualAppStream from '../VirtualAppStream';
import { generateAsset } from '../../services/geminiService';
import { AppMode, PanelLayout } from '../../types';
import { useGenerationStore, useAppStore } from '../../services/store';

const defaultSystemPrompt = `You are an AI assistant for the Aethereal Nexus platform. Your capabilities span creative coding, data analysis, and system operations. Provide concise, accurate, and helpful responses based on the user's request and the current operational mode.`;

const appRequirements: Partial<Record<AppMode, { type: 'image' | 'audio', message: string }>> = {
  scene: { type: 'image', message: 'This mode requires one image file.' },
  image_to_3d: { type: 'image', message: 'This mode requires one image file.' },
  image_to_animation: { type: 'image', message: 'Audio is optional, but an image is required.' },
  audio_enhance: { type: 'audio', message: 'This mode requires one audio file.' },
};

const PanelHeader: React.FC<{ title: string; onMoveLeft: () => void; onMoveRight: () => void; isLocked: boolean }> = ({ title, onMoveLeft, onMoveRight, isLocked }) => (
    <Group justify="space-between" p="xs" style={{borderBottom: '1px solid var(--border-color)', height: '48px'}}>
        <Text size="lg" fw={700} ml="xs">{title}</Text>
        {!isLocked && (
            <Group gap="xs">
                <Tooltip label="Move Left"><ActionIcon onClick={onMoveLeft} variant="light"><MoveHorizontal size={16} style={{transform: 'rotate(180deg)'}} /></ActionIcon></Tooltip>
                <Tooltip label="Move Right"><ActionIcon onClick={onMoveRight} variant="light"><MoveHorizontal size={16} /></ActionIcon></Tooltip>
            </Group>
        )}
    </Group>
);

const CreativeWorkspace: React.FC = () => {
  const [systemPrompt, setSystemPrompt] = useState<string>(defaultSystemPrompt);
  const [assets, setAssets] = useState<File[]>([]);
  const [userInput, setUserInput] = useState<string>('');
  const [isSoundboardView, setIsSoundboardView] = useState(false);
  
  // Zustand Store Hooks
  const { 
    outputs, 
    chatHistory, 
    selectedOutput, 
    isLoading, 
    loadingMessage, 
    addOutput, 
    addChatMessage, 
    setSelectedOutput, 
    setLoading, 
    deleteOutput, 
    clearSession 
  } = useGenerationStore();
  
  const { 
    appMode, 
    isLayoutLocked, 
    settings, 
    setSettings 
  } = useAppStore();

  const [panelOrder, setPanelOrder] = useState(settings.layout.map(p => p.id));
  
  const creativeApps: AppMode[] = ['scene', 'image', 'avatar', 'image_to_3d', 'image_to_animation', 'ai_toolbench', 'music', 'sound_fx', 'audio_enhance', 'voiceover'];
  const hasAudioOutputs = useMemo(() => outputs.some(o => o.type === 'audio_url'), [outputs]);

  useEffect(() => { if (isSoundboardView) setSelectedOutput(null); }, [isSoundboardView, setSelectedOutput]);
  useEffect(() => { if (!hasAudioOutputs && isSoundboardView) setIsSoundboardView(false); }, [hasAudioOutputs, isSoundboardView]);
  useEffect(() => { setSelectedOutput(null); setIsSoundboardView(false); }, [appMode, setSelectedOutput]);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    setAssets(prevAssets => [...prevAssets, ...acceptedFiles]);
    notifications.show({ title: 'Upload Success', message: `Uploaded ${acceptedFiles.length} asset(s).`, color: 'teal' });
  }, []);
  
  const removeAsset = (fileName: string) => setAssets(prevAssets => prevAssets.filter(f => f.name !== fileName));
  const { getRootProps, getInputProps, isDragActive } = useDropzone({ onDrop, accept: { 'image/*': [], 'audio/*': [], 'video/*': [] } });
  
  const handleGenerate = async () => {
    const currentUserInput = userInput.trim();
    if (!currentUserInput && assets.length === 0) {
        notifications.show({ title: 'Input Required', message: 'Please enter a prompt or upload an asset.', color: 'yellow' });
        return;
    }
    
    const requirement = appRequirements[appMode];
    if (settings.app.sceneAssetValidation && appMode === 'scene' && !assets.some(f => f.type.startsWith('image'))) {
        notifications.show({ title: 'Asset Required', message: appRequirements.scene.message, color: 'red' });
        return;
    }
    if (requirement && appMode !== 'scene' && !assets.some(f => f.type.startsWith(requirement.type))) {
        notifications.show({ title: 'Asset Required', message: requirement.message, color: 'red' });
        return;
    }

    setLoading(true);
    addChatMessage({ sender: 'user', text: currentUserInput || `Execute '${appMode}' with uploaded asset(s).` });
    setUserInput('');

    try {
        const generatedOutputs = await generateAsset(appMode, assets, currentUserInput);
        generatedOutputs.forEach(output => addOutput(output));

        if (generatedOutputs.length > 0) {
            generatedOutputs[0].type === 'audio_url' ? setIsSoundboardView(true) : setSelectedOutput(generatedOutputs[0]);
            notifications.show({ title: 'Generation Complete', message: `Successfully generated ${generatedOutputs.length} asset(s).`, color: 'green' });
        }
    } catch (e: any) {
        notifications.show({ title: 'Generation Error', message: e.message || "An unknown error occurred.", color: 'red' });
        addChatMessage({ sender: 'agent', text: `Error: ${e.message}`, type: 'error' });
    } finally {
        setLoading(false);
    }
  };
  
  const handleClearSession = () => { setAssets([]); clearSession(); setIsSoundboardView(false); };
  
  const handlePanelLayout = (sizes: number[]) => {
      const newLayout = settings.layout.map((panel, index) => ({ ...panel, width: sizes[index] }));
      setSettings({ ...settings, layout: newLayout });
  };
  
  const handleMovePanel = (panelId: 'left' | 'center' | 'right', direction: 'left' | 'right') => {
    const currentIndex = panelOrder.findIndex(id => id === panelId);
    const newIndex = direction === 'left' ? currentIndex - 1 : currentIndex + 1;
    if (newIndex < 0 || newIndex >= panelOrder.length) return;
    const newOrder = [...panelOrder];
    [newOrder[currentIndex], newOrder[newIndex]] = [newOrder[newIndex], newOrder[currentIndex]];
    setPanelOrder(newOrder);
  };

  const assetUploaderIsVisible = ['scene', 'image_to_3d', 'image_to_animation', 'audio_enhance'].includes(appMode);

  const renderCenterPanelContent = () => {
    if (appMode === 'ai_toolbench') return <VirtualAppStream appName="AI Toolbench" />;
    if (isLoading) return <Paper h="100%" w="100%" withBorder className="flex items-center justify-center"><Box className="text-center p-4"><Text mt="md" fz="sm">{loadingMessage}</Text></Box></Paper>;
    if (isSoundboardView) return <Soundboard outputs={outputs} />;
    if (selectedOutput) {
        switch (selectedOutput.type) {
          case 'p5js': return <P5Viewer code={selectedOutput.code!} />;
          case 'gltf_url': return <ModelViewer url={selectedOutput.url!} />;
          case 'audio_url': return <div className="p-4 flex items-center justify-center h-full"><audio controls src={selectedOutput.url!} className="w-full max-w-md"/></div>;
          case 'video_url': return <video controls src={selectedOutput.url!} className="max-w-full max-h-full rounded-lg" />;
          case 'image_url': return <img src={selectedOutput.url!} alt={selectedOutput.fullResponse} className="max-w-full max-h-full object-contain rounded-lg" />;
          default: return <Center h="100%" className="p-4 text-slate-500">Cannot preview this output type.</Center>;
        }
    }
    return <NexusIdleAnimation />;
  };
  
  const panelContent: Record<string, React.ReactNode> = {
    left: (
        <div className="flex flex-col h-full glass-panel rounded-2xl overflow-hidden">
            <PanelHeader title="Controls" isLocked={isLayoutLocked} onMoveLeft={() => handleMovePanel('left', 'left')} onMoveRight={() => handleMovePanel('left', 'right')} />
            <ScrollArea className="flex-grow">
            <div className="flex flex-col space-y-4 p-4">
                <AppSelector availableApps={creativeApps}/>
                {assetUploaderIsVisible && (
                    <div className="space-y-3">
                        <div {...getRootProps()} className={`p-6 border-2 border-dashed rounded-xl cursor-pointer text-center ${isDragActive ? 'border-indigo-500 bg-white/30' : 'border-black/10 hover:border-indigo-500'}`}><input {...getInputProps()} /><UploadCloud className="mx-auto h-10 w-10 text-slate-700" /><p className="mt-2 text-sm">Drop files here</p></div>
                        {assets.length > 0 && <div className="space-y-2 max-h-32 overflow-y-auto">{assets.map((file, index) => <AssetPreview key={file.name + index} file={file} onRemove={() => removeAsset(file.name)} />)}</div>}
                    </div>
                )}
                <Paper p="sm" withBorder className="flex-grow flex flex-col">
                    <Group justify="space-between" mb="sm">
                       <Text fw={700}>Aethereal Agent</Text>
                        <Group gap="xs">
                            {hasAudioOutputs && <Tooltip label="Soundboard"><ActionIcon variant="light" onClick={() => setIsSoundboardView(!isSoundboardView)} color={isSoundboardView ? 'blue': 'gray'}><Grid size={16} /></ActionIcon></Tooltip>}
                            <Tooltip label="Clear Session"><ActionIcon variant="light" onClick={handleClearSession}><RotateCcw size={16} /></ActionIcon></Tooltip>
                        </Group>
                    </Group>
                    <ScrollArea style={{height: 200}} mb="sm" p="xs" className="bg-white/20 rounded">
                        {chatHistory.map((msg, index) => (<div key={index} className={`flex items-start gap-2 mb-2 ${msg.sender === 'user' ? 'justify-end': ''}`}><Text size="sm" c={msg.type === 'error' ? 'red' : undefined}>{msg.text}</Text></div>))}
                    </ScrollArea>
                    <Textarea placeholder="Describe an asset to generate..." value={userInput} onChange={(e) => setUserInput(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && (e.preventDefault(), handleGenerate())} rightSection={<ActionIcon onClick={handleGenerate} loading={isLoading}><Send size={16}/></ActionIcon>} />
                </Paper>
            </div>
            </ScrollArea>
        </div>
    ),
    center: (
        <div className="flex flex-col h-full glass-panel rounded-2xl overflow-hidden">
            <PanelHeader title="Viewport" isLocked={isLayoutLocked} onMoveLeft={() => handleMovePanel('center', 'left')} onMoveRight={() => handleMovePanel('center', 'right')} />
            <div className="relative flex-grow flex items-center justify-center p-2">
                 {renderCenterPanelContent()}
                {selectedOutput && !isSoundboardView && <ActionIcon onClick={() => setSelectedOutput(null)} className="absolute top-4 right-4 z-10" variant="light" radius="xl"><X size={20} /></ActionIcon>}
            </div>
        </div>
    ),
    right: (
        <div className="flex flex-col h-full overflow-hidden">
            <RightPanel systemPrompt={systemPrompt} onSystemPromptChange={setSystemPrompt} />
        </div>
    ),
  };

  return (
    <div className="w-full h-full p-4">
      <PanelGroup direction="horizontal" onLayout={handlePanelLayout}>
        {panelOrder.map((panelId, index) => (
          <React.Fragment key={panelId}>
            <Panel defaultSize={settings.layout.find(p => p.id === panelId)?.width || 33} minSize={15}>
              {panelContent[panelId]}
            </Panel>
            {index < panelOrder.length - 1 && (
              <PanelResizeHandle
                className="w-3 flex items-center justify-center bg-transparent"
                disabled={isLayoutLocked}
              >
                <div className={`h-8 w-1 rounded-full ${isLayoutLocked ? 'bg-transparent' : 'bg-[var(--border-color)] group-hover:bg-[var(--primary-color)]'}`} />
              </PanelResizeHandle>
            )}
          </React.Fragment>
        ))}
      </PanelGroup>
    </div>
  );
};

export default CreativeWorkspace;