/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/

import React from 'react';
import { Book, TerminalSquare, Wrench } from 'lucide-react';
import { Tabs, ScrollArea, Textarea, Center, Skeleton } from '@mantine/core';

import ResultCard from './ObjectCard';
import { useGenerationStore, useAppStore } from '../services/store';

interface RightPanelProps {
    systemPrompt: string;
    onSystemPromptChange: (value: string) => void;
}

const RightPanel: React.FC<RightPanelProps> = ({ systemPrompt, onSystemPromptChange }) => {
    
    const { 
        outputs, 
        selectedOutput, 
        setSelectedOutput, 
        updateOutputCode,
        deleteOutput,
        isLoading
    } = useGenerationStore();

    const { appMode, workspaceMode } = useAppStore();

    const handleSelectOutput = (output: any) => {
        // Here you might add logic to set a soundboard view if you re-introduce it
        setSelectedOutput(output);
    }
    
    const LibraryContent = () => {
        if (isLoading && outputs.length === 0) {
            return (
                <>
                    <Skeleton height={256} mb="md" radius="lg" />
                    <Skeleton height={256} mb="md" radius="lg" />
                </>
            );
        }
        if (outputs.length > 0) {
            return [...outputs].reverse().map(output => (
                <ResultCard 
                    key={output.id}
                    output={output}
                    onCodeChange={updateOutputCode}
                    isSelected={selectedOutput?.id === output.id}
                    onSelect={() => handleSelectOutput(output)}
                    onDelete={deleteOutput}
                />
            ))
        }
        return (
            <Center h="100%">
                <p>Generated assets will appear here.</p>
            </Center>
        )
    };

    return (
        <div className="flex flex-col h-full glass-panel rounded-2xl overflow-hidden">
            <Tabs defaultValue="library" variant="pills" m="xs">
                <Tabs.List grow>
                    <Tabs.Tab value="library" leftSection={<Book size={16} />}>
                        Library
                    </Tabs.Tab>
                    <Tabs.Tab value="prompt" leftSection={<TerminalSquare size={16} />}>
                        Prompt
                    </Tabs.Tab>
                </Tabs.List>

                <Tabs.Panel value="library" pt="xs" style={{height: 'calc(100vh - 200px)'}}>
                    <ScrollArea h="100%" p="xs">
                        <LibraryContent />
                    </ScrollArea>
                </Tabs.Panel>
                
                <Tabs.Panel value="prompt" pt="xs" style={{height: 'calc(100vh - 200px)'}}>
                    <Textarea
                        value={systemPrompt}
                        onChange={(event) => onSystemPromptChange(event.currentTarget.value)}
                        autosize
                        minRows={10}
                        maxRows={20}
                        variant="filled"
                        styles={{input: {height: '100% !important'}}}
                    />
                </Tabs.Panel>
            </Tabs>
        </div>
    );
};

export default RightPanel;
