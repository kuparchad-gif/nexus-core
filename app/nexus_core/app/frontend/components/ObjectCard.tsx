/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/

import React, { useState } from 'react';
import Editor from '@monaco-editor/react';
import { Output } from '../types';
import { Music, Film, Box, Image as ImageIcon, Code, Eye, X } from 'lucide-react';
import { Card, Text, Group, Button, ActionIcon, Center } from '@mantine/core';
import P5Viewer from './P5Viewer';

interface ResultCardProps {
  output: Output;
  onCodeChange: (id: number, newCode: string) => void;
  isSelected: boolean;
  onSelect: () => void;
  onDelete: (id: number) => void;
}

const ResultCard: React.FC<ResultCardProps> = ({ output, onCodeChange, isSelected, onSelect, onDelete }) => {
  const [isCodeView, setIsCodeView] = useState(false);
  
  const Placeholder: React.FC<{icon: React.ReactNode, title: string, subtitle: string}> = ({icon, title, subtitle}) => (
    <Center h={256} style={{flexDirection: 'column'}}>
      {icon}
      <Text fw={500} mt="md">{title}</Text>
      <Text size="xs" c="dimmed">{subtitle}</Text>
    </Center>
  );
  
  const renderP5Content = () => {
    if (!output.code) return <Placeholder icon={<Code size={48}/>} title="p5.js Result" subtitle="Code not available" />;

    if (isCodeView) {
      return (
        <div style={{height: 256, borderRadius: 'var(--mantine-radius-md)', overflow: 'hidden'}} onClick={(e) => e.stopPropagation()}>
           <Editor
              height="100%"
              language="javascript"
              theme="vs-light"
              value={output.code}
              onChange={(value) => onCodeChange(output.id, value || '')}
              options={{ minimap: { enabled: false }, fontSize: 12, wordWrap: 'on' }}
            />
        </div>
      );
    }
    return (
      <div style={{height: 256, borderRadius: 'var(--mantine-radius-md)', overflow: 'hidden'}}>
        <P5Viewer code={output.code} />
      </div>
    );
  };

  const renderContent = () => {
    switch (output.type) {
      case 'p5js': return renderP5Content();
      case 'gltf_url': return <Placeholder icon={<Box size={48}/>} title="3D Model" subtitle="Select to view model" />;
      case 'audio_url': return <Placeholder icon={<Music size={48}/>} title="Audio Generated" subtitle="Select to play" />;
      case 'video_url': return <Placeholder icon={<Film size={48}/>} title="Animation" subtitle="Select to play video" />;
      case 'image_url': return <Placeholder icon={<ImageIcon size={48}/>} title="Image" subtitle="Select to view image" />;
      default: return <Placeholder icon={<Box size={48}/>} title="Result Log" subtitle="Select to view details" />;
    }
  };
  
  const canShowCode = output.type === 'p5js' && output.code;

  return (
    <Card
      shadow="sm"
      padding="lg"
      radius="lg"
      withBorder
      mb="md"
      onClick={onSelect}
      style={{
        cursor: 'pointer',
        borderColor: isSelected ? 'var(--mantine-color-primary-5)' : undefined,
        boxShadow: isSelected ? '0 0 10px var(--mantine-color-primary-2)' : undefined
      }}
    >
      <Card.Section withBorder inheritPadding py="xs">
        <Group justify="space-between">
          <Text fw={500}>Result #{output.id}</Text>
          <Group gap="xs">
              {canShowCode && (
                <Button
                    variant="light"
                    size="xs"
                    leftSection={isCodeView ? <Eye size={16} /> : <Code size={16} />}
                    onClick={(e) => {
                      e.stopPropagation();
                      setIsCodeView(!isCodeView);
                    }}
                >
                  {isCodeView ? 'View Sketch' : 'View Code'}
                </Button>
              )}
              <ActionIcon
                variant="light"
                color="red"
                size="md"
                title="Delete result"
                onClick={(e) => {
                  e.stopPropagation();
                  onDelete(output.id);
                }}
              >
                <X size={16} />
              </ActionIcon>
          </Group>
        </Group>
      </Card.Section>
      
      <Card.Section mt="sm">
        {renderContent()}
      </Card.Section>

      <Text size="xs" c="dimmed" mt="sm" truncate="end" title={output.fullResponse}>
        {output.fullResponse}
      </Text>
    </Card>
  );
};

export default ResultCard;
