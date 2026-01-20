/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/

import { MantineTheme } from '@mantine/core';

export type WorkspaceMode = 'creative' | 'coding' | 'admin' | 'operations' | 'game' | 'environments';

export type AppMode = 
  // Creative Suite
  'scene' | 'image' | 'avatar' | 'image_to_3d' | 'image_to_animation' | 
  'music' | 'sound_fx' | 'audio_enhance' | 'voiceover' | 'game_asset' | 'ai_toolbench' |
  // Dev & Ops Suite
  'code' | 'database' | 'metrics' | 'shell' | 'vscode' |
  // Professional App Suite
  'email' | 'documents' | 'spreadsheets' | 'storage' |
  // System
  'chat';

export interface Output {
  id: number;
  type: 'p5js' | 'gltf_url' | 'audio_url' | 'video_url' | 'image_url' | 'log' | 'unknown';
  code?: string; 
  url?: string; 
  fullResponse: string;
}

export interface ChatMessage {
  sender: 'user' | 'agent';
  text: string;
  type?: 'log' | 'error';
}

export interface QueryResult {
  headers: string[];
  rows: (string | number)[][];
}

export interface VirtualEnvironment {
    id: string;
    name: string;
    status: 'Running' | 'Stopped' | 'Provisioning';
    baseImage: 'Ubuntu 22.04' | 'Windows Server 2022' | 'GPU Optimized';
    installedApps: AppMode[];
}

// Types for Settings
export interface APIConfig {
  baseUrl: string;
  endpoints: Record<AppMode, string>;
}

export interface Theme {
  primaryColor: string;
  accentColor: string;
  backgroundColor: string;
  textColor: string;
  borderColor: string;
  fontFamily: string;
  fontFamilyMonospace: string;
  radius: number; // in rem
  shadow: number; // intensity
  opacity: number;
  // FIX: Added optional property to match usage in App.tsx
  backgroundImageUrl?: string;
}

export interface AppConfig {
  defaultCreativeMode: AppMode;
  sceneAssetValidation: boolean;
  funMode: boolean;
  // FIX: Add isLayoutLocked to AppConfig to resolve type error in services/store.ts
  isLayoutLocked: boolean;
}

export interface PanelLayout {
  id: 'left' | 'center' | 'right';
  width: number; // Percentage
}

export interface Settings {
  api: APIConfig;
  ui: Theme;
  app: AppConfig;
  layout: PanelLayout[];
}

// Auth Types
export interface User {
    id: string;
    name: string;
    role: 'admin' | 'user';
}


// Types for Coding Workspace File System
export interface FileNode {
  type: 'file';
  id: string;
  name: string;
  content: string;
}
export interface FolderNode {
  type: 'folder';
  id: string;
  name: string;
  children: FileSystemNode[];
}
export type FileSystemNode = FileNode | FolderNode;