

import React from 'react';

// Core Application
export type Page = 
  // Main
  'dashboard' | 'detail' | 'chat' | 'tools' | 'config' | 'nexus' | 'cloud' | 'personalization' |
  // Dev & Creative Suites
  'ide' | 'canvasEditor' | 'avatarStudio' | 'photoStudio' | 'videoEditor' | 'webDevSuite' |
  // System
  'systemTemplating';

// Theming
export interface ThemeConfig {
    wallpaper: string;
    accentColor: string; // Hex code
    glassOpacity: number; // 0.1 to 1.0
    iconStyle: 'glass-light' | 'glass-dark' | 'solid';
}

// Desktop Management
export interface IconPosition {
    x: number;
    y: number;
}

export interface ContextMenuState {
    visible: boolean;
    x: number;
    y: number;
    type: 'desktop' | 'icon';
    targetId?: string;
}

// Agent Model
export enum AgentStatus {
  IDLE = 'IDLE',
  PLANNING = 'PLANNING',
  RUNNING = 'RUNNING',
  AWAITING_USER_INPUT = 'AWAITING_USER_INPUT',
  CORRECTING = 'CORRECTING',
  COMPLETED = 'COMPLETED',
  ERROR = 'ERROR',
}

export enum TaskStatus {
  PENDING = 'PENDING',
  IN_PROGRESS = 'IN_PROGRESS',
  COMPLETED = 'COMPLETED',
  FAILED = 'FAILED',
  SKIPPED = 'SKIPPED',
}

export type TaskType = 'BROWSER' | 'SHELL' | 'MEMORY' | 'APPROVAL' | 'CONTAINER' | 'COGNIKUBE';

export interface Task {
  id: string;
  description: string;
  status: TaskStatus;
  type: TaskType;
  output?: string;
  params?: any;
}

export type ChatMessageSender = 'USER' | 'AGENT';

export interface ToolCall {
    toolName: string;
    args: { [key: string]: any };
}

export interface ToolResult {
    toolName: string;
    output: any;
}

export interface ChatMessage {
    id: string;
    sender: ChatMessageSender;
    message: string;
    timestamp: string;
    toolCall?: ToolCall;
    toolResult?: ToolResult;
    isThinking?: boolean;
}

export type AgentType = 'STANDARD' | 'FINANCE';
export type SecurityProfile = 'Restricted' | 'Web Researcher' | 'Developer';
export type AgentConnectionStatus = 'CONNECTED' | 'NEEDS_CONFIG' | 'DISCONNECTED';

export interface Agent {
  id: string;
  goal: string;
  status: AgentStatus;
  tasks: Task[];
  logs: string[];
  tools: string[];
  chatHistory: ChatMessage[];
  createdAt: string;
  type: AgentType;
  connectionStatus: AgentConnectionStatus;
  cpuAllocation: number;
  ramAllocation: number;
  storageAllocation: number;
  gpuAllocation: number;
  currentTaskIndex: number;
  financials: {
      revenue: number;
      profit: number;
      goal: number;
  };
}

// Configuration
export interface UserProfile {
  fullName: string;
  email: string;
}

export interface AIModelEndpoint {
  id:string;
  name: string;
  gatewayUrl: string;
  apiKey?: string;
  type: 'Local' | 'Cloud';
}

export interface CloudCredentials {
    aws?: { accessKeyId: string; secretAccessKey: string; region: string };
    gcp?: { serviceAccountJson: string; projectId: string };
    azure?: { subscriptionId: string; tenantId: string; clientId: string; clientSecret: string };
    digitalocean?: { token: string };
    upcloud?: { username: string; password: string };
    modal?: { tokenDates: string; tokenSecret: string };
    oracle?: { userOcid: string; tenancyOcid: string; fingerprint: string; privateKey: string };
}

export interface Tool {
    id: string;
    name:string;
    description: string;
    tags?: string[];
    input_schema?: object;
}

export interface CogniKubeService {
    key: string;
    name: string;
    description: string;
    category: 'LOGICAL' | 'ANALYTICAL' | 'CREATIVE' | 'DATA' | 'CODE';
    enabled: boolean;
}

// Cloud Provider Interface (Dynamic)
export interface CloudProvider {
    id: string;
    name: string;
    type: 'IaaS' | 'Serverless' | 'Platform' | 'API';
    status: 'connected' | 'disconnected' | 'connecting' | 'missing_creds';
    resources: number;
    region: string;
    color: string; // tailwind class or hex
    requiredFields: string[]; // Array of keys needed in CloudCredentials
    iconType?: 'cloud' | 'server' | 'database' | 'generic'; 
}

// Observability
export interface SystemResources {
    cpu: { used: number };
    ram: { used: number; total: number };
    storage: { used: number; total: number };
    latency: number[];
    concurrentRequests?: number;
    trafficWeight?: number;
}

// Developer Tools
export interface LogEntry {
  type: 'log' | 'warn' | 'error';
  message: string;
  timestamp: string;
}