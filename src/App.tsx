
import React, { useState, useEffect, useCallback, ReactNode, useRef } from 'react';
import AgentDashboard from './components/pages/AgentDashboard';
import AgentDetailView from './components/pages/AgentDetailView';
import ChatView from './components/pages/ChatView';
import ToolRegistryView from './components/pages/ToolRegistryView';
import ConfigurationView from './components/pages/ConfigurationView';
import SetupView from './components/pages/SetupView';
import IDEView from './components/pages/IDEView';
import CanvasEditorView from './components/pages/CanvasEditorView';
import AvatarStudioView from './components/pages/AvatarStudioView';
import PhotoStudioView from './components/pages/PhotoStudioView';
import VideoEditorView from './components/pages/VideoEditorView';
import WebDevSuiteView from './components/pages/WebDevSuiteView';
import SystemTemplatingView from './components/pages/SystemTemplatingView';
import NexusWorkspaceView from './components/pages/NexusWorkspaceView';
import CloudCommandView from './components/pages/CloudCommandView';
import PersonalizationView from './components/pages/PersonalizationView';
import DeveloperTools from './components/DeveloperTools';
import AgentCreator from './components/AgentCreator';

import * as virenService from './services/virenService';
import { RealNexusAdapter } from './services/RealNexusAdapter'; 
import { Page, Agent, Tool, AIModelEndpoint, UserProfile, LogEntry, ChatMessage, TaskStatus, AgentStatus, CloudCredentials, ThemeConfig, IconPosition, ContextMenuState } from './types';
import { v4 as uuidv4 } from 'uuid';
import { 
  HomeIcon, ChatBubbleLeftRightIcon, WrenchScrewdriverIcon, 
  CodeBracketIcon, PencilSquareIcon, UserCircleIcon, CameraIcon, 
  VideoCameraIcon, CodeBracketSquareIcon, CpuChipIcon, 
  CubeTransparentIcon, MinusIcon, SquareIcon, XMarkIcon, AILogoIcon, Cog6ToothIcon, CloudIcon, PaintBrushIcon, CommandLineIcon, PlusIcon
} from './components/icons';

// --- Types for Window Management ---
interface WindowInstance {
  id: string; 
  appId: string; 
  title: string;
  component: ReactNode;
  isMinimized: boolean;
  zIndex: number;
  position: { x: number, y: number };
  size: { width: string, height: string };
}

interface AppDefinition {
  id: string;
  name: string;
  icon: ReactNode;
  component: (props: any) => ReactNode;
  defaultTitle: string;
}

// --- Window Frame Component ---
const WindowFrame: React.FC<{
  window: WindowInstance;
  isActive: boolean;
  onClose: (id: string) => void;
  onMinimize: (id: string) => void;
  onFocus: (id: string) => void;
  onMove: (id: string, x: number, y: number) => void;
}> = ({ window, isActive, onClose, onMinimize, onFocus, onMove }) => {
  const [isDragging, setIsDragging] = useState(false);
  const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 });

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!isDragging) return;
      onMove(window.id, e.clientX - dragOffset.x, e.clientY - dragOffset.y);
    };

    const handleMouseUp = () => {
      setIsDragging(false);
    };

    if (isDragging) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
    }
    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isDragging, dragOffset, onMove, window.id]);

  const handleMouseDown = (e: React.MouseEvent) => {
    onFocus(window.id);
    if ((e.target as HTMLElement).closest('.window-controls')) return;
    setIsDragging(true);
    setDragOffset({
      x: e.clientX - window.position.x,
      y: e.clientY - window.position.y
    });
  };

  if (window.isMinimized) return null;

  return (
    <div
      className={`os-window ${isActive ? 'active' : ''}`}
      style={{
        zIndex: window.zIndex,
        top: window.position.y,
        left: window.position.x,
        width: window.size.width,
        height: window.size.height,
        transition: isDragging ? 'none' : 'width 0.3s, height 0.3s, opacity 0.2s, transform 0.2s',
      }}
      onMouseDown={() => onFocus(window.id)}
    >
      <div className="window-header" onMouseDown={handleMouseDown}>
        <div className="flex items-center gap-2 overflow-hidden">
            <span className="window-title truncate">{window.title}</span>
        </div>
        <div className="window-controls">
          <button className="window-control-btn btn-min" onClick={(e) => { e.stopPropagation(); onMinimize(window.id); }} title="Minimize">
            <MinusIcon className="w-3 h-3" />
          </button>
          <button className="window-control-btn btn-max" title="Maximize">
             <SquareIcon className="w-2.5 h-2.5" />
          </button>
          <button className="window-control-btn btn-close" onClick={(e) => { e.stopPropagation(); onClose(window.id); }} title="Close">
            <XMarkIcon className="w-4 h-4" />
          </button>
        </div>
      </div>
      <div className="window-content">
        {window.component}
      </div>
    </div>
  );
};

// --- Context Menu Component ---
const ContextMenu: React.FC<{
    state: ContextMenuState;
    onClose: () => void;
    onAction: (action: string, targetId?: string) => void;
}> = ({ state, onClose, onAction }) => {
    if (!state.visible) return null;

    return (
        <div 
            className="context-menu" 
            style={{ top: state.y, left: state.x }}
            onClick={(e) => e.stopPropagation()}
        >
            {state.type === 'desktop' && (
                <>
                    <div className="context-menu-item" onClick={() => onAction('terminal')}>
                        <CommandLineIcon className="w-4 h-4 text-slate-500" /> Open Terminal
                    </div>
                    <div className="context-menu-item" onClick={() => onAction('personalize')}>
                        <PaintBrushIcon className="w-4 h-4 text-slate-500" /> Personalize
                    </div>
                    <div className="context-menu-separator"></div>
                    <div className="context-menu-item" onClick={() => onAction('refresh')}>
                        Refresh
                    </div>
                    <div className="context-menu-item" onClick={() => onAction('new_folder')}>
                        <PlusIcon className="w-4 h-4 text-slate-500" /> New Folder
                    </div>
                </>
            )}
            {state.type === 'icon' && (
                <>
                    <div className="context-menu-item" onClick={() => onAction('open', state.targetId)}>
                        <SquareIcon className="w-4 h-4 text-slate-500" /> Open
                    </div>
                    <div className="context-menu-separator"></div>
                    <div className="context-menu-item" onClick={() => onAction('properties', state.targetId)}>
                        Properties
                    </div>
                </>
            )}
        </div>
    )
}

// --- Taskbar Component ---
const Taskbar: React.FC<{
  windows: WindowInstance[];
  activeWindowId: string | null;
  onWindowClick: (id: string) => void;
  onStartClick: () => void;
  isStartOpen: boolean;
  apps: AppDefinition[];
}> = ({ windows, activeWindowId, onWindowClick, onStartClick, isStartOpen, apps }) => {
    const getAppIcon = (appId: string) => {
        const app = apps.find(a => a.id === appId);
        return app ? app.icon : <CubeTransparentIcon className="w-5 h-5" />;
    };

  return (
    <div className="os-taskbar">
      <div className={`start-btn ${isStartOpen ? 'active' : ''}`} onClick={onStartClick} title="Start">
        <AILogoIcon className="w-5 h-5 text-white" />
      </div>
      <div className="flex items-center flex-1 h-full pl-2 gap-1">
          {windows.map(win => (
            <div
              key={win.id}
              className={`taskbar-app ${activeWindowId === win.id && !win.isMinimized ? 'active' : ''}`}
              onClick={() => onWindowClick(win.id)}
            >
              <div className="w-5 h-5 flex items-center justify-center">
                  {getAppIcon(win.appId)}
              </div>
              <span className="truncate hidden sm:block">{win.title}</span>
            </div>
          ))}
      </div>
      <div className="flex items-center px-4 gap-3 text-text-secondary text-xs font-medium cursor-default">
         <span>ENG</span>
         <span className="hover:bg-white/10 px-2 py-1 rounded transition">{new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}</span>
      </div>
    </div>
  );
};

// --- Start Menu Component ---
const StartMenu: React.FC<{
  isOpen: boolean;
  onClose: () => void;
  apps: AppDefinition[];
  onLaunch: (appId: string) => void;
  onLogout: () => void;
}> = ({ isOpen, onClose, apps, onLaunch, onLogout }) => {
  if (!isOpen) return null;

  return (
    <div className="start-menu" onClick={(e) => e.stopPropagation()}>
        <div className="flex h-full">
            <div className="w-16 bg-slate-100/50 border-r border-white/50 display flex flex-col items-center py-4">
                <div className="mt-auto mb-2 w-full flex flex-col items-center gap-4">
                     <button onClick={onLogout} className="p-2 rounded hover:bg-white/20 text-slate-500 hover:text-brand-primary transition" title="Sign Out">
                        <Cog6ToothIcon className="w-5 h-5" />
                    </button>
                    <div className="w-8 h-8 rounded-full bg-brand-primary flex items-center justify-center text-white font-bold text-xs shadow-lg">AI</div>
                </div>
            </div>
            <div className="flex-1 p-5 overflow-y-auto">
                <h3 className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-3 px-2">Applications</h3>
                <div className="space-y-1">
                    {apps.map(app => (
                        <div 
                            key={app.id} 
                            onClick={() => { onLaunch(app.id); onClose(); }}
                            className="flex items-center gap-3 p-2 rounded-lg hover:bg-white/60 cursor-pointer transition-colors group"
                        >
                            <div className="w-8 h-8 rounded bg-white/80 shadow-sm flex items-center justify-center text-brand-secondary group-hover:text-brand-primary transition-colors">
                                {React.cloneElement(app.icon as React.ReactElement, { className: 'w-5 h-5' })}
                            </div>
                            <div className="flex flex-col">
                                <span className="text-sm font-medium text-slate-800">{app.name}</span>
                                <span className="text-xs text-slate-500">Application</span>
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    </div>
  );
};

// --- Desktop Icon Component ---
const DesktopIcon: React.FC<{ 
    app: AppDefinition; 
    position: IconPosition;
    onMouseDown: (e: React.MouseEvent, id: string) => void;
    onLaunch: (id: string) => void; 
    onContextMenu: (e: React.MouseEvent, id: string) => void;
    iconStyle: string 
}> = ({ app, position, onMouseDown, onLaunch, onContextMenu, iconStyle }) => (
  <div 
    className={`desktop-icon group icon-style-${iconStyle}`} 
    style={{ top: position.y, left: position.x }}
    onMouseDown={(e) => onMouseDown(e, app.id)}
    onDoubleClick={() => onLaunch(app.id)}
    onContextMenu={(e) => onContextMenu(e, app.id)}
  >
    <div className="desktop-icon-img-wrapper pointer-events-none">
        {React.cloneElement(app.icon as React.ReactElement, { className: `w-12 h-12 drop-shadow-md ${app.icon.props.className}` })}
    </div>
    <div className="desktop-icon-label pointer-events-none">{app.name}</div>
  </div>
);


const App: React.FC = () => {
    // --- Default App Definitions ---
    // MODIFY ICONS HERE: Change the 'icon' property to any SVG component or adjust the 'className' for color.
    const DEFAULT_APPS: AppDefinition[] = [
        { id: 'portal', name: 'OS Portal', icon: <CubeTransparentIcon className="text-violet-500"/>, defaultTitle: 'Oz OS Portal', component: (props) => <NexusWorkspaceView addLog={addLog} {...props} /> },
        { id: 'cloud', name: 'Cloud Command', icon: <CloudIcon className="text-sky-500"/>, defaultTitle: 'Cloud Infrastructure', component: (props) => <CloudCommandView cloudCredentials={cloudCredentials} addLog={addLog} {...props} /> },
        { id: 'dashboard', name: 'Agent Dashboard', icon: <HomeIcon className="text-emerald-500"/>, defaultTitle: 'Nexus Dashboard', component: (props) => <AgentDashboard agents={agents} onSelectAgent={(id) => { setSelectedAgentId(id); launchApp('agent_detail', { agentId: id }); }} onOpenCreator={() => setIsCreatorOpen(true)} {...props} /> },
        { id: 'chat', name: 'Chat Hub', icon: <ChatBubbleLeftRightIcon className="text-pink-500"/>, defaultTitle: 'Nexus Chat Hub', component: (props) => <ChatView agents={agents} modelEndpoints={modelEndpoints} {...props} /> },
        { id: 'tools', name: 'Tool Registry', icon: <WrenchScrewdriverIcon className="text-slate-500"/>, defaultTitle: 'Tool Registry', component: (props) => <ToolRegistryView tools={tools} setTools={setTools} {...props} /> },
        { id: 'ide', name: 'Engineering Suite', icon: <CodeBracketIcon className="text-blue-600"/>, defaultTitle: 'IDE - Engineering Suite', component: (props) => <IDEView {...props} /> },
        { id: 'webDevSuite', name: 'Web Dev Suite', icon: <CodeBracketSquareIcon className="text-indigo-500"/>, defaultTitle: 'Web Development Suite', component: (props) => <WebDevSuiteView {...props} /> },
        { id: 'canvasEditor', name: 'Canvas', icon: <PencilSquareIcon className="text-orange-500"/>, defaultTitle: 'Canvas Editor', component: (props) => <CanvasEditorView {...props} /> },
        { id: 'avatarStudio', name: 'Avatar Studio', icon: <UserCircleIcon className="text-teal-500"/>, defaultTitle: 'Avatar Studio', component: (props) => <AvatarStudioView {...props} /> },
        { id: 'photoStudio', name: 'Photo Studio', icon: <CameraIcon className="text-rose-500"/>, defaultTitle: 'Photo Studio', component: (props) => <PhotoStudioView {...props} /> },
        { id: 'videoEditor', name: 'Video Editor', icon: <VideoCameraIcon className="text-red-500"/>, defaultTitle: 'Video Editor', component: (props) => <VideoEditorView {...props} /> },
        { id: 'systemTemplating', name: 'System Templating', icon: <CpuChipIcon className="text-cyan-500"/>, defaultTitle: 'System Templating', component: (props) => <SystemTemplatingView {...props} /> },
        { id: 'personalization', name: 'Personalization', icon: <PaintBrushIcon className="text-fuchsia-500"/>, defaultTitle: 'Appearance', component: (props) => <PersonalizationView theme={theme} setTheme={setTheme} {...props} /> },
        { id: 'config', name: 'Configuration', icon: <Cog6ToothIcon className="text-gray-600"/>, defaultTitle: 'System Configuration', component: (props) => <ConfigurationView profile={profile} onSaveProfile={setProfile} modelEndpoints={modelEndpoints} setAiModelEndpoints={setModelEndpoints} cloudCredentials={cloudCredentials} setCloudCredentials={setCloudCredentials} {...props} /> },
    ];

    const [installedApps, setInstalledApps] = useState<AppDefinition[]>(DEFAULT_APPS);

    // --- Application State ---
    const [selectedAgentId, setSelectedAgentId] = useState<string | null>(null);
    const [agents, setAgents] = useState<Agent[]>([]);
    const [tools, setTools] = useState<Tool[]>([]);
    const [profile, setProfile] = useState<UserProfile>({ fullName: '', email: '' });
    const [modelEndpoints, setModelEndpoints] = useState<AIModelEndpoint[]>([
        { id: 'qwen-default', name: 'Qwen 2.5 72B (Thinking)', gatewayUrl: 'http://localhost:11434/v1', type: 'Local' },
        { id: 'deepseek-coder', name: 'DeepSeek Coder V2', gatewayUrl: 'http://localhost:11434/v1', type: 'Local', apiKey: '' }
    ]);
    const [cloudCredentials, setCloudCredentials] = useState<CloudCredentials>({});
    const [virenStatus, setVirenStatus] = useState<'connecting' | 'connected' | 'disconnected'>('connecting');
    const [isCreatorOpen, setIsCreatorOpen] = useState(false);
    const [isCreatingAgent, setIsCreatingAgent] = useState(false);
    const [isAgentResponding, setIsAgentResponding] = useState(false);
    const [logs, setLogs] = useState<LogEntry[]>([]);
    const [isDevToolsOpen, setDevToolsOpen] = useState(false);
    const [isInitialSetupDone, setIsInitialSetupDone] = useState(() => {
        return localStorage.getItem('nexusSetupDone') === 'true';
    });

    // --- Theming State ---
    const [theme, setTheme] = useState<ThemeConfig>(() => {
        const saved = localStorage.getItem('nexusTheme');
        return saved ? JSON.parse(saved) : {
            wallpaper: 'radial-gradient(circle at 10% 10%, rgba(224, 231, 255, 0.8) 0%, transparent 40%), radial-gradient(circle at 90% 90%, rgba(237, 233, 254, 0.8) 0%, transparent 40%), linear-gradient(135deg, #f8fafc 0%, #e2e8f0 50%, #f1f5f9 100%)',
            accentColor: '#a78bfa',
            glassOpacity: 0.8, // Increased opacity default
            iconStyle: 'glass-dark'
        };
    });

    // --- Desktop Icon State ---
    const [iconLayout, setIconLayout] = useState<Record<string, IconPosition>>(() => {
        const layout: Record<string, IconPosition> = {};
        DEFAULT_APPS.forEach((app, index) => {
            const col = Math.floor(index / 6);
            const row = index % 6;
            layout[app.id] = { x: 16 + (col * 100), y: 16 + (row * 110) };
        });
        return layout;
    });
    const [draggingIconId, setDraggingIconId] = useState<string | null>(null);
    const [iconDragOffset, setIconDragOffset] = useState({ x: 0, y: 0 });

    // --- Context Menu State ---
    const [contextMenu, setContextMenu] = useState<ContextMenuState>({
        visible: false, x: 0, y: 0, type: 'desktop'
    });

    // --- Neural Link: Fetch System Manifest from Oz ---
    useEffect(() => {
        const syncWithOz = async () => {
            const manifest = await RealNexusAdapter.getSystemManifest();
            if (manifest && manifest.apps) {
                // Logic to merge/update apps based on manifest (Simplified for now)
                addLog('log', 'Oz System Manifest Synced.');
            }
        };
        syncWithOz();
    }, []);

    // Apply Theme & Persist
    useEffect(() => {
        document.documentElement.style.setProperty('--brand-primary', theme.accentColor);
        document.documentElement.style.setProperty('--glass-bg', `rgba(255, 255, 255, ${theme.glassOpacity})`);
        document.body.style.background = theme.wallpaper.startsWith('url') ? theme.wallpaper : theme.wallpaper;
        if (theme.wallpaper.startsWith('url')) {
             document.body.style.backgroundSize = 'cover';
             document.body.style.backgroundPosition = 'center';
        }
        
        // Apply Unified Icon Theme if selected
        if (theme.iconStyle === 'unified') {
            document.body.classList.add('theme-icons-unified');
        } else {
            document.body.classList.remove('theme-icons-unified');
        }

        localStorage.setItem('nexusTheme', JSON.stringify(theme));
    }, [theme]);

    // --- Window Manager State ---
    const [windows, setWindows] = useState<WindowInstance[]>([]);
    const [activeWindowId, setActiveWindowId] = useState<string | null>(null);
    const [isStartMenuOpen, setIsStartMenuOpen] = useState(false);
    const [nextZIndex, setNextZIndex] = useState(10);

    const addLog = useCallback((type: LogEntry['type'], ...args: any[]) => {
      const message = args.map(arg => typeof arg === 'object' ? JSON.stringify(arg) : String(arg)).join(' ');
      setLogs(prev => [...prev, { type, message, timestamp: new Date().toLocaleTimeString() }]);
    }, []);

    // Initial Data Fetch
    useEffect(() => {
        RealNexusAdapter.checkHealth().then(isAlive => {
            if (isAlive) {
                addLog('log', 'Real Oz Backend detected. Live Mode Active.');
                setVirenStatus('connected');
            } else {
                addLog('warn', 'Oz Backend not found. Running in Simulation Mode.');
                virenService.getAgents().then(setAgents).catch(err => addLog('error', 'Failed to fetch agents', err));
                setTimeout(() => setVirenStatus('connected'), 1500);
            }
        });
        virenService.getTools().then(setTools).catch(err => addLog('error', 'Failed to fetch tools', err));
    }, [addLog]);

    // --- Icon Dragging Logic ---
    const handleIconMouseDown = (e: React.MouseEvent, id: string) => {
        if (e.button !== 0) return; // Only left click drags
        e.preventDefault();
        const pos = iconLayout[id] || { x: 0, y: 0 };
        setDraggingIconId(id);
        setIconDragOffset({
            x: e.clientX - pos.x,
            y: e.clientY - pos.y
        });
        setContextMenu({ ...contextMenu, visible: false });
    };

    useEffect(() => {
        const handleMouseMove = (e: MouseEvent) => {
            if (draggingIconId) {
                setIconLayout(prev => ({
                    ...prev,
                    [draggingIconId]: {
                        x: e.clientX - iconDragOffset.x,
                        y: e.clientY - iconDragOffset.y
                    }
                }));
            }
        };
        const handleMouseUp = () => {
            setDraggingIconId(null);
        };

        if (draggingIconId) {
            window.addEventListener('mousemove', handleMouseMove);
            window.addEventListener('mouseup', handleMouseUp);
        }
        return () => {
            window.removeEventListener('mousemove', handleMouseMove);
            window.removeEventListener('mouseup', handleMouseUp);
        };
    }, [draggingIconId, iconDragOffset]);

    // --- Context Menu Logic ---
    const handleContextMenu = (e: React.MouseEvent, type: 'desktop' | 'icon', targetId?: string) => {
        e.preventDefault();
        setContextMenu({
            visible: true,
            x: e.clientX,
            y: e.clientY,
            type,
            targetId
        });
    };

    const handleContextMenuAction = (action: string, targetId?: string) => {
        setContextMenu({ ...contextMenu, visible: false });
        switch(action) {
            case 'open':
                if(targetId) launchApp(targetId);
                break;
            case 'refresh':
                setVirenStatus('connecting');
                setTimeout(() => setVirenStatus('connected'), 500);
                break;
            case 'personalize':
                launchApp('personalization');
                break;
            case 'terminal':
                launchApp('ide'); 
                break;
            case 'properties':
                alert(`Properties for ${targetId} (Mock)`);
                break;
            case 'new_folder':
                alert("New Folder creation not yet implemented in filesystem.");
                break;
        }
    };

    const handleGlobalClick = () => {
        if (contextMenu.visible) setContextMenu({ ...contextMenu, visible: false });
        if (isStartMenuOpen) setIsStartMenuOpen(false);
    };

    // --- Agent Logic ---
    const handleCreateAgent = async (goal: string, enabledTools: string[], profile: string, agentType: any, financialGoal: number, modelId: string) => {
        setIsCreatingAgent(true);
        try {
            const newAgent = await virenService.createAgent(goal, enabledTools, agentType, financialGoal);
            setAgents(prev => [newAgent, ...prev]);
            addLog('log', `New agent created with goal: ${goal}`);
            setIsCreatorOpen(false);
        } catch (error) {
            addLog('error', 'Failed to create agent', error);
        } finally {
            setIsCreatingAgent(false);
        }
    };

    const handleSendMessage = (agentId: string, message: string) => {
        setIsAgentResponding(true);
        const newMessage: ChatMessage = { id: uuidv4(), sender: 'USER', message, timestamp: new Date().toISOString() };
        setAgents(prev => prev.map(agent => agent.id === agentId ? { ...agent, chatHistory: [...agent.chatHistory, newMessage] } : agent));
        setTimeout(() => {
             const responseMessage: ChatMessage = { id: uuidv4(), sender: 'AGENT', message: `Acknowledged: "${message}". I will proceed.`, timestamp: new Date().toISOString() };
             setAgents(prev => prev.map(agent => agent.id === agentId ? { ...agent, chatHistory: [...agent.chatHistory, responseMessage], logs: [...agent.logs, `User interaction: ${message}`] } : agent));
            setIsAgentResponding(false);
        }, 1500);
    };

    const handleUserApproval = (agentId: string, approved: boolean) => {
        setAgents(prev => prev.map(agent => {
            if (agent.id === agentId) {
                const currentTask = agent.tasks[agent.currentTaskIndex];
                const taskDescription = currentTask ? currentTask.description : 'Unknown Task';
                const newLogs = [...agent.logs, `User ${approved ? 'approved' : 'denied'} task: ${taskDescription}`];
                if (approved) {
                    const newTasks = agent.tasks.map((task, index) => index === agent.currentTaskIndex ? { ...task, status: TaskStatus.COMPLETED, output: "User approved." } : task);
                    return { ...agent, status: AgentStatus.RUNNING, tasks: newTasks, logs: newLogs, currentTaskIndex: agent.currentTaskIndex + 1 };
                } else {
                     const newTasks = agent.tasks.map((task, index) => index === agent.currentTaskIndex ? { ...task, status: TaskStatus.SKIPPED } : task);
                    return { ...agent, status: AgentStatus.IDLE, tasks: newTasks, logs: newLogs };
                }
            }
            return agent;
        }));
    };

    const getComponentForApp = (appId: string, extraProps: any = {}) => {
        if (appId === 'agent_detail') {
             const agent = agents.find(a => a.id === extraProps.agentId);
             if (!agent) return <div>Agent not found</div>;
             return <AgentDetailView agent={agent} onBack={() => {}} onSendMessage={handleSendMessage} onUserApproval={handleUserApproval} isAgentResponding={isAgentResponding} />;
        }
        const app = installedApps.find(a => a.id === appId);
        return app ? app.component(extraProps) : null;
    }

    // --- Window Management Logic ---
    const launchApp = (appId: string, extraProps: any = {}) => {
        const existingWindow = windows.find(w => w.appId === appId && (appId !== 'agent_detail' || (w.id.includes(extraProps.agentId))));
        
        if (existingWindow) {
            if (existingWindow.isMinimized) {
                setWindows(prev => prev.map(w => w.id === existingWindow.id ? { ...w, isMinimized: false, zIndex: nextZIndex } : w));
            } else {
                 setWindows(prev => prev.map(w => w.id === existingWindow.id ? { ...w, zIndex: nextZIndex } : w));
            }
            setActiveWindowId(existingWindow.id);
            setNextZIndex(prev => prev + 1);
            return;
        }

        const appDef = installedApps.find(a => a.id === appId);
        const title = appId === 'agent_detail' 
            ? `Agent: ${agents.find(a => a.id === extraProps.agentId)?.goal.substring(0, 20)}...` 
            : (appDef?.defaultTitle || 'Application');
        
        const offset = (windows.length % 8) * 30;
        const widthVal = window.innerWidth > 800 ? 900 : window.innerWidth - 40;
        const heightVal = window.innerHeight > 600 ? 600 : window.innerHeight - 100;

        const newWindow: WindowInstance = {
            id: appId === 'agent_detail' ? `detail-${extraProps.agentId}` : `win-${appId}-${Date.now()}`,
            appId,
            title,
            component: getComponentForApp(appId, extraProps),
            isMinimized: false,
            zIndex: nextZIndex,
            position: { x: 120 + offset, y: 40 + offset },
            size: { width: `${widthVal}px`, height: `${heightVal}px` }
        };

        setWindows(prev => [...prev, newWindow]);
        setActiveWindowId(newWindow.id);
        setNextZIndex(prev => prev + 1);
    };

    const closeWindow = (id: string) => {
        setWindows(prev => prev.filter(w => w.id !== id));
        if (activeWindowId === id) setActiveWindowId(null);
    };

    const minimizeWindow = (id: string) => {
        setWindows(prev => prev.map(w => w.id === id ? { ...w, isMinimized: true } : w));
        if (activeWindowId === id) setActiveWindowId(null);
    };

    const focusWindow = (id: string) => {
        const win = windows.find(w => w.id === id);
        if (!win) return;
        if (win.isMinimized) {
            setWindows(prev => prev.map(w => w.id === id ? { ...w, isMinimized: false, zIndex: nextZIndex } : w));
        } else {
            setWindows(prev => prev.map(w => w.id === id ? { ...w, zIndex: nextZIndex } : w));
        }
        setActiveWindowId(id);
        setNextZIndex(prev => prev + 1);
    };

    const moveWindow = (id: string, x: number, y: number) => {
        setWindows(prev => prev.map(w => w.id === id ? { ...w, position: { x, y } } : w));
    }

    if (!isInitialSetupDone) {
        return <SetupView 
            onConfigure={() => { setIsInitialSetupDone(true); launchApp('config'); }} 
            onBoot={() => { setIsInitialSetupDone(true); launchApp('dashboard'); }} 
            onTerminal={() => { setIsInitialSetupDone(true); launchApp('ide'); }}
        />;
    }

    return (
        <>
            <div className="desktop-area" onClick={handleGlobalClick} onContextMenu={(e) => handleContextMenu(e, 'desktop')}>
                {installedApps.map(app => (
                    <DesktopIcon 
                        key={app.id} 
                        app={app} 
                        position={iconLayout[app.id] || {x:0, y:0}}
                        onMouseDown={handleIconMouseDown}
                        onLaunch={launchApp} 
                        onContextMenu={(e, id) => handleContextMenu(e, 'icon', id)}
                        iconStyle={theme.iconStyle} 
                    />
                ))}

                {windows.map(win => (
                    <WindowFrame
                        key={win.id}
                        window={win}
                        isActive={activeWindowId === win.id}
                        onClose={closeWindow}
                        onMinimize={minimizeWindow}
                        onFocus={focusWindow}
                        onMove={moveWindow}
                    />
                ))}
                
                <ContextMenu state={contextMenu} onClose={() => setContextMenu({ ...contextMenu, visible: false })} onAction={handleContextMenuAction} />
            </div>

            <StartMenu 
                isOpen={isStartMenuOpen} 
                onClose={() => setIsStartMenuOpen(false)} 
                apps={installedApps} 
                onLaunch={launchApp}
                onLogout={() => { setIsInitialSetupDone(false); setWindows([]); }} 
            />

            <Taskbar 
                windows={windows} 
                activeWindowId={activeWindowId} 
                onWindowClick={(id) => {
                    const win = windows.find(w => w.id === id);
                    if (win?.id === activeWindowId && !win.isMinimized) {
                        minimizeWindow(id);
                    } else {
                        focusWindow(id);
                    }
                }}
                onStartClick={() => setIsStartMenuOpen(!isStartMenuOpen)}
                isStartOpen={isStartMenuOpen}
                apps={installedApps}
            />

            {isCreatorOpen && (
                <AgentCreator 
                    onClose={() => setIsCreatorOpen(false)}
                    onCreateAgent={handleCreateAgent}
                    isCreating={isCreatingAgent}
                    tools={tools}
                    modelEndpoints={modelEndpoints}
                />
            )}
            <DeveloperTools isOpen={isDevToolsOpen} onToggle={() => setDevToolsOpen(!isDevToolsOpen)} logs={logs} />
        </>
    );
};

export default App;
