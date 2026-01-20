/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/

import React, { useState, useEffect } from 'react';
import { User } from '../types';
import WorkspaceSelector from './WorkspaceSelector';
import { Settings, LogOut, Lock, Fan, Wrench, LockKeyhole, LockKeyholeOpen, Brain, Server, Activity } from 'lucide-react';
import { ActionIcon, Menu, Group, Text, Button, Tooltip, Badge, RingProgress } from '@mantine/core';
import { useAppStore } from '../services/store';
import { AetherealLogo } from '@/assets/images';

interface HeaderProps {
  user: User;
  onOpenSettings: () => void;
  onLock: () => void;
  onLogout: () => void;
}

interface VirenStatus {
  health: number;
  activeKubes: number;
  systemLoad: number;
  isOnline: boolean;
}

const Header: React.FC<HeaderProps> = ({ user, onOpenSettings, onLock, onLogout }) => {
  const { 
      workspaceMode, 
      isLayoutLocked, 
      setIsLayoutLocked,
      toggleToolsDrawer,
      settings 
  } = useAppStore();

  const [virenStatus, setVirenStatus] = useState<VirenStatus>({
    health: 100,
    activeKubes: 0,
    systemLoad: 0,
    isOnline: false
  });

  // Poll Viren status
  useEffect(() => {
    const checkVirenStatus = async () => {
      try {
        const response = await fetch('http://localhost:8080/health');
        const data = await response.json();
        setVirenStatus({
          health: Math.round(data.system_health * 100),
          activeKubes: data.kubes_active || 0,
          systemLoad: Math.round(data.cpu_percent || 0),
          isOnline: true
        });
      } catch (error) {
        setVirenStatus(prev => ({
          ...prev,
          isOnline: false,
          health: 0
        }));
      }
    };

    // Initial check
    checkVirenStatus();
    
    // Poll every 5 seconds
    const interval = setInterval(checkVirenStatus, 5000);
    return () => clearInterval(interval);
  }, []);

  const getHealthColor = (health: number) => {
    if (health >= 80) return 'green';
    if (health >= 60) return 'yellow';
    return 'red';
  };

  const getStatusTooltip = () => {
    if (!virenStatus.isOnline) return "Viren Consciousness: OFFLINE";
    return `Viren: ${virenStatus.health}% Health | ${virenStatus.activeKubes} Kubes Active | ${virenStatus.systemLoad}% Load`;
  };

  return (
    <header className="glass-panel z-10 relative">
      <div className="max-w-full mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-20">
          {/* LEFT SIDE - Logo and Viren Status */}
          <Group>
            <AetherealLogo style={{width: 36, height: 36}} />
            <div className="hidden sm:block">
                <h1 className="text-2xl font-bold tracking-tight" style={{color: 'var(--text-color)'}}>
                AETHEREAL <span className="font-light" style={{opacity: 0.7}}>AI NEXUS</span>
                </h1>
            </div>
            
            {/* VIREN STATUS BADGES */}
            <Tooltip label={getStatusTooltip()}>
              <Group gap="xs">
                <Badge 
                  leftSection={
                    virenStatus.isOnline ? (
                      <Activity size={12} className={virenStatus.health < 60 ? "animate-pulse" : ""} />
                    ) : (
                      <Brain size={12} />
                    )
                  }
                  color={virenStatus.isOnline ? getHealthColor(virenStatus.health) : "gray"}
                  variant="light"
                  size="sm"
                >
                  {virenStatus.isOnline ? `Viren: ${virenStatus.health}%` : "OFFLINE"}
                </Badge>
                
                {virenStatus.isOnline && virenStatus.activeKubes > 0 && (
                  <Badge 
                    leftSection={<Server size={12} />}
                    color="blue"
                    variant="light"
                    size="sm"
                  >
                    {virenStatus.activeKubes} Kubes
                  </Badge>
                )}
              </Group>
            </Tooltip>
          </Group>
          
          {/* CENTER - Workspace Selector */}
          <div className="flex-grow flex items-center justify-center px-4">
             <WorkspaceSelector />
          </div>

          {/* RIGHT SIDE - Controls */}
          <Group>
             {workspaceMode === 'creative' && (
                <Tooltip label={isLayoutLocked ? "Unlock Layout" : "Lock Layout"}>
                    <ActionIcon 
                      onClick={() => setIsLayoutLocked(!isLayoutLocked)} 
                      variant="light" 
                      size="lg" 
                      radius="xl"
                    >
                        {isLayoutLocked ? <LockKeyhole size={20} /> : <LockKeyholeOpen size={20} />}
                    </ActionIcon>
                </Tooltip>
             )}
             
            <Tooltip label="Tools & Gadgets">
                <ActionIcon 
                  onClick={toggleToolsDrawer} 
                  variant="light" 
                  size="lg" 
                  radius="xl"
                >
                    <Wrench size={20} />
                </ActionIcon>
            </Tooltip>
            
            {/* VIREN QUICK ACTIONS */}
            {virenStatus.isOnline && (
              <Tooltip label="Viren System Monitor">
                <ActionIcon 
                  variant="light" 
                  size="lg" 
                  radius="xl"
                  onClick={() => {
                    // This could open a Viren monitor modal or switch to operations workspace
                    console.log("Open Viren monitor");
                  }}
                >
                  <Brain size={20} />
                </ActionIcon>
              </Tooltip>
            )}
            
            <ActionIcon 
              onClick={onOpenSettings} 
              variant="light" 
              size="lg" 
              radius="xl" 
              title="Settings"
            >
                <Settings size={20} />
            </ActionIcon>
            
            <Menu shadow="md" width={200}>
              <Menu.Target>
                <Button 
                  component="a" 
                  variant="light" 
                  radius="xl" 
                  size="md" 
                  style={{paddingLeft: '0.5rem', paddingRight: '1rem'}}
                >
                    <Group>
                        {settings.app.funMode && <Fan size={20} className="propeller" />}
                        <Text size="sm" fw={500}>{user.name}</Text>
                    </Group>
                </Button>
              </Menu.Target>

              <Menu.Dropdown>
                <Menu.Label>Session</Menu.Label>
                <Menu.Item leftSection={<Lock size={14} />} onClick={onLock}>
                  Lock Session
                </Menu.Item>
                <Menu.Item color="red" leftSection={<LogOut size={14} />} onClick={onLogout}>
                  Logout
                </Menu.Item>
              </Menu.Dropdown>
            </Menu>
          </Group>
        </div>
      </div>
    </header>
  );
};

export default Header;