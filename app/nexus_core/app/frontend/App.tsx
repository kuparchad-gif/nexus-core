/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/

import React, { useState, useEffect } from 'react';
import { MantineProvider, AppShell, Drawer } from '@mantine/core';
import { Notifications } from '@mantine/notifications';
import { useIdle } from '@mantine/hooks';

import Header from '@/components/Header';
import CreativeWorkspace from '@/components/workspaces/CreativeWorkspace';
import CodingWorkspace from '@/components/workspaces/CodingWorkspace';
import AdminWorkspace from '@/components/workspaces/AdminWorkspace';
import OperationsWorkspace from '@/components/workspaces/OperationsWorkspace';
import GameWorkspace from '@/components/workspaces/GameWorkspace';
import EnvironmentsWorkspace from '@/components/workspaces/EnvironmentsWorkspace';
import SettingsModal from '@/components/SettingsModal';
import LoginScreen from '@/components/LoginScreen';
import ToolsDrawer from '@/components/ToolsDrawer';
import { AetherealLogo } from '@/assets/images';

import { User } from './types';
import { createMantineTheme } from './services/settingsService';
import { authService } from './services/authService';
import { useAppStore } from './services/store';

const App: React.FC = () => {
  const [user, setUser] = useState<User | null>(authService.checkSession());
  const [isLocked, setIsLocked] = useState<boolean>(!user);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  
  const { 
    workspaceMode, 
    setWorkspaceMode,
    appMode,
    isToolsDrawerOpen, 
    toggleToolsDrawer,
    settings,
    setSettings
  } = useAppStore();

  const idle = useIdle(5 * 60 * 1000); // 5 minutes

  // Auto-lock on idle
  useEffect(() => {
    if (idle && user && !isLocked) {
      setIsLocked(true);
    }
  }, [idle, user, isLocked]);

  // Apply theme and settings from state
  const applyCssVariables = (s: typeof settings) => {
    const root = document.documentElement;
    root.style.setProperty('--font-family', s.ui.fontFamily);
    root.style.setProperty('--font-family-monospace', s.ui.fontFamilyMonospace);
    root.style.setProperty('--background-color', s.ui.backgroundColor);
    root.style.setProperty('--text-color', s.ui.textColor);
    root.style.setProperty('--border-color', s.ui.borderColor);
    root.style.setProperty('--primary-color', s.ui.primaryColor);
    root.style.setProperty('--accent-color', s.ui.accentColor);
    root.style.setProperty('--border-radius', `${s.ui.radius}rem`);
    root.style.setProperty('--shadow-intensity', s.ui.shadow.toString());
    root.style.setProperty('--glass-opacity', s.ui.opacity.toString());
    
    document.body.style.backgroundImage = s.ui.backgroundImageUrl ? `url('${s.ui.backgroundImageUrl}')` : 'none';
    document.body.style.backgroundSize = 'cover';
    document.body.style.backgroundPosition = 'center';
  };

  useEffect(() => {
    applyCssVariables(settings);
  }, [settings]);

  const handleLogin = (loggedInUser: User) => {
    setUser(loggedInUser);
    setIsLocked(false);
  };

  const handleLogout = () => {
    authService.logout();
    setUser(null);
    setIsLocked(true);
  };
  
  const handleUnlock = () => {
      setIsLocked(false);
  }

  const renderWorkspace = () => {
    switch (workspaceMode) {
      case 'creative': return <CreativeWorkspace />;
      case 'coding': return <CodingWorkspace />;
      case 'admin': return <AdminWorkspace />;
      case 'operations': return <OperationsWorkspace />;
      case 'game': return <GameWorkspace />;
      case 'environments': return <EnvironmentsWorkspace />;
      default: return <CreativeWorkspace />;
    }
  };

  if (!user || isLocked) {
    return (
      <MantineProvider theme={createMantineTheme(settings.ui)}>
         <LoginScreen 
            onLogin={handleLogin} 
            onUnlock={handleUnlock} 
            isLockScreen={isLocked && !!user} 
            user={user} 
         />
      </MantineProvider>
    );
  }

  return (
    <MantineProvider theme={createMantineTheme(settings.ui)}>
      <Notifications position="top-right" />
      <AppShell
        header={{ height: 80 }}
        padding="md"
        style={{
          width: '100vw',
          height: '100vh',
          overflow: 'hidden',
          display: 'flex',
          flexDirection: 'column',
          backgroundColor: 'transparent'
        }}
      >
        <AppShell.Header style={{ backgroundColor: 'transparent', borderBottom: 'none' }}>
           <Header 
            user={user}
            onOpenSettings={() => setIsSettingsOpen(true)}
            onLock={() => setIsLocked(true)}
            onLogout={handleLogout}
          />
        </AppShell.Header>

        <AppShell.Main style={{ backgroundColor: 'transparent', height: 'calc(100% - 80px)' }}>
            {renderWorkspace()}
        </AppShell.Main>
      </AppShell>
      
      <SettingsModal
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        onSettingsChange={setSettings}
      />

      <Drawer
        opened={isToolsDrawerOpen}
        onClose={toggleToolsDrawer}
        title="Tools & Gadgets"
        position="right"
        size="md"
        overlayProps={{ backgroundOpacity: 0.5, blur: 4 }}
      >
        <ToolsDrawer />
      </Drawer>
    </MantineProvider>
  );
};

export default App;