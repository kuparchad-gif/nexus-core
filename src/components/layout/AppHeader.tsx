

import React from 'react';
import { Page } from '../../types';
import { WifiIcon, SpinnerIcon } from '../icons';

interface AppHeaderProps {
    virenStatus: 'connecting' | 'connected' | 'disconnected';
    currentPage: Page;
}

const VirenConnectionIndicator: React.FC<{ status: 'connecting' | 'connected' | 'disconnected' }> = ({ status }) => {
    const statusMap = {
        connecting: { icon: <SpinnerIcon className="w-5 h-5 text-brand-secondary" />, text: 'Connecting...', color: 'text-text-secondary' },
        connected: { icon: <WifiIcon className="w-5 h-5 text-green-500" />, text: 'Agents Connected', color: 'text-text-secondary' },
        disconnected: { icon: <WifiIcon className="w-5 h-5 text-red-500" />, text: 'Agents Disconnected', color: 'text-red-500' },
    };

    const { icon, text, color } = statusMap[status];

    return (
        <div className="flex items-center space-x-2">
            {icon}
            <span className={`text-sm font-medium ${color}`}>{text}</span>
        </div>
    );
}

const AppHeader: React.FC<AppHeaderProps> = ({ virenStatus, currentPage }) => {
  return (
    <header className="glass-card border-b border-border-color sticky top-0 z-10">
      <div className="mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center space-x-3">
             <h2 className="text-lg font-semibold capitalize">{currentPage.replace(/([A-Z])/g, ' $1').trim()}</h2>
          </div>
          <div className="flex items-center space-x-4">
             <VirenConnectionIndicator status={virenStatus} />
          </div>
        </div>
      </div>
    </header>
  );
};

export default AppHeader;