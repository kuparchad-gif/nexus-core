

import React from 'react';
import { Page } from '../../types';
import { HomeIcon, Cog6ToothIcon, WrenchScrewdriverIcon, ChatBubbleLeftRightIcon, UserCircleIcon, PencilSquareIcon, CodeBracketIcon, CubeTransparentIcon, CameraIcon, VideoCameraIcon, CpuChipIcon, CodeBracketSquareIcon } from '../icons';

interface SidebarProps {
  currentPage: Page;
  onNavigate: (page: Page) => void;
}

const NavItem: React.FC<{
  icon: React.ReactNode;
  label: string;
  isActive: boolean;
  onClick: () => void;
}> = ({ icon, label, isActive, onClick }) => {
  return (
    <li>
      <a
        href="#"
        onClick={(e) => {
          e.preventDefault();
          onClick();
        }}
        className={`flex items-center p-3 my-1 rounded-lg text-text-secondary transition-colors duration-200 ${
          isActive
            ? 'bg-brand-primary/10 text-brand-primary font-semibold'
            : 'hover:bg-slate-200 hover:text-text-primary'
        }`}
      >
        <span className="w-6 h-6">{icon}</span>
        <span className="ml-4 text-sm">{label}</span>
      </a>
    </li>
  );
};

const NavDivider: React.FC<{label: string}> = ({label}) => (
    <li className="pt-4 pb-2 px-3">
        <span className="text-xs font-bold text-text-secondary/60 uppercase tracking-wider">{label}</span>
    </li>
)

const Sidebar: React.FC<SidebarProps> = ({ currentPage, onNavigate }) => {
  const mainNavItems = [
    { id: 'dashboard', icon: <HomeIcon />, label: 'Dashboard' },
    { id: 'chat', icon: <ChatBubbleLeftRightIcon />, label: 'Chat Hub' },
    { id: 'tools', icon: <WrenchScrewdriverIcon />, label: 'Tool Registry' },
  ];

  const workspaceNavItems = [
    { id: 'nexus', icon: <CubeTransparentIcon />, label: 'Nexus Workspace' },
  ]
  
  const devNavItems = [
    { id: 'ide', icon: <CodeBracketIcon />, label: 'IDE' },
    { id: 'webDevSuite', icon: <CodeBracketSquareIcon />, label: 'Web Dev Suite' },
    { id: 'canvasEditor', icon: <PencilSquareIcon />, label: 'Canvas' },
  ];
  
  const creativeNavItems = [
      { id: 'avatarStudio', icon: <UserCircleIcon />, label: 'Avatar Studio' },
      { id: 'photoStudio', icon: <CameraIcon />, label: 'Photo Studio' },
      { id: 'videoEditor', icon: <VideoCameraIcon />, label: 'Video Editor' },
  ]
  
  const systemNavItems = [
      { id: 'systemTemplating', icon: <CpuChipIcon />, label: 'System Templating' },
  ]

  return (
    <aside className="w-64 bg-bg-light-card border-r border-border-color h-screen flex-col sticky top-0 hidden sm:flex">
      <div className="flex items-center space-x-3 p-4 h-16 border-b border-border-color">
        <img src="https://i.imgur.com/8p8YJkP.png" alt="Aethereal Icon" className="w-8 h-8"/>
        <h1 className="text-xl font-bold text-text-primary tracking-tight">
          AI Nexus
        </h1>
      </div>
      <nav className="flex-grow p-4 overflow-y-auto">
        <ul>
          <NavDivider label="Core" />
          {mainNavItems.map(item => (
             <NavItem
                key={item.id}
                icon={item.icon}
                label={item.label}
                isActive={currentPage === item.id}
                onClick={() => onNavigate(item.id as Page)}
            />
          ))}

          <NavDivider label="Workspace" />
          {workspaceNavItems.map(item => (
             <NavItem
                key={item.id}
                icon={item.icon}
                label={item.label}
                isActive={currentPage === item.id}
                onClick={() => onNavigate(item.id as Page)}
            />
          ))}
          
          <NavDivider label="Development" />
           {devNavItems.map(item => (
             <NavItem
                key={item.id}
                icon={item.icon}
                label={item.label}
                isActive={currentPage === item.id}
                onClick={() => onNavigate(item.id as Page)}
            />
          ))}

          <NavDivider label="Creative Suite" />
           {creativeNavItems.map(item => (
             <NavItem
                key={item.id}
                icon={item.icon}
                label={item.label}
                isActive={currentPage === item.id}
                onClick={() => onNavigate(item.id as Page)}
            />
          ))}
          
          <NavDivider label="System" />
           {systemNavItems.map(item => (
             <NavItem
                key={item.id}
                icon={item.icon}
                label={item.label}
                isActive={currentPage === item.id}
                onClick={() => onNavigate(item.id as Page)}
            />
          ))}
        </ul>
      </nav>
      <div className="p-4 border-t border-border-color">
        <ul>
          <NavItem
            icon={<Cog6ToothIcon />}
            label="Configuration"
            isActive={currentPage === 'config'}
            onClick={() => onNavigate('config')}
          />
        </ul>
      </div>
    </aside>
  );
};

export default Sidebar;