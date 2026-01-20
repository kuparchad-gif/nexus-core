

import React, { useState } from 'react';
import { CurrencyDollarIcon, SpinnerIcon, ShieldCheckIcon, GlobeAltIcon, CodeBracketSquareIcon } from './icons';
import { Tool, SecurityProfile, AgentType, AIModelEndpoint } from '../types';

interface AgentCreatorProps {
  onClose: () => void;
  onCreateAgent: (goal: string, tools: string[], profile: SecurityProfile, agentType: AgentType, financialGoal: number, modelEndpointId: string) => void;
  isCreating: boolean;
  tools: Tool[];
  modelEndpoints: AIModelEndpoint[];
}

const securityProfiles: { name: SecurityProfile, description: string, icon: React.ReactNode, allowedTags: string[] }[] = [
  { name: 'Restricted', description: 'No network or file access. Can only use internal memory.', icon: <ShieldCheckIcon className="w-5 h-5" />, allowedTags: ['memory'] },
  { name: 'Web Researcher', description: 'Can access the web to browse and read content.', icon: <GlobeAltIcon className="w-5 h-5" />, allowedTags: ['memory', 'web'] },
  { name: 'Developer', description: 'Full access to shell and other advanced tools.', icon: <CodeBracketSquareIcon className="w-5 h-5" />, allowedTags: ['memory', 'web', 'system', 'finance', 'code'] },
];

const MAX_GOAL_LENGTH = 1000;

const AgentCreator: React.FC<AgentCreatorProps> = ({ onClose, onCreateAgent, isCreating, tools, modelEndpoints }) => {
  const [goal, setGoal] = useState('');
  const [selectedTools, setSelectedTools] = useState<string[]>([]);
  const [selectedProfile, setSelectedProfile] = useState<SecurityProfile>('Developer');
  const [agentType, setAgentType] = useState<AgentType>('STANDARD');
  const [financialGoal, setFinancialGoal] = useState<number>(1500);
  const [selectedModelId, setSelectedModelId] = useState<string>(modelEndpoints[0]?.id || '');
  
  const activeProfile = securityProfiles.find(p => p.name === selectedProfile)!;
  const availableTools = tools.filter(tool => 
    tool.tags?.some(tag => activeProfile.allowedTags.includes(tag))
  );
  
  const handleToolToggle = (toolName: string) => {
    setSelectedTools(prev => 
      prev.includes(toolName) 
        ? prev.filter(t => t !== toolName)
        : [...prev, toolName]
    );
  };
  
  const handleProfileChange = (profileName: SecurityProfile) => {
    setSelectedProfile(profileName);
    const newActiveProfile = securityProfiles.find(p => p.name === profileName)!;
    const allowedToolsForNewProfile = tools.filter(tool =>
        tool.tags?.some(tag => newActiveProfile.allowedTags.includes(tag))
    ).map(t => t.name);

    setSelectedTools(prev => prev.filter(toolName => allowedToolsForNewProfile.includes(toolName)));
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (goal.trim() && selectedModelId && !isCreating) {
      onCreateAgent(goal.trim(), selectedTools, selectedProfile, agentType, agentType === 'FINANCE' ? financialGoal : 0, selectedModelId);
    }
  };

  return (
    <div 
      className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 animate-fade-in"
      onClick={onClose}
    >
      <div 
        className="bg-bg-light-card rounded-xl shadow-2xl w-full max-w-2xl mx-4 transform transition-all duration-300 flex flex-col glass-card"
        onClick={(e) => e.stopPropagation()}
        style={{ maxHeight: '90vh' }}
      >
        <div className="p-8 border-b border-border-color">
          <h2 className="text-2xl font-bold text-text-primary mb-2">New Autonomous Agent</h2>
          <p className="text-text-secondary">Define the primary objective. The Nexus will generate and execute a workflow to achieve it.</p>
        </div>
        <form onSubmit={handleSubmit} className="flex-grow overflow-y-auto">
          <div className="p-8 space-y-6">
            <div>
              <label className="block text-sm font-medium text-text-primary mb-2">Agent Type</label>
              <div className="flex rounded-lg bg-bg-light-bg p-1">
                <button type="button" onClick={() => setAgentType('STANDARD')} className={`w-full p-2 text-sm font-semibold rounded-md transition-colors ${agentType === 'STANDARD' ? 'bg-brand-primary text-white' : 'text-text-secondary hover:bg-slate-200'}`}>Standard Workflow</button>
                <button type="button" onClick={() => setAgentType('FINANCE')} className={`w-full p-2 text-sm font-semibold rounded-md transition-colors ${agentType === 'FINANCE' ? 'bg-brand-primary text-white' : 'text-text-secondary hover:bg-slate-200'}`}>Financial Goal</button>
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-text-primary mb-2">Agent's Goal</label>
              <div className="relative">
                <textarea
                  value={goal}
                  onChange={(e) => setGoal(e.target.value)}
                  placeholder={agentType === 'FINANCE' ? "e.g., 'Achieve $1,500 in revenue by selling T-shirts'" : "e.g., 'Research AI trends and write a report'"}
                  className="w-full h-24 bg-bg-light-bg border border-border-color rounded-lg p-4 pr-10 text-text-primary placeholder-text-secondary focus:ring-2 focus:ring-brand-primary focus:border-brand-primary transition resize-none"
                  disabled={isCreating}
                  required
                  maxLength={MAX_GOAL_LENGTH}
                />
              </div>
              <p className="text-right text-xs text-text-secondary mt-1">
                {goal.length} / {MAX_GOAL_LENGTH}
              </p>
            </div>
            
            <div>
                <label className="block text-sm font-medium text-text-primary mb-2">AI Model</label>
                <select 
                    value={selectedModelId}
                    onChange={(e) => setSelectedModelId(e.target.value)}
                    className="w-full bg-bg-light-bg border border-border-color rounded-lg p-3 text-text-primary focus:ring-2 focus:ring-brand-primary"
                    disabled={isCreating || modelEndpoints.length === 0}
                >
                    {modelEndpoints.length === 0 ? (
                        <option>No models configured</option>
                    ) : (
                        modelEndpoints.map(model => (
                            <option key={model.id} value={model.id}>{model.name}</option>
                        ))
                    )}
                </select>
            </div>

            {agentType === 'FINANCE' && (
              <div className="animate-fade-in">
                <label htmlFor="financial-goal" className="block text-sm font-medium text-text-primary mb-2">Revenue Goal</label>
                <div className="relative">
                  <div className="pointer-events-none absolute inset-y-0 left-0 flex items-center pl-3">
                    <CurrencyDollarIcon className="h-5 w-5 text-text-secondary" />
                  </div>
                  <input
                    type="number"
                    id="financial-goal"
                    value={financialGoal}
                    onChange={(e) => setFinancialGoal(parseFloat(e.target.value))}
                    className="w-full bg-bg-light-bg border border-border-color rounded-lg p-3 pl-10 text-text-primary placeholder-text-secondary focus:ring-2 focus:ring-brand-primary"
                    min="0"
                    step="100"
                  />
                </div>
              </div>
            )}

            <div>
              <h3 className="text-sm font-medium text-text-primary mb-2">Security Profile</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                {securityProfiles.map(profile => (
                  <label key={profile.name} className={`flex flex-col text-left p-3 rounded-lg border-2 transition-all duration-200 cursor-pointer ${selectedProfile === profile.name ? 'border-brand-primary bg-brand-primary/5' : 'border-border-color hover:border-slate-300 bg-bg-light-bg'}`}>
                    <div className="flex items-center space-x-2">
                       <input 
                          type="radio"
                          name="security-profile"
                          checked={selectedProfile === profile.name}
                          onChange={() => handleProfileChange(profile.name)}
                          className="h-4 w-4 border-gray-400 text-brand-primary focus:ring-brand-primary bg-bg-light-bg"
                        />
                        <span className="font-semibold text-text-primary">{profile.name}</span>
                        <span className="text-brand-primary">{profile.icon}</span>
                    </div>
                    <p className="text-xs text-text-secondary mt-2">{profile.description}</p>
                  </label>
                ))}
              </div>
            </div>

            {availableTools.length > 0 && (
              <div>
                <h3 className="text-sm font-medium text-text-primary mb-2">Enable Tools</h3>
                <p className="text-xs text-text-secondary mb-3">The following tools are available for the '{selectedProfile}' profile.</p>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                  {availableTools.map(tool => (
                    <label 
                      key={tool.id} 
                      className={`flex items-center space-x-3 p-3 rounded-lg border transition-all duration-200 cursor-pointer ${selectedTools.includes(tool.name) ? 'bg-brand-primary/10 border-brand-primary' : 'bg-bg-light-bg border-border-color hover:border-slate-300'}`}
                    >
                      <input 
                        type="checkbox"
                        checked={selectedTools.includes(tool.name)}
                        onChange={() => handleToolToggle(tool.name)}
                        className="h-4 w-4 rounded bg-bg-light-bg border-border-color text-brand-primary focus:ring-brand-primary"
                      />
                      <span className="text-sm font-medium text-text-primary truncate">{tool.name}</span>
                    </label>
                  ))}
                </div>
              </div>
            )}
          </div>
        </form>
        <div className="p-6 border-t border-border-color bg-bg-light-bg/50 rounded-b-xl flex justify-end items-center space-x-4">
            <button type="button" onClick={onClose} disabled={isCreating} className="text-text-secondary hover:text-text-primary transition">Cancel</button>
            <button 
              type="submit" 
              onClick={handleSubmit}
              disabled={!goal.trim() || isCreating || !selectedModelId}
              className="w-36 h-11 bg-brand-primary text-white rounded-lg flex items-center justify-center hover:bg-brand-primary/90 disabled:bg-slate-400 disabled:cursor-not-allowed transition-colors font-semibold"
            >
              {isCreating ? <SpinnerIcon className="w-5 h-5" /> : 'Create Agent'}
            </button>
        </div>
      </div>
    </div>
  );
};

export default AgentCreator;