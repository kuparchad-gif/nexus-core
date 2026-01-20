/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/

import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { Output, ChatMessage, AppMode, WorkspaceMode, Settings } from '../types';
import { getSettings, saveSettings as saveSettingsToStorage } from './settingsService';

// State for generated assets and chat
interface GenerationState {
  outputs: Output[];
  chatHistory: ChatMessage[];
  selectedOutput: Output | null;
  isLoading: boolean;
  loadingMessage: string;
}

interface GenerationActions {
  addOutput: (output: Output) => void;
  addChatMessage: (message: ChatMessage) => void;
  setSelectedOutput: (output: Output | null) => void;
  setLoading: (loading: boolean, message?: string) => void;
  updateOutputCode: (id: number, newCode: string) => void;
  deleteOutput: (id: number) => void;
  clearSession: () => void;
}

export const useGenerationStore = create<GenerationState & GenerationActions>((set) => ({
  outputs: [],
  chatHistory: [],
  selectedOutput: null,
  isLoading: false,
  loadingMessage: '',
  addOutput: (output) => set((state) => ({ outputs: [...state.outputs, output] })),
  addChatMessage: (message) => set((state) => ({ chatHistory: [...state.chatHistory, message] })),
  setSelectedOutput: (output) => set({ selectedOutput: output }),
  setLoading: (loading, message = '') => set({ isLoading: loading, loadingMessage: message }),
  updateOutputCode: (id, newCode) => set((state) => ({
    outputs: state.outputs.map(o => o.id === id ? { ...o, code: newCode } : o),
    selectedOutput: state.selectedOutput?.id === id ? { ...state.selectedOutput, code: newCode } : state.selectedOutput,
  })),
  deleteOutput: (id) => set((state) => ({
    outputs: state.outputs.filter(o => o.id !== id),
    selectedOutput: state.selectedOutput?.id === id ? null : state.selectedOutput,
  })),
  clearSession: () => set({ outputs: [], chatHistory: [], selectedOutput: null, isLoading: false }),
}));


// State for UI and App configuration
interface AppState {
    workspaceMode: WorkspaceMode;
    appMode: AppMode;
    isToolsDrawerOpen: boolean;
    isLayoutLocked: boolean;
    settings: Settings;
}

interface AppActions {
    setWorkspaceMode: (mode: WorkspaceMode) => void;
    setAppMode: (mode: AppMode) => void;
    toggleToolsDrawer: () => void;
    setIsLayoutLocked: (locked: boolean) => void;
    setSettings: (settings: Settings) => void;
}

export const useAppStore = create<AppState & AppActions>()(
    persist(
        (set) => ({
            workspaceMode: 'creative',
            appMode: getSettings().app.defaultCreativeMode,
            isToolsDrawerOpen: false,
            isLayoutLocked: getSettings().app.isLayoutLocked,
            settings: getSettings(),

            setWorkspaceMode: (mode) => set({ workspaceMode: mode }),
            setAppMode: (mode) => set({ appMode: mode }),
            toggleToolsDrawer: () => set((state) => ({ isToolsDrawerOpen: !state.isToolsDrawerOpen })),
            setIsLayoutLocked: (locked) => {
                set((state) => {
                    const newSettings = { ...state.settings, app: { ...state.settings.app, isLayoutLocked: locked }};
                    saveSettingsToStorage(newSettings);
                    return { isLayoutLocked: locked, settings: newSettings };
                });
            },
            setSettings: (settings) => {
                saveSettingsToStorage(settings);
                set({ settings, isLayoutLocked: settings.app.isLayoutLocked });
            },
        }),
        {
            name: 'aethereal-nexus-app-storage',
            // Only persist a subset of the state to avoid conflicts with settingsService
            partialize: (state) => ({
                workspaceMode: state.workspaceMode,
                appMode: state.appMode,
            }),
        }
    )
);

// Sync settings from localStorage on startup or when changed in another tab
window.addEventListener('storage', () => {
    const freshSettings = getSettings();
    useAppStore.getState().setSettings(freshSettings);
});
