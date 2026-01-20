/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/

import { Settings, AppMode, Theme } from '../types';
import { MantineTheme, MantineThemeOverride, createTheme, rem } from '@mantine/core';

const SETTINGS_KEY = 'aethereal-nexus-settings';

const DEFAULT_ENDPOINTS: Record<AppMode, string> = {
    // Creative
    scene: '/fusion',
    image: '/generate_image',
    avatar: '/generate_avatar',
    image_to_3d: '/image_to_3d',
    image_to_animation: '/image_to_animation',
    music: '/generate_music',
    sound_fx: '/generate_sound_fx',
    audio_enhance: '/audio_enhance',
    voiceover: '/voiceover',
    game_asset: '/game_asset',
    ai_toolbench: '/generate_ar_effect',
    // Dev & Ops
    code: '/execute_code',
    database: '/query_database',
    metrics: '/get_metrics',
    shell: '/execute_shell',
    vscode: '/vscode_action',
    // Professional Apps
    email: '/email_action',
    documents: '/document_action',
    spreadsheets: '/spreadsheet_action',
    storage: '/storage_action',
    // System
    chat: '/chat',
};

const defaultThemeSettings: Theme = {
    primaryColor: '#4f46e5', // Indigo 600
    accentColor: '#10b981', // Emerald 500
    backgroundColor: '#f8fafc', // Slate 50
    textColor: '#1e293b', // Slate 800
    borderColor: 'rgba(0, 0, 0, 0.1)',
    fontFamily: 'Inter, sans-serif',
    fontFamilyMonospace: 'Fira Code, monospace',
    radius: 1.0, // rem
    shadow: 0.05,
    opacity: 0.4,
};

const defaultSettings: Settings = {
  api: {
    baseUrl: 'https://aethereal-nexus-viren-db0--metatron-ulam-fastapi-app.modal.run',
    endpoints: DEFAULT_ENDPOINTS,
  },
  ui: defaultThemeSettings,
  app: {
    defaultCreativeMode: 'scene',
    sceneAssetValidation: false,
    funMode: false,
    // FIX: Add isLayoutLocked to default settings to avoid type errors.
    isLayoutLocked: false,
  },
  layout: [
    { id: 'left', width: 25 },
    { id: 'center', width: 50 },
    { id: 'right', width: 25 },
  ],
};

export const getSettings = (): Settings => {
  try {
    const storedSettings = localStorage.getItem(SETTINGS_KEY);
    if (storedSettings) {
      const parsed = JSON.parse(storedSettings);
      // Deep merge with defaults to ensure new settings are included
      return {
        ...defaultSettings,
        ...parsed,
        api: { ...defaultSettings.api, ...(parsed.api || {}) },
        ui: { ...defaultSettings.ui, ...(parsed.ui || {}) },
        app: { ...defaultSettings.app, ...(parsed.app || {}) },
        layout: parsed.layout || defaultSettings.layout,
      };
    }
  } catch (error) {
    console.error("Failed to parse settings from localStorage", error);
  }
  return defaultSettings;
};

export const saveSettings = (settings: Settings): void => {
  try {
    const settingsString = JSON.stringify(settings);
    localStorage.setItem(SETTINGS_KEY, settingsString);
    window.dispatchEvent(new StorageEvent('storage', { key: SETTINGS_KEY }));
  } catch (error)
    {
    console.error("Failed to save settings to localStorage", error);
  }
};

// Function to convert our simple Theme object into a Mantine theme
export const createMantineTheme = (theme: Theme): MantineThemeOverride => {
    return createTheme({
        primaryColor: 'custom',
        colors: {
            custom: [
              theme.primaryColor, // 0
              theme.primaryColor, // 1
              theme.primaryColor, // 2
              theme.primaryColor, // 3
              theme.primaryColor, // 4
              theme.primaryColor, // 5
              theme.primaryColor, // 6
              theme.primaryColor, // 7
              theme.primaryColor, // 8
              theme.primaryColor, // 9
            ],
        },
        fontFamily: theme.fontFamily,
        fontFamilyMonospace: theme.fontFamilyMonospace,
        headings: {
            fontFamily: theme.fontFamily,
        },
        radius: {
            xs: `calc(${theme.radius}rem * 0.4)`,
            sm: `calc(${theme.radius}rem * 0.6)`,
            md: `calc(${theme.radius}rem * 0.8)`,
            lg: `${theme.radius}rem`,
            xl: `calc(${theme.radius}rem * 1.2)`,
        },
    });
};