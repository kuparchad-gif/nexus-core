

import { produce } from 'immer';

export interface ChatMessage {
  role: 'user' | 'model';
  text: string;
}

export interface Diagnostics {
  backendOk: boolean;
  checked: boolean;
  error: string | null;
}

export interface GeneratedProp {
  name: string;
  description: string;
  imageUrl: string;
}

export interface AssetBundle {
  videoUrl: string | null;
  characterSheetUrl: string | null;
  description: string | null;
  lore: string | null;
  props: GeneratedProp[];
  socialPostContent: string;
}

export interface Settings {
    enableTTS: boolean;
}

export interface AppState {
  selectedFile: File | null;
  previewUrl: string | null;
  prompt: string;
  assets: AssetBundle;
  histories: {
    assistant: ChatMessage[];
    avatar: ChatMessage[];
    support: ChatMessage[];
  };
  sessionId: string;
  isLoading: boolean;
  loadingMessage: string;
  diagnostics: Diagnostics;
  notifications: AppNotification[];
  nowPlaying: string | null; // ID of the currently playing TTS audio
  settings: Settings;
}

export interface AppNotification {
    id: number;
    type: 'success' | 'error' | 'info';
    message: string;
}

export type AppAction =
  | { type: 'SET_FILE'; payload: File }
  | { type: 'SET_PROMPT'; payload: string }
  | { type: 'SET_LOADING'; payload: string }
  | { type: 'SET_DIAGNOSTICS'; payload: Diagnostics }
  | { type: 'ANALYZE_SUCCESS'; payload: string }
  | { type: 'GENERATE_SUCCESS'; payload: { videoUrl: string, description: string, characterSheetUrl: string } }
  | { type: 'EDIT_SHEET_SUCCESS'; payload: string }
  | { type: 'RESET_GENERATED_ASSETS' }
  | { type: 'SEND_CHAT_MESSAGE'; payload: { mode: 'assistant' | 'avatar' | 'support', text: string } }
  | { type: 'RECEIVE_CHAT_MESSAGE'; payload: { mode: 'assistant' | 'avatar' | 'support', text: string } }
  | { type: 'SET_LORE'; payload: string | null }
  | { type: 'ADD_PROP'; payload: GeneratedProp }
  | { type: 'ADD_NOTIFICATION', payload: Omit<AppNotification, 'id'> }
  | { type: 'REMOVE_NOTIFICATION', payload: number }
  | { type: 'SET_NOW_PLAYING', payload: string | null }
  | { type: 'UPDATE_SETTINGS', payload: Partial<Settings> }
  // Fix: Add actions for social post content
  | { type: 'GENERATE_SOCIAL_POST_SUCCESS'; payload: string }
  | { type: 'SET_SOCIAL_POST_CONTENT'; payload: string };

export const initialState: AppState = {
  selectedFile: null,
  previewUrl: null,
  prompt: '',
  // Fix: Add socialPostContent to initial state
  assets: { videoUrl: null, characterSheetUrl: null, description: null, lore: null, props: [], socialPostContent: '' },
  histories: {
    assistant: [{ role: 'model', text: "Welcome to the Forge! I am your studio assistant. To begin, provide a reference file, describe your vision, or both." }],
    avatar: [],
    support: [{ role: 'model', text: "Welcome to the AI Support Terminal. I can help diagnose and resolve platform issues. How can I assist you?" }],
  },
  sessionId: `session_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`,
  isLoading: false,
  loadingMessage: '',
  diagnostics: { backendOk: false, checked: false, error: null },
  notifications: [],
  nowPlaying: null,
  settings: { enableTTS: true },
};

function getInitialState(): AppState {
    try {
        const savedState = localStorage.getItem('aiForgeSession');
        if (savedState) {
            const parsed = JSON.parse(savedState);
            // Merge saved state with initial state to ensure all keys are present
            const mergedState = { ...initialState, ...parsed };
            // We don't persist file objects, so reset them.
            mergedState.selectedFile = null;
            mergedState.previewUrl = null;
            mergedState.isLoading = false;
            mergedState.loadingMessage = '';
            return mergedState;
        }
    } catch (error) {
        console.error("Could not load state from localStorage", error);
    }
    return initialState;
}

export const appReducer = produce((draft: AppState, action: AppAction) => {
  switch (action.type) {
    case 'SET_FILE':
      draft.selectedFile = action.payload;
      if (draft.previewUrl) URL.revokeObjectURL(draft.previewUrl);
      draft.previewUrl = URL.createObjectURL(action.payload);
      break;
    case 'SET_PROMPT':
      draft.prompt = action.payload;
      break;
    case 'SET_LOADING':
      draft.isLoading = !!action.payload;
      draft.loadingMessage = action.payload;
      break;
    case 'SET_DIAGNOSTICS':
      draft.diagnostics = action.payload;
      break;
    case 'ANALYZE_SUCCESS':
      draft.prompt = action.payload;
      draft.histories.assistant.push({ role: 'model', text: "Analysis complete! I've populated the prompt for you." });
      break;
    case 'GENERATE_SUCCESS':
      draft.assets.videoUrl = action.payload.videoUrl;
      draft.assets.characterSheetUrl = action.payload.characterSheetUrl;
      draft.assets.description = action.payload.description;
      draft.histories.avatar = [{ role: 'model', text: 'I am forged! Interact with me here or explore other workspaces.' }];
      break;
    case 'EDIT_SHEET_SUCCESS':
        draft.assets.characterSheetUrl = action.payload;
        break;
    case 'RESET_GENERATED_ASSETS':
        draft.assets = initialState.assets;
        draft.histories.avatar = [];
        if (draft.previewUrl) URL.revokeObjectURL(draft.previewUrl);
        draft.previewUrl = null;
        draft.selectedFile = null;
        break;
    case 'SEND_CHAT_MESSAGE':
        draft.histories[action.payload.mode].push({ role: 'user', text: action.payload.text });
        draft.isLoading = true;
        break;
    case 'RECEIVE_CHAT_MESSAGE':
        draft.histories[action.payload.mode].push({ role: 'model', text: action.payload.text });
        draft.isLoading = false;
        break;
    case 'SET_LORE':
        draft.assets.lore = action.payload;
        break;
    case 'ADD_PROP':
        draft.assets.props.push(action.payload);
        break;
    case 'ADD_NOTIFICATION':
        draft.notifications.push({ ...action.payload, id: Date.now() });
        break;
    case 'REMOVE_NOTIFICATION':
        draft.notifications = draft.notifications.filter(n => n.id !== action.payload);
        break;
    case 'SET_NOW_PLAYING':
        draft.nowPlaying = action.payload;
        break;
    case 'UPDATE_SETTINGS':
        draft.settings = { ...draft.settings, ...action.payload };
        break;
    // Fix: Add reducers for social post actions
    case 'GENERATE_SOCIAL_POST_SUCCESS':
        draft.assets.socialPostContent = action.payload;
        break;
    case 'SET_SOCIAL_POST_CONTENT':
        draft.assets.socialPostContent = action.payload;
        break;
  }
});

// Wrap reducer with a function that also saves to localStorage
export const rootReducer = (state: AppState, action: AppAction) => {
    const nextState = appReducer(state, action);
    try {
        const stateToSave = { ...nextState };
        // Clean non-serializable and transient state before saving
        delete stateToSave.selectedFile;
        delete stateToSave.previewUrl;
        delete stateToSave.isLoading;
        delete stateToSave.loadingMessage;
        delete stateToSave.notifications;
        delete stateToSave.nowPlaying;
        localStorage.setItem('aiForgeSession', JSON.stringify(stateToSave));
    } catch (error) {
        console.error("Could not save state to localStorage", error);
    }
    return nextState;
};

export const loadedState = getInitialState();