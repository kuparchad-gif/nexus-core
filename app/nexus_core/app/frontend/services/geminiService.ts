/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/

import { Output, AppMode } from '../types';
import { getSettings } from './settingsService';
import { useGenerationStore } from './store';

const getApiConfig = () => getSettings().api;

export const generateAsset = async (
  mode: AppMode,
  assets: File[],
  prompt: string,
): Promise<Output[]> => {
  const { setLoading, addChatMessage } = useGenerationStore.getState();
  try {
    setLoading(true, "Connecting to Aethereal Nexus backend...");
    
    const { baseUrl, endpoints } = getApiConfig();
    const endpoint = endpoints[mode];

    if (!endpoint) {
        throw new Error(`Invalid generation mode: ${mode}`);
    }
    const URL = baseUrl + endpoint;

    const formData = new FormData();
    formData.append('prompt', prompt);

    if (assets.length > 0) {
      formData.append('file', assets[0]);
    }
    
    setLoading(true, `Generating asset via ${endpoint}...`);
    
    const response = await fetch(URL, { method: 'POST', body: formData });

    if (!response.ok) {
      let errorDetail = 'The server returned an error response.';
      try {
        const errorData = await response.json();
        errorDetail = errorData.message || errorData.detail || JSON.stringify(errorData);
      } catch (e) {
        errorDetail = response.statusText;
      }
      throw new Error(`Server Error: ${response.status} - ${errorDetail}`);
    }

    const data = await response.json();
    setLoading(true, "Processing results...");

    let result: Output[] = [];
    // Creative Modes
    if (mode === 'scene' && data.status === 'ok' && data.payload?.sketch) {
        result = [{ id: Date.now(), type: 'p5js', code: data.payload.sketch, fullResponse: 'Generated p5.js sketch from image.' }];
    } else if (mode === 'image_to_3d' && data.url) {
        result = [{ id: Date.now(), type: 'gltf_url', url: data.url, fullResponse: `Generated 3D model from image.` }];
    } else if (mode === 'image_to_animation' && data.video_url) {
        result = [{ id: Date.now(), type: 'video_url', url: data.video_url, fullResponse: `Animation generated.` }];
    } else if (mode === 'audio_enhance' && data.enhanced_audio_url) {
        result = [{ id: Date.now(), type: 'audio_url', url: data.enhanced_audio_url, fullResponse: 'Audio enhancement complete.' }];
    } else if (mode === 'voiceover' && data.voiceover_url) {
        result = [{ id: Date.now(), type: 'audio_url', url: data.voiceover_url, fullResponse: `Voiceover generated` }];
    } else if ((mode === 'image' || mode === 'avatar') && data.image_url) {
        result = [{ id: Date.now(), type: 'image_url', url: data.image_url, fullResponse: `Generated ${mode} successfully.` }];
    } else if ((mode === 'sound_fx' || mode === 'music') && data.audio_url) {
        result = [{ id: Date.now(), type: 'audio_url', url: data.audio_url, fullResponse: `Generated ${mode}` }];
    } else {
        const fullResponse = JSON.stringify(data, null, 2);
        result = [{ id: Date.now(), type: 'log', fullResponse }];
        addChatMessage({sender: 'agent', text: `Received data: \n\`\`\`json\n${fullResponse}\n\`\`\``});
    }

    setLoading(false);
    return result;

  } catch (e: any) {
    console.error(`Error in generateAsset for mode ${mode}:`, e);
    setLoading(false);
    throw e;
  }
};

export const sendChatMessage = async (
    prompt: string,
    systemPrompt: string,
): Promise<string> => {
    const { baseUrl, endpoints } = getApiConfig();
    const URL = baseUrl + endpoints.chat;

    try {
        const response = await fetch(URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt, system_prompt: systemPrompt }),
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `Server error: ${response.status}`);
        }

        const data = await response.json();
        return data.response || "No response text found.";

    } catch (error) {
        console.error("Chat API error:", error);
        throw error;
    }
};
