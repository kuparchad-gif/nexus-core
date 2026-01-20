

import React, { useState, useEffect } from 'react';
import { GoogleGenAI, GenerateContentResponse } from "@google/genai";

const loadingMessages = [ "Initializing generation core...", "Allocating resources...", "Beginning diffusion process...", "Rendering keyframes...", "Encoding video stream...", "Finalizing output..." ];

// FIX: Removed LogEntry import and updated addLog prop type.
export default function MediaGenerationView({ addLog }: { addLog: (type: 'log' | 'error', ...args: any[]) => void; }) {
    const [prompt, setPrompt] = useState('');
    const [mediaType, setMediaType] = useState<'image' | 'video'>('image');
    const [aspectRatio, setAspectRatio] = useState('1:1');
    const [isLoading, setIsLoading] = useState(false);
    const [loadingStatus, setLoadingStatus] = useState('');
    const [generatedMedia, setGeneratedMedia] = useState<string | null>(null);
    const ai = React.useRef<GoogleGenAI | null>(null);

    useEffect(() => {
        if (process.env.API_KEY) {
            ai.current = new GoogleGenAI({ apiKey: process.env.API_KEY });
        }
    }, []);

    useEffect(() => {
        // FIX: Replaced NodeJS.Timeout with `number | undefined` which is the correct type for setInterval return value in a browser environment.
        let interval: number | undefined;
        if (isLoading && mediaType === 'video') {
            let i = 0;
            interval = setInterval(() => {
                setLoadingStatus(loadingMessages[i % loadingMessages.length]);
                i++;
            }, 2500);
        }
        return () => clearInterval(interval);
    }, [isLoading, mediaType]);
    
    const handleGenerate = async () => {
        if (!prompt.trim() || isLoading) return;
        
        setIsLoading(true);
        setGeneratedMedia(null);
        setLoadingStatus('Initializing...');

        if (mediaType === 'image') {
            try {
                if (!ai.current) throw new Error("API Key not configured.");
                
                addLog('log', `Generating image with prompt: "${prompt}"`);
                const response = await ai.current.models.generateImages({
                    model: 'imagen-4.0-generate-001',
                    prompt: prompt,
                    config: {
                      numberOfImages: 1,
                      outputMimeType: 'image/png',
                      aspectRatio: aspectRatio as "1:1" | "16:9" | "9:16" | "4:3" | "3:4",
                    },
                });

                const base64Image = response.generatedImages[0].image.imageBytes;
                setGeneratedMedia(`data:image/png;base64,${base64Image}`);

            } catch (error) {
                const errorMessage = error instanceof Error ? error.message : "An unknown error occurred.";
                addLog('error', 'Image generation failed:', errorMessage);
                alert(`Image generation failed: ${errorMessage}`);
            } finally {
                setIsLoading(false);
            }
        } else { // Video generation (mock)
            setTimeout(() => {
                const resultUrl = 'https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/Big_Buck_Bunny_360_10s_1MB.mp4';
                setGeneratedMedia(resultUrl);
                setIsLoading(false);
            }, 10000);
        }
    };

    return (
      <div className="app-window">
        <div className="app-header">
            <img src="https://i.imgur.com/8p8YJkP.png" alt="Aethereal Icon" className="w-7 h-7" />
            <span>Media Generation</span>
        </div>
        <div className="media-gen-layout">
            <div className="media-gen-controls">
                <textarea
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    placeholder="Describe the media you want to create..."
                    className="media-gen-prompt"
                    rows={3}
                />
                <div className="media-gen-options">
                    <div className="flex items-center gap-4">
                        <select value={mediaType} onChange={(e) => setMediaType(e.target.value as any)} className="p-2 rounded-lg border border-gray-300">
                            <option value="image">Image</option>
                            <option value="video">Video (Mock)</option>
                        </select>
                        <select value={aspectRatio} onChange={(e) => setAspectRatio(e.target.value)} className="p-2 rounded-lg border border-gray-300">
                            <option value="1:1">1:1 (Square)</option>
                            <option value="16:9">16:9 (Widescreen)</option>
                            <option value="9:16">9:16 (Vertical)</option>
                            <option value="4:3">4:3 (Landscape)</option>
                            <option value="3:4">3:4 (Portrait)</option>
                        </select>
                    </div>
                    <button onClick={handleGenerate} className="media-gen-button" disabled={isLoading}>
                        {isLoading ? 'Generating...' : 'Generate'}
                    </button>
                </div>
            </div>
            <div className="media-preview-area">
                {isLoading && (
                    <div className="loading-overlay">
                        <div className="loading-spinner"></div>
                        <p className="loading-status">{loadingStatus}</p>
                    </div>
                )}
                {!generatedMedia && !isLoading && <p className="text-gray-500">Preview will appear here</p>}
                {generatedMedia && mediaType === 'image' && <img src={generatedMedia} alt={prompt} className="generated-image p-2" />}
                {generatedMedia && mediaType === 'video' && <video src={generatedMedia} controls autoPlay loop className="generated-image" />}
            </div>
        </div>
      </div>
    );
}