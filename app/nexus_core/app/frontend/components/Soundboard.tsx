/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/
import React, { useState, useRef, useEffect } from 'react';
import { Output } from '../types';
import { Play, Volume2 } from 'lucide-react';

interface SoundboardPadProps {
    output: Output;
}

const SoundboardPad: React.FC<SoundboardPadProps> = ({ output }) => {
    const [isPlaying, setIsPlaying] = useState(false);
    const audioRef = useRef<HTMLAudioElement | null>(null);

    // Initialize Audio object only once
    useEffect(() => {
        if (!audioRef.current && output.url) {
            audioRef.current = new Audio(output.url);
            
            const onEnded = () => setIsPlaying(false);
            audioRef.current.addEventListener('ended', onEnded);

            return () => {
                // Cleanup: remove event listener
                if (audioRef.current) {
                    audioRef.current.removeEventListener('ended', onEnded);
                }
            };
        }
    }, [output.url]);
    
    const playSound = () => {
        if (audioRef.current) {
            audioRef.current.currentTime = 0;
            audioRef.current.play().catch(e => console.error("Audio play failed:", e));
            setIsPlaying(true);
        }
    };

    return (
        <button 
            onClick={playSound}
            className={`relative w-full h-full p-4 rounded-lg flex flex-col items-center justify-center text-center transition-all duration-150 transform focus:outline-none focus:ring-4 focus:ring-indigo-500/50 glass-panel
            ${isPlaying 
                ? 'bg-indigo-600/80 text-white shadow-lg scale-105 glow-border' 
                : 'text-slate-700 hover:bg-white/40 active:scale-95'
            }`}
        >
            <div className="absolute top-2 right-2">
                {isPlaying ? <Volume2 size={16} className="animate-pulse"/> : <Play size={16}/>}
            </div>
            <p className="font-semibold text-sm break-words">{output.fullResponse}</p>
        </button>
    );
};


interface SoundboardProps {
  outputs: Output[];
}

const Soundboard: React.FC<SoundboardProps> = ({ outputs }) => {
  const audioOutputs = outputs.filter(o => o.type === 'audio_url');

  if (audioOutputs.length === 0) {
      return (
        <div className="flex flex-col items-center justify-center h-full text-slate-600 p-4">
            <Volume2 size={48} className="mb-4" />
            <h3 className="text-xl font-bold mb-2">Soundboard</h3>
            <p>Generate audio assets to add them here.</p>
        </div>
      );
  }

  return (
    <div className="w-full h-full p-4 overflow-y-auto">
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4">
            {audioOutputs.map(output => (
                <div key={output.id} className="aspect-square">
                    <SoundboardPad output={output} />
                </div>
            ))}
        </div>
    </div>
  );
};

export default Soundboard;