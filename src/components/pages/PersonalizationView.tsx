
import React, { useState } from 'react';
import { ThemeConfig } from '../../types';
import { CheckCircleIcon, PlusIcon } from '../icons';

interface PersonalizationViewProps {
    theme: ThemeConfig;
    setTheme: React.Dispatch<React.SetStateAction<ThemeConfig>>;
}

const WALLPAPERS = [
    { id: 'default', name: 'Nexus Aurora', css: 'radial-gradient(circle at 10% 10%, rgba(224, 231, 255, 0.8) 0%, transparent 40%), radial-gradient(circle at 90% 90%, rgba(237, 233, 254, 0.8) 0%, transparent 40%), linear-gradient(135deg, #f8fafc 0%, #e2e8f0 50%, #f1f5f9 100%)' },
    { id: 'dark_void', name: 'Dark Void', css: 'radial-gradient(circle at 50% 50%, #1e293b 0%, #0f172a 100%)' },
    { id: 'cyber_blue', name: 'Cyber Blue', css: 'linear-gradient(135deg, #0ea5e9 0%, #2563eb 100%)' },
    { id: 'sunset', name: 'Solar Flare', css: 'linear-gradient(to top, #f43f5e 0%, #f97316 100%)' },
    { id: 'mint', name: 'Clean Slate', css: 'linear-gradient(120deg, #d4fc79 0%, #96e6a1 100%)' }
];

const COLORS = [
    { name: 'Violet', hex: '#a78bfa', class: 'bg-violet-400' },
    { name: 'Sky', hex: '#38bdf8', class: 'bg-sky-400' },
    { name: 'Emerald', hex: '#34d399', class: 'bg-emerald-400' },
    { name: 'Rose', hex: '#fb7185', class: 'bg-rose-400' },
    { name: 'Amber', hex: '#fbbf24', class: 'bg-amber-400' },
    { name: 'Slate', hex: '#94a3b8', class: 'bg-slate-400' },
];

const PersonalizationView: React.FC<PersonalizationViewProps> = ({ theme, setTheme }) => {
    const [customUrl, setCustomUrl] = useState('');

    const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onloadend = () => {
            const base64 = reader.result as string;
            setTheme(prev => ({ ...prev, wallpaper: `url(${base64})` }));
        };
        reader.readAsDataURL(file);
    };

    const handleUrlApply = () => {
        if (customUrl) {
            setTheme(prev => ({ ...prev, wallpaper: `url(${customUrl})` }));
        }
    };

    return (
        <div className="p-4 sm:p-6 lg:p-8 animate-scale-in h-full flex flex-col">
            <header className="mb-6 flex-shrink-0">
                <h2 className="text-3xl font-bold text-text-primary">Personalization</h2>
                <p className="text-text-secondary">Customize the aesthetics of your Nexus environment.</p>
            </header>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 flex-grow overflow-y-auto">
                {/* Background Selection */}
                <div className="bg-bg-light-card glass-card rounded-xl shadow-aura border border-border-color p-6">
                    <h3 className="text-lg font-bold text-text-primary mb-4">Wallpaper</h3>
                    <div className="grid grid-cols-2 gap-4 mb-4">
                        {WALLPAPERS.map(wp => (
                            <div 
                                key={wp.id}
                                onClick={() => setTheme(prev => ({ ...prev, wallpaper: wp.css }))}
                                className={`h-24 rounded-lg cursor-pointer border-2 transition-all relative overflow-hidden shadow-sm hover:shadow-md ${theme.wallpaper === wp.css ? 'border-brand-primary scale-[1.02]' : 'border-transparent hover:scale-[1.02]'}`}
                                style={{ background: wp.css }}
                            >
                                {theme.wallpaper === wp.css && (
                                    <div className="absolute inset-0 flex items-center justify-center bg-black/20 backdrop-blur-[1px]">
                                        <CheckCircleIcon className="w-8 h-8 text-white" />
                                    </div>
                                )}
                                <span className="absolute bottom-2 left-2 text-xs font-bold text-white drop-shadow-md">{wp.name}</span>
                            </div>
                        ))}
                    </div>
                    
                    {/* Custom Upload */}
                    <div className="space-y-3 border-t border-border-color pt-4">
                        <h4 className="text-sm font-semibold text-text-secondary">Custom Image</h4>
                        <div className="flex gap-2">
                            <input 
                                type="text" 
                                placeholder="Paste Image URL..." 
                                value={customUrl}
                                onChange={(e) => setCustomUrl(e.target.value)}
                                className="flex-1 bg-bg-light-bg border border-border-color rounded-lg px-3 py-2 text-sm"
                            />
                            <button onClick={handleUrlApply} className="bg-brand-primary text-white px-4 rounded-lg text-sm font-medium">Set</button>
                        </div>
                        <div className="flex items-center gap-2">
                            <span className="text-xs text-text-secondary uppercase font-bold">OR</span>
                            <label className="flex items-center gap-2 cursor-pointer bg-bg-light-bg hover:bg-slate-200 border border-border-color px-3 py-2 rounded-lg text-sm text-text-primary transition-colors">
                                <PlusIcon className="w-4 h-4" /> Upload File
                                <input type="file" accept="image/*" onChange={handleFileUpload} className="hidden" />
                            </label>
                        </div>
                    </div>
                </div>

                {/* Accent Colors */}
                <div className="bg-bg-light-card glass-card rounded-xl shadow-aura border border-border-color p-6">
                    <h3 className="text-lg font-bold text-text-primary mb-4">System Accent Color</h3>
                    <div className="flex flex-wrap gap-4">
                        {COLORS.map(color => (
                            <button
                                key={color.name}
                                onClick={() => setTheme(prev => ({ ...prev, accentColor: color.hex }))}
                                className={`w-12 h-12 rounded-full ${color.class} shadow-sm hover:shadow-md transition-transform flex items-center justify-center border-2 ${theme.accentColor === color.hex ? 'border-white ring-2 ring-brand-primary scale-110' : 'border-transparent hover:scale-110'}`}
                                title={color.name}
                            >
                                {theme.accentColor === color.hex && <CheckCircleIcon className="w-6 h-6 text-white" />}
                            </button>
                        ))}
                    </div>
                </div>

                {/* Advanced Appearance */}
                <div className="bg-bg-light-card glass-card rounded-xl shadow-aura border border-border-color p-6">
                    <h3 className="text-lg font-bold text-text-primary mb-4">Appearance</h3>
                    <div className="space-y-6">
                        <div>
                            <label className="block text-sm font-medium text-text-secondary mb-2">Window Glass Opacity ({Math.round(theme.glassOpacity * 100)}%)</label>
                            <input 
                                type="range" 
                                min="0.1" 
                                max="1" 
                                step="0.05" 
                                value={theme.glassOpacity}
                                onChange={(e) => setTheme(prev => ({ ...prev, glassOpacity: parseFloat(e.target.value) }))}
                                className="w-full"
                            />
                            <div className="flex justify-between text-xs text-text-secondary mt-1">
                                <span>Transparent</span>
                                <span>Opaque</span>
                            </div>
                        </div>

                        <div>
                            <label className="block text-sm font-medium text-text-secondary mb-2">Icon Theme</label>
                            <div className="flex gap-3">
                                <button 
                                    onClick={() => setTheme(prev => ({ ...prev, iconStyle: 'glass-dark' }))} // Reusing prop for logic simplicity
                                    className={`px-4 py-2 rounded-lg border text-sm font-medium transition-colors ${theme.iconStyle !== 'unified' ? 'bg-brand-primary text-white border-brand-primary' : 'bg-white border-border-color text-text-secondary hover:bg-slate-50'}`}
                                >
                                    Vibrant (Original)
                                </button>
                                <button 
                                    onClick={() => setTheme(prev => ({ ...prev, iconStyle: 'unified' }))}
                                    className={`px-4 py-2 rounded-lg border text-sm font-medium transition-colors ${theme.iconStyle === 'unified' ? 'bg-brand-primary text-white border-brand-primary' : 'bg-white border-border-color text-text-secondary hover:bg-slate-50'}`}
                                >
                                    Unified (Accent Color)
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default PersonalizationView;
