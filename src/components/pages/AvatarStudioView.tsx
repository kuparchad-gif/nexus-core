

import React, { useState } from 'react';

const AvatarStudioView: React.FC = () => {
    const [activeTab, setActiveTab] = useState('sculpt');

    const renderPanel = () => {
        switch (activeTab) {
            case 'sculpt':
                return (
                    <div className="space-y-4">
                        <div>
                            <label className="text-sm font-medium">Symmetry</label>
                            <input type="checkbox" className="ml-2" defaultChecked />
                        </div>
                        <div>
                            <label className="text-sm font-medium">Nose Bridge</label>
                            <input type="range" className="w-full" />
                        </div>
                        <div>
                            <label className="text-sm font-medium">Jaw Width</label>
                            <input type="range" className="w-full" />
                        </div>
                        <div>
                            <label className="text-sm font-medium">Eye Socket Depth</label>
                            <input type="range" className="w-full" />
                        </div>
                    </div>
                );
            case 'materials':
                 return (
                    <div className="space-y-4">
                        <div>
                            <label className="text-sm font-medium">Skin Texture</label>
                            <select className="w-full bg-bg-light-bg border border-border-color rounded p-1 text-sm">
                                <option>Porcelain_01</option>
                                <option>Olive_03</option>
                            </select>
                        </div>
                        <div>
                            <label className="text-sm font-medium">Subsurface Scattering</label>
                            <input type="range" className="w-full" />
                        </div>
                         <div>
                            <label className="text-sm font-medium">Hair Shader</label>
                            <select className="w-full bg-bg-light-bg border border-border-color rounded p-1 text-sm">
                                <option>Anisotropic</option>
                                <option>Phong</option>
                            </select>
                        </div>
                    </div>
                );
            case 'lighting':
                return (
                     <div className="space-y-4">
                        <div>
                            <label className="text-sm font-medium">Environment</label>
                            <select className="w-full bg-bg-light-bg border border-border-color rounded p-1 text-sm">
                                <option>Studio</option>
                                <option>Outdoor</option>
                                <option>Sunset</option>
                            </select>
                        </div>
                        <div>
                            <label className="text-sm font-medium">Key Light Intensity</label>
                            <input type="range" className="w-full" />
                        </div>
                        <div>
                            <label className="text-sm font-medium">Rim Light Color</label>
                            <input type="color" className="w-full h-8 p-0 border-none rounded" defaultValue="#8B5CF6" />
                        </div>
                    </div>
                );
            default: return null;
        }
    }

    return (
        <div className="p-4 sm:p-6 lg:p-8 animate-scale-in h-full flex flex-col">
            <header className="mb-6 flex-shrink-0">
                <h2 className="text-3xl font-bold text-text-primary">Avatar Studio</h2>
                <p className="text-text-secondary">High-fidelity, cinematic character creator.</p>
            </header>
            <div className="flex-grow grid grid-cols-1 lg:grid-cols-4 gap-6">
                {/* Viewport */}
                <div className="lg:col-span-3 bg-black rounded-xl shadow-aura border border-border-color flex items-center justify-center">
                    <p className="text-slate-600 font-mono">[ Photorealistic 3D Viewport ]</p>
                </div>

                {/* Controls */}
                <div className="lg:col-span-1 bg-bg-light-card glass-card rounded-xl shadow-aura border border-border-color p-4 flex flex-col">
                    <div className="border-b border-border-color mb-4">
                        <nav className="-mb-px flex space-x-4">
                            <button onClick={() => setActiveTab('sculpt')} className={`whitespace-nowrap py-3 px-1 border-b-2 font-medium text-sm ${activeTab === 'sculpt' ? 'border-brand-primary text-brand-primary' : 'border-transparent text-text-secondary'}`}>Sculpt</button>
                            <button onClick={() => setActiveTab('materials')} className={`whitespace-nowrap py-3 px-1 border-b-2 font-medium text-sm ${activeTab === 'materials' ? 'border-brand-primary text-brand-primary' : 'border-transparent text-text-secondary'}`}>Materials</button>
                            <button onClick={() => setActiveTab('lighting')} className={`whitespace-nowrap py-3 px-1 border-b-2 font-medium text-sm ${activeTab === 'lighting' ? 'border-brand-primary text-brand-primary' : 'border-transparent text-text-secondary'}`}>Lighting</button>
                        </nav>
                    </div>
                    <div className="flex-grow">
                        {renderPanel()}
                    </div>
                    <div className="mt-auto pt-4">
                         <button className="w-full bg-brand-primary text-white font-semibold py-3 rounded-lg hover:bg-brand-primary/90 transition-colors shadow-aura">
                            Render & Export
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default AvatarStudioView;