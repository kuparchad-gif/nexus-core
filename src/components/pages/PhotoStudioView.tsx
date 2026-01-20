import React from 'react';
import { AILogoIcon, CpuChipIcon } from '../icons';

const PhotoStudioView: React.FC = () => {
    return (
        <div className="p-4 sm:p-6 lg:p-8 animate-scale-in h-full flex flex-col">
            <header className="mb-6 flex-shrink-0">
                <h2 className="text-3xl font-bold text-text-primary">Photo Studio</h2>
                <p className="text-text-secondary">AI-assisted, professional-grade photo editing and enhancement.</p>
            </header>
            <div className="flex-grow grid grid-cols-1 lg:grid-cols-4 gap-6">
                {/* Main Viewport */}
                <div className="lg:col-span-3 bg-bg-light-bg rounded-xl shadow-inner border border-border-color flex items-center justify-center p-4">
                    <img 
                        src="https://picsum.photos/seed/aethereal/1200/800" 
                        alt="Main photograph"
                        className="max-w-full max-h-full object-contain rounded-lg shadow-lg"
                    />
                </div>

                {/* Control Panel */}
                <div className="lg:col-span-1 bg-bg-light-card glass-card rounded-xl shadow-aura border border-border-color p-4 flex flex-col">
                    <h3 className="text-lg font-semibold text-text-primary border-b border-border-color pb-2">Adjustments</h3>
                    <div className="space-y-4 overflow-y-auto pr-2 flex-grow">
                        {['Exposure', 'Contrast', 'Highlights', 'Shadows', 'Clarity', 'Vibrance'].map(label => (
                             <div key={label}>
                                <label className="block text-sm font-medium text-text-secondary mb-1">{label}</label>
                                <input type="range" className="w-full" defaultValue={50} />
                            </div>
                        ))}
                    </div>
                    
                    {/* Vision Core Panel */}
                    <div className="mt-4 pt-4 border-t border-border-color">
                         <h3 className="text-sm font-semibold text-text-primary mb-2 flex items-center space-x-2"><CpuChipIcon className="w-5 h-5" /><span>Vision Core</span></h3>
                         <div className="text-xs text-text-secondary space-y-1">
                            <p><span className="font-semibold text-green-600">●</span> LightGlue Feature Matching</p>
                            <p><span className="font-semibold text-green-600">●</span> Segment Anything Model</p>
                         </div>
                    </div>

                    <div className="mt-auto pt-4 border-t border-border-color space-y-3">
                        <button className="w-full flex items-center justify-center space-x-2 bg-brand-primary/10 text-brand-primary font-semibold px-4 py-2.5 rounded-lg hover:bg-brand-primary/20 transition-colors">
                            <AILogoIcon className="w-5 h-5" />
                            <span>Lilith: Suggest Edits</span>
                        </button>
                         <button className="w-full flex items-center justify-center space-x-2 bg-brand-secondary/10 text-brand-secondary font-semibold px-4 py-2.5 rounded-lg hover:bg-brand-secondary/20 transition-colors">
                            <AILogoIcon className="w-5 h-5" />
                            <span>Viren: Auto-Enhance</span>
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default PhotoStudioView;