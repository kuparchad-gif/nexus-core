import React from 'react';
import { AILogoIcon } from '../icons';

const VideoEditorView: React.FC = () => {
    return (
        <div className="p-4 sm:p-6 lg:p-8 animate-scale-in h-full flex flex-col">
            <header className="mb-6 flex-shrink-0">
                <h2 className="text-3xl font-bold text-text-primary">Video Editor</h2>
                <p className="text-text-secondary">Intuitive timeline-based video editing powered by Nexus agents.</p>
            </header>
            <div className="flex-grow grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Left Panel: Media Bin & Effects */}
                <div className="lg:col-span-1 bg-bg-light-card glass-card rounded-xl shadow-aura border border-border-color p-4 flex flex-col space-y-4">
                    <h3 className="text-lg font-semibold text-text-primary border-b border-border-color pb-2">Media Bin</h3>
                    <div className="grid grid-cols-3 gap-2">
                        {[1, 2, 3, 4, 5, 6].map(i => (
                            <img key={i} src={`https://picsum.photos/seed/clip${i}/100/80`} className="rounded aspect-video object-cover cursor-pointer" />
                        ))}
                    </div>
                    <div className="mt-auto pt-4 border-t border-border-color space-y-3">
                       <button className="w-full flex items-center justify-center space-x-2 bg-brand-primary/10 text-brand-primary font-semibold px-4 py-2.5 rounded-lg hover:bg-brand-primary/20 transition-colors">
                            <AILogoIcon className="w-5 h-5" />
                            <span>Lilith: Create Storyboard</span>
                        </button>
                         <button className="w-full flex items-center justify-center space-x-2 bg-brand-secondary/10 text-brand-secondary font-semibold px-4 py-2.5 rounded-lg hover:bg-brand-secondary/20 transition-colors">
                            <AILogoIcon className="w-5 h-5" />
                            <span>Viren: Auto-Cut Scene</span>
                        </button>
                    </div>
                </div>

                {/* Right Panel: Monitor & Timeline */}
                <div className="lg:col-span-2 flex flex-col gap-6">
                    <div className="flex-grow bg-black rounded-xl shadow-inner border border-border-color flex items-center justify-center p-4">
                        <p className="text-slate-600 font-mono">[ Video Preview Monitor ]</p>
                    </div>
                    <div className="h-48 bg-bg-light-card glass-card rounded-xl shadow-aura border border-border-color p-4 flex flex-col">
                         <h3 className="text-lg font-semibold text-text-primary border-b border-border-color pb-2 mb-2 flex-shrink-0">Timeline</h3>
                         <div className="flex-grow flex items-center justify-center">
                            <p className="text-slate-500 font-mono">[ Timeline & Audio Tracks ]</p>
                         </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default VideoEditorView;