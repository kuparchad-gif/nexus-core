import React from 'react';
import { AILogoIcon } from '../icons';

const WebDevSuiteView: React.FC = () => {
    return (
        <div className="p-4 sm:p-6 lg:p-8 animate-scale-in h-full flex flex-col">
            <header className="mb-6 flex-shrink-0">
                <h2 className="text-3xl font-bold text-text-primary">Web Development Suite</h2>
                <p className="text-text-secondary">An integrated environment for building and previewing web applications with agent assistance.</p>
            </header>
            <div className="flex-grow grid grid-cols-1 lg:grid-cols-4 gap-6">
                {/* File Explorer */}
                <div className="lg:col-span-1 bg-bg-light-card glass-card rounded-xl shadow-aura border border-border-color p-4 flex flex-col">
                    <h3 className="text-lg font-semibold text-text-primary border-b border-border-color pb-2 mb-2">File Explorer</h3>
                    <div className="font-mono text-sm space-y-1 text-text-secondary">
                        <p>src/</p>
                        <p className="pl-4">components/</p>
                        <p className="pl-8 text-brand-primary">Button.tsx</p>
                        <p className="pl-4">App.tsx</p>
                        <p>package.json</p>
                    </div>
                     <div className="mt-auto pt-4 border-t border-border-color space-y-3">
                       <button className="w-full flex items-center justify-center space-x-2 bg-brand-primary/10 text-brand-primary font-semibold px-4 py-2.5 rounded-lg hover:bg-brand-primary/20 transition-colors">
                            <AILogoIcon className="w-5 h-5" />
                            <span>Lilith: Suggest UI</span>
                        </button>
                         <button className="w-full flex items-center justify-center space-x-2 bg-brand-secondary/10 text-brand-secondary font-semibold px-4 py-2.5 rounded-lg hover:bg-brand-secondary/20 transition-colors">
                            <AILogoIcon className="w-5 h-5" />
                            <span>Viren: Generate Component</span>
                        </button>
                    </div>
                </div>
                
                {/* Code & Preview */}
                <div className="lg:col-span-3 flex flex-col gap-6">
                    <div className="flex-grow bg-[#1e1e1e] text-white rounded-xl shadow-inner border border-border-color p-4 font-mono text-sm">
                        <pre>
                            <code>
{`const Button = () => {
  return <button className="btn">Click Me</button>
}`}
                            </code>
                        </pre>
                    </div>
                    <div className="h-64 bg-white rounded-xl shadow-aura border border-border-color p-4 flex items-center justify-center">
                        <p className="text-slate-500 font-mono">[ Live Browser Preview ]</p>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default WebDevSuiteView;