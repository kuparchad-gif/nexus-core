/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/

import React, { useState, useEffect } from 'react';
import Spinner from './Spinner';
import { HardDrive, Server, Wifi } from 'lucide-react';

interface VirtualAppStreamProps {
    appName: string;
    appUrl?: string; // Optional URL for web-based apps like VS Code
}

const loadingSteps = [
    { message: "Provisioning Environment...", icon: <HardDrive className="h-16 w-16 text-slate-600" /> },
    { message: "Starting Application Server...", icon: <Server className="h-16 w-16 text-slate-600" /> },
    { message: "Establishing Secure Connection...", icon: <Wifi className="h-16 w-16 text-slate-600" /> },
];

const VirtualAppStream: React.FC<VirtualAppStreamProps> = ({ appName, appUrl }) => {
    const [step, setStep] = useState(0);
    const [isLoaded, setIsLoaded] = useState(false);

    useEffect(() => {
        setIsLoaded(false);
        setStep(0);
        // FIX: Replaced NodeJS.Timeout with 'number' for browser compatibility.
        const timers: number[] = [];

        // Simulate the connection process
        for (let i = 0; i < loadingSteps.length; i++) {
            timers.push(setTimeout(() => setStep(i + 1), (i + 1) * 2000));
        }

        // After all steps, show the app
        timers.push(setTimeout(() => setIsLoaded(true), (loadingSteps.length + 1) * 2000));
        
        return () => timers.forEach(clearTimeout);
    }, [appName]);

    if (!isLoaded) {
        const currentStep = loadingSteps[step] || loadingSteps[loadingSteps.length - 1];
        return (
            <div className="w-full h-full flex flex-col items-center justify-center text-center p-4">
                <div className="mb-8">
                    {currentStep.icon}
                </div>
                <h2 className="text-xl font-bold text-slate-800 mb-2">Connecting to {appName}</h2>
                <p className="text-slate-600 font-medium animate-pulse">{currentStep.message}</p>
            </div>
        );
    }
    
    // For real web apps, show an iframe. For desktop apps, show a placeholder.
    if (appUrl) {
        return (
             <iframe 
                src={appUrl}
                title={`${appName} Stream`}
                className="w-full h-full"
                style={{ border: 'none' }}
                // A strong sandbox is critical for security when embedding external sites
                sandbox="allow-scripts allow-same-origin allow-forms allow-popups allow-modals"
            />
        );
    }

    return (
        <div className="w-full h-full flex flex-col items-center justify-center text-center p-4 bg-slate-900 text-white">
            <h2 className="text-2xl font-bold">{appName} Stream</h2>
            <p className="text-slate-400 mt-2">Virtual desktop application is now active.</p>
            <p className="text-xs text-slate-500 mt-4">(This is a placeholder for a real desktop streaming technology like WebRTC)</p>
        </div>
    );
};

export default VirtualAppStream;