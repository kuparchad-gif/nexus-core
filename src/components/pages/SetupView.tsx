
import React, { useState } from 'react';
import { AILogoIcon, Cog6ToothIcon, CommandLineIcon, CpuChipIcon, RadarIcon, WifiIcon, DatabaseIcon, SatelliteIcon } from '../icons';
import { RealNexusAdapter } from '../../services/RealNexusAdapter';

interface SetupViewProps {
  onConfigure: () => void;
  onBoot: () => void;
  onTerminal: () => void;
}

const SetupView: React.FC<SetupViewProps> = ({ onConfigure, onBoot, onTerminal }) => {
  const [isScanning, setIsScanning] = useState(false);
  const [scanStatus, setScanStatus] = useState('');
  const [signalLocked, setSignalLocked] = useState(false);
  const [showQdrantInput, setShowQdrantInput] = useState(false);
  const [qdrantUrl, setQdrantUrl] = useState('');
  const [qdrantKey, setQdrantKey] = useState('');

  const handleSignalTrace = async () => {
      setIsScanning(true);
      setSignalLocked(false);
      setScanStatus('Initializing Scan...');
      
      const foundUrl = await RealNexusAdapter.seekSignal((url) => {
          setScanStatus(`Pinging ${url}...`);
      });

      if (foundUrl) {
          setSignalLocked(true);
          setScanStatus(`Signal Locked: ${foundUrl}`);
          setTimeout(() => setIsScanning(false), 1500);
      } else {
          setScanStatus('No local signal found.');
          setTimeout(() => setIsScanning(false), 2000);
      }
  };

  const handleQdrantConnect = async () => {
      if(!qdrantUrl || !qdrantKey) return;
      
      setIsScanning(true);
      setScanStatus('Establishing Vector Uplink...');
      
      const foundUrl = await RealNexusAdapter.connectViaQdrant(qdrantUrl, qdrantKey);
      
      if (foundUrl) {
          setSignalLocked(true);
          setScanStatus(`Uplink Established: ${foundUrl}`);
          setShowQdrantInput(false);
          setTimeout(() => setIsScanning(false), 1500);
      } else {
          setScanStatus('Uplink Failed: No active beacon found.');
          setTimeout(() => setIsScanning(false), 2000);
      }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen text-center p-4 relative overflow-hidden">
       {/* Background Ambience */}
       <div className="absolute inset-0 z-0 pointer-events-none">
          <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-brand-primary/10 rounded-full blur-3xl animate-pulse"></div>
          <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-brand-secondary/10 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '1s' }}></div>
       </div>

       <div className="bg-bg-light-card/80 glass-card border border-border-color rounded-2xl shadow-aura-lg p-8 md:p-12 max-w-2xl w-full animate-scale-in z-10 backdrop-blur-xl">
         <div className="flex justify-center mb-6 relative">
            <div className={`absolute inset-0 bg-brand-primary/20 blur-xl rounded-full transform scale-150 transition-all duration-1000 ${signalLocked ? 'bg-green-500/30' : ''}`}></div>
            <img src="https://i.imgur.com/8p8YJkP.png" alt="Aethereal Icon" className="w-20 h-20 relative z-10 drop-shadow-lg"/>
         </div>
        
        <h1 className="text-4xl font-bold text-text-primary mb-2 tracking-tight">Oz OS <span className="text-brand-primary">v1.313</span></h1>
        <p className="text-sm font-mono text-brand-secondary mb-6 tracking-widest uppercase opacity-80">Aethereal Nexus Core • System Online</p>
        
        <p className="text-lg text-text-secondary mb-8 max-w-lg mx-auto leading-relaxed">
          The mesh is active. 545 nodes engaged. <br/>
          Initialize the graphical interface or bypass directly to the shell.
        </p>
        
        <div className="flex flex-col gap-4 w-full max-w-md mx-auto">
          <button
            onClick={onBoot}
            className="group relative w-full flex items-center justify-center gap-3 bg-brand-primary text-white font-semibold px-6 py-4 rounded-xl hover:bg-brand-primary/90 transition-all shadow-aura hover:shadow-aura-lg hover:-translate-y-0.5 active:translate-y-0"
          >
            <CpuChipIcon className="w-5 h-5 opacity-90" />
            <span>Initialize Desktop Environment</span>
            <div className="absolute inset-0 rounded-xl ring-2 ring-white/20 group-hover:ring-white/40 transition-all"></div>
          </button>

          {showQdrantInput ? (
              <div className="bg-bg-light-bg border border-border-color rounded-xl p-4 animate-fade-in text-left space-y-3">
                  <div className="flex items-center gap-2 text-sm font-bold text-text-primary mb-1">
                      <SatelliteIcon className="w-4 h-4 text-purple-500"/> Vector Rendezvous Protocol
                  </div>
                  <input 
                    type="text" 
                    placeholder="Qdrant Cluster URL" 
                    value={qdrantUrl}
                    onChange={e => setQdrantUrl(e.target.value)}
                    className="w-full bg-white border border-border-color rounded-lg p-2 text-sm"
                  />
                  <input 
                    type="password" 
                    placeholder="API Key" 
                    value={qdrantKey}
                    onChange={e => setQdrantKey(e.target.value)}
                    className="w-full bg-white border border-border-color rounded-lg p-2 text-sm"
                  />
                  <div className="flex gap-2">
                      <button onClick={() => setShowQdrantInput(false)} className="flex-1 py-2 text-sm text-text-secondary hover:text-text-primary">Cancel</button>
                      <button onClick={handleQdrantConnect} disabled={isScanning} className="flex-1 bg-brand-secondary text-white py-2 rounded-lg text-sm font-semibold hover:bg-brand-secondary/90">
                          {isScanning ? 'Triangulating...' : 'Establish Uplink'}
                      </button>
                  </div>
              </div>
          ) : (
            <div className="flex gap-2">
                <button
                    onClick={handleSignalTrace}
                    disabled={isScanning}
                    className={`relative overflow-hidden flex-1 flex items-center justify-center gap-2 font-semibold px-4 py-3 rounded-xl transition-all border ${
                        signalLocked 
                        ? 'bg-green-100 border-green-300 text-green-700' 
                        : 'bg-white border-border-color text-text-primary hover:bg-slate-50'
                    }`}
                >
                    {isScanning && !showQdrantInput ? (
                        <>
                            <RadarIcon className="w-5 h-5 animate-spin text-brand-secondary" />
                            <span className="text-sm">{scanStatus || 'Scanning...'}</span>
                        </>
                    ) : signalLocked ? (
                        <>
                            <WifiIcon className="w-5 h-5" />
                            <span>{scanStatus}</span>
                        </>
                    ) : (
                        <>
                            <RadarIcon className="w-5 h-5 text-brand-secondary" />
                            <span>Trace Signal</span>
                        </>
                    )}
                </button>
                <button
                    onClick={() => setShowQdrantInput(true)}
                    className="flex items-center justify-center px-4 py-3 rounded-xl bg-white border border-border-color text-text-secondary hover:text-purple-600 hover:border-purple-200 transition-colors"
                    title="Connect via Qdrant Satellite"
                >
                    <DatabaseIcon className="w-5 h-5" />
                </button>
            </div>
          )}

          <div className="grid grid-cols-2 gap-4">
            <button
                onClick={onTerminal}
                className="flex items-center justify-center gap-2 bg-slate-800 text-slate-200 font-mono text-sm font-medium px-4 py-3 rounded-xl hover:bg-slate-700 transition-colors border border-slate-600 hover:border-slate-500 shadow-lg"
            >
                <CommandLineIcon className="w-4 h-4" />
                <span>System Shell</span>
            </button>
            
            <button
                onClick={onConfigure}
                className="flex items-center justify-center gap-2 bg-white text-text-secondary font-medium px-4 py-3 rounded-xl border border-border-color hover:bg-slate-50 hover:text-brand-primary transition-colors shadow-sm"
            >
                <Cog6ToothIcon className="w-4 h-4" />
                <span>Config</span>
            </button>
          </div>
        </div>
        
        <div className="mt-8 pt-6 border-t border-border-color/50 flex justify-between text-xs font-mono text-text-secondary opacity-60">
            <span>MEM: OK</span>
            <span>QWEN: STBY</span>
            <span>{signalLocked ? 'LINK: ACTIVE' : 'LINK: SEARCHING'}</span>
        </div>
      </div>
    </div>
  );
};

export default SetupView;
