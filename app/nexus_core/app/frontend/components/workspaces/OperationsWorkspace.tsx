/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/

import React, { useState, useEffect } from 'react';
import { Globe, Cpu, Server, Database } from 'lucide-react';
import { SegmentedControl, Paper, Title } from '@mantine/core';
import HoloGlobe from '../HoloGlobe';
import Gauge from '../Gauge';

const OperationsWorkspace: React.FC = () => {
  const [cpu, setCpu] = useState(75.3);
  const [memory, setMemory] = useState(62.1);
  const [api, setApi] = useState(98.2);

  useEffect(() => {
      const interval = setInterval(() => {
          setCpu(c => Math.max(10, Math.min(95, c + (Math.random() - 0.5) * 5)));
          setMemory(m => Math.max(20, Math.min(90, m + (Math.random() - 0.5) * 3)));
          setApi(a => Math.max(90, Math.min(100, a + (Math.random() - 0.4) * 2)));
      }, 2000);
      return () => clearInterval(interval);
  }, []);

  return (
    <div className="w-full h-full p-4 grid grid-cols-3 grid-rows-1 gap-4">
      {/* Globe */}
      <div className="col-span-2 glass-panel rounded-2xl p-4 flex flex-col">
        <Title order={3} className="flex items-center gap-2"><Globe size={24}/> Global Traffic</Title>
        <div className="flex-grow mt-4">
            <HoloGlobe />
        </div>
      </div>
      
      {/* Metrics */}
      <div className="col-span-1 flex flex-col gap-4">
        <Paper withBorder p="md" radius="lg" className="flex-grow glass-panel flex flex-col items-center justify-around">
            <Gauge value={cpu} label="CPU Utilization" color="var(--primary-color)" />
            <Gauge value={memory} label="Memory Usage" color="var(--accent-color)" />
            <Gauge value={api} label="API Health" color="#f59f0b" />
        </Paper>
      </div>
    </div>
  );
};

// Re-export HoloGlobe to be used in other files if needed, but keep it local to this workspace.
const HoloGlobe: React.FC = () => {
    const mountRef = React.useRef<HTMLDivElement>(null);

    React.useEffect(() => {
        const THREE = (window as any).THREE;
        if (!mountRef.current || !THREE) return;
        
        // This is a placeholder for the actual Three.js implementation to avoid making this file too large.
        // The logic from the original file should be placed here.
        const div = document.createElement('div');
        div.innerText = "HoloGlobe Animation";
        div.style.width = '100%';
        div.style.height = '100%';
        div.style.display = 'flex';
        div.style.alignItems = 'center';
        div.style.justifyContent = 'center';
        mountRef.current.appendChild(div);

    }, []);

    return <div ref={mountRef} className="w-full h-full" />;
};


export default OperationsWorkspace;
