/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/

import React, { useState, useRef, useCallback } from 'react';

interface KnobProps {
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
  size?: number;
}

const Knob: React.FC<KnobProps> = ({
  value,
  onChange,
  min = 0,
  max = 1,
  step = 0.01,
  size = 60,
}) => {
  const knobRef = useRef<HTMLDivElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const angle = ((value - min) / (max - min)) * 270 - 135;

  const handleMouseMove = useCallback((e: MouseEvent) => {
    if (!isDragging || !knobRef.current) return;
    const rect = knobRef.current.getBoundingClientRect();
    const centerX = rect.left + rect.width / 2;
    const centerY = rect.top + rect.height / 2;
    const angleRad = Math.atan2(e.clientY - centerY, e.clientX - centerX);
    let angleDeg = angleRad * (180 / Math.PI) + 90;
    if (angleDeg < 0) angleDeg += 360;

    if (angleDeg > 225 && angleDeg < 315) return; // Dead zone at the bottom
    
    let currentAngle = angleDeg - 45;
    if (currentAngle < 0) currentAngle += 360;
    
    let newValue = (currentAngle / 270) * (max - min) + min;
    newValue = Math.round(newValue / step) * step;
    newValue = Math.max(min, Math.min(max, newValue));
    onChange(newValue);
  }, [isDragging, min, max, step, onChange]);

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
    document.removeEventListener('mousemove', handleMouseMove);
    document.removeEventListener('mouseup', handleMouseUp);
  }, [handleMouseMove]);

  const handleMouseDown = (e: React.MouseEvent) => {
    e.preventDefault();
    setIsDragging(true);
    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
  };
  
  const tickCount = 11;
  const ticks = Array.from({length: tickCount}, (_, i) => {
      const tickAngle = (i / (tickCount-1)) * 270 - 135;
      return <div key={i} className="absolute w-0.5 h-1.5 bg-[var(--border-color)] origin-bottom"
          style={{
              height: i % 2 === 0 ? '8px' : '4px',
              transform: `translate(-50%, -${size/2}px) rotate(${tickAngle}deg)`,
              top: '50%',
              left: '50%'
          }}
      />
  });

  return (
    <div
      ref={knobRef}
      onMouseDown={handleMouseDown}
      className="relative rounded-full cursor-pointer select-none"
      style={{ width: size, height: size, touchAction: 'none' }}
    >
      <div className="absolute inset-0 rounded-full bg-slate-200 shadow-inner" />
      {ticks}
      <div
        className="absolute top-1/2 left-1/2 w-full h-full origin-center"
        style={{ transform: `translate(-50%, -50%) rotate(${angle}deg)` }}
      >
        <div 
          className="absolute h-1 w-1/2 bg-[var(--primary-color)] rounded-l-full"
          style={{ top: 'calc(50% - 2px)', left: 0 }}
        />
      </div>
       <div className="absolute inset-2 rounded-full bg-slate-100" />
    </div>
  );
};

export default Knob;
