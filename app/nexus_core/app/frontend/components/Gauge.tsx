/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/

import React from 'react';

interface GaugeProps {
  value: number; // 0 to 100
  label: string;
  color: string;
}

const Gauge: React.FC<GaugeProps> = ({ value, label, color }) => {
  const clampedValue = Math.max(0, Math.min(100, value));
  const circumference = 2 * Math.PI * 45; // r = 45
  const offset = circumference - (clampedValue / 100) * circumference;

  return (
    <div className="relative flex flex-col items-center justify-center">
      <svg className="w-full h-full" viewBox="0 0 100 60">
        <defs>
            <linearGradient id={`gradient-${color.replace('#', '')}`} x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor={color} stopOpacity="0.5" />
                <stop offset="100%" stopColor={color} stopOpacity="1" />
            </linearGradient>
        </defs>
        {/* Background Arc */}
        <path
          d="M 5 50 A 45 45 0 0 1 95 50"
          fill="none"
          stroke="var(--border-color)"
          strokeWidth="8"
          strokeLinecap="round"
        />
        {/* Foreground Arc */}
        <path
          d="M 5 50 A 45 45 0 0 1 95 50"
          fill="none"
          stroke={`url(#gradient-${color.replace('#', '')})`}
          strokeWidth="8"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          style={{ transition: 'stroke-dashoffset 0.5s ease-in-out' }}
        />
      </svg>
      <div className="absolute bottom-0 text-center">
        <span className="text-2xl font-bold" style={{ color: 'var(--text-color)' }}>
          {clampedValue.toFixed(1)}
        </span>
        <span className="text-lg font-bold" style={{ color: color }}>%</span>
        <p className="text-xs font-semibold" style={{ color: 'var(--text-color)', opacity: 0.7 }}>
          {label}
        </p>
      </div>
    </div>
  );
};

export default Gauge;
