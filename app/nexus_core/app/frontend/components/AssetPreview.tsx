/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/

import React, { useState, useEffect } from 'react';
import { FileImage, Music, X } from 'lucide-react';

interface AssetPreviewProps {
  file: File;
  onRemove: () => void;
}

const AssetPreview: React.FC<AssetPreviewProps> = ({ file, onRemove }) => {
  const isImage = file.type.startsWith('image/');
  const isAudio = file.type.startsWith('audio/');
  const [preview, setPreview] = useState<string | null>(null);

  useEffect(() => {
    if (isImage) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result as string);
      };
      // FIX: Corrected typo in method name from readDataURL to readAsDataURL.
      reader.readAsDataURL(file);
    }
  }, [file, isImage]);

  const formatBytes = (bytes: number, decimals = 2) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
  };

  return (
    <div className="flex items-center p-2 bg-white/30 rounded-lg border border-white/40 animate-fade-in">
      <div className="flex-shrink-0 w-12 h-12 bg-white/20 rounded-md flex items-center justify-center mr-3 overflow-hidden">
        {isImage && preview ? (
          <img src={preview} alt={file.name} className="w-full h-full object-cover" />
        ) : isAudio ? (
          <Music className="w-6 h-6 text-slate-600" />
        ) : (
          <FileImage className="w-6 h-6 text-slate-600" />
        )}
      </div>
      <div className="flex-grow overflow-hidden">
        <p className="text-sm font-medium text-slate-800 truncate">{file.name}</p>
        <p className="text-xs text-slate-600">{formatBytes(file.size)}</p>
      </div>
      <button
        onClick={onRemove}
        className="ml-3 flex-shrink-0 p-1.5 text-slate-600 hover:text-red-600 hover:bg-red-500/20 rounded-full transition-colors"
        title="Remove asset"
      >
        <X size={16} />
      </button>
    </div>
  );
};

export default AssetPreview;