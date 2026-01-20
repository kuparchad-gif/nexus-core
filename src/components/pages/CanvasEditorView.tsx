

import React, { useState, useEffect, useRef } from 'react';
import { TrashIcon } from '../icons';

const CanvasEditorView: React.FC = () => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [isDrawing, setIsDrawing] = useState(false);
    const [brushColor, setBrushColor] = useState('#1e293b');
    const [brushSize, setBrushSize] = useState(5);

    const getCanvasContext = () => {
        const canvas = canvasRef.current;
        return canvas ? canvas.getContext('2d') : null;
    };

    useEffect(() => {
        const canvas = canvasRef.current;
        if (canvas) {
            const parent = canvas.parentElement;
            if (parent) {
                canvas.width = parent.clientWidth;
                canvas.height = parent.clientHeight;
            }
        }
    }, []);

    const startDrawing = (event: React.MouseEvent<HTMLCanvasElement>) => {
        const context = getCanvasContext();
        if (context) {
            const { offsetX, offsetY } = event.nativeEvent;
            context.beginPath();
            context.moveTo(offsetX, offsetY);
            setIsDrawing(true);
        }
    };

    const stopDrawing = () => {
        const context = getCanvasContext();
        if (context) {
            context.closePath();
            setIsDrawing(false);
        }
    };

    const draw = (event: React.MouseEvent<HTMLCanvasElement>) => {
        if (!isDrawing) return;
        const context = getCanvasContext();
        if (context) {
            const { offsetX, offsetY } = event.nativeEvent;
            context.lineWidth = brushSize;
            context.lineCap = 'round';
            context.strokeStyle = brushColor;
            context.lineTo(offsetX, offsetY);
            context.stroke();
        }
    };

    const clearCanvas = () => {
        const canvas = canvasRef.current;
        const context = getCanvasContext();
        if (canvas && context) {
            context.clearRect(0, 0, canvas.width, canvas.height);
        }
    };

    return (
        <div className="p-4 sm:p-6 lg:p-8 animate-scale-in h-full flex flex-col">
            <header className="mb-6 flex-shrink-0">
                <h2 className="text-3xl font-bold text-text-primary">Canvas</h2>
                <p className="text-text-secondary">A unified interface for technical drawing and conceptualization.</p>
            </header>
            <div className="flex-grow flex flex-col lg:flex-row gap-6">
                <div className="w-full lg:w-56 flex-shrink-0 bg-bg-light-card glass-card rounded-xl shadow-aura border border-border-color p-4 space-y-6">
                    <h3 className="text-lg font-semibold text-text-primary">Toolbar</h3>
                    <div>
                        <label htmlFor="brushColor" className="block text-sm font-medium text-text-secondary mb-2">Color</label>
                        <input
                            type="color"
                            id="brushColor"
                            value={brushColor}
                            onChange={(e) => setBrushColor(e.target.value)}
                            className="w-full h-10 p-1 bg-white border border-border-color rounded-lg cursor-pointer"
                        />
                    </div>
                    <div>
                        <label htmlFor="brushSize" className="block text-sm font-medium text-text-secondary mb-2">Brush Size</label>
                        <div className="flex items-center space-x-2">
                            <input
                                type="range"
                                id="brushSize"
                                min="1"
                                max="50"
                                value={brushSize}
                                onChange={(e) => setBrushSize(parseInt(e.target.value, 10))}
                                className="w-full"
                            />
                            <span className="text-sm font-semibold w-8 text-right">{brushSize}</span>
                        </div>
                    </div>
                     <button
                        onClick={clearCanvas}
                        className="w-full flex items-center justify-center space-x-2 bg-red-100 text-red-700 font-semibold px-4 py-2 rounded-lg hover:bg-red-200 transition-colors"
                    >
                        <TrashIcon className="w-5 h-5" />
                        <span>Clear Canvas</span>
                    </button>
                </div>
                <div className="flex-grow bg-white rounded-xl shadow-aura border border-border-color overflow-hidden">
                    <canvas
                        ref={canvasRef}
                        onMouseDown={startDrawing}
                        onMouseUp={stopDrawing}
                        onMouseLeave={stopDrawing}
                        onMouseMove={draw}
                        className="w-full h-full cursor-crosshair"
                    />
                </div>
            </div>
        </div>
    );
};

export default CanvasEditorView;