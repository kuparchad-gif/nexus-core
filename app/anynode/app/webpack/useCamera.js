import { useState, useRef, useCallback } from "react";

export default function useCamera() {
  const [isActive, setIsActive] = useState(false);
  const [stream, setStream] = useState(null);
  const videoRef = useRef(null);

  const startCamera = useCallback(async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { width: 1280, height: 720 },
        audio: false
      });
      
      setStream(mediaStream);
      setIsActive(true);
      
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
      }
      
      return mediaStream;
    } catch (error) {
      console.error("Camera access denied:", error);
      return null;
    }
  }, []);

  const stopCamera = useCallback(() => {
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      setStream(null);
      setIsActive(false);
      
      if (videoRef.current) {
        videoRef.current.srcObject = null;
      }
    }
  }, [stream]);

  const captureFrame = useCallback(() => {
    if (!videoRef.current) return null;
    
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;
    
    ctx.drawImage(videoRef.current, 0, 0);
    
    return {
      dataUrl: canvas.toDataURL('image/jpeg', 0.8),
      timestamp: new Date().toISOString(),
      width: canvas.width,
      height: canvas.height
    };
  }, []);

  const toggle = useCallback(() => {
    if (isActive) {
      stopCamera();
    } else {
      startCamera();
    }
  }, [isActive, startCamera, stopCamera]);

  return {
    isActive,
    videoRef,
    startCamera,
    stopCamera,
    captureFrame,
    toggle
  };
}