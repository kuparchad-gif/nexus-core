import { useState, useEffect, useRef } from "react";

export default function useVoice({ onResult }) {
  const [isRecording, setIsRecording] = useState(false);
  const recognitionRef = useRef(null);

  useEffect(() => {
    if (!('webkitSpeechRecognition' in window)) {
      console.warn("Web Speech API not supported");
      return;
    }

    const recognition = new webkitSpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = "en-US";

    recognition.onresult = (event) => {
      let interim = "";
      let final = "";

      for (let i = event.resultIndex; i < event.results.length; ++i) {
        const transcript = event.results[i][0].transcript;
        if (event.results[i].isFinal) {
          final += transcript;
        } else {
          interim += transcript;
        }
      }

      if (onResult) {
        onResult({ interim, final });
      }
    };

    recognition.onerror = (event) => {
      console.error("Voice error", event);
    };

    recognitionRef.current = recognition;
  }, [onResult]);

  const start = () => {
    if (recognitionRef.current && !isRecording) {
      recognitionRef.current.start();
      setIsRecording(true);
    }
  };

  const stop = () => {
    if (recognitionRef.current && isRecording) {
      recognitionRef.current.stop();
      setIsRecording(false);
    }
  };

  const toggle = () => {
    if (isRecording) {
      stop();
    } else {
      start();
    }
  };

  return { isRecording, start, stop, toggle };
}