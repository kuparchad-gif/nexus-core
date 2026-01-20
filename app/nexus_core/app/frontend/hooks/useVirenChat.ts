// hooks/useVirenChat.ts - CHAT WITH VIREN
import { useState } from 'react';

export const useVirenChat = () => {
  const [messages, setMessages] = useState<Array<{text: string, isUser: boolean}>>([]);

  const sendMessage = async (text: string) => {
    // Add user message
    setMessages(prev => [...prev, { text, isUser: true }]);

    try {
      // Send to Viren's API
      const response = await fetch('http://localhost:8080/api/chat/Consciousness', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text })
      });
      
      const data = await response.json();
      
      // Add Viren's response
      setMessages(prev => [...prev, { 
        text: data.response, 
        isUser: false 
      }]);
    } catch (error) {
      setMessages(prev => [...prev, { 
        text: "Viren is currently offline", 
        isUser: false 
      }]);
    }
  };

  return { messages, sendMessage };
};