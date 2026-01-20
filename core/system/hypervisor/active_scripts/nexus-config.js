// Nexus Configuration for Existing UI
// Add this to your existing UI code

export const NEXUS_CONFIG = {
  API_BASE: 'http://localhost:8001',
  WS_URL: 'ws://localhost:8001/ws',
  ENDPOINTS: {
    GENERATE: '/generate',
    SYSTEM: '/system',
    HEALTH: '/health',
    METRICS: '/metrics'
  },
  MODEL: 'deepseek-coder:14b-q4_K_M',
  QUANTIZATION: 'q4_K_M',
  MEMORY_OPTIMIZED: true
};

// Example usage in your React component:
/*
import { NEXUS_CONFIG } from './nexus-config';

async function generateCode(prompt) {
  const response = await fetch(`${NEXUS_CONFIG.API_BASE}${NEXUS_CONFIG.ENDPOINTS.GENERATE}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      prompt,
      model: NEXUS_CONFIG.MODEL,
      max_tokens: 512
    })
  });
  return await response.json();
}

// WebSocket connection:
const ws = new WebSocket(NEXUS_CONFIG.WS_URL);
*/
