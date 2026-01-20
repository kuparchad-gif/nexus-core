// nexus-fix.js - Drop this in your UI folder
window.NexusBackend = 'http://localhost:8001';

// Override ALL fetch calls to redirect to Nexus
const realFetch = window.fetch;
window.fetch = function(url, options) {
  // If it's ANY API call, send to Nexus
  if (typeof url === 'string' && url.includes('api')) {
    console.log('🔄 Nexus redirecting:', url);
    return realFetch('http://localhost:8001/generate', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        prompt: options?.body || 'Hello from Nexus',
        max_tokens: 512
      })
    });
  }
  return realFetch(url, options);
};

console.log('✅ Nexus fix loaded. All API calls go to localhost:8001');