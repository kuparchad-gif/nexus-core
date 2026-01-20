// Add this to your existing chat interface JavaScript

// Enable sending with Enter key
document.getElementById('message').addEventListener('keydown', function(event) {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    sendMessage();
  }
});

// Bridge LLMs function
async function bridgeLLMs(sourceModel, targetModel, message) {
  const response = await fetch('/bridge', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      source: sourceModel,
      target: targetModel,
      message: message
    })
  });
  
  return await response.json();
}

// Update send function to bridge LLMs
async function sendMessage() {
  const source = document.getElementById('source').value;
  const destination = document.getElementById('destination').value;
  const message = document.getElementById('message').value;
  const priority = document.getElementById('priority').value;
  
  if (!message.trim()) return;
  
  // Get source and target models
  const sourceModel = document.getElementById('source').options[document.getElementById('source').selectedIndex].dataset.model;
  const targetValue = parseInt(destination);
  
  // Find target model
  let targetModel = null;
  document.querySelectorAll('#source option').forEach(option => {
    if (parseInt(option.dataset.value) === targetValue) {
      targetModel = option.dataset.model;
    }
  });
  
  // Bridge LLMs if both models are available
  let bridgeResult = null;
  if (sourceModel && targetModel) {
    bridgeResult = await bridgeLLMs(sourceModel, targetModel, message);
  }
  
  // Send message through network
  const response = await fetch('/send', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      source,
      destination_value: targetValue,
      content: message,
      priority,
      bridge_result: bridgeResult
    })
  });
  
  const data = await response.json();
  document.getElementById('results').textContent = JSON.stringify(data, null, 2);
  document.getElementById('message').value = '';
}