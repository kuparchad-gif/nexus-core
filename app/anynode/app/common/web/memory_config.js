// Shared JS for memory config (e.g., sliders, API calls)

function updateMemoryConfig(threshold) {
  fetch('http://localhost:8011/shard', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({emotional: true, threshold})  // Simulated config update
  }).then(res => res.json())
    .then(data => console.log('Config updated:', data))
    .catch(err => console.error('Config error:', err));
}