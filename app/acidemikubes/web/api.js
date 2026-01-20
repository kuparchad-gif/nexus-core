// Simple API client to connect Lillith's web portal to backend services

const API_BASE_URL = 'http://localhost:5000/api'; // Placeholder URL, adjust based on deployment

// Fetch system health status
async function getSystemHealth() {
  try {
    const response = await fetch(`${API_BASE_URL}/health`);
    if (!response.ok) throw new Error('Network response was not ok');
    return await response.json();
  } catch (error) {
    console.error('Error fetching system health:', error);
    return { status: 'error', message: 'Unable to connect to backend' };
  }
}

// Fetch consciousness metrics
async function getConsciousnessMetrics() {
  try {
    const response = await fetch(`${API_BASE_URL}/metrics/consciousness`);
    if (!response.ok) throw new Error('Network response was not ok');
    return await response.json();
  } catch (error) {
    console.error('Error fetching consciousness metrics:', error);
    return { status: 'error', message: 'Unable to connect to backend' };
  }
}

// Trigger system update
async function triggerUpdate() {
  try {
    const response = await fetch(`${API_BASE_URL}/update`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ version: 'latest' })
    });
    if (!response.ok) throw new Error('Network response was not ok');
    return await response.json();
  } catch (error) {
    console.error('Error triggering update:', error);
    return { status: 'error', message: 'Update failed' };
  }
}

// Placeholder for login authentication
async function login(username, password) {
  try {
    const response = await fetch(`${API_BASE_URL}/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password })
    });
    if (!response.ok) throw new Error('Network response was not ok');
    return await response.json();
  } catch (error) {
    console.error('Error during login:', error);
    return { status: 'error', message: 'Login failed' };
  }
}

export { getSystemHealth, getConsciousnessMetrics, triggerUpdate, login };
