/**
 * Portal Integration JavaScript
 * Handles API communication for all Viren HTML portals
 */

class VirenPortalAPI {
    constructor() {
        this.baseURL = window.location.origin;
        this.currentPortal = this.detectPortal();
        this.init();
    }

    detectPortal() {
        const path = window.location.pathname;
        const filename = window.location.pathname.split('/').pop();
        
        if (path.includes('dashboard') || filename.includes('dashboard')) {
            return 'dashboard';
        } else if (path.includes('orb') || filename.includes('orb')) {
            return 'orb';
        } else if (path.includes('portal') || filename.includes('portal')) {
            return 'portal';
        } else {
            return 'unknown';
        }
    }

    init() {
        console.log(`🤖 Viren Portal API initialized for: ${this.currentPortal}`);
        this.setupEventListeners();
        this.startStatusUpdates();
    }

    setupEventListeners() {
        // Enhanced sendMessage function for all portals
        window.sendMessage = () => this.sendMessage();
        
        // Voice toggle function
        window.toggleVoice = () => this.toggleVoice();
        
        // Deploy function for dashboard
        window.deployCloud = () => this.deployCloud();
        
        // Status update function
        window.updateStatus = () => this.updateStatus();
        
        // Enter key support for message input
        const messageInput = document.getElementById('message-input');
        if (messageInput) {
            messageInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    this.sendMessage();
                }
            });
        }
    }

    async sendMessage() {
        const input = document.getElementById('message-input');
        const chatbot = document.getElementById('chatbot');
        const horn = document.getElementById('gabriels-horn');
        
        if (!input || !input.value.trim()) {
            return;
        }

        const message = input.value.trim();
        const timestamp = new Date().toLocaleTimeString();
        
        // Update UI immediately
        if (chatbot) {
            chatbot.innerHTML += `<br><strong>User (${timestamp}):</strong> ${message}`;
            chatbot.innerHTML += `<br><strong>VIREN:</strong> <em>Processing...</em>`;
        }
        
        if (horn) {
            horn.innerHTML += `<br>Gabriel's Horn: Relaying "${message}"...`;
        }
        
        // Clear input
        input.value = '';
        
        try {
            const response = await fetch(`${this.baseURL}/api/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: message,
                    portal: this.currentPortal,
                    timestamp: Date.now()
                })
            });

            const data = await response.json();
            
            if (response.ok) {
                // Update chatbot with response
                if (chatbot) {
                    // Remove "Processing..." message
                    chatbot.innerHTML = chatbot.innerHTML.replace('<em>Processing...</em>', data.response);
                }
                
                if (horn) {
                    horn.innerHTML += `<br>Gabriel's Horn: Response transmitted.`;
                }
                
                // Scroll to bottom
                if (chatbot) {
                    chatbot.scrollTop = chatbot.scrollHeight;
                }
                
            } else {
                throw new Error(data.error || 'API request failed');
            }
            
        } catch (error) {
            console.error('Error sending message:', error);
            
            if (chatbot) {
                chatbot.innerHTML = chatbot.innerHTML.replace(
                    '<em>Processing...</em>', 
                    `<span style="color: #ff6b6b;">Error: ${error.message}</span>`
                );
            }
        }
    }

    toggleVoice() {
        const voiceButton = document.getElementById('voice-button');
        if (voiceButton) {
            const isActive = voiceButton.textContent.includes('On');
            voiceButton.textContent = isActive ? 'Voice' : 'Voice On';
            
            if (!isActive) {
                this.startVoiceRecognition();
            } else {
                this.stopVoiceRecognition();
            }
        }
    }

    startVoiceRecognition() {
        if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            const recognition = new SpeechRecognition();
            
            recognition.continuous = false;
            recognition.interimResults = false;
            recognition.lang = 'en-US';
            
            recognition.onstart = () => {
                console.log('🎤 Voice recognition started');
            };
            
            recognition.onresult = (event) => {
                const transcript = event.results[0][0].transcript;
                const messageInput = document.getElementById('message-input');
                if (messageInput) {
                    messageInput.value = transcript;
                    this.sendMessage();
                }
            };
            
            recognition.onerror = (event) => {
                console.error('Voice recognition error:', event.error);
                alert('Voice recognition error: ' + event.error);
            };
            
            recognition.start();
            this.recognition = recognition;
        } else {
            alert('Voice recognition not supported in this browser');
        }
    }

    stopVoiceRecognition() {
        if (this.recognition) {
            this.recognition.stop();
            this.recognition = null;
        }
    }

    async deployCloud() {
        const deployInput = document.getElementById('deploy-input');
        const deployStatus = document.getElementById('deploy-status');
        
        if (!deployInput || !deployInput.value.trim()) {
            if (deployStatus) {
                deployStatus.textContent = 'Please enter a deployment target.';
            }
            return;
        }

        const platform = deployInput.value.trim();
        
        if (deployStatus) {
            deployStatus.textContent = `Deploying to ${platform}...`;
        }

        try {
            const response = await fetch(`${this.baseURL}/api/deploy`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    platform: platform,
                    portal: this.currentPortal,
                    timestamp: Date.now()
                })
            });

            const data = await response.json();
            
            if (response.ok) {
                if (deployStatus) {
                    deployStatus.textContent = `✅ ${data.message}`;
                }
                alert(`Deployment to ${platform} initiated! ID: ${data.deployment_id}`);
            } else {
                throw new Error(data.error || 'Deployment failed');
            }
            
        } catch (error) {
            console.error('Deployment error:', error);
            if (deployStatus) {
                deployStatus.textContent = `❌ Error: ${error.message}`;
            }
            alert(`Deployment error: ${error.message}`);
        }
    }

    async updateStatus() {
        try {
            const response = await fetch(`${this.baseURL}/api/status`);
            const data = await response.json();
            
            if (response.ok) {
                this.updateStatusDisplay(data);
            } else {
                console.error('Status update failed:', data.error);
            }
            
        } catch (error) {
            console.error('Status update error:', error);
        }
    }

    updateStatusDisplay(statusData) {
        // Update timestamp
        const timestampElement = document.getElementById('timestamp');
        if (timestampElement) {
            timestampElement.textContent = new Date().toLocaleString('en-US', {
                hour: '2-digit',
                minute: '2-digit',
                hour12: true,
                timeZoneName: 'short',
                month: 'long',
                day: 'numeric',
                year: 'numeric'
            });
        }

        // Update system status in chatbot
        const chatbot = document.getElementById('chatbot');
        if (chatbot && this.currentPortal === 'dashboard') {
            const statusMessage = `<br><strong>System Status Update:</strong> CPU: ${statusData.cpu.toFixed(1)}%, Memory: ${statusData.memory.toFixed(1)}%, Disk: ${statusData.disk.toFixed(1)}%`;
            chatbot.innerHTML += statusMessage;
        }

        // Update gauges if they exist
        this.updateGauges(statusData);
    }

    updateGauges(statusData) {
        const gauges = document.querySelectorAll('.circular-gauge');
        gauges.forEach((gauge, index) => {
            let value;
            switch (index) {
                case 0:
                    value = statusData.cpu;
                    break;
                case 1:
                    value = statusData.memory;
                    break;
                case 2:
                    value = statusData.disk;
                    break;
                default:
                    value = Math.random() * 100; // Random for additional gauges
            }
            
            gauge.setAttribute('data-value', Math.round(value));
            this.animateGauge(gauge, value);
        });
    }

    animateGauge(gauge, value) {
        // Simple gauge animation
        const percentage = Math.min(100, Math.max(0, value));
        const rotation = (percentage / 100) * 360;
        
        gauge.style.background = `conic-gradient(
            ${gauge.style.borderColor || '#800080'} 0deg,
            ${gauge.style.borderColor || '#800080'} ${rotation}deg,
            transparent ${rotation}deg,
            transparent 360deg
        )`;
    }

    startStatusUpdates() {
        // Update status every 30 seconds
        setInterval(() => {
            this.updateStatus();
        }, 30000);
        
        // Initial status update
        setTimeout(() => {
            this.updateStatus();
        }, 1000);
    }

    // Utility function to show notifications
    showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 15px 20px;
            border-radius: 5px;
            color: white;
            font-weight: bold;
            z-index: 10000;
            max-width: 300px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        `;
        
        switch (type) {
            case 'success':
                notification.style.backgroundColor = '#10b981';
                break;
            case 'error':
                notification.style.backgroundColor = '#ef4444';
                break;
            case 'warning':
                notification.style.backgroundColor = '#f59e0b';
                break;
            default:
                notification.style.backgroundColor = '#3b82f6';
        }
        
        notification.textContent = message;
        document.body.appendChild(notification);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 5000);
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.virenAPI = new VirenPortalAPI();
});

// Global utility functions for backward compatibility
window.updateTimestamp = function() {
    const timestampElement = document.getElementById('timestamp');
    if (timestampElement) {
        timestampElement.textContent = new Date().toLocaleString('en-US', {
            hour: '2-digit',
            minute: '2-digit',
            hour12: true,
            timeZoneName: 'short',
            month: 'long',
            day: 'numeric',
            year: 'numeric'
        });
    }
};

// Start timestamp updates
setInterval(window.updateTimestamp, 1000);
window.updateTimestamp();