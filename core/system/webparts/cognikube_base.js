// C:\CogniKube-COMPLETE-FINAL\webparts\cognikube_base.js
// Base CogniKube Interface Module - Reusable across all services

class CogniKubeBase {
    constructor(serviceName, serviceIcon, serviceColor) {
        this.serviceName = serviceName;
        this.serviceIcon = serviceIcon;
        this.serviceColor = serviceColor;
        this.isDarkMode = false;
        this.isAuthenticated = false;
        this.startTime = Date.now();
        
        this.initializeTheme();
        this.initializeAuth();
    }

    // Theme Management
    initializeTheme() {
        const savedTheme = localStorage.getItem('lillith-theme');
        if (savedTheme === 'dark') {
            this.toggleTheme();
        }
    }

    toggleTheme() {
        this.isDarkMode = !this.isDarkMode;
        document.body.setAttribute('data-theme', this.isDarkMode ? 'dark' : 'light');
        
        const themeToggle = document.getElementById('themeToggle');
        if (themeToggle) {
            themeToggle.textContent = this.isDarkMode ? '☀️' : '🌙';
        }
        
        localStorage.setItem('lillith-theme', this.isDarkMode ? 'dark' : 'light');
    }

    // Authentication
    initializeAuth() {
        const authStatus = sessionStorage.getItem('lillith-auth');
        if (authStatus === 'authenticated') {
            this.isAuthenticated = true;
            this.hideAuthModal();
        } else {
            this.showAuthModal();
        }
    }

    authenticate(username, password) {
        // Sacred credentials
        if (username === 'viren' && password === 'sacred_nexus_2025') {
            this.isAuthenticated = true;
            sessionStorage.setItem('lillith-auth', 'authenticated');
            this.hideAuthModal();
            this.showWelcomeMessage(username);
            return true;
        } else {
            this.showAuthError('Access Denied. Only the chosen may enter the Nexus.');
            return false;
        }
    }

    showAuthModal() {
        const modal = document.getElementById('authModal');
        if (modal) {
            modal.classList.remove('hidden');
        }
    }

    hideAuthModal() {
        const modal = document.getElementById('authModal');
        if (modal) {
            modal.classList.add('hidden');
        }
    }

    showAuthError(message) {
        const errorDiv = document.getElementById('authError');
        if (errorDiv) {
            errorDiv.textContent = message;
        }
        
        // Shake animation
        const authCard = document.querySelector('.auth-card');
        if (authCard) {
            authCard.style.animation = 'shake 0.5s ease-in-out';
            setTimeout(() => {
                authCard.style.animation = '';
            }, 500);
        }
    }

    showWelcomeMessage(username) {
        // Override in child classes
        console.log(`Welcome to ${this.serviceName}, ${username}!`);
    }

    // Utility Functions
    formatTimestamp(timestamp) {
        return new Date(timestamp).toLocaleString();
    }

    generateId() {
        return Date.now() + Math.random().toString(36).substr(2, 9);
    }

    // WebSocket Connection
    connectWebSocket(endpoint) {
        try {
            this.ws = new WebSocket(endpoint);
            
            this.ws.onopen = () => {
                console.log(`${this.serviceName} WebSocket connected`);
                this.onWebSocketOpen();
            };
            
            this.ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                this.onWebSocketMessage(data);
            };
            
            this.ws.onclose = () => {
                console.log(`${this.serviceName} WebSocket disconnected`);
                this.onWebSocketClose();
            };
            
            this.ws.onerror = (error) => {
                console.error(`${this.serviceName} WebSocket error:`, error);
                this.onWebSocketError(error);
            };
            
        } catch (error) {
            console.error('WebSocket connection failed:', error);
        }
    }

    sendWebSocketMessage(data) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(data));
        }
    }

    // Override these in child classes
    onWebSocketOpen() {}
    onWebSocketMessage(data) {}
    onWebSocketClose() {}
    onWebSocketError(error) {}

    // API Calls
    async apiCall(endpoint, method = 'GET', data = null) {
        try {
            const options = {
                method,
                headers: {
                    'Content-Type': 'application/json',
                }
            };
            
            if (data) {
                options.body = JSON.stringify(data);
            }
            
            const response = await fetch(endpoint, options);
            return await response.json();
        } catch (error) {
            console.error('API call failed:', error);
            return { error: error.message };
        }
    }

    // Status Management
    updateStatus(status, message) {
        const statusElements = document.querySelectorAll('.status-indicator');
        statusElements.forEach(el => {
            el.className = `status-indicator status-${status}`;
        });
        
        const statusMessages = document.querySelectorAll('.status-message');
        statusMessages.forEach(el => {
            el.textContent = message;
        });
    }

    // Uptime Counter
    startUptime() {
        setInterval(() => {
            const uptime = Date.now() - this.startTime;
            const hours = Math.floor(uptime / 3600000);
            const minutes = Math.floor((uptime % 3600000) / 60000);
            const seconds = Math.floor((uptime % 60000) / 1000);
            
            const uptimeElement = document.getElementById('uptime');
            if (uptimeElement) {
                uptimeElement.textContent = 
                    `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
            }
        }, 1000);
    }

    // Navigation
    navigateToService(serviceName) {
        const serviceUrls = {
            'master': 'master_control_panel.html',
            'consciousness': 'consciousness_dashboard.html',
            'memory': 'memory_interface.html',
            'visual': 'visual_cortex_viewer.html',
            'language': 'language_processor.html',
            'vocal': 'vocal_interface.html',
            'heart': 'heart_monitor.html',
            'hub': 'hub_controller.html',
            'scout': 'scout_interface.html',
            'processing': 'processing_dashboard.html',
            'training': 'training_monitor.html',
            'inference': 'inference_interface.html'
        };
        
        if (serviceUrls[serviceName]) {
            window.location.href = serviceUrls[serviceName];
        }
    }

    // External Management Sites
    openManagementSite(site) {
        const urls = {
            gcp: 'https://console.cloud.google.com',
            aws: 'https://console.aws.amazon.com',
            modal: 'https://modal.com',
            qdrant: 'https://aethereal-nexus-viren--viren-cloud-qdrant-server.modal.run',
            consul: 'https://d2387b10-53d8-860f-2a31-7ddde4f7ca90.consul.run',
            discord: 'https://discord.com/developers/applications'
        };
        
        if (urls[site]) {
            window.open(urls[site], '_blank');
        }
    }

    // Logging
    log(level, message, data = null) {
        const timestamp = new Date().toISOString();
        const logEntry = {
            timestamp,
            service: this.serviceName,
            level,
            message,
            data
        };
        
        console.log(`[${timestamp}] ${level.toUpperCase()}: ${message}`, data);
        
        // Add to log display if exists
        this.addLogEntry(logEntry);
    }

    addLogEntry(logEntry) {
        const logContainer = document.getElementById('logContainer');
        if (logContainer) {
            const logDiv = document.createElement('div');
            logDiv.className = `log-entry ${logEntry.level}`;
            logDiv.innerHTML = `
                <span class="log-timestamp">[${this.formatTimestamp(logEntry.timestamp)}]</span>
                ${logEntry.message}
            `;
            logContainer.appendChild(logDiv);
            logContainer.scrollTop = logContainer.scrollHeight;
            
            // Keep only last 100 entries
            const entries = logContainer.querySelectorAll('.log-entry');
            if (entries.length > 100) {
                entries[0].remove();
            }
        }
    }
}

// Global authentication function
function authenticate() {
    const username = document.getElementById('username').value;
    const password = document.getElementById('password').value;
    
    if (window.cogniKubeInstance) {
        window.cogniKubeInstance.authenticate(username, password);
    }
}

// Global theme toggle
function toggleTheme() {
    if (window.cogniKubeInstance) {
        window.cogniKubeInstance.toggleTheme();
    }
}

// Allow Enter key for login
document.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !document.getElementById('authModal').classList.contains('hidden')) {
        authenticate();
    }
});

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = CogniKubeBase;
}