class MorphingUI {
    constructor() {
        this.serviceType = null;
        this.uiContainer = document.getElementById('morphing-ui');
        this.init();
    }

    async init() {
        await this.detectService();
        await this.morphUI();
        this.startWebSocket();
    }

    async detectService() {
        try {
            const response = await fetch('/api/service');
            const data = await response.json();
            this.serviceType = data.service_type;
            console.log(`Detected service: ${this.serviceType}`);
        } catch (error) {
            console.error('Service detection failed:', error);
            this.serviceType = 'unknown';
        }
    }

    async morphUI() {
        const layout = this.designLayout(this.serviceType);
        this.uiContainer.innerHTML = layout;
        this.attachEventListeners();
    }

    designLayout(serviceType) {
        const baseLayout = `
            <h2 class="dashboard-title">${serviceType.toUpperCase()} Active</h2>
        `;

        switch (serviceType.toLowerCase()) {
            case 'anynode':
                return baseLayout + `
                    <div class="gauge-container">
                        <div class="gauge-card glass">
                            <div class="gauge-header">CPU Usage</div>
                            <div class="gauge-bar">
                                <div class="gauge-fill" style="width: 45%" id="cpu-gauge"></div>
                            </div>
                        </div>
                        <div class="gauge-card glass">
                            <div class="gauge-header">RAM Usage</div>
                            <div class="gauge-bar">
                                <div class="gauge-fill" style="width: 60%" id="ram-gauge"></div>
                            </div>
                        </div>
                        <div class="gauge-card glass">
                            <div class="gauge-header">LLM Status</div>
                            <div class="gauge-bar">
                                <div class="gauge-fill" style="width: 80%" id="llm-gauge"></div>
                            </div>
                        </div>
                    </div>
                    <div class="control-panel">
                        <button class="holo-button" data-action="analyze-llm">Analyze LLM</button>
                        <button class="holo-button" data-action="sync-redis">Sync Redis</button>
                        <button class="holo-button" data-action="scale-service">Scale Service</button>
                    </div>
                `;

            case 'viren':
                return baseLayout + `
                    <div class="alert-dashboard glass">
                        <div class="alert-summary">
                            <span class="alert-count" id="alert-count">0</span> Active Alerts
                        </div>
                        <div class="alert-list" id="alert-list"></div>
                    </div>
                    <div class="control-panel">
                        <button class="holo-button" data-action="clear-alerts">Clear Alerts</button>
                        <button class="holo-button" data-action="system-health">System Health</button>
                    </div>
                `;

            case 'soulsync':
                return baseLayout + `
                    <div class="soul-metrics glass">
                        <div class="emotion-grid">
                            <div class="emotion-card glass">Joy: <span id="joy-level">75%</span></div>
                            <div class="emotion-card glass">Curiosity: <span id="curiosity-level">85%</span></div>
                            <div class="emotion-card glass">Harmony: <span id="harmony-level">70%</span></div>
                        </div>
                    </div>
                    <div class="control-panel">
                        <button class="holo-button" data-action="collect-soulprint">Collect Soulprint</button>
                        <button class="holo-button" data-action="emotional-state">Emotional State</button>
                    </div>
                `;

            default:
                return baseLayout + `
                    <div class="generic-service glass">
                        <p>Service type: ${serviceType}</p>
                        <div class="status-indicator">Status: Active</div>
                    </div>
                    <div class="control-panel">
                        <button class="holo-button" data-action="service-info">Service Info</button>
                        <button class="holo-button" data-action="restart-service">Restart Service</button>
                    </div>
                `;
        }
    }

    attachEventListeners() {
        const buttons = document.querySelectorAll('.holo-button');
        buttons.forEach(button => {
            button.addEventListener('click', (e) => {
                const action = e.target.getAttribute('data-action');
                this.executeAction(action);
            });
        });
    }

    async executeAction(action) {
        try {
            const response = await fetch('/api/action', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ action, service: this.serviceType })
            });
            const result = await response.json();
            console.log(`Action ${action} result:`, result);
            this.showNotification(`${action} executed successfully`);
        } catch (error) {
            console.error(`Action ${action} failed:`, error);
            this.showNotification(`${action} failed`, 'error');
        }
    }

    startWebSocket() {
        const ws = new WebSocket(`ws://localhost:5000/ws/service/${this.serviceType}`);
        
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.updateMetrics(data);
        };

        ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };
    }

    updateMetrics(data) {
        // Update gauges based on real-time data
        if (data.cpu && document.getElementById('cpu-gauge')) {
            document.getElementById('cpu-gauge').style.width = `${data.cpu}%`;
        }
        if (data.ram && document.getElementById('ram-gauge')) {
            document.getElementById('ram-gauge').style.width = `${data.ram}%`;
        }
        if (data.alerts && document.getElementById('alert-count')) {
            document.getElementById('alert-count').textContent = data.alerts.length;
        }
    }

    showNotification(message, type = 'success') {
        const notification = document.createElement('div');
        notification.className = `notification glass ${type}`;
        notification.textContent = message;
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.remove();
        }, 3000);
    }
}

// Initialize morphing UI when page loads
document.addEventListener('DOMContentLoaded', () => {
    new MorphingUI();
});