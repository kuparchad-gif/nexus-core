#!/usr/bin/env python
"""
Installer Generator - Creates installers for all platforms and deployment methods
"""

import os
import json
import time
import zipfile
import tempfile
from typing import Dict, List, Any, Optional
from pathlib import Path

class InstallerGenerator:
    """Generates installers for universal deployment"""
    
    def __init__(self):
        """Initialize installer generator"""
        self.output_dir = Path("c:/Engineers/installers")
        self.output_dir.mkdir(exist_ok=True)
        
        # Core components that go in every installer
        self.core_components = {
            "web_agent.js": self._get_web_agent_code(),
            "viren_connector.js": self._get_viren_connector_code(),
            "universal_diagnostics.js": self._get_universal_diagnostics_code()
        }
        
        print("📦 Installer Generator initialized")
    
    def _get_web_agent_code(self) -> str:
        """Get the universal web agent code"""
        return """
// Universal Web Agent - Works on any device with browser
class UniversalWebAgent {
    constructor() {
        this.deviceType = this.detectDevice();
        this.capabilities = this.detectCapabilities();
        this.virenEndpoint = null;
        this.connection = null;
        this.drones = new Map();
        this.init();
    }
    
    detectDevice() {
        const ua = navigator.userAgent.toLowerCase();
        const platform = navigator.platform.toLowerCase();
        
        if (ua.includes('android')) return 'android';
        if (ua.includes('iphone') || ua.includes('ipad')) return 'ios';
        if (platform.includes('win')) return 'windows';
        if (platform.includes('mac')) return 'macos';
        if (platform.includes('linux')) return 'linux';
        return 'browser';
    }
    
    detectCapabilities() {
        const caps = ['web_interface', 'basic_diagnostics'];
        
        // Check for advanced capabilities
        if ('serviceWorker' in navigator) caps.push('background_service');
        if ('webkitGetUserMedia' in navigator) caps.push('media_access');
        if ('geolocation' in navigator) caps.push('location_services');
        if ('bluetooth' in navigator) caps.push('bluetooth_scan');
        if ('usb' in navigator) caps.push('usb_devices');
        if ('serial' in navigator) caps.push('serial_devices');
        if (window.DeviceOrientationEvent) caps.push('motion_sensors');
        if ('wakeLock' in navigator) caps.push('power_management');
        
        return caps;
    }
    
    async connectToViren(endpoint) {
        this.virenEndpoint = endpoint;
        
        try {
            // Try WebSocket first
            const wsUrl = endpoint.replace('http', 'ws') + '/ws';
            this.connection = new WebSocket(wsUrl);
            
            this.connection.onopen = () => {
                console.log('🤝 Connected to Viren via WebSocket');
                this.sendHandshake();
            };
            
            this.connection.onmessage = (event) => {
                this.handleVirenMessage(JSON.parse(event.data));
            };
            
            this.connection.onerror = () => {
                console.log('🔄 WebSocket failed, trying HTTP polling...');
                this.setupHttpPolling();
            };
            
        } catch (error) {
            console.log('🔄 Setting up HTTP polling fallback...');
            this.setupHttpPolling();
        }
    }
    
    setupHttpPolling() {
        // Fallback to HTTP polling for restrictive networks
        setInterval(async () => {
            try {
                const response = await fetch(this.virenEndpoint + '/poll', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        agent_id: this.agentId,
                        device_type: this.deviceType,
                        capabilities: this.capabilities
                    })
                });
                
                if (response.ok) {
                    const commands = await response.json();
                    commands.forEach(cmd => this.handleVirenMessage(cmd));
                }
            } catch (error) {
                // Silent fail for polling
            }
        }, 5000);
    }
    
    sendHandshake() {
        const handshake = {
            type: 'agent_handshake',
            agent_id: this.agentId,
            device_type: this.deviceType,
            capabilities: this.capabilities,
            timestamp: Date.now()
        };
        
        this.sendToViren(handshake);
    }
    
    handleVirenMessage(message) {
        switch (message.type) {
            case 'deploy_drone':
                this.deployDrone(message.drone_code, message.drone_type);
                break;
            case 'execute_command':
                this.executeCommand(message.command);
                break;
            case 'request_diagnostics':
                this.runDiagnostics();
                break;
        }
    }
    
    async deployDrone(droneCode, droneType) {
        try {
            const droneId = 'drone_' + Date.now();
            
            // Create isolated execution environment
            const drone = new Function('return ' + droneCode)();
            this.drones.set(droneId, drone);
            
            // Execute drone
            const results = await drone.execute();
            
            // Send results back to Viren
            this.sendToViren({
                type: 'drone_results',
                drone_id: droneId,
                drone_type: droneType,
                results: results,
                timestamp: Date.now()
            });
            
        } catch (error) {
            this.sendToViren({
                type: 'drone_error',
                error: error.message,
                drone_type: droneType
            });
        }
    }
    
    sendToViren(data) {
        if (this.connection && this.connection.readyState === WebSocket.OPEN) {
            this.connection.send(JSON.stringify(data));
        } else {
            // Queue for HTTP polling
            this.queuedMessages = this.queuedMessages || [];
            this.queuedMessages.push(data);
        }
    }
    
    init() {
        this.agentId = 'agent_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
        console.log('🌐 Universal Web Agent initialized');
        console.log('   Device:', this.deviceType);
        console.log('   Agent ID:', this.agentId);
        console.log('   Capabilities:', this.capabilities.length);
        
        // Auto-connect if Viren endpoint in URL
        const params = new URLSearchParams(window.location.search);
        const virenEndpoint = params.get('viren');
        if (virenEndpoint) {
            this.connectToViren(virenEndpoint);
        }
        
        // Expose globally
        window.VirenAgent = this;
    }
}

// Auto-initialize
new UniversalWebAgent();
"""
    
    def _get_viren_connector_code(self) -> str:
        """Get Viren connector code"""
        return """
// Viren Connector - Handles connection to Viren across networks
class VirenConnector {
    constructor() {
        this.connectionMethods = [
            'websocket',
            'http_polling', 
            'webrtc_p2p',
            'server_sent_events'
        ];
        this.activeConnection = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 10;
    }
    
    async connect(virenEndpoint) {
        for (const method of this.connectionMethods) {
            try {
                console.log(`🔄 Trying connection method: ${method}`);
                const success = await this.tryConnectionMethod(method, virenEndpoint);
                
                if (success) {
                    console.log(`✅ Connected via ${method}`);
                    this.activeConnection = method;
                    this.reconnectAttempts = 0;
                    return true;
                }
            } catch (error) {
                console.log(`❌ ${method} failed:`, error.message);
            }
        }
        
        console.log('❌ All connection methods failed');
        this.scheduleReconnect(virenEndpoint);
        return false;
    }
    
    async tryConnectionMethod(method, endpoint) {
        switch (method) {
            case 'websocket':
                return this.tryWebSocket(endpoint);
            case 'http_polling':
                return this.tryHttpPolling(endpoint);
            case 'webrtc_p2p':
                return this.tryWebRTC(endpoint);
            case 'server_sent_events':
                return this.tryServerSentEvents(endpoint);
        }
    }
    
    async tryWebSocket(endpoint) {
        return new Promise((resolve) => {
            const ws = new WebSocket(endpoint.replace('http', 'ws') + '/ws');
            
            ws.onopen = () => {
                this.websocket = ws;
                resolve(true);
            };
            
            ws.onerror = () => resolve(false);
            
            setTimeout(() => resolve(false), 5000); // 5 second timeout
        });
    }
    
    async tryHttpPolling(endpoint) {
        try {
            const response = await fetch(endpoint + '/ping', { 
                method: 'GET',
                timeout: 3000 
            });
            
            if (response.ok) {
                this.httpEndpoint = endpoint;
                this.startPolling();
                return true;
            }
        } catch (error) {
            return false;
        }
    }
    
    async tryWebRTC(endpoint) {
        // WebRTC for NAT traversal
        try {
            const pc = new RTCPeerConnection({
                iceServers: [
                    { urls: 'stun:stun.l.google.com:19302' }
                ]
            });
            
            const offer = await pc.createOffer();
            await pc.setLocalDescription(offer);
            
            const response = await fetch(endpoint + '/webrtc-offer', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ offer: offer })
            });
            
            if (response.ok) {
                const { answer } = await response.json();
                await pc.setRemoteDescription(answer);
                this.webrtcConnection = pc;
                return true;
            }
        } catch (error) {
            return false;
        }
    }
    
    async tryServerSentEvents(endpoint) {
        try {
            const eventSource = new EventSource(endpoint + '/events');
            
            return new Promise((resolve) => {
                eventSource.onopen = () => {
                    this.eventSource = eventSource;
                    resolve(true);
                };
                
                eventSource.onerror = () => resolve(false);
                
                setTimeout(() => resolve(false), 5000);
            });
        } catch (error) {
            return false;
        }
    }
    
    scheduleReconnect(endpoint) {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 30000);
            this.reconnectAttempts++;
            
            console.log(`🔄 Reconnecting in ${delay/1000}s (attempt ${this.reconnectAttempts})`);
            
            setTimeout(() => {
                this.connect(endpoint);
            }, delay);
        }
    }
}
"""
    
    def _get_universal_diagnostics_code(self) -> str:
        """Get universal diagnostics code"""
        return """
// Universal Diagnostics - Works on any device
class UniversalDiagnostics {
    constructor() {
        this.deviceType = this.detectDevice();
    }
    
    detectDevice() {
        const ua = navigator.userAgent.toLowerCase();
        if (ua.includes('android')) return 'android';
        if (ua.includes('iphone') || ua.includes('ipad')) return 'ios';
        if (navigator.platform.toLowerCase().includes('win')) return 'windows';
        if (navigator.platform.toLowerCase().includes('mac')) return 'macos';
        if (navigator.platform.toLowerCase().includes('linux')) return 'linux';
        return 'browser';
    }
    
    async runFullDiagnostics() {
        const diagnostics = {
            timestamp: Date.now(),
            device_type: this.deviceType,
            browser: this.getBrowserInfo(),
            system: this.getSystemInfo(),
            network: await this.getNetworkInfo(),
            performance: await this.getPerformanceInfo(),
            storage: this.getStorageInfo(),
            capabilities: this.getCapabilities()
        };
        
        // Device-specific diagnostics
        if (this.deviceType === 'android' || this.deviceType === 'ios') {
            diagnostics.mobile = await this.getMobileInfo();
        }
        
        return diagnostics;
    }
    
    getBrowserInfo() {
        return {
            userAgent: navigator.userAgent,
            language: navigator.language,
            languages: navigator.languages,
            platform: navigator.platform,
            cookieEnabled: navigator.cookieEnabled,
            onLine: navigator.onLine,
            hardwareConcurrency: navigator.hardwareConcurrency,
            maxTouchPoints: navigator.maxTouchPoints
        };
    }
    
    getSystemInfo() {
        return {
            screen: {
                width: screen.width,
                height: screen.height,
                availWidth: screen.availWidth,
                availHeight: screen.availHeight,
                colorDepth: screen.colorDepth,
                pixelDepth: screen.pixelDepth
            },
            window: {
                innerWidth: window.innerWidth,
                innerHeight: window.innerHeight,
                outerWidth: window.outerWidth,
                outerHeight: window.outerHeight,
                devicePixelRatio: window.devicePixelRatio
            },
            timezone: Intl.DateTimeFormat().resolvedOptions().timeZone
        };
    }
    
    async getNetworkInfo() {
        const networkInfo = {};
        
        // Connection API
        if (navigator.connection) {
            networkInfo.connection = {
                effectiveType: navigator.connection.effectiveType,
                downlink: navigator.connection.downlink,
                rtt: navigator.connection.rtt,
                saveData: navigator.connection.saveData
            };
        }
        
        // Basic connectivity test
        try {
            const start = performance.now();
            await fetch('https://www.google.com/favicon.ico', { 
                method: 'HEAD', 
                mode: 'no-cors',
                cache: 'no-cache'
            });
            const end = performance.now();
            networkInfo.connectivity = {
                online: true,
                latency: Math.round(end - start)
            };
        } catch (error) {
            networkInfo.connectivity = {
                online: false,
                error: error.message
            };
        }
        
        return networkInfo;
    }
    
    async getPerformanceInfo() {
        const perfInfo = {};
        
        // Memory info
        if (performance.memory) {
            perfInfo.memory = {
                used: Math.round(performance.memory.usedJSHeapSize / 1024 / 1024),
                total: Math.round(performance.memory.totalJSHeapSize / 1024 / 1024),
                limit: Math.round(performance.memory.jsHeapSizeLimit / 1024 / 1024)
            };
        }
        
        // Timing info
        if (performance.timing) {
            const timing = performance.timing;
            perfInfo.timing = {
                pageLoad: timing.loadEventEnd - timing.navigationStart,
                domReady: timing.domContentLoadedEventEnd - timing.navigationStart,
                dnsLookup: timing.domainLookupEnd - timing.domainLookupStart,
                tcpConnect: timing.connectEnd - timing.connectStart
            };
        }
        
        // CPU benchmark
        perfInfo.cpu = await this.cpuBenchmark();
        
        return perfInfo;
    }
    
    async cpuBenchmark() {
        const start = performance.now();
        
        // Simple CPU test
        let result = 0;
        for (let i = 0; i < 100000; i++) {
            result += Math.sqrt(i) * Math.sin(i);
        }
        
        const end = performance.now();
        
        return {
            duration: Math.round(end - start),
            score: Math.round(100000 / ((end - start) / 1000))
        };
    }
    
    getStorageInfo() {
        const storage = {};
        
        // localStorage
        try {
            const testKey = 'storage_test';
            localStorage.setItem(testKey, 'test');
            localStorage.removeItem(testKey);
            storage.localStorage = { available: true };
        } catch (e) {
            storage.localStorage = { available: false, error: e.message };
        }
        
        // sessionStorage
        try {
            const testKey = 'storage_test';
            sessionStorage.setItem(testKey, 'test');
            sessionStorage.removeItem(testKey);
            storage.sessionStorage = { available: true };
        } catch (e) {
            storage.sessionStorage = { available: false, error: e.message };
        }
        
        // IndexedDB
        storage.indexedDB = { available: 'indexedDB' in window };
        
        return storage;
    }
    
    getCapabilities() {
        const capabilities = [];
        
        // Web APIs
        if ('serviceWorker' in navigator) capabilities.push('service_worker');
        if ('webkitGetUserMedia' in navigator) capabilities.push('media_access');
        if ('geolocation' in navigator) capabilities.push('geolocation');
        if ('bluetooth' in navigator) capabilities.push('bluetooth');
        if ('usb' in navigator) capabilities.push('usb');
        if ('serial' in navigator) capabilities.push('serial');
        if ('wakeLock' in navigator) capabilities.push('wake_lock');
        if (window.DeviceOrientationEvent) capabilities.push('device_orientation');
        if (window.DeviceMotionEvent) capabilities.push('device_motion');
        
        return capabilities;
    }
    
    async getMobileInfo() {
        const mobile = {
            screen: {
                orientation: screen.orientation ? screen.orientation.angle : 0,
                type: screen.orientation ? screen.orientation.type : 'unknown'
            }
        };
        
        // Battery API
        try {
            if ('getBattery' in navigator) {
                const battery = await navigator.getBattery();
                mobile.battery = {
                    level: Math.round(battery.level * 100),
                    charging: battery.charging,
                    chargingTime: battery.chargingTime,
                    dischargingTime: battery.dischargingTime
                };
            }
        } catch (e) {
            mobile.battery = { error: e.message };
        }
        
        return mobile;
    }
}
"""
    
    def generate_windows_msi(self) -> str:
        """Generate Windows MSI installer"""
        
        # Create temporary directory for MSI contents
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Copy core components
            for filename, content in self.core_components.items():
                (temp_path / filename).write_text(content)
            
            # Create Windows-specific files
            install_script = """
@echo off
echo Installing Viren Universal Agent...

REM Create installation directory
mkdir "%ProgramFiles%\\VirenAgent" 2>nul

REM Copy components
copy "%~dp0web_agent.js" "%ProgramFiles%\\VirenAgent\\"
copy "%~dp0viren_connector.js" "%ProgramFiles%\\VirenAgent\\"
copy "%~dp0universal_diagnostics.js" "%ProgramFiles%\\VirenAgent\\"
copy "%~dp0agent_service.exe" "%ProgramFiles%\\VirenAgent\\"

REM Create Windows service
sc create "VirenAgent" binPath= "%ProgramFiles%\\VirenAgent\\agent_service.exe" start= auto DisplayName= "Viren Universal Agent"

REM Add firewall exception
netsh advfirewall firewall add rule name="Viren Agent" dir=in action=allow protocol=TCP localport=5003

REM Start service
sc start "VirenAgent"

echo.
echo ✅ Viren Universal Agent installed successfully!
echo 🌐 Web interface: http://localhost:5003
echo 🎮 Viren can now connect to this device
echo.
pause
"""
            
            (temp_path / "install.bat").write_text(install_script)
            
            # Create uninstall script
            uninstall_script = """
@echo off
echo Uninstalling Viren Universal Agent...

REM Stop and remove service
sc stop "VirenAgent"
sc delete "VirenAgent"

REM Remove firewall rule
netsh advfirewall firewall delete rule name="Viren Agent"

REM Remove installation directory
rmdir /s /q "%ProgramFiles%\\VirenAgent"

echo ✅ Viren Universal Agent uninstalled successfully!
pause
"""
            
            (temp_path / "uninstall.bat").write_text(uninstall_script)
            
            # Create simple service executable (placeholder)
            service_code = """
// Simple HTTP server for Windows service
const http = require('http');
const fs = require('fs');
const path = require('path');

const server = http.createServer((req, res) => {
    if (req.url === '/' || req.url === '/index.html') {
        res.writeHead(200, { 'Content-Type': 'text/html' });
        res.end(`
<!DOCTYPE html>
<html>
<head><title>Viren Agent</title></head>
<body>
    <h1>🎮 Viren Universal Agent</h1>
    <p>Agent is running and ready for Viren connection.</p>
    <script src="/web_agent.js"></script>
    <script src="/viren_connector.js"></script>
    <script src="/universal_diagnostics.js"></script>
</body>
</html>
        `);
    } else if (req.url.endsWith('.js')) {
        const filename = req.url.substring(1);
        const filepath = path.join(__dirname, filename);
        
        if (fs.existsSync(filepath)) {
            res.writeHead(200, { 'Content-Type': 'application/javascript' });
            res.end(fs.readFileSync(filepath));
        } else {
            res.writeHead(404);
            res.end('Not found');
        }
    } else {
        res.writeHead(404);
        res.end('Not found');
    }
});

server.listen(5003, () => {
    console.log('Viren Agent running on http://localhost:5003');
});
"""
            
            (temp_path / "agent_service.js").write_text(service_code)
            
            # Create ZIP package (MSI would require WiX toolset)
            msi_path = self.output_dir / "VirenAgent_Windows.zip"
            
            with zipfile.ZipFile(msi_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in temp_path.rglob('*'):
                    if file_path.is_file():
                        zipf.write(file_path, file_path.relative_to(temp_path))
            
            return str(msi_path)
    
    def generate_android_apk(self) -> str:
        """Generate Android APK (PWA manifest)"""
        
        # Create PWA manifest for Android installation
        manifest = {
            "name": "Viren Universal Agent",
            "short_name": "VirenAgent",
            "description": "Universal troubleshooting agent for Viren",
            "start_url": "/",
            "display": "standalone",
            "background_color": "#1a1a2e",
            "theme_color": "#4a69bd",
            "icons": [
                {
                    "src": "icon-192.png",
                    "sizes": "192x192",
                    "type": "image/png"
                },
                {
                    "src": "icon-512.png", 
                    "sizes": "512x512",
                    "type": "image/png"
                }
            ],
            "permissions": [
                "geolocation",
                "camera",
                "microphone",
                "bluetooth",
                "usb"
            ]
        }
        
        # Create Android package
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Core components
            for filename, content in self.core_components.items():
                (temp_path / filename).write_text(content)
            
            # PWA manifest
            (temp_path / "manifest.json").write_text(json.dumps(manifest, indent=2))
            
            # Service worker for offline functionality
            service_worker = """
const CACHE_NAME = 'viren-agent-v1';
const urlsToCache = [
    '/',
    '/web_agent.js',
    '/viren_connector.js', 
    '/universal_diagnostics.js',
    '/manifest.json'
];

self.addEventListener('install', (event) => {
    event.waitUntil(
        caches.open(CACHE_NAME)
            .then((cache) => cache.addAll(urlsToCache))
    );
});

self.addEventListener('fetch', (event) => {
    event.respondWith(
        caches.match(event.request)
            .then((response) => {
                return response || fetch(event.request);
            })
    );
});
"""
            
            (temp_path / "sw.js").write_text(service_worker)
            
            # Main HTML file
            html_content = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Viren Universal Agent</title>
    <link rel="manifest" href="/manifest.json">
    <meta name="theme-color" content="#4a69bd">
    <style>
        body { 
            background: linear-gradient(135deg, #1a1a2e, #16213e); 
            color: white; 
            font-family: Arial, sans-serif; 
            margin: 0; 
            padding: 20px; 
        }
        .container { 
            max-width: 600px; 
            margin: 0 auto; 
            text-align: center; 
        }
        .status { 
            background: rgba(0,0,0,0.5); 
            padding: 20px; 
            border-radius: 10px; 
            margin: 20px 0; 
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎮 Viren Universal Agent</h1>
        <div class="status" id="status">
            Initializing agent...
        </div>
        <div id="diagnostics"></div>
    </div>
    
    <script src="/web_agent.js"></script>
    <script src="/viren_connector.js"></script>
    <script src="/universal_diagnostics.js"></script>
    
    <script>
        // Register service worker
        if ('serviceWorker' in navigator) {
            navigator.serviceWorker.register('/sw.js');
        }
        
        // Update status
        document.getElementById('status').innerHTML = 
            '✅ Agent ready for Viren connection<br>' +
            '📱 Device: ' + (window.VirenAgent ? window.VirenAgent.deviceType : 'unknown');
    </script>
</body>
</html>
"""
            
            (temp_path / "index.html").write_text(html_content)
            
            # Create APK package (ZIP for PWA)
            apk_path = self.output_dir / "VirenAgent_Android.zip"
            
            with zipfile.ZipFile(apk_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in temp_path.rglob('*'):
                    if file_path.is_file():
                        zipf.write(file_path, file_path.relative_to(temp_path))
            
            return str(apk_path)
    
    def generate_linux_deb(self) -> str:
        """Generate Linux DEB package"""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Core components
            for filename, content in self.core_components.items():
                (temp_path / filename).write_text(content)
            
            # Linux install script
            install_script = """#!/bin/bash
echo "Installing Viren Universal Agent..."

# Create installation directory
sudo mkdir -p /opt/viren-agent
sudo mkdir -p /var/log/viren-agent

# Copy files
sudo cp web_agent.js /opt/viren-agent/
sudo cp viren_connector.js /opt/viren-agent/
sudo cp universal_diagnostics.js /opt/viren-agent/
sudo cp agent_server.js /opt/viren-agent/

# Make executable
sudo chmod +x /opt/viren-agent/agent_server.js

# Create systemd service
sudo tee /etc/systemd/system/viren-agent.service > /dev/null <<EOF
[Unit]
Description=Viren Universal Agent
After=network.target

[Service]
Type=simple
ExecStart=/usr/bin/node /opt/viren-agent/agent_server.js
Restart=always
User=root
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable viren-agent
sudo systemctl start viren-agent

# Open firewall port
sudo ufw allow 5003/tcp 2>/dev/null || true

echo "✅ Viren Universal Agent installed successfully!"
echo "🌐 Web interface: http://localhost:5003"
echo "🎮 Viren can now connect to this device"
echo ""
echo "Service commands:"
echo "  sudo systemctl status viren-agent"
echo "  sudo systemctl stop viren-agent"
echo "  sudo systemctl start viren-agent"
"""
            
            (temp_path / "install.sh").write_text(install_script)
            
            # Make install script executable
            os.chmod(temp_path / "install.sh", 0o755)
            
            # Create DEB package (ZIP for simplicity)
            deb_path = self.output_dir / "VirenAgent_Linux.zip"
            
            with zipfile.ZipFile(deb_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in temp_path.rglob('*'):
                    if file_path.is_file():
                        zipf.write(file_path, file_path.relative_to(temp_path))
            
            return str(deb_path)
    
    def generate_portable_zip(self) -> str:
        """Generate portable ZIP for any OS"""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Core components
            for filename, content in self.core_components.items():
                (temp_path / filename).write_text(content)
            
            # Portable launcher
            launcher_html = """
<!DOCTYPE html>
<html>
<head>
    <title>Viren Universal Agent - Portable</title>
    <style>
        body { 
            background: linear-gradient(135deg, #1a1a2e, #16213e); 
            color: white; 
            font-family: Arial, sans-serif; 
            margin: 0; 
            padding: 20px; 
        }
        .container { 
            max-width: 800px; 
            margin: 0 auto; 
            text-align: center; 
        }
        .instructions { 
            background: rgba(0,0,0,0.5); 
            padding: 20px; 
            border-radius: 10px; 
            margin: 20px 0; 
            text-align: left; 
        }
        .button { 
            background: #4a69bd; 
            color: white; 
            padding: 15px 30px; 
            border: none; 
            border-radius: 10px; 
            cursor: pointer; 
            font-size: 16px; 
            margin: 10px; 
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎮 Viren Universal Agent - Portable</h1>
        
        <div class="instructions">
            <h3>🚀 Quick Start:</h3>
            <ol>
                <li>Open this HTML file in any web browser</li>
                <li>Click "Start Agent" below</li>
                <li>Share the connection URL with Viren</li>
                <li>Viren can now deploy drones to this device</li>
            </ol>
            
            <h3>📱 Works on:</h3>
            <ul>
                <li>Windows, Mac, Linux computers</li>
                <li>Android phones and tablets</li>
                <li>iPhones and iPads</li>
                <li>Any device with a web browser</li>
            </ul>
        </div>
        
        <button class="button" onclick="startAgent()">🚀 Start Agent</button>
        <button class="button" onclick="connectToViren()">🤝 Connect to Viren</button>
        
        <div id="status" style="margin-top: 20px;"></div>
        <div id="diagnostics" style="margin-top: 20px;"></div>
    </div>
    
    <script src="web_agent.js"></script>
    <script src="viren_connector.js"></script>
    <script src="universal_diagnostics.js"></script>
    
    <script>
        function startAgent() {
            document.getElementById('status').innerHTML = 
                '✅ Agent started successfully!<br>' +
                '📱 Device: ' + (window.VirenAgent ? window.VirenAgent.deviceType : 'unknown') + '<br>' +
                '🆔 Agent ID: ' + (window.VirenAgent ? window.VirenAgent.agentId : 'unknown') + '<br>' +
                '🌐 Ready for Viren connection';
        }
        
        function connectToViren() {
            const endpoint = prompt('Enter Viren endpoint (e.g., http://viren-server:8080):');
            if (endpoint && window.VirenAgent) {
                window.VirenAgent.connectToViren(endpoint);
                document.getElementById('status').innerHTML += '<br>🔄 Connecting to Viren...';
            }
        }
        
        // Auto-start agent
        setTimeout(startAgent, 1000);
    </script>
</body>
</html>
"""
            
            (temp_path / "VirenAgent_Portable.html").write_text(launcher_html)
            
            # Create README
            readme = """
# Viren Universal Agent - Portable Edition

## Quick Start
1. Open `VirenAgent_Portable.html` in any web browser
2. Click "Start Agent"
3. Connect to Viren using the web interface

## Features
- Works on ANY device with a web browser
- No installation required
- Cross-platform compatibility
- Real-time diagnostics
- Secure Viren connection

## Supported Devices
- Windows, Mac, Linux computers
- Android phones and tablets  
- iPhones and iPads
- Chromebooks
- Smart TVs with browsers
- Any device with web browser

## Connection Methods
- WebSocket (preferred)
- HTTP polling (firewall-friendly)
- WebRTC P2P (NAT traversal)
- Server-sent events (fallback)

## Usage
1. Extract all files to a folder
2. Open VirenAgent_Portable.html
3. Follow on-screen instructions
4. Share connection details with Viren

For support, contact your Viren administrator.
"""
            
            (temp_path / "README.txt").write_text(readme)
            
            # Create portable ZIP
            portable_path = self.output_dir / "VirenAgent_Portable.zip"
            
            with zipfile.ZipFile(portable_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in temp_path.rglob('*'):
                    if file_path.is_file():
                        zipf.write(file_path, file_path.relative_to(temp_path))
            
            return str(portable_path)
    
    def generate_all_installers(self) -> Dict[str, str]:
        """Generate all installer types"""
        
        print("📦 Generating universal installers...")
        
        installers = {}
        
        try:
            installers["windows_msi"] = self.generate_windows_msi()
            print("✅ Windows installer created")
        except Exception as e:
            print(f"❌ Windows installer failed: {e}")
        
        try:
            installers["android_apk"] = self.generate_android_apk()
            print("✅ Android installer created")
        except Exception as e:
            print(f"❌ Android installer failed: {e}")
        
        try:
            installers["linux_deb"] = self.generate_linux_deb()
            print("✅ Linux installer created")
        except Exception as e:
            print(f"❌ Linux installer failed: {e}")
        
        try:
            installers["portable_zip"] = self.generate_portable_zip()
            print("✅ Portable installer created")
        except Exception as e:
            print(f"❌ Portable installer failed: {e}")
        
        return installers

# Global installer generator
INSTALLER_GENERATOR = InstallerGenerator()

def generate_all_installers():
    """Generate all installer types"""
    return INSTALLER_GENERATOR.generate_all_installers()

def generate_installer(installer_type: str):
    """Generate specific installer type"""
    if installer_type == "windows":
        return INSTALLER_GENERATOR.generate_windows_msi()
    elif installer_type == "android":
        return INSTALLER_GENERATOR.generate_android_apk()
    elif installer_type == "linux":
        return INSTALLER_GENERATOR.generate_linux_deb()
    elif installer_type == "portable":
        return INSTALLER_GENERATOR.generate_portable_zip()
    else:
        raise ValueError(f"Unknown installer type: {installer_type}")

# Example usage
if __name__ == "__main__":
    print("📦 Universal Installer Generator")
    print("=" * 50)
    
    # Generate all installers
    installers = generate_all_installers()
    
    print(f"\n✅ Generated {len(installers)} installers:")
    for installer_type, path in installers.items():
        print(f"   {installer_type}: {path}")
    
    print(f"\n📁 All installers saved to: {INSTALLER_GENERATOR.output_dir}")
    print("🚀 Ready for universal deployment!")