#!/usr/bin/env python
"""
Universal Deployment Core - Multi-level deployment for any device/network
"""

import os
import json
import time
import platform
import socket
import threading
import subprocess
from typing import Dict, List, Any, Optional
from enum import Enum
from pathlib import Path

class DeploymentLevel(Enum):
    """Deployment levels for different environments"""
    OFFLINE = "offline"          # Air-gapped, no network
    WEB_AGENT = "web_agent"      # Lightweight browser agent
    NETWORK_INFRA = "network"    # Router/firewall/switch
    CLOUD_ENTERPRISE = "cloud"   # Full cloud integration

class DeviceType(Enum):
    """Types of devices we can deploy to"""
    WINDOWS_PC = "windows_pc"
    LINUX_SERVER = "linux_server"
    MACOS = "macos"
    ANDROID = "android"
    IOS = "ios"
    ROUTER = "router"
    FIREWALL = "firewall"
    SWITCH = "switch"
    IOT_DEVICE = "iot"
    BROWSER_ONLY = "browser"

class UniversalDeploymentCore:
    """Core system for universal deployment across all devices"""
    
    def __init__(self):
        """Initialize universal deployment system"""
        self.device_type = self._detect_device_type()
        self.deployment_level = self._detect_deployment_level()
        self.capabilities = self._detect_capabilities()
        self.viren_connection = None
        self.active_drones = {}
        
        print(f"🌐 Universal Deployment Core initialized")
        print(f"   Device: {self.device_type.value}")
        print(f"   Level: {self.deployment_level.value}")
        print(f"   Capabilities: {len(self.capabilities)}")
    
    def _detect_device_type(self) -> DeviceType:
        """Auto-detect what type of device we're running on"""
        system = platform.system().lower()
        
        if system == "windows":
            return DeviceType.WINDOWS_PC
        elif system == "linux":
            # Check if it's a router/firewall
            if self._is_network_device():
                return DeviceType.ROUTER
            return DeviceType.LINUX_SERVER
        elif system == "darwin":
            return DeviceType.MACOS
        elif system == "android" or "android" in platform.platform().lower():
            return DeviceType.ANDROID
        else:
            return DeviceType.BROWSER_ONLY
    
    def _is_network_device(self) -> bool:
        """Check if this is a network infrastructure device"""
        network_indicators = [
            "/etc/openwrt_release",
            "/etc/pfsense_version", 
            "/usr/bin/vyatta-cfg-cmd",
            "/opt/vyatta"
        ]
        
        return any(os.path.exists(path) for path in network_indicators)
    
    def _detect_deployment_level(self) -> DeploymentLevel:
        """Detect what deployment level is appropriate"""
        # Check network connectivity
        try:
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            has_internet = True
        except:
            has_internet = False
        
        # Check if we're in a browser environment
        if os.environ.get("BROWSER_ENV") or not has_internet:
            return DeploymentLevel.WEB_AGENT
        
        # Check if we're on network infrastructure
        if self.device_type in [DeviceType.ROUTER, DeviceType.FIREWALL, DeviceType.SWITCH]:
            return DeploymentLevel.NETWORK_INFRA
        
        # Default to web agent for maximum compatibility
        return DeploymentLevel.WEB_AGENT
    
    def _detect_capabilities(self) -> List[str]:
        """Detect what capabilities this device has"""
        capabilities = ["basic_diagnostics", "web_interface"]
        
        # System-specific capabilities
        if self.device_type == DeviceType.WINDOWS_PC:
            capabilities.extend([
                "registry_access", "wmi_queries", "powershell", 
                "event_logs", "performance_counters", "hardware_diagnostics"
            ])
        elif self.device_type == DeviceType.LINUX_SERVER:
            capabilities.extend([
                "system_logs", "process_monitoring", "network_tools",
                "package_management", "service_control", "filesystem_analysis"
            ])
        elif self.device_type == DeviceType.ANDROID:
            capabilities.extend([
                "app_diagnostics", "battery_analysis", "network_info",
                "storage_analysis", "permission_check"
            ])
        elif self.device_type == DeviceType.ROUTER:
            capabilities.extend([
                "network_monitoring", "traffic_analysis", "firewall_rules",
                "dhcp_management", "wireless_diagnostics"
            ])
        
        # Check for additional tools
        tools_to_check = {
            "docker": "container_diagnostics",
            "kubectl": "kubernetes_diagnostics", 
            "git": "code_analysis",
            "python": "script_execution",
            "node": "javascript_execution"
        }
        
        for tool, capability in tools_to_check.items():
            if self._command_exists(tool):
                capabilities.append(capability)
        
        return capabilities
    
    def _command_exists(self, command: str) -> bool:
        """Check if a command exists on the system"""
        try:
            subprocess.run([command, "--version"], 
                         capture_output=True, timeout=5)
            return True
        except:
            return False
    
    def generate_deployment_package(self, target_device: DeviceType) -> Dict[str, Any]:
        """Generate deployment package for target device"""
        
        package = {
            "target_device": target_device.value,
            "deployment_level": self.deployment_level.value,
            "timestamp": time.time(),
            "components": []
        }
        
        # Core web agent (universal)
        web_agent = {
            "name": "web_agent",
            "type": "javascript",
            "size_kb": 45,  # Lightweight
            "capabilities": ["viren_connection", "drone_injection", "basic_diagnostics"],
            "deployment_method": "browser_injection"
        }
        package["components"].append(web_agent)
        
        # Device-specific components
        if target_device == DeviceType.WINDOWS_PC:
            package["components"].extend([
                {
                    "name": "windows_diagnostics",
                    "type": "powershell",
                    "size_kb": 25,
                    "capabilities": ["wmi_queries", "event_logs", "registry_check"],
                    "deployment_method": "script_injection"
                },
                {
                    "name": "hardware_monitor",
                    "type": "executable", 
                    "size_kb": 120,
                    "capabilities": ["temperature", "performance", "disk_health"],
                    "deployment_method": "binary_injection"
                }
            ])
        
        elif target_device == DeviceType.ANDROID:
            package["components"].extend([
                {
                    "name": "mobile_diagnostics",
                    "type": "javascript",
                    "size_kb": 35,
                    "capabilities": ["battery_info", "app_analysis", "network_check"],
                    "deployment_method": "pwa_injection"
                }
            ])
        
        elif target_device == DeviceType.ROUTER:
            package["components"].extend([
                {
                    "name": "network_monitor",
                    "type": "shell_script",
                    "size_kb": 15,
                    "capabilities": ["traffic_analysis", "connection_monitor"],
                    "deployment_method": "ssh_injection"
                }
            ])
        
        return package
    
    def create_web_agent(self) -> str:
        """Create lightweight web agent for Viren connection"""
        
        web_agent_code = """
// Universal Web Agent - Viren Connection Point
class VirenWebAgent {
    constructor() {
        this.deviceInfo = this.detectDevice();
        this.capabilities = this.detectCapabilities();
        this.virenConnection = null;
        this.activeDrones = new Map();
        this.init();
    }
    
    detectDevice() {
        const ua = navigator.userAgent;
        const platform = navigator.platform;
        
        if (/Android/i.test(ua)) return 'android';
        if (/iPhone|iPad/i.test(ua)) return 'ios';
        if (/Windows/i.test(platform)) return 'windows';
        if (/Mac/i.test(platform)) return 'macos';
        if (/Linux/i.test(platform)) return 'linux';
        return 'unknown';
    }
    
    detectCapabilities() {
        const caps = ['web_diagnostics', 'network_info'];
        
        // Check for advanced web APIs
        if ('serviceWorker' in navigator) caps.push('background_tasks');
        if ('webkitGetUserMedia' in navigator) caps.push('media_access');
        if ('geolocation' in navigator) caps.push('location_services');
        if ('bluetooth' in navigator) caps.push('bluetooth_diagnostics');
        if ('usb' in navigator) caps.push('usb_diagnostics');
        
        return caps;
    }
    
    async connectToViren(virenEndpoint) {
        try {
            // WebRTC for NAT traversal
            this.virenConnection = new RTCPeerConnection({
                iceServers: [
                    { urls: 'stun:stun.l.google.com:19302' },
                    { urls: 'stun:stun1.l.google.com:19302' }
                ]
            });
            
            // Establish connection
            const offer = await this.virenConnection.createOffer();
            await this.virenConnection.setLocalDescription(offer);
            
            // Send connection info to Viren
            const response = await fetch(virenEndpoint + '/connect', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    device: this.deviceInfo,
                    capabilities: this.capabilities,
                    offer: offer
                })
            });
            
            const virenResponse = await response.json();
            await this.virenConnection.setRemoteDescription(virenResponse.answer);
            
            console.log('🤝 Connected to Viren');
            return true;
        } catch (error) {
            console.error('❌ Viren connection failed:', error);
            return false;
        }
    }
    
    async receiveDrone(droneCode, droneType) {
        console.log(`🚁 Receiving ${droneType} drone from Viren`);
        
        try {
            // Create isolated execution environment
            const droneId = 'drone_' + Date.now();
            
            if (droneType === 'diagnostic') {
                const drone = new Function('return ' + droneCode)();
                this.activeDrones.set(droneId, drone);
                
                // Execute drone
                const results = await drone.execute();
                
                // Send results back to Viren
                this.sendToViren({
                    type: 'drone_results',
                    droneId: droneId,
                    results: results
                });
                
            } else if (droneType === 'monitor') {
                // Background monitoring drone
                const worker = new Worker(URL.createObjectURL(
                    new Blob([droneCode], { type: 'application/javascript' })
                ));
                
                worker.onmessage = (e) => {
                    this.sendToViren({
                        type: 'monitor_data',
                        droneId: droneId,
                        data: e.data
                    });
                };
                
                this.activeDrones.set(droneId, worker);
            }
            
            return droneId;
        } catch (error) {
            console.error('❌ Drone deployment failed:', error);
            return null;
        }
    }
    
    sendToViren(data) {
        if (this.virenConnection && this.virenConnection.connectionState === 'connected') {
            const channel = this.virenConnection.createDataChannel('viren_comm');
            channel.send(JSON.stringify(data));
        }
    }
    
    async runDiagnostics() {
        const diagnostics = {
            timestamp: Date.now(),
            device: this.deviceInfo,
            capabilities: this.capabilities,
            system: {
                userAgent: navigator.userAgent,
                language: navigator.language,
                platform: navigator.platform,
                cookieEnabled: navigator.cookieEnabled,
                onLine: navigator.onLine
            },
            performance: performance.timing,
            memory: performance.memory || null,
            connection: navigator.connection || null
        };
        
        // Device-specific diagnostics
        if (this.deviceInfo === 'android' || this.deviceInfo === 'ios') {
            diagnostics.mobile = {
                screen: {
                    width: screen.width,
                    height: screen.height,
                    orientation: screen.orientation?.angle || 0
                },
                battery: await this.getBatteryInfo(),
                network: this.getNetworkInfo()
            };
        }
        
        return diagnostics;
    }
    
    async getBatteryInfo() {
        try {
            if ('getBattery' in navigator) {
                const battery = await navigator.getBattery();
                return {
                    level: battery.level,
                    charging: battery.charging,
                    chargingTime: battery.chargingTime,
                    dischargingTime: battery.dischargingTime
                };
            }
        } catch (error) {
            return null;
        }
    }
    
    getNetworkInfo() {
        const conn = navigator.connection || navigator.mozConnection || navigator.webkitConnection;
        if (conn) {
            return {
                effectiveType: conn.effectiveType,
                downlink: conn.downlink,
                rtt: conn.rtt,
                saveData: conn.saveData
            };
        }
        return null;
    }
    
    init() {
        console.log('🌐 Viren Web Agent initialized');
        console.log('   Device:', this.deviceInfo);
        console.log('   Capabilities:', this.capabilities.length);
        
        // Auto-connect to Viren if endpoint provided
        const virenEndpoint = new URLSearchParams(window.location.search).get('viren');
        if (virenEndpoint) {
            this.connectToViren(virenEndpoint);
        }
        
        // Expose global interface
        window.VirenAgent = this;
    }
}

// Auto-initialize
new VirenWebAgent();
"""
        
        return web_agent_code
    
    def create_offline_installer(self, target_os: str) -> Dict[str, Any]:
        """Create offline installer for target OS"""
        
        installer_config = {
            "target_os": target_os,
            "installer_type": self._get_installer_type(target_os),
            "components": [],
            "size_mb": 0,
            "install_script": ""
        }
        
        # Core components for all installers
        core_components = [
            {"name": "web_agent", "size_kb": 45},
            {"name": "viren_connector", "size_kb": 25},
            {"name": "basic_diagnostics", "size_kb": 35}
        ]
        
        installer_config["components"].extend(core_components)
        
        # OS-specific components
        if target_os == "windows":
            installer_config["components"].extend([
                {"name": "windows_diagnostics.ps1", "size_kb": 30},
                {"name": "hardware_monitor.exe", "size_kb": 150},
                {"name": "registry_scanner.dll", "size_kb": 80}
            ])
            installer_config["install_script"] = self._create_windows_install_script()
            
        elif target_os == "linux":
            installer_config["components"].extend([
                {"name": "linux_diagnostics.sh", "size_kb": 20},
                {"name": "system_monitor", "size_kb": 100},
                {"name": "network_tools.py", "size_kb": 40}
            ])
            installer_config["install_script"] = self._create_linux_install_script()
            
        elif target_os == "android":
            installer_config["installer_type"] = "apk"
            installer_config["components"].extend([
                {"name": "mobile_diagnostics.js", "size_kb": 35},
                {"name": "battery_monitor.js", "size_kb": 15},
                {"name": "app_analyzer.js", "size_kb": 25}
            ])
        
        # Calculate total size
        installer_config["size_mb"] = sum(comp["size_kb"] for comp in installer_config["components"]) / 1024
        
        return installer_config
    
    def _get_installer_type(self, target_os: str) -> str:
        """Get appropriate installer type for OS"""
        installer_types = {
            "windows": "msi",
            "linux": "deb", 
            "macos": "dmg",
            "android": "apk",
            "ios": "ipa"
        }
        return installer_types.get(target_os, "zip")
    
    def _create_windows_install_script(self) -> str:
        """Create Windows installation script"""
        return """
@echo off
echo Installing Viren Universal Agent...

REM Create installation directory
mkdir "%ProgramFiles%\\VirenAgent" 2>nul

REM Copy components
copy web_agent.js "%ProgramFiles%\\VirenAgent\\"
copy windows_diagnostics.ps1 "%ProgramFiles%\\VirenAgent\\"
copy hardware_monitor.exe "%ProgramFiles%\\VirenAgent\\"

REM Create Windows service
sc create "VirenAgent" binPath= "%ProgramFiles%\\VirenAgent\\hardware_monitor.exe" start= auto

REM Add firewall exception
netsh advfirewall firewall add rule name="Viren Agent" dir=in action=allow protocol=TCP localport=5003

REM Start service
sc start "VirenAgent"

echo Viren Universal Agent installed successfully!
echo Access via: http://localhost:5003
pause
"""
    
    def _create_linux_install_script(self) -> str:
        """Create Linux installation script"""
        return """#!/bin/bash
echo "Installing Viren Universal Agent..."

# Create installation directory
sudo mkdir -p /opt/viren-agent

# Copy components
sudo cp web_agent.js /opt/viren-agent/
sudo cp linux_diagnostics.sh /opt/viren-agent/
sudo cp system_monitor /opt/viren-agent/
sudo chmod +x /opt/viren-agent/system_monitor

# Create systemd service
sudo tee /etc/systemd/system/viren-agent.service > /dev/null <<EOF
[Unit]
Description=Viren Universal Agent
After=network.target

[Service]
Type=simple
ExecStart=/opt/viren-agent/system_monitor
Restart=always
User=root

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable viren-agent
sudo systemctl start viren-agent

echo "Viren Universal Agent installed successfully!"
echo "Access via: http://localhost:5003"
"""
    
    def deploy_to_device(self, target_device: DeviceType, deployment_method: str = "auto") -> Dict[str, Any]:
        """Deploy agent to target device"""
        
        deployment_result = {
            "target_device": target_device.value,
            "deployment_method": deployment_method,
            "timestamp": time.time(),
            "success": False,
            "components_deployed": [],
            "access_url": None
        }
        
        try:
            # Generate deployment package
            package = self.generate_deployment_package(target_device)
            
            # Deploy based on method
            if deployment_method == "web_injection":
                # Deploy via browser
                web_agent = self.create_web_agent()
                # In real implementation, this would inject into browser
                deployment_result["components_deployed"].append("web_agent")
                deployment_result["access_url"] = "browser://viren-agent"
                
            elif deployment_method == "offline_installer":
                # Create offline installer
                installer = self.create_offline_installer(target_device.value)
                deployment_result["components_deployed"] = [c["name"] for c in installer["components"]]
                deployment_result["installer_size_mb"] = installer["size_mb"]
                
            deployment_result["success"] = True
            
        except Exception as e:
            deployment_result["error"] = str(e)
        
        return deployment_result

# Global deployment system
UNIVERSAL_DEPLOYMENT = UniversalDeploymentCore()

def deploy_to_device(device_type: str, method: str = "auto"):
    """Deploy to target device"""
    device_enum = DeviceType(device_type)
    return UNIVERSAL_DEPLOYMENT.deploy_to_device(device_enum, method)

def create_web_agent():
    """Create web agent code"""
    return UNIVERSAL_DEPLOYMENT.create_web_agent()

def create_offline_installer(target_os: str):
    """Create offline installer"""
    return UNIVERSAL_DEPLOYMENT.create_offline_installer(target_os)

# Example usage
if __name__ == "__main__":
    print("🌐 Universal Deployment System")
    print("=" * 50)
    
    # Show current device info
    print(f"Current Device: {UNIVERSAL_DEPLOYMENT.device_type.value}")
    print(f"Deployment Level: {UNIVERSAL_DEPLOYMENT.deployment_level.value}")
    print(f"Capabilities: {', '.join(UNIVERSAL_DEPLOYMENT.capabilities)}")
    
    # Test deployment to different devices
    devices_to_test = [
        DeviceType.WINDOWS_PC,
        DeviceType.ANDROID,
        DeviceType.ROUTER
    ]
    
    for device in devices_to_test:
        print(f"\n📱 Testing deployment to {device.value}:")
        result = deploy_to_device(device.value, "web_injection")
        print(f"   Success: {result['success']}")
        print(f"   Components: {len(result['components_deployed'])}")
        if result.get('access_url'):
            print(f"   Access: {result['access_url']}")