#!/usr/bin/env python3
"""
Viren Diagnostic System
Main executable for Cloud Viren diagnostic component
"""

import os
import sys
import json
import time
import logging
import argparse
import threading
import signal
import webbrowser
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("viren_diagnostic.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("VirenDiagnostic")

# Import components
try:
    from diagnostic_core import DiagnosticCore
    from research_tentacles import ResearchTentacles
    from blockchain_relay import BlockchainRelay
    from llm_client import LLMClient
except ImportError as e:
    logger.error(f"Failed to import required components: {e}")
    logger.error("Please ensure all component files are in the correct location")
    sys.exit(1)

class VirenDiagnosticSystem:
    """
    Main class for Viren Diagnostic System
    Integrates all components into a unified system
    """
    
    def __init__(self, config_path: str = None):
        """Initialize the diagnostic system"""
        self.config_path = config_path or os.path.join("config", "viren_diagnostic_config.json")
        self.config = self._load_config()
        self.running = False
        self.diagnostic_core = None
        self.research_tentacles = None
        self.blockchain_relay = None
        self.llm_client = None
        self.web_server = None
        self.cloud_sync_thread = None
        self.last_activity = time.time()
        self.system_status = "initializing"
        self.diagnostic_results = {}
        self.research_results = {}
        self.relay_status = {}
        
        # Initialize components
        self._init_components()
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)
        
        logger.info("Viren Diagnostic System initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        default_config = {
            "diagnostic_interval": 3600,  # 1 hour
            "cloud_sync_interval": 86400,  # 24 hours
            "idle_threshold": 1800,  # 30 minutes
            "web_interface": {
                "enabled": True,
                "port": 8080,
                "open_browser": True
            },
            "components": {
                "diagnostic_core": True,
                "research_tentacles": True,
                "blockchain_relay": True,
                "llm_client": True
            },
            "theme_colors": {
                "plumb": "#A2799A",
                "primer": "#93AEC5",
                "silver": "#AFC5DC",
                "putty": "#C6D6E2",
                "dried_putty": "#D8E3EB",
                "white": "#EBF2F6"
            }
        }
        
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    # Merge with defaults for any missing keys
                    for key, value in default_config.items():
                        if key not in config:
                            config[key] = value
                        elif isinstance(value, dict) and isinstance(config.get(key), dict):
                            for subkey, subvalue in value.items():
                                if subkey not in config[key]:
                                    config[key][subkey] = subvalue
                    
                    logger.info("Configuration loaded successfully")
                    return config
            except Exception as e:
                logger.error(f"Error loading configuration: {e}")
        
        logger.info("Using default configuration")
        return default_config
    
    def _save_config(self) -> bool:
        """Save configuration to file"""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info("Configuration saved successfully")
            return True
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            return False
    
    def _init_components(self) -> None:
        """Initialize all components"""
        # Initialize diagnostic core
        if self.config["components"]["diagnostic_core"]:
            try:
                self.diagnostic_core = DiagnosticCore()
                logger.info("Diagnostic core initialized")
            except Exception as e:
                logger.error(f"Failed to initialize diagnostic core: {e}")
        
        # Initialize research tentacles
        if self.config["components"]["research_tentacles"]:
            try:
                self.research_tentacles = ResearchTentacles()
                logger.info("Research tentacles initialized")
            except Exception as e:
                logger.error(f"Failed to initialize research tentacles: {e}")
        
        # Initialize blockchain relay
        if self.config["components"]["blockchain_relay"]:
            try:
                self.blockchain_relay = BlockchainRelay()
                logger.info("Blockchain relay initialized")
            except Exception as e:
                logger.error(f"Failed to initialize blockchain relay: {e}")
        
        # Initialize LLM client
        if self.config["components"]["llm_client"]:
            try:
                self.llm_client = LLMClient()
                logger.info("LLM client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize LLM client: {e}")
    
    def start(self) -> bool:
        """Start the diagnostic system"""
        if self.running:
            logger.warning("Diagnostic system is already running")
            return False
        
        logger.info("Starting Viren Diagnostic System")
        self.running = True
        self.system_status = "starting"
        
        # Start web interface if enabled
        if self.config["web_interface"]["enabled"]:
            self._start_web_interface()
        
        # Start cloud sync thread
        self.cloud_sync_thread = threading.Thread(target=self._cloud_sync_loop)
        self.cloud_sync_thread.daemon = True
        self.cloud_sync_thread.start()
        
        # Start main loop
        self.system_status = "running"
        self._main_loop()
        
        return True
    
    def stop(self) -> bool:
        """Stop the diagnostic system"""
        if not self.running:
            logger.warning("Diagnostic system is not running")
            return False
        
        logger.info("Stopping Viren Diagnostic System")
        self.running = False
        self.system_status = "stopping"
        
        # Stop blockchain relay if running
        if self.blockchain_relay and hasattr(self.blockchain_relay, "stop"):
            self.blockchain_relay.stop()
        
        # Stop web interface if running
        if self.web_server:
            self._stop_web_interface()
        
        # Wait for cloud sync thread to finish
        if self.cloud_sync_thread and self.cloud_sync_thread.is_alive():
            self.cloud_sync_thread.join(timeout=5)
        
        self.system_status = "stopped"
        logger.info("Viren Diagnostic System stopped")
        return True
    
    def _main_loop(self) -> None:
        """Main loop for the diagnostic system"""
        logger.info("Entering main loop")
        
        last_diagnostic_time = 0
        
        while self.running:
            try:
                current_time = time.time()
                
                # Run diagnostics at regular intervals
                if current_time - last_diagnostic_time > self.config["diagnostic_interval"]:
                    self._run_diagnostics()
                    last_diagnostic_time = current_time
                
                # Check if system is idle
                is_idle = current_time - self.last_activity > self.config["idle_threshold"]
                
                # Start or stop blockchain relay based on idle status
                if is_idle and self.blockchain_relay and hasattr(self.blockchain_relay, "start"):
                    if not getattr(self.blockchain_relay, "running", False):
                        logger.info("System idle, starting blockchain relay")
                        self.blockchain_relay.start()
                elif not is_idle and self.blockchain_relay and hasattr(self.blockchain_relay, "stop"):
                    if getattr(self.blockchain_relay, "running", False):
                        logger.info("System active, stopping blockchain relay")
                        self.blockchain_relay.stop()
                
                # Update relay status
                if self.blockchain_relay and hasattr(self.blockchain_relay, "get_status"):
                    self.relay_status = self.blockchain_relay.get_status()
                
                # Sleep for a bit
                time.sleep(10)
            
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(30)  # Wait longer after an error
    
    def _run_diagnostics(self) -> None:
        """Run system diagnostics"""
        logger.info("Running system diagnostics")
        self.system_status = "diagnosing"
        
        try:
            # Run diagnostics using the diagnostic core
            if self.diagnostic_core:
                # Get system info
                system_info = getattr(self.diagnostic_core, "system_info", {})
                
                # Run CPU diagnostics
                cpu_module = self.diagnostic_core.diagnostic_modules.get("cpu", {})
                cpu_results = {}
                for check_name, check_func in cpu_module.get("checks", {}).items():
                    cpu_results[check_name] = check_func()
                
                # Run memory diagnostics
                memory_module = self.diagnostic_core.diagnostic_modules.get("memory", {})
                memory_results = {}
                for check_name, check_func in memory_module.get("checks", {}).items():
                    memory_results[check_name] = check_func()
                
                # Run disk diagnostics
                disk_module = self.diagnostic_core.diagnostic_modules.get("disk", {})
                disk_results = {}
                for check_name, check_func in disk_module.get("checks", {}).items():
                    disk_results[check_name] = check_func()
                
                # Run network diagnostics
                network_module = self.diagnostic_core.diagnostic_modules.get("network", {})
                network_results = {}
                for check_name, check_func in network_module.get("checks", {}).items():
                    network_results[check_name] = check_func()
                
                # Compile results
                self.diagnostic_results = {
                    "timestamp": time.time(),
                    "system_info": system_info,
                    "cpu": cpu_results,
                    "memory": memory_results,
                    "disk": disk_results,
                    "network": network_results
                }
                
                # Analyze results with LLM
                if self.llm_client:
                    analysis_prompt = "Analyze the following diagnostic results and identify any issues or potential problems:\n"
                    analysis_prompt += json.dumps(self.diagnostic_results, indent=2)
                    
                    analysis_response = self.llm_client.query(analysis_prompt)
                    self.diagnostic_results["llm_analysis"] = analysis_response.get("response", "")
                
                logger.info("Diagnostics completed successfully")
            else:
                logger.warning("Diagnostic core not available, skipping diagnostics")
        
        except Exception as e:
            logger.error(f"Error running diagnostics: {e}")
            self.diagnostic_results = {
                "timestamp": time.time(),
                "error": str(e)
            }
        
        self.system_status = "running"
        self.last_activity = time.time()
    
    def research_issue(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Research an issue using research tentacles"""
        logger.info(f"Researching issue: {query}")
        self.system_status = "researching"
        self.last_activity = time.time()
        
        try:
            if self.research_tentacles:
                # Deploy research tentacles
                research_results = self.research_tentacles.deploy(query, context)
                
                # Store results
                self.research_results = research_results
                
                # Analyze results with LLM
                if self.llm_client:
                    analysis_prompt = "Analyze the following research results and provide a comprehensive solution:\n"
                    analysis_prompt += json.dumps(research_results, indent=2)
                    
                    analysis_response = self.llm_client.query(analysis_prompt)
                    research_results["llm_analysis"] = analysis_response.get("response", "")
                
                logger.info("Research completed successfully")
                self.system_status = "running"
                return research_results
            else:
                logger.warning("Research tentacles not available")
                self.system_status = "running"
                return {
                    "error": "Research tentacles not available",
                    "timestamp": time.time()
                }
        
        except Exception as e:
            logger.error(f"Error researching issue: {e}")
            self.system_status = "running"
            return {
                "error": str(e),
                "timestamp": time.time()
            }
    
    def _cloud_sync_loop(self) -> None:
        """Sync with Cloud Viren periodically"""
        logger.info("Starting cloud sync loop")
        
        while self.running:
            try:
                # Sync with cloud
                self._sync_with_cloud()
                
                # Sleep until next sync
                time.sleep(self.config["cloud_sync_interval"])
            
            except Exception as e:
                logger.error(f"Error in cloud sync loop: {e}")
                time.sleep(300)  # Wait 5 minutes before retrying
    
    def _sync_with_cloud(self) -> None:
        """Sync data with Cloud Viren"""
        logger.info("Syncing with Cloud Viren")
        
        # This is a placeholder for actual cloud sync implementation
        # In a real implementation, this would send diagnostic data to the cloud
        # and receive updates and knowledge base updates
        
        # Simulate successful sync
        logger.info("Cloud sync completed successfully")
    
    def _start_web_interface(self) -> None:
        """Start the web interface"""
        if self.web_server:
            logger.warning("Web interface is already running")
            return
        
        try:
            # Import web server modules
            import http.server
            import socketserver
            
            # Create web directory if it doesn't exist
            web_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
            os.makedirs(web_dir, exist_ok=True)
            
            # Create index.html if it doesn't exist
            index_path = os.path.join(web_dir, "index.html")
            if not os.path.exists(index_path):
                with open(index_path, 'w') as f:
                    f.write(self._generate_index_html())
            
            # Create handler
            handler = http.server.SimpleHTTPRequestHandler
            
            # Create server
            port = self.config["web_interface"]["port"]
            self.web_server = socketserver.TCPServer(("", port), handler)
            
            # Start server in a separate thread
            web_thread = threading.Thread(target=self.web_server.serve_forever)
            web_thread.daemon = True
            web_thread.start()
            
            logger.info(f"Web interface started on port {port}")
            
            # Open browser if enabled
            if self.config["web_interface"]["open_browser"]:
                webbrowser.open(f"http://localhost:{port}")
        
        except Exception as e:
            logger.error(f"Failed to start web interface: {e}")
            self.web_server = None
    
    def _stop_web_interface(self) -> None:
        """Stop the web interface"""
        if not self.web_server:
            logger.warning("Web interface is not running")
            return
        
        try:
            self.web_server.shutdown()
            self.web_server = None
            logger.info("Web interface stopped")
        except Exception as e:
            logger.error(f"Error stopping web interface: {e}")
    
    def _generate_index_html(self) -> str:
        """Generate index.html for the web interface"""
        colors = self.config["theme_colors"]
        
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Viren Diagnostic System</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: {colors["white"]};
            color: #333;
        }}
        
        header {{
            background-color: {colors["plumb"]};
            color: white;
            padding: 1rem;
            text-align: center;
        }}
        
        nav {{
            background-color: {colors["primer"]};
            padding: 0.5rem;
        }}
        
        nav ul {{
            list-style-type: none;
            margin: 0;
            padding: 0;
            display: flex;
            justify-content: center;
        }}
        
        nav ul li {{
            margin: 0 1rem;
        }}
        
        nav ul li a {{
            color: white;
            text-decoration: none;
            padding: 0.5rem;
        }}
        
        nav ul li a:hover {{
            background-color: {colors["silver"]};
            border-radius: 4px;
        }}
        
        main {{
            padding: 1rem;
            max-width: 1200px;
            margin: 0 auto;
        }}
        
        .dashboard {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}
        
        .card {{
            background-color: {colors["dried_putty"]};
            border-radius: 8px;
            padding: 1rem;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }}
        
        .card h2 {{
            margin-top: 0;
            color: {colors["plumb"]};
            border-bottom: 2px solid {colors["silver"]};
            padding-bottom: 0.5rem;
        }}
        
        footer {{
            background-color: {colors["putty"]};
            color: #333;
            text-align: center;
            padding: 1rem;
            margin-top: 2rem;
        }}
        
        .status {{
            display: inline-block;
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-weight: bold;
        }}
        
        .status-normal {{
            background-color: #4caf50;
            color: white;
        }}
        
        .status-warning {{
            background-color: #ff9800;
            color: white;
        }}
        
        .status-critical {{
            background-color: #f44336;
            color: white;
        }}
        
        .button {{
            background-color: {colors["plumb"]};
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 4px;
            cursor: pointer;
            font-weight: bold;
        }}
        
        .button:hover {{
            background-color: #8a6382;
        }}
    </style>
</head>
<body>
    <header>
        <h1>Viren Diagnostic System</h1>
        <p>Comprehensive system diagnostics with Gemma 3 3B</p>
    </header>
    
    <nav>
        <ul>
            <li><a href="#dashboard">Dashboard</a></li>
            <li><a href="#diagnostics">Diagnostics</a></li>
            <li><a href="#research">Research</a></li>
            <li><a href="#blockchain">Blockchain</a></li>
            <li><a href="#settings">Settings</a></li>
        </ul>
    </nav>
    
    <main>
        <section id="dashboard">
            <h1>System Dashboard</h1>
            
            <div class="dashboard">
                <div class="card">
                    <h2>System Status</h2>
                    <p><strong>Status:</strong> <span class="status status-normal">Running</span></p>
                    <p><strong>Last Activity:</strong> <span id="last-activity">Just now</span></p>
                    <p><strong>Uptime:</strong> <span id="uptime">0 days, 0 hours, 0 minutes</span></p>
                    <button class="button" onclick="runDiagnostics()">Run Diagnostics</button>
                </div>
                
                <div class="card">
                    <h2>CPU</h2>
                    <p><strong>Usage:</strong> <span id="cpu-usage">0%</span></p>
                    <p><strong>Temperature:</strong> <span id="cpu-temp">N/A</span></p>
                    <p><strong>Status:</strong> <span class="status status-normal" id="cpu-status">Normal</span></p>
                </div>
                
                <div class="card">
                    <h2>Memory</h2>
                    <p><strong>Usage:</strong> <span id="memory-usage">0%</span></p>
                    <p><strong>Available:</strong> <span id="memory-available">0 GB</span></p>
                    <p><strong>Status:</strong> <span class="status status-normal" id="memory-status">Normal</span></p>
                </div>
                
                <div class="card">
                    <h2>Disk</h2>
                    <p><strong>Usage:</strong> <span id="disk-usage">0%</span></p>
                    <p><strong>Free Space:</strong> <span id="disk-free">0 GB</span></p>
                    <p><strong>Status:</strong> <span class="status status-normal" id="disk-status">Normal</span></p>
                </div>
                
                <div class="card">
                    <h2>Network</h2>
                    <p><strong>Connectivity:</strong> <span id="network-connectivity">Online</span></p>
                    <p><strong>Latency:</strong> <span id="network-latency">0 ms</span></p>
                    <p><strong>Status:</strong> <span class="status status-normal" id="network-status">Normal</span></p>
                </div>
                
                <div class="card">
                    <h2>Blockchain Relay</h2>
                    <p><strong>Status:</strong> <span id="blockchain-status">Inactive</span></p>
                    <p><strong>Peers:</strong> <span id="blockchain-peers">0</span></p>
                    <p><strong>Transactions:</strong> <span id="blockchain-transactions">0</span></p>
                </div>
            </div>
        </section>
        
        <section id="diagnostics">
            <h1>Diagnostic Results</h1>
            <div class="card">
                <h2>Latest Diagnostics</h2>
                <p><strong>Last Run:</strong> <span id="diagnostics-last-run">Never</span></p>
                <div id="diagnostics-results">
                    <p>No diagnostic results available. Click "Run Diagnostics" to start.</p>
                </div>
            </div>
        </section>
        
        <section id="research">
            <h1>Research Tentacles</h1>
            <div class="card">
                <h2>Issue Research</h2>
                <p>Use research tentacles to find solutions to unknown issues.</p>
                <div>
                    <input type="text" id="research-query" placeholder="Describe the issue to research..." style="width: 80%; padding: 0.5rem; margin-bottom: 1rem;">
                    <button class="button" onclick="researchIssue()">Research</button>
                </div>
                <div id="research-results">
                    <p>No research results available. Enter an issue to research.</p>
                </div>
            </div>
        </section>
        
        <section id="blockchain">
            <h1>Blockchain Relay</h1>
            <div class="card">
                <h2>Relay Status</h2>
                <p>The blockchain relay activates when the system is idle.</p>
                <div id="blockchain-details">
                    <p>No blockchain relay data available.</p>
                </div>
            </div>
        </section>
        
        <section id="settings">
            <h1>Settings</h1>
            <div class="card">
                <h2>System Configuration</h2>
                <p>Configure the Viren Diagnostic System.</p>
                <button class="button" onclick="saveSettings()">Save Settings</button>
            </div>
        </section>
    </main>
    
    <footer>
        <p>Viren Diagnostic System &copy; 2023</p>
    </footer>
    
    <script>
        // Placeholder for actual JavaScript functionality
        function runDiagnostics() {
            alert('Running diagnostics...');
            // In a real implementation, this would make an API call to the backend
        }
        
        function researchIssue() {
            const query = document.getElementById('research-query').value;
            alert('Researching issue: ' + query);
            // In a real implementation, this would make an API call to the backend
        }
        
        function saveSettings() {
            alert('Settings saved');
            // In a real implementation, this would make an API call to the backend
        }
        
        // Update UI with mock data (would be replaced with actual API calls)
        document.getElementById('cpu-usage').textContent = '25%';
        document.getElementById('memory-usage').textContent = '40%';
        document.getElementById('disk-usage').textContent = '65%';
        document.getElementById('uptime').textContent = '0 days, 1 hours, 30 minutes';
    </script>
</body>
</html>
"""
    
    def _handle_signal(self, signum, frame) -> None:
        """Handle signals (SIGINT, SIGTERM)"""
        logger.info(f"Received signal {signum}, shutting down")
        self.stop()
        sys.exit(0)
    
    def get_status(self) -> Dict[str, Any]:
        """Get system status"""
        return {
            "status": self.system_status,
            "running": self.running,
            "last_activity": self.last_activity,
            "components": {
                "diagnostic_core": self.diagnostic_core is not None,
                "research_tentacles": self.research_tentacles is not None,
                "blockchain_relay": self.blockchain_relay is not None,
                "llm_client": self.llm_client is not None
            },
            "web_interface": self.web_server is not None,
            "diagnostic_results": bool(self.diagnostic_results),
            "research_results": bool(self.research_results),
            "relay_status": self.relay_status
        }

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Viren Diagnostic System")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--no-web", action="store_true", help="Disable web interface")
    parser.add_argument("--no-relay", action="store_true", help="Disable blockchain relay")
    parser.add_argument("--diagnose", action="store_true", help="Run diagnostics and exit")
    parser.add_argument("--research", help="Research an issue and exit")
    
    args = parser.parse_args()
    
    # Create diagnostic system
    system = VirenDiagnosticSystem(config_path=args.config)
    
    # Apply command line options
    if args.no_web:
        system.config["web_interface"]["enabled"] = False
    
    if args.no_relay:
        system.config["components"]["blockchain_relay"] = False
    
    # Run diagnostics if requested
    if args.diagnose:
        system._run_diagnostics()
        print(json.dumps(system.diagnostic_results, indent=2))
        return 0
    
    # Research issue if requested
    if args.research:
        results = system.research_issue(args.research)
        print(json.dumps(results, indent=2))
        return 0
    
    # Start the system
    try:
        system.start()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        system.stop()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())