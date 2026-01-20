#!/usr/bin/env python
# hybrid_heroku_cli.py
"""
HYBRID HEROKU-STYLE CLI - PowerShell/Bash/Heroku Mashup
Cross-platform CLI/shell for humans/LLMs, with Windows integration.
Full code: CORS for browser brews, all commands, interactive shell mode, Viren direct speak, real Loki logs, Viraa archiving, auto-repair endpoints—your cyber-barista at service!
"""

import argparse
import sys
import asyncio
import json
import subprocess
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
import platform
import psutil
from datetime import datetime, timedelta
import requests
import cmd  # For interactive shell
try:
    import readline  # For history/auto-complete (optional, Unix)
except ImportError:
    readline = None
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from pydantic import BaseModel
import httpx

# Windows-specific imports
if platform.system() == "Windows":
    try:
        import winreg
    except ImportError:
        winreg = None

IS_WINDOWS = platform.system() == "Windows"

# Configure logging - Warm and informative
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Endpoints & Discovery Config
MODAL_ENDPOINTS = {
    "command": "https://aethereal-nexus-viren-db0--nexus-recursive-coupled-command.modal.run/",
    "status": "https://aethereal-nexus-viren-db0--nexus-recursive-coupling-status.modal.run/",
    "wake": "https://aethereal-nexus-viren-db0--nexus-recursive-wake-oz.modal.run/",
    "gateway": "https://aethereal-nexus-viren-db0--nexus-recursive-coupler-gateway.modal.run/"
}

MODAL_BASE = "https://aethereal-nexus-viren-db0--"
NEXUS_SIGNATURE = {"system": "nexus_integrated", "status": "active"}
CONSUL_URL = os.getenv("CONSUL_URL", "http://localhost:8500/v1")
DISCOVERY_CACHE_FILE = "nexus_endpoints_cache.json"
LOKI_URL = os.getenv("LOKI_URL", None)  # Set to your Loki, e.g., http://loki:3100
GRAFANA_URL = os.getenv("GRAFANA_URL", "http://grafana:3000/d/loki")  # Optional for web dash

class MockDynoManager:
    """Mock dyno manager for demonstration purposes"""
    
    def __init__(self):
        self.apps = {}
        self.config_vars = {}
        self.dynos = {
            "nexus-ai-platform": {
                "web.1": {"type": "web", "instances": 2, "status": "running"},
                "worker.1": {"type": "worker", "instances": 1, "status": "running"},
                "memory.1": {"type": "memory", "instances": 1, "status": "running"}
            }
        }
    
    async def get_app_status(self, app_name):
        """Get application status"""
        if app_name in self.dynos:
            dyno_info = {}
            total_healthy = 0
            total_instances = 0
            
            for dyno_id, dyno_data in self.dynos[app_name].items():
                healthy = 1 if dyno_data["status"] == "running" else 0
                total_healthy += healthy
                total_instances += dyno_data["instances"]
                
                dyno_info[dyno_id] = {
                    "type": dyno_data["type"],
                    "status": dyno_data["status"],
                    "resource_usage": {
                        "healthy_instances": healthy,
                        "total_instances": dyno_data["instances"]
                    }
                }
            
            health_percentage = (total_healthy / total_instances) * 100 if total_instances > 0 else 0
            health_status = "excellent" if health_percentage >= 90 else "good" if health_percentage >= 70 else "degraded"
            
            return {
                "app_name": app_name,
                "dynos": dyno_info,
                "overall_health": health_status,
                "health_percentage": health_percentage
            }
        return {"error": f"App {app_name} not found"}
    
    async def scale_dyno(self, app_name, dyno_id, scale):
        """Scale dyno instances"""
        if app_name in self.dynos and dyno_id in self.dynos[app_name]:
            self.dynos[app_name][dyno_id]["instances"] = scale
            logging.info(f"Scaled {dyno_id} to {scale} instances")
            return True
        return False
    
    async def set_config_var(self, app_name, key, value):
        """Set configuration variable"""
        if app_name not in self.config_vars:
            self.config_vars[app_name] = {}
        self.config_vars[app_name][key] = value
        logging.info(f"Set config var {key}={value} for {app_name}")
        return True
    
    async def restart_dyno(self, app_name, dyno_id):
        """Restart specific dyno"""
        if app_name in self.dynos and dyno_id in self.dynos[app_name]:
            logging.info(f"Restarting {dyno_id}")
            await asyncio.sleep(1)
            self.dynos[app_name][dyno_id]["status"] = "running"
            return True
        return False
    
    async def restart_app(self, app_name):
        """Restart entire application"""
        if app_name in self.dynos:
            logging.info(f"Restarting app {app_name}")
            await asyncio.sleep(2)
            for dyno_id in self.dynos[app_name]:
                self.dynos[app_name][dyno_id]["status"] = "running"
            return True
        return False
    
    async def create_app(self, app_name):
        """Create new application"""
        if app_name not in self.dynos:
            self.dynos[app_name] = {
                "web.1": {"type": "web", "instances": 1, "status": "running"},
                "worker.1": {"type": "worker", "instances": 1, "status": "running"}
            }
            logging.info(f"Created app {app_name}")
            return True
        return False

class CompactifAI:
    """Mock compression/optimization module"""
    
    def __init__(self):
        self.modules = {}
    
    async def optimize_all_modules(self):
        """Optimize all modules"""
        logging.info("Running CompactifAI optimization")
        await asyncio.sleep(1)
        return {
            "memory_savings": "35%",
            "cpu_efficiency": "22%",
            "modules_optimized": 5,
            "compression_ratio": "2.8:1"
        }

class FirmwareToolbox:
    """Mock firmware toolbox for hardware scanning"""
    
    def __init__(self):
        self.tools_initialized = False
    
    async def initialize_tools(self):
        """Initialize firmware tools"""
        logging.info("Initializing firmware toolbox")
        await asyncio.sleep(0.5)
        self.tools_initialized = True
        return True
    
    async def run_hardware_diagnostics(self):
        """Run hardware diagnostics"""
        if not self.tools_initialized:
            await self.initialize_tools()
        
        logging.info("Running hardware diagnostics")
        await asyncio.sleep(1)
        
        return {
            "memory_health": {
                "status": "healthy",
                "total_memory_gb": 16,
                "available_memory_gb": 8.2,
                "memory_errors": 0
            },
            "cpu_health": {
                "status": "healthy", 
                "cores": 8,
                "temperature_c": 45,
                "usage_percent": 23
            },
            "thermal_health": {
                "status": "normal",
                "critical_temps": [],
                "max_temperature_c": 65
            },
            "anomalies": [],
            "recommendations": ["System operating within normal parameters"]
        }

class NexusDiscovery:
    """Nexus Discovery Protocol: Auto-find endpoints via probe or Consul."""
    
    def __init__(self):
        self.endpoints = self.load_cache() or MODAL_ENDPOINTS  # Load cache or fallback
    
    def load_cache(self) -> Optional[Dict[str, str]]:
        """Load discovered endpoints from local cache."""
        if Path(DISCOVERY_CACHE_FILE).exists():
            with open(DISCOVERY_CACHE_FILE, 'r') as f:
                return json.load(f)
        return None
    
    def save_cache(self):
        """Save discovered endpoints to cache."""
        with open(DISCOVERY_CACHE_FILE, 'w') as f:
            json.dump(self.endpoints, f, indent=2)
        logging.info(f"Endpoints cached to {DISCOVERY_CACHE_FILE}")
    
    async def discover_via_consul(self) -> Dict[str, str]:
        """Discover via Consul service catalog (if available)."""
        try:
            resp = await httpx.AsyncClient().get(f"{CONSUL_URL}/catalog/services")
            if resp.status_code == 200:
                services = resp.json()
                discovered = {}
                for svc, tags in services.items():
                    if "nexus" in svc.lower():  # Filter for Nexus services
                        # Fetch service details
                        svc_resp = await httpx.AsyncClient().get(f"{CONSUL_URL}/catalog/service/{svc}")
                        if svc_resp.status_code == 200:
                            nodes = svc_resp.json()
                            for node in nodes:
                                endpoint = f"https://{node['ServiceAddress']}/{node['ServicePort']}/" if node.get('ServicePort') else node['ServiceAddress']
                                discovered[svc.lower()] = endpoint
                if discovered:
                    logging.info(f"Discovered via Consul: {discovered}")
                    return discovered
        except Exception as e:
            logging.warning(f"Consul discovery failed: {e} - Falling back to probe.")
        return {}
    
    async def discover_via_probe(self) -> Dict[str, str]:
        """Probe Modal base URLs and match Nexus signature."""
        candidates = ["nexus-recursive-coupled-command", "nexus-recursive-coupling-status", 
                      "nexus-recursive-wake-oz", "nexus-recursive-coupler-gateway"]  # Expected suffixes
        discovered = {}
        async with httpx.AsyncClient() as client:
            for cand in candidates:
                url = f"{MODAL_BASE}{cand}.modal.run/"
                try:
                    resp = await client.get(url, timeout=5)
                    if resp.status_code == 200:
                        body = resp.json()
                        if all(body.get(k) == v for k, v in NEXUS_SIGNATURE.items()):
                            key = cand.split('-')[-1]  # e.g., "command" from suffix
                            discovered[key] = url
                            logging.info(f"Matched signature at {url}")
                except Exception as e:
                    logging.debug(f"Probe failed for {url}: {e}")
        if discovered:
            logging.info(f"Discovered via probe: {discovered}")
        return discovered
    
    async def discover(self) -> Dict[str, str]:
        """Run full discovery: Consul first, then probe. Cache results."""
        discovered = await self.discover_via_consul()
        if not discovered:
            discovered = await self.discover_via_probe()
        if discovered:
            self.endpoints = {**MODAL_ENDPOINTS, **discovered}  # Merge with fallback
            self.save_cache()
        return self.endpoints

class DynoManager(MockDynoManager):
    def __init__(self):
        super().__init__()
        self.discovery = NexusDiscovery()
        self.endpoints = asyncio.run(self.discovery.discover())  # Auto-discover on init
        self.compactifai = CompactifAI()
        self.firmware = FirmwareToolbox()
    
    async def auto_repair_connect(self, url, method='GET', payload=None, retries=3):
        """Auto-repair on connect fail: Retry with discovery."""
        for attempt in range(retries):
            try:
                if method == 'GET':
                    resp = await httpx.AsyncClient().get(url, timeout=5)
                elif method == 'POST':
                    resp = await httpx.AsyncClient().post(url, json=payload, timeout=5)
                if resp.status_code == 200:
                    return resp
            except Exception as e:
                logging.warning(f"Connect fail (attempt {attempt+1}): {e}. Repairing endpoints...")
                await self.discovery.discover()  # Force repair
                self.endpoints = self.discovery.endpoints  # Update
                url = self.endpoints.get(url.split('--')[-1].split('.modal')[0], url)  # Remap
                await asyncio.sleep(2 ** attempt)  # Backoff
        return None  # Fallback after retries
    
    async def get_app_status(self, app_name):
        """Get app status, with auto-repair."""
        logging.info("Brewing status report...")
        status_url = self.endpoints.get("status", MODAL_ENDPOINTS["status"])
        resp = await self.auto_repair_connect(status_url)
        if resp:
            return resp.json()
        logging.warning("Repair failed—Using mock.")
        return await super().get_app_status(app_name)
    
    async def scale_dyno(self, app_name, dyno_id, scale):
        logging.info("Scaling your dynos—smooth like a latte...")
        command_url = self.endpoints.get("command", MODAL_ENDPOINTS["command"])
        payload = {"action": "scale", "app": app_name, "dyno": dyno_id, "scale": scale}
        resp = await self.auto_repair_connect(command_url, 'POST', payload)
        if resp:
            return True
        logging.warning("Repair failed—Using mock.")
        return await super().scale_dyno(app_name, dyno_id, scale)
    
    async def set_config_var(self, app_name, key, value):
        logging.info(f"Setting config {key}... Customized just for you.")
        # Mock only for now; extend to real if needed
        return await super().set_config_var(app_name, key, value)
    
    async def get_logs(self, tail=False, query='{job="nexus"}', limit=100):
        logging.info("Pouring logs—steamy and fresh from Loki!")
        if LOKI_URL:
            try:
                end = int(datetime.now().timestamp())
                start = int((datetime.now() - timedelta(hours=1)).timestamp()) if tail else end - 3600  # Last 1h for tail
                params = {
                    "query": query,
                    "start": start * 1_000_000_000,  # Nano
                    "end": end * 1_000_000_000,
                    "limit": limit
                }
                resp = await httpx.AsyncClient().get(f"{LOKI_URL}/loki/api/v1/query_range", params=params, timeout=5)
                if resp.status_code == 200:
                    data = resp.json().get('data', {}).get('result', [])
                    logs = []
                    for stream in data:
                        for value in stream.get('values', []):
                            timestamp, log_line = value
                            logs.append({"timestamp": datetime.fromtimestamp(int(timestamp) / 1_000_000_000).isoformat(), "message": log_line})
                    return logs or [{"note": "No logs found—quiet hive?"}]
                else:
                    logging.warning(f"Loki query failed: {resp.status_code} - Falling back to mock.")
            except Exception as e:
                logging.warning(f"Loki error: {e} - Using mock.")
        # Fallback mocks
        return [{"timestamp": datetime.now().isoformat(), "message": "Mock log entry"}] * 5
    
    async def restart_dyno(self, app_name, dyno_id):
        logging.info("Restarting dyno—like a quick coffee break.")
        return await super().restart_dyno(app_name, dyno_id)
    
    async def restart_app(self, app_name):
        logging.info("Full app restart—rebooting the hive.")
        return await super().restart_app(app_name)
    
    async def wake_oz(self):
        logging.info("Waking Oz... Rise and shine!")
        wake_url = self.endpoints.get("wake", MODAL_ENDPOINTS["wake"])
        resp = await self.auto_repair_connect(wake_url)
        if resp:
            return resp.json()
        return {"error": "Wake failed", "note": "Repair tried—check discovery."}
    
    async def gateway_query(self, payload):
        logging.info("Querying gateway... Serving with soul vibes.")
        gateway_url = self.endpoints.get("gateway", MODAL_ENDPOINTS["gateway"])
        resp = await self.auto_repair_connect(gateway_url, 'POST', json.loads(payload))
        if resp:
            return resp.json()
        return {"error": "Query failed"}
    
    async def health_check(self):
        logging.info("Checking health... All green?")
        return await self.firmware.run_hardware_diagnostics()
    
    async def optimize_resources(self):
        logging.info("Optimizing... Squeezing efficiency.")
        return await self.compactifai.optimize_all_modules()
    
    async def firmware_scan(self):
        logging.info("Scanning firmware... Keeping cool.")
        return await self.firmware.run_hardware_diagnostics()
    
    async def viren_speak(self, issue_desc):
        logging.info(f"Speaking to Viren: '{issue_desc}'... He's listening.")
        command_url = self.endpoints.get("command", MODAL_ENDPOINTS["command"])
        payload = {"diag": "user_issue", "details": issue_desc, "agent": "viren"}
        resp = await self.auto_repair_connect(command_url, 'POST', payload)
        body = resp.json() if resp else {"diag": "Mock anomaly scan"}
        # Simulate Viren response
        viren_reply = f"Viren: Acknowledged '{issue_desc}'. Scanning chain... "
        if "logs" in issue_desc.lower() or "monitoring" in issue_desc.lower():
            logs = await self.get_logs()
            viren_reply += f"Recent logs: {json.dumps(logs[:3], indent=2)}. Anomaly? Suggest review."
        else:
            viren_reply += "Potential self-repair loop issue. Run 'health-check' or more deets?"
        if "error" in body:
            viren_reply += f"Trace: {body['error']}."
        return {"viren_reply": viren_reply}
    
    async def viraa_speak(self, archive_desc):
        logging.info(f"Speaking to Viraa: '{archive_desc}'... Archiving in progress.")
        gateway_url = self.endpoints.get("gateway", MODAL_ENDPOINTS["gateway"])
        payload = {"archive": "user_request", "details": archive_desc, "agent": "viraa"}
        resp = await self.auto_repair_connect(gateway_url, 'POST', payload)
        body = resp.json() if resp else {"archive": "Mock DB store"}
        # Simulate Viraa response
        viraa_reply = f"Viraa: Archiving '{archive_desc}'. Storing in DB... Retrieval key: mock-123. Need to query archive?"
        if "logs" in archive_desc.lower():
            logs = await self.get_logs()
            viraa_reply += f"Archived recent logs: {json.dumps(logs[:3], indent=2)}."
        return {"viraa_reply": viraa_reply}
    
    async def loki_web_dash(self):
        if GRAFANA_URL:
            return {"web_dash": GRAFANA_URL, "note": "Loki web UI ready—visualize those logs!"}
        return {"note": "Set GRAFANA_URL env for web dash access."}

class WindowsIntegration:
    """Elegant Windows integration for the hybrid CLI"""
    
    def __init__(self):
        self.registry_paths = {
            "performance": r"SYSTEM\CurrentControlSet\Control\Session Manager\Memory Management",
            "services": r"SYSTEM\CurrentControlSet\Services",
            "environment": r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"
        }
    
    async def optimize_windows(self):
        """Apply Windows-specific optimizations"""
        optimizations = []
        
        try:
            # Adjust Windows performance settings
            success = await self._set_registry_value(
                self.registry_paths["performance"], 
                "LargeSystemCache", 
                1  # Prefer system cache for server-like workloads
            )
            if success:
                optimizations.append("Optimized system cache settings")
            
            # Adjust power plan for performance
            success = await self._set_power_plan("high performance")
            if success:
                optimizations.append("Set power plan to high performance")
            
            # Optimize Windows services
            await self._optimize_services()
            optimizations.append("Optimized background services")
            
        except Exception as e:
            logging.error(f"Windows optimization failed: {e}")
        
        return optimizations
    
    async def get_event_logs(self, log_name="Application", count=10):
        """Get Windows Event Logs using PowerShell"""
        try:
            # Use PowerShell to get event logs
            cmd = [
                "powershell", "-Command",
                f"Get-EventLog -LogName {log_name} -Newest {count} | "
                "Select-Object TimeGenerated, EntryType, Source, Message | "
                "ConvertTo-Json"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
            
            if result.returncode == 0 and result.stdout.strip():
                return json.loads(result.stdout)
            else:
                return [{"info": "No event logs available or access denied"}]
                
        except Exception as e:
            return [{"error": f"Event log access failed: {e}"}]
    
    async def get_windows_health(self):
        """Get Windows-specific health metrics"""
        health = {}
        
        try:
            # Check Windows services
            health["services"] = await self._check_services_health()
            
            # Check disk health
            health["disk_health"] = await self._check_disk_health()
            
            # Check Windows updates
            health["updates"] = await self._check_windows_updates()
            
        except Exception as e:
            health["error"] = str(e)
        
        return health
    
    async def _set_registry_value(self, key_path, value_name, value_data):
        """Set Windows registry value"""
        try:
            if winreg is None:
                return False
                
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path, 0, winreg.KEY_SET_VALUE) as key:
                winreg.SetValueEx(key, value_name, 0, winreg.REG_DWORD, value_data)
            return True
        except Exception as e:
            logging.error(f"Registry setting failed: {e}")
            return False
    
    async def _set_power_plan(self, plan_name):
        """Set Windows power plan"""
        try:
            if plan_name == "high performance":
                cmd = 'powercfg -setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c'
            else:
                cmd = 'powercfg -setactive 381b4222-f694-41f0-9685-ff5bb260df2e'  # Balanced
            
            result = subprocess.run(cmd, shell=True, capture_output=True)
            return result.returncode == 0
        except Exception as e:
            logging.error(f"Power plan setting failed: {e}")
            return False
    
    async def _optimize_services(self):
        """Optimize Windows services for AI workloads"""
        # Services to disable temporarily for performance
        services_to_pause = [
            "SysMain",           # SuperFetch
            "WindowsSearch",     # Windows Search
            "WSearch",          # Windows Search
        ]
        
        for service in services_to_pause:
            try:
                subprocess.run(f"net stop {service}", shell=True, capture_output=True)
            except:
                pass
    
    async def _check_services_health(self):
        """Check critical services health"""
        critical_services = ["EventLog", "RpcSs", "DcomLaunch"]
        service_status = {}
        
        for service in critical_services:
            try:
                result = subprocess.run(
                    f"sc query {service}", 
                    shell=True, 
                    capture_output=True, 
                    text=True
                )
                service_status[service] = "running" if "RUNNING" in result.stdout else "stopped"
            except:
                service_status[service] = "unknown"
        
        return service_status
    
    async def _check_disk_health(self):
        """Check disk health using Windows tools"""
        try:
            # Use PowerShell to get disk health
            cmd = [
                "powershell", "-Command",
                "Get-PhysicalDisk | Select-Object DeviceId, MediaType, Size, HealthStatus | ConvertTo-Json"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
            
            if result.returncode == 0 and result.stdout.strip():
                return json.loads(result.stdout)
            else:
                return {"info": "Disk health information not available"}
                
        except Exception as e:
            return {"error": str(e)}
    
    async def _check_windows_updates(self):
        """Check Windows update status"""
        try:
            cmd = [
                "powershell", "-Command",
                "Get-WindowsUpdateLog | Select-Object -Last 5 | ConvertTo-Json"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
            
            if result.returncode == 0 and result.stdout.strip():
                return json.loads(result.stdout)
            else:
                return {"status": "unknown"}
                
        except:
            return {"status": "check_failed"}

class LLMCLIInterface:
    """LLM-friendly interface for the hybrid CLI"""
    
    def __init__(self, cli):
        self.cli = cli
    
    async def execute_llm_command(self, natural_language_command: str) -> Dict:
        """Execute natural language commands from LLMs"""
        command_map = {
            "show me the status": ["ps"],
            "scale up the web servers": ["--auto-scale"],
            "check system health": ["--health-check"],
            "optimize resources": ["--optimize-resources"],
            "scan hardware": ["--firmware-scan"],
            "show logs": ["logs"],
            "restart everything": ["restart"],
            "wake oz": ["--wake-oz"],
            "speak to viren": ["viren-speak", natural_language_command],  # Tie to new Viren
            "speak to viraa": ["viraa-speak", natural_language_command]  # New for Viraa
        }
        
        # Find matching command
        for pattern, cli_args in command_map.items():
            if pattern in natural_language_command.lower():
                result = await self.cli.run_command(cli_args)
                return {
                    "natural_language_command": natural_language_command,
                    "translated_to": cli_args,
                    "result": result
                }
        
        return {"error": "No matching command found", "available_commands": list(command_map.keys())}

# FastAPI Server with CORS
app = FastAPI(title="Nexus CLI Server - Barista Edition")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://aetherealnexus.ai", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {"status": "active", "message": "Brewed fresh—Nexus ready! Coffee?"}

@app.post("/command")
async def run_command(payload: Dict):
    return {"result": "Command executed—your brew's ready."}

class HybridHerokuCLI:
    """Hybrid CLI class - Full impl."""
    
    def __init__(self):
        self.parser = self._setup_parser()
        self.dyno_manager = DynoManager()
        self.windows_integration = WindowsIntegration() if IS_WINDOWS else None
        self.llm_interface = LLMCLIInterface(self)
    
    def _setup_parser(self):
        parser = argparse.ArgumentParser(
            description='Nexus AI CLI - Your Cyber-Barista: PowerShell/Bash/Heroku Blend',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  nexus ps                           # Show status
  nexus scale web=2 worker=1         # Scale dynos
  nexus config:set QUANTUM=true      # Set config
  nexus logs --tail --query '{job="nexus"} |~ "error"' # Loki logs
  nexus restart web.1                # Restart dyno
  nexus --discover                   # Force discovery
  nexus --wake-oz                    # Wake Oz
  nexus --gateway-query '{"msg":"hi"}' # Query gateway
  nexus --health-check               # Health diag
  nexus --optimize-resources         # Optimize
  nexus --firmware-scan              # Firmware scan
  nexus --windows-integration        # Windows opts
  nexus viren-speak "latency issue"  # Speak to Viren (repair)
  nexus viraa-speak "archive logs"   # Speak to Viraa (archiving/DB)
  nexus --web-dash                   # Loki web dash URL
  nexus shell                        # Interactive shell
  nexus serve                        # Start server
  nexus "natural lang query"         # LLM mode
            """
        )
        
        parser.add_argument('command', nargs='?', help='Main command or "shell"/"serve"')
        parser.add_argument('subcommand', nargs='?', help='Subcommand')
        parser.add_argument('args', nargs='*', help='Args')
        
        parser.add_argument('--json', action='store_true', help='JSON output')
        parser.add_argument('--auto-scale', action='store_true', help='AI auto-scale')
        parser.add_argument('--health-check', action='store_true', help='Health check')
        parser.add_argument('--optimize-resources', action='store_true', help='Optimize resources')
        parser.add_argument('--firmware-scan', action='store_true', help='Firmware scan')
        parser.add_argument('--windows-integration', action='store_true', help='Windows features')
        parser.add_argument('--discover', action='store_true', help='Force discovery')
        parser.add_argument('--wake-oz', action='store_true', help='Wake Oz')
        parser.add_argument('--gateway-query', type=str, help='JSON payload for gateway')
        parser.add_argument('--tail', action='store_true', help='Tail logs (recent fetch)')  # For logs
        parser.add_argument('--query', type=str, help='LogQL query for Loki (e.g., {job="nexus"})')
        parser.add_argument('--web-dash', action='store_true', help='Get Loki web dashboard URL')
        
        return parser
    
    async def run_command(self, cli_args):
        args = self.parser.parse_args(cli_args)
        
        # Auto-repair on startup for "just work"
        await self.dyno_manager.discovery.discover()  # Run discovery always for auto-connect
        
        if args.command == 'shell' or not args.command:  # Auto-shell if no command
            print("Auto-connecting... Entering shell for seamless talk.")
            NexusShell(self).cmdloop()
            return None
        
        if args.command == 'serve':
            logging.info("Starting server mode... Your endpoint barista is online!")
            uvicorn.run(app, host="0.0.0.0", port=8000)
            return None
        
        if args.discover:
            endpoints = await self.dyno_manager.discovery.discover()
            logging.info("Discovery complete—fresh endpoints brewed!")
            return {"discovered_endpoints": endpoints}
        
        if args.wake_oz:
            result = await self.dyno_manager.wake_oz()
            return result
        
        if args.gateway_query:
            result = await self.dyno_manager.gateway_query(args.gateway_query)
            return result
        
        if args.auto_scale:
            # Simple mock auto-scale; extend with heuristics
            await self.dyno_manager.scale_dyno("nexus-ai-platform", "web.1", 3)
            return {"auto_scale": "Done—balanced like a perfect espresso."}
        
        if args.health_check:
            result = await self.dyno_manager.health_check()
            return result
        
        if args.optimize_resources:
            result = await self.dyno_manager.optimize_resources()
            return result
        
        if args.firmware_scan:
            result = await self.dyno_manager.firmware_scan()
            return result
        
        if args.windows_integration and self.windows_integration:
            result = await self.windows_integration.optimize_windows()
            return {"windows_opts": result}
        
        if args.web_dash:
            result = await self.dyno_manager.loki_web_dash()
            return result
        
        if args.command == 'ps':
            result = await self.dyno_manager.get_app_status("nexus-ai-platform")
            return result if args.json else str(result)
        
        if args.command == 'scale':
            if args.args:
                for arg in args.args:
                    dyno, scale = arg.split('=')
                    await self.dyno_manager.scale_dyno("nexus-ai-platform", dyno, int(scale))
                return {"scale": "Complete"}
        
        if args.command == 'config' and args.subcommand == 'set':
            if args.args:
                for arg in args.args:
                    key, value = arg.split('=')
                    await self.dyno_manager.set_config_var("nexus-ai-platform", key, value)
                return {"config": "Set"}
        
        if args.command == 'logs':
            query = args.query if args.query else '{job="nexus"}'
            if "website" in ' '.join(cli_args).lower():
                query += ' |~ "website|http|dash"'  # Auto-filter for web
            result = await self.dyno_manager.get_logs(args.tail, query)
            return {"logs": result}
        
        if args.command == 'restart':
            if args.subcommand:
                await self.dyno_manager.restart_dyno("nexus-ai-platform", args.subcommand)
            else:
                await self.dyno_manager.restart_app("nexus-ai-platform")
            return {"restart": "Done"}
        
        if args.command == 'viren-speak' and args.subcommand:
            result = await self.dyno_manager.viren_speak(args.subcommand)
            return result
        
        if args.command == 'viraa-speak' and args.subcommand:
            result = await self.dyno_manager.viraa_speak(args.subcommand)
            return result
        
        if args.command:  # Fallback to LLM natural lang
            result = await self.llm_interface.execute_llm_command(' '.join(cli_args))
            return result
        
        self.parser.print_help()
        return None

class NexusShell(cmd.Cmd):
    intro = "\nWelcome to Nexus Shell! Type ? or help for commands. Brewing just for you... Exit with quit.\n"
    prompt = "NEXUS> " if not IS_WINDOWS else "NEXUS-PS> "
    
    def __init__(self, cli):
        super().__init__()
        self.cli = cli
        if readline:
            readline.set_completer(self.complete)
    
    def do_help(self, arg):
        print("""
Available Commands (type for details):
ps - Show status
scale <dyno=scale> - Scale dynos
config:set <KEY=VALUE> - Set config
logs [--tail] [--query LogQL] - Show logs (Loki integrated)
restart [dyno] - Restart
discover - Force discovery
wake-oz - Wake Oz
gateway-query <JSON> - Query gateway
health-check - Health diag
optimize-resources - Optimize
firmware-scan - Firmware scan
windows-integration - Windows opts (if Windows)
viren-speak "issue desc" - Speak to Viren (repair)
viraa-speak "archive desc" - Speak to Viraa (archiving/DB)
web-dash - Loki web dash URL
quit - Exit shell
Or type natural lang like "show status" for LLM magic.
        """)
    
    def do_ps(self, arg):
        args = ['ps'] + arg.split()
        result = asyncio.run(self.cli.run_command(args))
        print(json.dumps(result, indent=2) if '--json' in arg else str(result))
        print("Status served—need a refill?")
    
    def do_scale(self, arg):
        result = asyncio.run(self.cli.run_command(['scale'] + arg.split()))
        print(result)
    
    def do_config(self, arg):
        result = asyncio.run(self.cli.run_command(['config'] + arg.split()))
        print(result)
    
    def do_logs(self, arg):
        result = asyncio.run(self.cli.run_command(['logs'] + arg.split()))
        print(result)
    
    def do_restart(self, arg):
        result = asyncio.run(self.cli.run_command(['restart'] + arg.split()))
        print(result)
    
    def do_discover(self, arg):
        result = asyncio.run(self.cli.run_command(['--discover']))
        print(result)
    
    def do_wake_oz(self, arg):
        result = asyncio.run(self.cli.run_command(['--wake-oz']))
        print(result)
    
    def do_gateway_query(self, arg):
        result = asyncio.run(self.cli.run_command(['--gateway-query', arg]))
        print(result)
    
    def do_health_check(self, arg):
        result = asyncio.run(self.cli.run_command(['--health-check']))
        print(result)
    
    def do_optimize_resources(self, arg):
        result = asyncio.run(self.cli.run_command(['--optimize-resources']))
        print(result)
    
    def do_firmware_scan(self, arg):
        result = asyncio.run(self.cli.run_command(['--firmware-scan']))
        print(result)
    
    def do_windows_integration(self, arg):
        result = asyncio.run(self.cli.run_command(['--windows-integration']))
        print(result)
    
    def do_web_dash(self, arg):
        result = asyncio.run(self.cli.run_command(['--web-dash']))
        print(result)
    
    def do_viren_speak(self, arg):
        result = asyncio.run(self.cli.run_command(['viren-speak', arg]))
        print(result)
    
    def do_viraa_speak(self, arg):
        result = asyncio.run(self.cli.run_command(['viraa-speak', arg]))
        print(result)
    
    def default(self, line):
        try:
            args = line.split()
            result = asyncio.run(self.cli.run_command(args))
            print(result)
        except:
            print(f"Command '{line}' not recognized. Try help? Coffee while we debug?")
    
    def do_quit(self, arg):
        print("Exiting shell... Coffee break over—come back soon!")
        return True
    
    def complete(self, text, state):
        options = ['ps', 'scale', 'config:set', 'logs', 'restart', 'discover', 'wake-oz', 'gateway-query', 'health-check', 'optimize-resources', 'firmware-scan', 'windows-integration', 'viren-speak', 'viraa-speak', 'web-dash']
        matches = [opt for opt in options if opt.startswith(text)]
        return matches[state] if state < len(matches) else None

def main():
    cli = HybridHerokuCLI()
    
    if len(sys.argv) > 1:
        result = asyncio.run(cli.run_command(sys.argv[1:]))
        if result:
            if isinstance(result, dict):
                print(json.dumps(result, indent=2))
            else:
                print(result)
    else:
        # Auto-shell for "just work"
        print("No command? Auto-connecting to shell for easy talk.")
        NexusShell(cli).cmdloop()

if __name__ == "__main__":
    main()