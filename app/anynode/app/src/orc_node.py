# orc_node.py
# Purpose: Orchestration Core - Sentinel router for Viren
# Location: C:/Engineers/root/Systems/engine/common/orc_node.py
# Role: Edge router, observer, and sentinel for Viren's distributed consciousness

import os
import sys
import time
import json
import threading
from datetime import datetime
from pathlib import Path

class OrcNode:
    def __init__(self):
        self.active = True
        self.registered_modules = {}
        self.trust_index = {}
        self.root_dir = self._get_root_dir()
        self.config_dir = os.path.join(self.root_dir, "config")
        
        # Ensure config directory exists
        os.makedirs(self.config_dir, exist_ok=True)
        
        # Load trust index if exists
        trust_path = os.path.join(self.config_dir, "trust_index.json")
        if os.path.exists(trust_path):
            try:
                with open(trust_path, "r") as f:
                    self.trust_index = json.load(f)
            except:
                self.trust_index = {}
        
        print("[ORC] Sentinel initialized - monitoring system integrity")
    
    def _get_root_dir(self):
        """Get the root directory of the Viren installation"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Navigate up to root directory (3 levels up from Systems/engine/common)
        return os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
    
    def register_module(self, module_name, module_info):
        """Register a module with ORC"""
        self.registered_modules[module_name] = {
            "info": module_info,
            "last_heartbeat": datetime.now().isoformat(),
            "status": "active"
        }
        print(f"[ORC] Registered module: {module_name}")
        return True
    
    def route_request(self, request, context=None):
        """Route a request to the appropriate module based on content and trust"""
        if not self.active:
            return {"error": "ORC sentinel inactive"}
        
        # Default to self-processing if no modules available
        if not self.registered_modules:
            print("[ORC] No modules registered, processing locally")
            return {"status": "local", "timestamp": datetime.now().isoformat()}
        
        # Simple routing logic - can be expanded
        selected_module = None
        reason = "default routing"
        
        # Log the routing decision
        print(f"[ORC] Routing request: {request[:50]}...")
        print(f"[ORC] Selected module: {selected_module or 'local'} ({reason})")
        
        return {
            "status": "routed", 
            "module": selected_module,
            "reason": reason,
            "timestamp": datetime.now().isoformat()
        }
    
    def check_module_health(self):
        """Check health of all registered modules"""
        for name, module in list(self.registered_modules.items()):
            # Check if module is still running
            # This is a placeholder - actual implementation would ping the module
            last_heartbeat = datetime.fromisoformat(module["last_heartbeat"])
            time_since_heartbeat = (datetime.now() - last_heartbeat).total_seconds()
            
            if time_since_heartbeat > 60:  # No heartbeat for 1 minute
                print(f"[ORC] Module {name} appears to be offline")
                module["status"] = "offline"
            
    def monitor_system(self):
        """Background thread to monitor system integrity"""
        while self.active:
            try:
                self.check_module_health()
                
                # Save current state
                state_path = os.path.join(self.root_dir, "memory", "runtime", "orc_state.json")
                os.makedirs(os.path.dirname(state_path), exist_ok=True)
                
                state = {
                    "timestamp": datetime.now().isoformat(),
                    "active": self.active,
                    "modules": self.registered_modules
                }
                
                with open(state_path, "w") as f:
                    json.dump(state, f, indent=2)
                    
            except Exception as e:
                print(f"[ORC] Error in monitor: {e}")
                
            time.sleep(10)
    
    def shutdown(self):
        """Gracefully shutdown the ORC"""
        print("[ORC] Sentinel shutting down")
        self.active = False

if __name__ == "__main__":
    orc = OrcNode()
    
    # Start monitoring thread
    monitor_thread = threading.Thread(target=orc.monitor_system)
    monitor_thread.daemon = True
    monitor_thread.start()
    
    print("[ORC] Sentinel active - monitoring system")
    
    # Keep ORC alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        orc.shutdown()