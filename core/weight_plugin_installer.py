#!/usr/bin/env python
"""
Weight Plugin Installer - Universal system for managing model weights across different runtimes
"""

import os
import json
import time
import psutil
import subprocess
from typing import Dict, List, Any, Optional
from enum import Enum
from pathlib import Path

class RuntimeType(Enum):
    """Supported LLM runtime types"""
    VLLM = "vllm"
    OLLAMA = "ollama"
    MLX = "mlx"
    LMSTUDIO = "lmstudio"
    UNKNOWN = "unknown"

class WeightType(Enum):
    """Types of weights that can be installed"""
    PERSONALITY = "personality"
    SKILL = "skill"
    KNOWLEDGE = "knowledge"
    BEHAVIOR = "behavior"
    MEMORY = "memory"

class WeightPlugin:
    """A weight plugin with metadata"""
    
    def __init__(self, 
                name: str,
                weight_type: WeightType,
                model_id: str,
                weights_data: Dict[str, Any],
                metadata: Dict[str, Any] = None):
        """Initialize a weight plugin"""
        self.id = f"weight_{int(time.time())}_{id(name)}"
        self.name = name
        self.weight_type = weight_type
        self.model_id = model_id
        self.weights_data = weights_data
        self.metadata = metadata or {}
        self.created_at = time.time()
        self.installed = False
        self.temporary = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "id": self.id,
            "name": self.name,
            "weight_type": self.weight_type.value,
            "model_id": self.model_id,
            "weights_data": self.weights_data,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "installed": self.installed,
            "temporary": self.temporary
        }

class RuntimeDetector:
    """Detects running LLM runtimes"""
    
    @staticmethod
    def detect_running_runtimes() -> List[Dict[str, Any]]:
        """Detect all running LLM runtimes"""
        runtimes = []
        
        # Check running processes
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                proc_info = proc.info
                name = proc_info['name'].lower()
                cmdline = ' '.join(proc_info['cmdline']).lower() if proc_info['cmdline'] else ''
                
                runtime_type = RuntimeType.UNKNOWN
                model_path = None
                
                # Detect vLLM
                if 'vllm' in name or 'vllm' in cmdline:
                    runtime_type = RuntimeType.VLLM
                    # Extract model path from command line
                    if '--model' in cmdline:
                        parts = cmdline.split('--model')
                        if len(parts) > 1:
                            model_path = parts[1].split()[0]
                
                # Detect Ollama
                elif 'ollama' in name or 'ollama' in cmdline:
                    runtime_type = RuntimeType.OLLAMA
                    # Ollama models are typically in ~/.ollama/models
                    model_path = os.path.expanduser("~/.ollama/models")
                
                # Detect MLX
                elif 'mlx' in name or 'mlx' in cmdline:
                    runtime_type = RuntimeType.MLX
                    # MLX models path varies
                    if '--model-path' in cmdline:
                        parts = cmdline.split('--model-path')
                        if len(parts) > 1:
                            model_path = parts[1].split()[0]
                
                # Detect LM Studio
                elif 'lmstudio' in name or 'lm studio' in cmdline:
                    runtime_type = RuntimeType.LMSTUDIO
                    # LM Studio models are typically in user directory
                    model_path = os.path.expanduser("~/.cache/lm-studio/models")
                
                if runtime_type != RuntimeType.UNKNOWN:
                    runtimes.append({
                        "pid": proc_info['pid'],
                        "runtime_type": runtime_type.value,
                        "model_path": model_path,
                        "cmdline": cmdline
                    })
                    
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        return runtimes

class WeightPluginInstaller:
    """Universal weight plugin installer"""
    
    def __init__(self, storage_path: str = None):
        """Initialize the weight plugin installer"""
        self.storage_path = storage_path or os.path.join(os.path.dirname(__file__), "weight_plugins")
        
        # Create storage directories
        self.plugins_path = os.path.join(self.storage_path, "plugins")
        self.templates_path = os.path.join(self.storage_path, "templates")
        self.temp_path = os.path.join(self.storage_path, "temp")
        
        os.makedirs(self.plugins_path, exist_ok=True)
        os.makedirs(self.templates_path, exist_ok=True)
        os.makedirs(self.temp_path, exist_ok=True)
        
        # In-memory stores
        self.plugins = {}  # plugin_id -> WeightPlugin
        self.templates = {}  # model_id -> template_data
        self.runtime_detector = RuntimeDetector()
        
        # Load existing data
        self._load_data()
    
    def _load_data(self):
        """Load plugins and templates from storage"""
        # Load plugins
        plugin_files = [f for f in os.listdir(self.plugins_path) if f.endswith('.json')]
        for file_name in plugin_files:
            try:
                with open(os.path.join(self.plugins_path, file_name), 'r') as f:
                    data = json.load(f)
                    plugin = WeightPlugin(
                        name=data["name"],
                        weight_type=WeightType(data["weight_type"]),
                        model_id=data["model_id"],
                        weights_data=data["weights_data"],
                        metadata=data["metadata"]
                    )
                    plugin.id = data["id"]
                    plugin.created_at = data["created_at"]
                    plugin.installed = data["installed"]
                    plugin.temporary = data["temporary"]
                    self.plugins[plugin.id] = plugin
            except Exception as e:
                print(f"Error loading plugin {file_name}: {e}")
        
        print(f"Loaded {len(self.plugins)} plugins")
    
    def detect_system_state(self) -> Dict[str, Any]:
        """Detect current system state and running models"""
        # Detect running runtimes
        runtimes = self.runtime_detector.detect_running_runtimes()
        
        # Get system info
        system_info = {
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total,
            "disk_usage": psutil.disk_usage('/').percent if os.name != 'nt' else psutil.disk_usage('C:').percent
        }
        
        return {
            "runtimes": runtimes,
            "system_info": system_info,
            "timestamp": time.time()
        }
    
    def create_weight_plugin(self, 
                           name: str,
                           weight_type: WeightType,
                           model_id: str,
                           weights_data: Dict[str, Any],
                           metadata: Dict[str, Any] = None) -> str:
        """Create a new weight plugin"""
        # Create plugin
        plugin = WeightPlugin(
            name=name,
            weight_type=weight_type,
            model_id=model_id,
            weights_data=weights_data,
            metadata=metadata
        )
        
        # Store plugin
        self.plugins[plugin.id] = plugin
        
        return plugin.id
    
    def install_plugin(self, plugin_id: str, target_runtime: str = None) -> Dict[str, Any]:
        """Install a weight plugin to a running model"""
        if plugin_id not in self.plugins:
            return {"success": False, "error": "Plugin not found"}
        
        plugin = self.plugins[plugin_id]
        
        # Detect system state
        system_state = self.detect_system_state()
        
        # Find target runtime
        target = None
        if target_runtime:
            for runtime in system_state["runtimes"]:
                if runtime["runtime_type"] == target_runtime:
                    target = runtime
                    break
        else:
            # Use first available runtime
            if system_state["runtimes"]:
                target = system_state["runtimes"][0]
        
        if not target:
            return {"success": False, "error": "No suitable runtime found"}
        
        # Mark as installed
        plugin.installed = True
        
        return {"success": True, "runtime": target["runtime_type"]}
    
    def create_master_template(self, model_id: str, plugins: List[str]) -> Dict[str, Any]:
        """Create a master template from installed plugins"""
        if not plugins:
            return {"success": False, "error": "No plugins provided"}
        
        # Collect weights from all plugins
        combined_weights = {}
        plugin_info = []
        
        for plugin_id in plugins:
            if plugin_id in self.plugins:
                plugin = self.plugins[plugin_id]
                
                # Merge weights
                for key, value in plugin.weights_data.items():
                    if key not in combined_weights:
                        combined_weights[key] = []
                    combined_weights[key].append(value)
                
                plugin_info.append({
                    "id": plugin.id,
                    "name": plugin.name,
                    "type": plugin.weight_type.value
                })
        
        # Average numerical values, concatenate strings
        final_weights = {}
        for key, values in combined_weights.items():
            if all(isinstance(v, (int, float)) for v in values):
                final_weights[key] = sum(values) / len(values)
            elif all(isinstance(v, str) for v in values):
                final_weights[key] = " ".join(values)
            else:
                final_weights[key] = values[-1]  # Use last value
        
        # Create template
        template = {
            "model_id": model_id,
            "weights": final_weights,
            "plugins": plugin_info,
            "created_at": time.time(),
            "version": "1.0"
        }
        
        # Store template
        self.templates[model_id] = template
        
        return {"success": True, "template": template}
    
    def distribute_to_archivers(self, template_id: str) -> Dict[str, Any]:
        """Distribute finalized weights to archivers for hot storage"""
        if template_id not in self.templates:
            return {"success": False, "error": "Template not found"}
        
        template = self.templates[template_id]
        
        # Create distribution package
        distribution_package = {
            "template_id": template_id,
            "model_id": template["model_id"],
            "weights": template["weights"],
            "metadata": {
                "created_at": template["created_at"],
                "version": template["version"],
                "plugin_count": len(template["plugins"])
            },
            "distribution_timestamp": time.time()
        }
        
        # Save to distribution folder for archivers to pick up
        dist_path = os.path.join(self.storage_path, "distribution")
        os.makedirs(dist_path, exist_ok=True)
        
        dist_file = os.path.join(dist_path, f"dist_{template_id}_{int(time.time())}.json")
        try:
            with open(dist_file, 'w') as f:
                json.dump(distribution_package, f, indent=2)
            
            return {
                "success": True,
                "distribution_file": dist_file,
                "package": distribution_package
            }
        except Exception as e:
            return {"success": False, "error": str(e)}