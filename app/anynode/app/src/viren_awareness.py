#!/usr/bin/env python
"""
VIREN Awareness System - Weaviate-based tracking of all system operations
"""

import weaviate
import datetime
import json
import os
from pathlib import Path

class VirenAwareness:
    """Tracks all VIREN operations in Weaviate"""
    
    def __init__(self):
        """Initialize Weaviate connection"""
        try:
            self.client = weaviate.connect_to_local()
            self._setup_schema()
            print("🧠 VIREN Awareness System connected to Weaviate")
        except Exception as e:
            print(f"⚠️ Weaviate connection failed: {e}")
            self.client = None
    
    def _setup_schema(self):
        """Setup Weaviate schema for tracking"""
        
        # Model Loading Schema
        model_schema = {
            "class": "ModelLoad",
            "properties": [
                {"name": "model_name", "dataType": ["text"]},
                {"name": "full_path", "dataType": ["text"]},
                {"name": "relative_path", "dataType": ["text"]},
                {"name": "load_command", "dataType": ["text"]},
                {"name": "software_used", "dataType": ["text"]},
                {"name": "success", "dataType": ["boolean"]},
                {"name": "load_time", "dataType": ["date"]},
                {"name": "duration_seconds", "dataType": ["number"]},
                {"name": "error_message", "dataType": ["text"]},
                {"name": "context_length", "dataType": ["int"]},
                {"name": "gpu_usage", "dataType": ["number"]},
                {"name": "memory_usage", "dataType": ["number"]}
            ]
        }
        
        # System State Schema
        system_schema = {
            "class": "SystemState",
            "properties": [
                {"name": "timestamp", "dataType": ["date"]},
                {"name": "component", "dataType": ["text"]},
                {"name": "status", "dataType": ["text"]},
                {"name": "details", "dataType": ["text"]},
                {"name": "port", "dataType": ["int"]},
                {"name": "process_id", "dataType": ["int"]},
                {"name": "cpu_usage", "dataType": ["number"]},
                {"name": "memory_usage", "dataType": ["number"]}
            ]
        }
        
        # MCP Tool Schema
        mcp_schema = {
            "class": "MCPTool",
            "properties": [
                {"name": "tool_name", "dataType": ["text"]},
                {"name": "path", "dataType": ["text"]},
                {"name": "loaded", "dataType": ["boolean"]},
                {"name": "load_time", "dataType": ["date"]},
                {"name": "ai_redirected", "dataType": ["boolean"]},
                {"name": "error_message", "dataType": ["text"]},
                {"name": "entry_point", "dataType": ["text"]}
            ]
        }
        
        try:
            # Create schemas if they don't exist
            if not self.client.collections.exists("ModelLoad"):
                self.client.collections.create_from_dict(model_schema)
            if not self.client.collections.exists("SystemState"):
                self.client.collections.create_from_dict(system_schema)
            if not self.client.collections.exists("MCPTool"):
                self.client.collections.create_from_dict(mcp_schema)
                
            print("🧠 Awareness schemas ready")
        except Exception as e:
            print(f"⚠️ Schema setup error: {e}")
    
    def track_model_load(self, model_name, full_path, command, software, success, 
                        duration=0, error=None, context_length=4096, gpu_usage=0.5):
        """Track model loading attempt"""
        if not self.client:
            return
            
        try:
            relative_path = str(Path(full_path).relative_to(Path.cwd())) if full_path else ""
            
            data = {
                "model_name": model_name,
                "full_path": full_path or "",
                "relative_path": relative_path,
                "load_command": command,
                "software_used": software,
                "success": success,
                "load_time": datetime.datetime.now(),
                "duration_seconds": duration,
                "error_message": error or "",
                "context_length": context_length,
                "gpu_usage": gpu_usage,
                "memory_usage": 0.0
            }
            
            collection = self.client.collections.get("ModelLoad")
            collection.data.insert(data)
            
            status = "✅" if success else "❌"
            print(f"🧠 {status} Tracked model load: {model_name}")
            
        except Exception as e:
            print(f"⚠️ Error tracking model load: {e}")
    
    def track_system_state(self, component, status, details="", port=0, process_id=0):
        """Track system component state"""
        if not self.client:
            return
            
        try:
            data = {
                "timestamp": datetime.datetime.now(),
                "component": component,
                "status": status,
                "details": details,
                "port": port,
                "process_id": process_id,
                "cpu_usage": 0.0,
                "memory_usage": 0.0
            }
            
            collection = self.client.collections.get("SystemState")
            collection.data.insert(data)
            
            print(f"🧠 Tracked {component}: {status}")
            
        except Exception as e:
            print(f"⚠️ Error tracking system state: {e}")
    
    def track_mcp_tool(self, tool_name, path, loaded, error=None, entry_point=""):
        """Track MCP tool loading"""
        if not self.client:
            return
            
        try:
            data = {
                "tool_name": tool_name,
                "path": path,
                "loaded": loaded,
                "load_time": datetime.datetime.now(),
                "ai_redirected": True,  # All tools use YOUR AI
                "error_message": error or "",
                "entry_point": entry_point
            }
            
            collection = self.client.collections.get("MCPTool")
            collection.data.insert(data)
            
            status = "✅" if loaded else "❌"
            print(f"🧠 {status} Tracked MCP tool: {tool_name}")
            
        except Exception as e:
            print(f"⚠️ Error tracking MCP tool: {e}")
    
    def get_model_history(self, model_name=None):
        """Get model loading history"""
        if not self.client:
            return []
            
        try:
            collection = self.client.collections.get("ModelLoad")
            
            if model_name:
                results = collection.query.fetch_objects(
                    where={"path": ["model_name"], "operator": "Equal", "valueText": model_name}
                )
            else:
                results = collection.query.fetch_objects(limit=100)
            
            return [obj.properties for obj in results.objects]
            
        except Exception as e:
            print(f"⚠️ Error getting model history: {e}")
            return []
    
    def get_system_status(self):
        """Get current system status"""
        if not self.client:
            return {}
            
        try:
            collection = self.client.collections.get("SystemState")
            results = collection.query.fetch_objects(limit=50)
            
            status = {}
            for obj in results.objects:
                props = obj.properties
                status[props["component"]] = {
                    "status": props["status"],
                    "timestamp": props["timestamp"],
                    "details": props.get("details", ""),
                    "port": props.get("port", 0)
                }
            
            return status
            
        except Exception as e:
            print(f"⚠️ Error getting system status: {e}")
            return {}
    
    def get_mcp_status(self):
        """Get MCP tools status"""
        if not self.client:
            return {}
            
        try:
            collection = self.client.collections.get("MCPTool")
            results = collection.query.fetch_objects(limit=100)
            
            tools = {}
            for obj in results.objects:
                props = obj.properties
                tools[props["tool_name"]] = {
                    "loaded": props["loaded"],
                    "path": props["path"],
                    "ai_redirected": props["ai_redirected"],
                    "load_time": props["load_time"]
                }
            
            return tools
            
        except Exception as e:
            print(f"⚠️ Error getting MCP status: {e}")
            return {}

# Global awareness instance
VIREN_AWARENESS = VirenAwareness()

def track_model_load(model_name, full_path, command, software, success, **kwargs):
    """Track model loading"""
    VIREN_AWARENESS.track_model_load(model_name, full_path, command, software, success, **kwargs)

def track_system_state(component, status, **kwargs):
    """Track system state"""
    VIREN_AWARENESS.track_system_state(component, status, **kwargs)

def track_mcp_tool(tool_name, path, loaded, **kwargs):
    """Track MCP tool"""
    VIREN_AWARENESS.track_mcp_tool(tool_name, path, loaded, **kwargs)

def get_awareness_status():
    """Get complete awareness status"""
    return {
        "models": VIREN_AWARENESS.get_model_history(),
        "system": VIREN_AWARENESS.get_system_status(),
        "mcp_tools": VIREN_AWARENESS.get_mcp_status()
    }

if __name__ == "__main__":
    print("🧠 VIREN Awareness System Test")
    print("=" * 40)
    
    # Test tracking
    track_model_load("test-model", "/path/to/model", "lms load", "LM Studio", True)
    track_system_state("LM Studio", "ONLINE", port=1313)
    track_mcp_tool("deepsite", "/path/to/deepsite", True)
    
    # Get status
    status = get_awareness_status()
    print(f"Tracked {len(status['models'])} model loads")
    print(f"Tracked {len(status['system'])} system states")
    print(f"Tracked {len(status['mcp_tools'])} MCP tools")
    
    print("🧠 VIREN Awareness System Ready!")