#!/usr/bin/env python
"""
MCP Bridge Loader - Loads all MCP tools into the bridge system
"""

import os
import sys
import importlib
from pathlib import Path

class MCPBridgeLoader:
    """Loads all MCP tools into Viren's bridge system"""
    
    def __init__(self):
        """Initialize MCP bridge loader"""
        self.mcp_tools = {}
        self.loaded_modules = {}
        self.bridge_ready = False
        
        # Get MCP utils path
        self.mcp_path = Path(__file__).parent / "mcp_utils"
        
        print("🌉 MCP Bridge Loader initialized")
    
    def scan_mcp_tools(self):
        """Scan all MCP tools in mcp_utils directory"""
        
        mcp_tools = []
        
        # Scan all directories in mcp_utils
        for item in self.mcp_path.iterdir():
            if item.is_dir() and not item.name.startswith('__'):
                mcp_tools.append({
                    "name": item.name,
                    "path": str(item),
                    "type": "mcp_tool",
                    "loaded": False
                })
        
        # Scan Python files in mcp_utils
        for item in self.mcp_path.glob("*.py"):
            if not item.name.startswith('__'):
                mcp_tools.append({
                    "name": item.stem,
                    "path": str(item),
                    "type": "mcp_module",
                    "loaded": False
                })
        
        self.mcp_tools = {tool["name"]: tool for tool in mcp_tools}
        
        print(f"🔍 Found {len(self.mcp_tools)} MCP tools:")
        for name in self.mcp_tools.keys():
            print(f"   • {name}")
        
        return self.mcp_tools
    
    def load_mcp_tool(self, tool_name):
        """Load specific MCP tool"""
        
        if tool_name not in self.mcp_tools:
            return {"success": False, "error": f"Tool {tool_name} not found"}
        
        tool = self.mcp_tools[tool_name]
        
        try:
            if tool["type"] == "mcp_module":
                # Load Python module
                import importlib.util
                spec = importlib.util.spec_from_file_location(tool_name, tool["path"])
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                self.loaded_modules[tool_name] = module
                tool["loaded"] = True
                
                return {
                    "success": True,
                    "tool": tool_name,
                    "type": "module",
                    "module": module
                }
            
            elif tool["type"] == "mcp_tool":
                # Load tool directory (check for main files)
                tool_path = Path(tool["path"])
                
                # Look for common entry points
                entry_points = ["main.py", "app.py", "server.py", "__init__.py"]
                
                for entry in entry_points:
                    entry_file = tool_path / entry
                    if entry_file.exists():
                        import importlib.util
                        from pathlib import Path
                        entry_path = Path(entry_file)
                        spec = importlib.util.spec_from_file_location(f"{tool_name}_{entry_path.stem}", str(entry_file))
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        
                        self.loaded_modules[tool_name] = module
                        tool["loaded"] = True
                        
                        return {
                            "success": True,
                            "tool": tool_name,
                            "type": "tool_directory",
                            "entry_point": entry,
                            "module": module
                        }
                
                # If no entry point found, mark as available but not loaded
                tool["loaded"] = "available"
                return {
                    "success": True,
                    "tool": tool_name,
                    "type": "tool_directory",
                    "status": "available_no_entry_point"
                }
        
        except Exception as e:
            return {
                "success": False,
                "tool": tool_name,
                "error": str(e)
            }
    
    def load_all_mcp_tools(self):
        """Load all MCP tools into bridge"""
        
        print("🌉 Loading all MCP tools into bridge...")
        
        # First scan for tools
        self.scan_mcp_tools()
        
        results = {
            "loaded": [],
            "available": [],
            "failed": [],
            "total": len(self.mcp_tools)
        }
        
        # Load each tool
        for tool_name in self.mcp_tools.keys():
            print(f"   Loading {tool_name}...")
            
            result = self.load_mcp_tool(tool_name)
            
            if result["success"]:
                if result.get("status") == "available_no_entry_point":
                    results["available"].append(tool_name)
                    print(f"   ✓ {tool_name}: Available (no entry point)")
                else:
                    results["loaded"].append(tool_name)
                    print(f"   ✅ {tool_name}: Loaded")
            else:
                results["failed"].append({
                    "tool": tool_name,
                    "error": result["error"]
                })
                print(f"   ❌ {tool_name}: Failed - {result['error']}")
        
        self.bridge_ready = True
        
        print(f"\n🌉 MCP Bridge Status:")
        print(f"   Loaded: {len(results['loaded'])}")
        print(f"   Available: {len(results['available'])}")
        print(f"   Failed: {len(results['failed'])}")
        print(f"   Total: {results['total']}")
        
        return results
    
    def get_loaded_tools(self):
        """Get list of loaded tools"""
        return {
            name: tool for name, tool in self.mcp_tools.items() 
            if tool.get("loaded") == True
        }
    
    def get_available_tools(self):
        """Get list of available tools"""
        return {
            name: tool for name, tool in self.mcp_tools.items() 
            if tool.get("loaded") in [True, "available"]
        }
    
    def call_mcp_tool(self, tool_name, method=None, *args, **kwargs):
        """Call method on loaded MCP tool"""
        
        if tool_name not in self.loaded_modules:
            return {"error": f"Tool {tool_name} not loaded"}
        
        module = self.loaded_modules[tool_name]
        
        try:
            if method:
                if hasattr(module, method):
                    func = getattr(module, method)
                    return func(*args, **kwargs)
                else:
                    return {"error": f"Method {method} not found in {tool_name}"}
            else:
                # Return module info
                return {
                    "module": tool_name,
                    "attributes": [attr for attr in dir(module) if not attr.startswith('_')],
                    "loaded": True
                }
        
        except Exception as e:
            return {"error": f"Error calling {tool_name}.{method}: {str(e)}"}

# Global bridge loader
MCP_BRIDGE = MCPBridgeLoader()

def load_all_mcp_tools():
    """Load all MCP tools"""
    return MCP_BRIDGE.load_all_mcp_tools()

def get_mcp_tools():
    """Get loaded MCP tools"""
    return MCP_BRIDGE.get_loaded_tools()

def call_mcp_tool(tool_name, method=None, *args, **kwargs):
    """Call MCP tool method"""
    return MCP_BRIDGE.call_mcp_tool(tool_name, method, *args, **kwargs)

# Example usage
if __name__ == "__main__":
    print("🌉 MCP Bridge Loader Test")
    print("=" * 40)
    
    # Load all tools
    results = load_all_mcp_tools()
    
    # Show loaded tools
    loaded_tools = get_mcp_tools()
    print(f"\n🔧 Loaded Tools:")
    for name, tool in loaded_tools.items():
        print(f"   {name}: {tool['path']}")
    
    print(f"\n🌉 MCP Bridge Ready with {len(loaded_tools)} tools!")