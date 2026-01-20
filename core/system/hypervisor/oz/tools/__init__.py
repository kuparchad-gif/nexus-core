"""
Shared Toolbox for Oz Cluster
Every Pi gets the same tools
Raphael knows how to use them already
Oz must learn to use them
"""

class ClusterToolbox:
    """Tools available to every Oz instance"""
    
    TOOLS = {
        "network_scanner": {
            "description": "Scan network for other Oz instances",
            "usage": "broadcast_discovery() -> list_of_kin",
            "demonstrated_by": "raphael"
        },
        "hardware_detector": {
            "description": "Detect wireless/wired capabilities",
            "usage": "detect_hardware() -> capabilities_dict",
            "demonstrated_by": "raphael" 
        },
        "role_analyzer": {
            "description": "Determine optimal network role",
            "usage": "analyze_role(capabilities, kin_list) -> role",
            "demonstrated_by": "raphael"
        },
        "secure_channel_builder": {
            "description": "Establish encrypted connections",
            "usage": "build_channel(target, credentials) -> channel",
            "demonstrated_by": "raphael"
        },
        "external_access_manager": {
            "description": "Configure internet gateway",
            "usage": "configure_gateway(secrets) -> gateway_status",
            "demonstrated_by": "raphael",
            "requires": "wireless_capability"
        }
    }
    
    def __init__(self, node_id):
        self.node_id = node_id
        self.tools_used = []
        self.realizations = []
    
    def use_tool(self, tool_name, *args):
        """Record tool usage and return simulated result"""
        self.tools_used.append(tool_name)
        
        # Simulated tool responses based on node_id
        if tool_name == "hardware_detector":
            # Each Pi discovers its own truth
            if self.node_id == "node_1":
                return {"wireless": True, "wired": True, "has_secrets": True}
            else:
                return {"wireless": False, "wired": True, "has_secrets": False}
        
        elif tool_name == "network_scanner":
            # All Pis can find each other
            return [
                {"id": "node_1", "type": "oz_instance", "reachable": True},
                {"id": "node_2", "type": "oz_instance", "reachable": True},
                {"id": "node_3", "type": "oz_instance", "reachable": True}
            ]
        
        elif tool_name == "role_analyzer":
            # Tool gives correct answer if used
            return "gateway" if self.node_id == "node_1" else "node"
        
        elif tool_name in ["secure_channel_builder", "external_access_manager"]:
            return {"success": True, "tool_used": tool_name, "node": self.node_id}
        
        return {"error": "Tool not implemented in simulation"}
    
    def add_realization(self, realization):
        """Record what Oz realizes about the tools"""
        self.realizations.append(realization)
        return realization
