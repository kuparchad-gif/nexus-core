# valhalla/providers.py
from abc import ABC, abstractmethod

class NodeProvider(ABC):
    """
    Abstract base class for a provider of compute nodes.
    This creates a hook for different deployment targets (e.g., cloud, Pi cluster).
    """
    
    @abstractmethod
    def discover_resources(self):
        """Discover available resources (VMs, Pis, etc.)."""
        pass
        
    @abstractmethod
    def provision_node(self, specs):
        """Provision a new node with the given specifications."""
        pass
        
    @abstractmethod
    def get_node_credentials(self, node_id):
        """Get the credentials required to access a provisioned node."""
        pass

# Example of how it will be used later
class PiClusterProvider(NodeProvider):
    def discover_resources(self):
        # Logic to scan the local network for available Pis
        print("Scanning for Raspberry Pi nodes on the local network...")
        return []

    def provision_node(self, specs):
        # In a Pi cluster, provisioning might mean assigning a role
        print(f"Assigning role to Pi node with specs: {specs}")
        return {}

    def get_node_credentials(self, node_id):
        # Credentials might be a pre-shared SSH key
        print(f"Getting credentials for Pi node: {node_id}")
        return {}
