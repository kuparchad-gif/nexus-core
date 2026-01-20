# valhalla/scouts.py
import random
import time
from .providers import NodeProvider

class CloudScout(NodeProvider):
    """A scout that discovers resources from major cloud providers."""
    
    def __init__(self, providers=["oracle", "aws", "gcp", "azure"]):
        self.providers = providers
        
    def discover_resources(self):
        """Simulates the discovery of free-tier cloud resources."""
        print("🕵️‍♂️  Deploying cloud scouts...")
        discoveries = []
        for provider in self.providers:
            time.sleep(random.uniform(0.1, 0.5))
            if provider == "oracle":
                discoveries.append({
                    "provider": "oracle",
                    "resource": "VM.Standard.A1.Flex",
                    "cores": 4,
                    "ram_gb": 24,
                    "region": random.choice(["us-phoenix-1", "us-ashburn-1"]),
                    "status": "available",
                })
            elif provider == "aws":
                discoveries.append({
                    "provider": "aws",
                    "resource": "t4g.micro",
                    "cores": 2,
                    "ram_gb": 1,
                    "region": random.choice(["us-east-1", "us-west-2"]),
                    "status": "available",
                })
        print(f"✅ Scouts returned with {len(discoveries)} potential nodes.")
        return discoveries
        
    def provision_node(self, specs):
        """Simulates provisioning a cloud VM."""
        print(f"  -> Provisioning {specs['resource']} in {specs['provider']}...")
        time.sleep(random.uniform(0.5, 1.5))
        node_id = f"{specs['provider']}-{random.randint(1000, 9999)}"
        print(f"  ✅ Provisioned node {node_id}")
        return {"id": node_id, "status": "running"}
        
    def get_node_credentials(self, node_id):
        """Generates mock credentials for a provisioned cloud node."""
        return {
            "ip_address": f"1{random.randint(10, 99)}.1{random.randint(10, 99)}.0.{random.randint(2, 254)}",
            "ssh_user": "ubuntu",
            "ssh_key_name": f"{node_id}-key"
        }

if __name__ == "__main__":
    scout = CloudScout()
    resources = scout.discover_resources()
    if resources:
        first_resource = resources[0]
        node = scout.provision_node(first_resource)
        credentials = scout.get_node_credentials(node["id"])
        print(f"Node '{node['id']}' is accessible at {credentials['ip_address']}")
