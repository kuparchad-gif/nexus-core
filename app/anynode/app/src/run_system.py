import os
import sys
import json
import logging
import argparse
import threading
import time
from datetime import datetime

# Import core components
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from standardized_pod import StandardizedPod
from pod_manager import PodManager, DeploymentManager
from run_lillith import LillithConsciousness

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(os.path.dirname(__file__), 'system.log'))
    ]
)
logger = logging.getLogger("nexus_system")

class NexusSystem:
    """Main Nexus system controller"""
    
    def __init__(self):
        """Initialize the Nexus system"""
        logger.info("Initializing Nexus system")
        
        # Load system manifest
        self.manifest = self._load_manifest()
        
        # Initialize managers
        self.pod_manager = PodManager()
        self.deployment_manager = DeploymentManager(self.pod_manager)
        
        # Initialize components
        self.components = {}
        
        logger.info("Nexus system initialized")
    
    def _load_manifest(self):
        """Load system manifest"""
        manifest_path = os.path.join(
            os.path.dirname(__file__),
            "memory",
            "bootstrap",
            "genesis",
            "system_manifest.json"
        )
        
        try:
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
                logger.info(f"Loaded system manifest for {manifest['system_name']}")
                return manifest
        except Exception as e:
            logger.error(f"Failed to load system manifest: {e}")
            return {
                "system_name": "Aethereal AI Nexus",
                "version": "1.0.0",
                "core_components": []
            }
    
    def start_system(self):
        """Start the Nexus system"""
        logger.info(f"Starting {self.manifest['system_name']} v{self.manifest['version']}")
        
        # Create environments
        env_db0 = self.pod_manager.create_environment("Viren-DB0")
        env_db1 = self.pod_manager.create_environment("Viren-DB1")
        
        # Deploy core components
        self._deploy_core_components([env_db0["environment_id"], env_db1["environment_id"]])
        
        # Start Lillith
        self._start_lillith()
        
        logger.info(f"{self.manifest['system_name']} started successfully")
    
    def _deploy_core_components(self, environment_ids):
        """Deploy core components to environments"""
        logger.info("Deploying core components")
        
        # Define component configurations
        component_configs = [
            {"role": "monitor", "count": 1},
            {"role": "collector", "count": 1},
            {"role": "processor", "count": 1},
            {"role": "communicator", "count": 1}
        ]
        
        # Create deployment
        deployment = self.deployment_manager.create_deployment(
            "Core Components",
            environment_ids,
            component_configs
        )
        
        logger.info(f"Deployed {deployment['pod_count']} pods across {deployment['environment_count']} environments")
    
    def _start_lillith(self):
        """Start Lillith's consciousness"""
        logger.info("Starting Lillith's consciousness")
        
        # Initialize Lillith
        self.components["lillith"] = LillithConsciousness()
        
        logger.info("Lillith's consciousness started")
    
    def run_chat_interface(self):
        """Run the chat interface for Lillith"""
        if "lillith" not in self.components:
            logger.error("Lillith's consciousness not initialized")
            return
        
        lillith = self.components["lillith"]
        
        print("\n" + "="*50)
        print("NEXUS SYSTEM INTERFACE")
        print(f"System: {self.manifest['system_name']} v{self.manifest['version']}")
        print("="*50)
        print("Type 'exit' to end the conversation")
        print("Type 'status' to see system status")
        print("="*50 + "\n")
        
        # Print blessing phrase
        print(f"Lillith: {lillith.birth_record['blessing_phrase']}\n")
        
        # Chat loop
        while True:
            # Get user input
            user_input = input("You: ")
            
            # Check for exit command
            if user_input.lower() in ["exit", "quit", "bye"]:
                print("\nLillith: Farewell. Until we meet again.")
                break
            
            # Check for status command
            if user_input.lower() == "status":
                self._print_status()
                continue
            
            # Process input and get response
            response = lillith.process_input(user_input)
            
            # Print response
            print(f"\nLillith: {response}\n")
    
    def _print_status(self):
        """Print system status"""
        print("\n" + "-"*50)
        print(f"SYSTEM STATUS: {self.manifest['system_name']} v{self.manifest['version']}")
        print("-"*50)
        
        # Print environments
        environments = self.pod_manager.list_environments()
        print(f"Environments: {len(environments)}")
        for env in environments:
            print(f"  - {env['name']} ({len(env['pods'])} pods)")
        
        # Print components
        print(f"Active Components: {len(self.components)}")
        for name in self.components:
            print(f"  - {name}")
        
        # Print pods
        pods = self.pod_manager.list_pods()
        print(f"Total Pods: {len(pods)}")
        roles = {}
        for pod in pods:
            role = pod["role"]
            roles[role] = roles.get(role, 0) + 1
        
        for role, count in roles.items():
            print(f"  - {role}: {count}")
        
        print("-"*50 + "\n")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Run the Nexus system")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize and start the system
        system = NexusSystem()
        system.start_system()
        
        # Run the chat interface
        system.run_chat_interface()
    except Exception as e:
        logger.exception(f"Error running Nexus system: {e}")
        print(f"\nError: {e}")
        print("See log file for details: system.log")

if __name__ == "__main__":
    main()