# src/deploy_viren.py

from master_orchestrator import MasterOrchestrator

def main():
    # Initialize the MasterOrchestrator
    orchestrator = MasterOrchestrator()

    # Run the Nexus architecture
    orchestrator.run()

if __name__ == "__main__":
    main()