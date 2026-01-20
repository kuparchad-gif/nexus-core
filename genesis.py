# genesis.py
import time
from valhalla import scouts, orchestrator, registry

def eternal_watch():
    """Displays the final status and waits for interruption."""
    print("\n======================================================")
    print("🏰 VALHALLA GENESIS COMPLETE")
    print("======================================================")
    print("\nLillith Core:    ACTIVE (http://localhost:8001)")
    print("Memory Cluster:  ACTIVE (http://localhost:6333)")
    print("Inference Engine: ACTIVE (http://localhost:7860)")
    print("\nTo chat with Lillith, POST to http://localhost:8001/chat")
    print("Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🔥 VALHALLA STANDS. THE GHOSTS REMEMBER. 🔥")

if __name__ == "__main__":
    print("🔥 VALHALLA GENESIS INITIATED (Cloud First) 🔥")

    # 1. Discover and provision nodes
    cloud_scout = scouts.CloudScout()
    resources = cloud_scout.discover_resources()
    nodes = [cloud_scout.provision_node(res) for res in resources]

    # 2. Orchestrate service deployment
    cloud_orchestrator = orchestrator.CloudOrchestrator(cloud_scout)
    if cloud_orchestrator.deploy_core_services(nodes):
        # 3. Deploy the Spirallaspan Genesis Node
        if cloud_orchestrator.deploy_spirallaspan_node():
            print("✅ Spirallaspan Genesis Node deployed.")
            # The Spirallaspan node will register itself, so no need to do it here.
            eternal_watch()
    else:
        print("\n❌ VALHALLA GENESIS FAILED. CHECK SERVICE DEPLOYMENT.")
