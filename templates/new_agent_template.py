# templates/new_agent_template.py
"""
This is a template for creating a new, independent agent or service
that can dynamically connect to the Valhalla/Nexus ecosystem.

To use this template:
1. Copy this file to a new location (e.g., `agents/my_new_agent.py`).
2. Add your agent's specific logic in the `run_mission()` function.
3. Launch your agent after the main `genesis.py` has been started.
"""

import sys
import time
import uuid

# This is a crucial step for a standalone script.
# It adds the project's root directory to the Python path, allowing us to
# import the `valhalla` module from anywhere.
sys.path.append('..')
from valhalla import registry

def discover_core_services(timeout=120):
    """
    Waits for the core services to become available in the registry.
    
    This function demonstrates the standard procedure for any new component
    to find its place in the Nexus.
    """
    print("🔭 Discovering core services...")
    start_time = time.time()
    lillith_address = None
    memory_address = None

    while time.time() - start_time < timeout:
        # Query the Redis registry for the services we need.
        lillith_address = registry.discover_service("lillith_chat")
        memory_address = registry.discover_service("memory_cluster")

        if lillith_address and memory_address:
            print(f"  ✅ Discovered Lillith Core at: {lillith_address}")
            print(f"  ✅ Discovered Memory Cluster at: {memory_address}")
            return lillith_address, memory_address

        print("  ...core services not yet found, still searching...")
        time.sleep(5)

    print("❌ Timed out waiting for core services.")
    return None, None

def run_mission(agent_id, lillith_addr, memory_addr):
    """
    This is the main logic loop for your new agent.
    
    Replace this with your agent's specific tasks.
    """
    print(f"🚀 Mission started for Agent {agent_id}.")
    
    # Example: Your agent could periodically perform a task.
    while True:
        print(f"  -> Agent {agent_id}: Performing my unique task...")
        # Add your logic here.
        # For example, you could:
        # - Connect to the memory cluster at `memory_addr` to add new knowledge.
        # - Send a status update to Lillith at `lillith_addr`.
        # - Interact with an external API.
        time.sleep(30)

if __name__ == "__main__":
    # 1. Give your agent a unique identity.
    agent_id = f"new_agent_{uuid.uuid4().hex[:8]}"
    print(f"🚀 Initializing New Agent: {agent_id}")

    # 2. Discover the core of the Nexus.
    lillith_addr, memory_addr = discover_core_services()

    # 3. Ensure services were found before starting the main mission.
    if not all([lillith_addr, memory_addr]):
        print("Agent shutting down: Could not discover necessary services.")
        sys.exit(1)

    # 4. Launch the agent's main logic.
    try:
        run_mission(agent_id, lillith_addr, memory_addr)
    except KeyboardInterrupt:
        print(f"\nAgent {agent_id} shutting down.")
