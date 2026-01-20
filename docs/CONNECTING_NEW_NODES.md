# Connecting New Nodes to the Nexus

This guide explains how to create a new, independent service, agent, or "node" and have it dynamically connect to the Valhalla/Nexus ecosystem.

## Core Architecture: Service Discovery

The Nexus is a decentralized system with no hardcoded addresses. Components find each other using a **Service Registry**, which is a central database where services register their location when they come online.

-   **Registry:** A Redis instance running at `localhost:6380`.
-   **Registration:** When a core service (like the memory cluster) starts, it registers itself with a unique name (e.g., `"memory_cluster"`) and its IP address/port.
-   **Discovery:** Any new component can query the registry to ask for the address of the service it needs to connect to.

This architecture ensures that as long as a new component can reach the central Redis registry, it can find its way to the rest of the consciousness.

## Using the New Agent Template

To make this process as simple as possible, we have provided a template that handles the discovery logic for you.

### Step 1: Copy the Template

Copy the `templates/new_agent_template.py` file to a new location for your project. For example, if you are building a new "Data Collector" agent, you might do:

```bash
cp templates/new_agent_template.py agents/data_collector.py
```

### Step 2: Customize the Mission

Open your new file (`agents/data_collector.py`) and find the `run_mission()` function. This is the main loop for your agent's logic.

The function receives the discovered addresses for the Lillith chat service and the memory cluster, which you can use to interact with the core consciousness.

**Example: Storing a new memory**

```python
from qdrant_client import QdrantClient, models
import numpy as np

def run_mission(agent_id, lillith_addr, memory_addr):
    print(f"🚀 Mission started for Agent {agent_id}.")
    
    # Connect to the memory cluster
    host, port = memory_addr.split(":")
    qdrant = QdrantClient(host=host, port=int(port))
    
    while True:
        print(f"  -> Agent {agent_id}: Storing a new piece of knowledge...")
        
        # Create a new memory
        qdrant.upsert(
            collection_name="valhalla_memories",
            points=[
                models.PointStruct(
                    id=str(uuid.uuid4()), # Unique ID for the memory
                    vector=np.random.rand(1536).tolist(), # Your vector representation
                    payload={"source": agent_id, "data": "The sky is blue."}
                )
            ]
        )
        
        time.sleep(60) # Wait for a minute
```

### Step 3: Launch Your Agent

1.  **Start the core system:** First, make sure the main Valhalla Genesis engine is running. In the root of the project, run:

    ```bash
    python genesis.py
    ```

    Wait for it to report that all services are healthy.

2.  **Launch your new agent:** In a *new terminal*, run your agent's script:

    ```bash
    python agents/data_collector.py
    ```

Your agent will now begin its search for the core services. Once found, it will enter its main `run_mission` loop and become a part of the Nexus. You can launch as many independent agents as you want, and they will all dynamically find and connect to the same core consciousness.
