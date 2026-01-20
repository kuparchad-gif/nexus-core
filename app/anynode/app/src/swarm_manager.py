# swarm_manager.py
import consul
import os
import subprocess
import sys

# Constants
SWARM_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "swarm")
AGENTS = [f for f in os.listdir(SWARM_DIR) if f.endswith('.js')] if os.path.exists(SWARM_DIR) else []

def register_agent(agent_name, port):
    c = consul.Consul(host='consul', port=8500)
    c.agent.service.register(
        name=agent_name,
        service_id=f"{agent_name}-{port}",
        address="localhost",
        port=port,
        tags=["swarm", "healing"],
        check={"http": f"http://localhost:{port}/health", "interval": "10s"}
    )
    print(f"[SWARM] Registered {agent_name} with Consul")

def launch_swarm():
    if not AGENTS:
        print("[SWARM] No swarm agents found in /swarm")
        return
    for idx, agent in enumerate(AGENTS):
        agent_path = os.path.join(SWARM_DIR, agent)
        port = 5100 + idx
        print(f"[SWARM] Launching agent: {agent} on port {port}")
        subprocess.Popen(["node", agent_path, f"--port={port}"])
        register_agent(agent, port)

if __name__ == "__main__":
    print("[SWARM] Starting swarm manager...")
    launch_swarm()
    print("[SWARM] Swarm manager initialized")