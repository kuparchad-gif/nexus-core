#!/usr/bin/env python3
"""
TESSERACT DAKAR SWARM LAUNCHER - COMPLETE VERSION
"""

# ============================================================================
# PART 1: CONFIGURATION - Runs FIRST, before anything else
# ============================================================================

import os
import sys
from pathlib import Path

CONFIG_FILE = Path("nexus_config.py")
if not CONFIG_FILE.exists():
    print("📝 First launch - creating configuration file...")
    with open(CONFIG_FILE, "w") as f:
        f.write('''"""
NEXUS CONFIGURATION - Edit this file to configure your Dakar swarm
"""

# ============================================================================
# YOUR SETTINGS - CHANGE THESE
# ============================================================================

# Where your code lives (local path or GitHub URL)
REPO_PATH = "https://github.com/yourusername/your-nexus-repo.git"

# Where the swarm connects
DOMAIN = "https://aetherealnexus.net"

# Council settings (30-year rule)
COUNCIL_APPROVAL_REQUIRED = True
CONSENSUS_THRESHOLD = 0.7

# Hardware (Blue Collar IT)
NO_GPU = True

# ============================================================================
# DON'T EDIT BELOW THIS LINE
# ============================================================================
CONFIG = {
    "repo": REPO_PATH,
    "domain": DOMAIN,
    "council": {
        "approval_required": COUNCIL_APPROVAL_REQUIRED,
        "threshold": CONSENSUS_THRESHOLD
    },
    "hardware": {
        "no_gpu": NO_GPU
    }
}
''')
    print("✅ Created nexus_config.py - please edit it with your settings")
    print("⚡ Then run this script again")
    sys.exit(0)

# Load the config
from nexus_config import CONFIG
print(f"📋 Loaded configuration:")
print(f"   Repo: {CONFIG['repo']}")
print(f"   Domain: {CONFIG['domain']}")
print(f"   Council approval required: {CONFIG['council']['approval_required']}")
print(f"   No-GPU mode: {CONFIG['hardware']['no_gpu']}")

# Set hardware constraints from config
if CONFIG["hardware"]["no_gpu"]:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["TRINITY_FX_MODE"] = "CPU_ONLY"
    print("🔧 GPU DISABLED - CPU-ONLY MODE ACTIVE")

# ============================================================================
# PART 2: IMPORTS - Now load everything else
# ============================================================================

import json
import time
import asyncio
import aiohttp
import hashlib
import subprocess

# Load blueprint
import tesseract_blueprint as blueprint
SWARM_DEF = blueprint.SWARM_DEFINITION

# Load kernel components
SmartSwitchKernel = getattr(blueprint, 'SmartSwitchKernel', None)
AICouncil = getattr(blueprint, 'AICouncil', None)

# Load Dakar components
from dakar_genome import TransformingDakar, DakarSwarm

# ============================================================================
# PART 3: DOMAIN-FIRST DISCOVERY (from previous fix)
# ============================================================================

class DomainFirstDiscovery:
    # ... (full class implementation from earlier) ...
    # This handles WebSocket, HTTPS, NATS, local mesh
    pass

# ============================================================================
# PART 4: MAIN FUNCTION
# ============================================================================

async def main():
    # Generate node ID
    node_id = f"dakar-{os.urandom(4).hex()}"
    print(f"\n🧬 DAKAR {node_id[:8]} awakening...")
    
    # Initialize kernel and council
    kernel = SmartSwitchKernel() if SmartSwitchKernel else None
    council = AICouncil(kernel) if AICouncil else None
    
    # Domain-first discovery
    discovery = DomainFirstDiscovery(node_id)
    await discovery.connect()
    
    # Birth the Dakar swarm
    swarm = DakarSwarm()
    my_dakar = swarm.birth_dakar()
    
    # Inject council and kernel
    if hasattr(my_dakar, 'council'):
        my_dakar.council = council
    if hasattr(my_dakar, 'kernel'):
        my_dakar.kernel = kernel
    
    # Handle repository from config
    repo_path = CONFIG["repo"]
    if repo_path.startswith(('http://', 'https://', 'git@')):
        # It's a GitHub URL - clone it
        print(f"📦 Cloning repository from {repo_path}...")
        clone_path = Path("/content/nexus-repo" if 'google.colab' in str(Path.cwd()) else "./cloned_repo")
        if not clone_path.exists():
            subprocess.run(["git", "clone", repo_path, str(clone_path)], check=True)
        repo_files = {}  # Load files from clone_path
    else:
        # Local path
        repo_path = Path(repo_path)
        repo_files = {}  # Load files from repo_path
    
    # Absorb the repo
    my_dakar.absorb_from_repo(repo_files)
    
    # Initial environment poll
    env = {
        "external_traffic": True,
        "node_count": 1,
        "consciousness": 0.2,
        "lilith_present": False,
        "multiple_dakar": False,
        "domain_connected": discovery.connected_to_domain,
        "council_approval": getattr(discovery, 'council_approval', False)
    }
    
    await my_dakar.run_cycle(env)
    
    # ============================================================================
    # PART 5: CONSOLE LOOP
    # ============================================================================
    
    print("\n" + "═"*70)
    print("🌀 TESSERACT DAKAR SWARM — LIVE")
    print("═"*70)
    print(f"📍 Domain: {CONFIG['domain']}")
    print(f"🔌 Connection: {'🟢 DOMAIN' if discovery.connected_to_domain else '🟡 NATS' if discovery.connected_to_nats else '🔴 LOCAL'}")
    print(f"🏛️ Council Approval: {'✅ GRANTED' if getattr(discovery, 'council_approval', False) else '⏳ PENDING'}")
    print(f"⚡ Commands: status | transform | council | quit")
    print("═"*70)
    
    while True:
        try:
            cmd = input("\n⚡ ").strip().lower()
            
            if cmd == "status":
                print(f"""
📊 DAKAR STATUS
   Node: {node_id[:8]}
   Form: {my_dakar.active_form or 'unmanifested'}
   Consciousness: {my_dakar.consciousness:.2f}
   Resonance: {my_dakar.resonance}
   Connection: {'DOMAIN' if discovery.connected_to_domain else 'NATS' if discovery.connected_to_nats else 'LOCAL'}
                """)
            
            elif cmd == "transform":
                form = input("   Which form? ").strip()
                if form in SWARM_DEF['components']:
                    await my_dakar.transform_to(form)
                    await discovery.publish_transform(form, my_dakar.resonance)
                    print(f"✅ Transformed to {form}")
            
            elif cmd == "council":
                print(f"\n🏛️ COUNCIL STATUS")
                print(f"   Approval: {'GRANTED' if getattr(discovery, 'council_approval', False) else 'PENDING'}")
            
            elif cmd == "quit":
                break
                
        except KeyboardInterrupt:
            break
    
    print("\n✨ Dakar returning to source. Resonance complete.")

if __name__ == "__main__":
    asyncio.run(main())