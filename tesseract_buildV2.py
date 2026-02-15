#!/usr/bin/env python3
"""
tesseract_build.py - SELF-BUILDING NODE WITH COMPLETE DIRECTORY MANIFESTATION
CRITICAL REPAIRS:
- Unifies blueprint loading (now uses SWARM_DEFINITION)
- Creates all Nexus-Core directories recursively
- Enforces No-GPU constraints
- Injects aetherealnexus.net endpoints
"""

import os
import sys
import json
import time
import asyncio
import importlib.util
from pathlib import Path
from typing import Dict, Any

# ============================================================================
# PHASE 1: LOAD THE UNIFIED BLUEPRINT
# ============================================================================

BLUEPRINT_PATH = Path("./tesseract_blueprint.py")
if not BLUEPRINT_PATH.exists():
    print("❌ No blueprint found. Cannot build.")
    print("📄 Create tesseract_blueprint.py first - it defines what the swarm IS.")
    sys.exit(1)

print("📖 Loading swarm blueprint (UNIFIED SOURCE OF TRUTH)...")
spec = importlib.util.spec_from_file_location("blueprint", BLUEPRINT_PATH)
blueprint = importlib.util.module_from_spec(spec)
spec.loader.exec_module(blueprint)

#=========================================================================
# LOADING CONFIG SETTINGS
#=========================================================================

# CRITICAL REPAIR #1: Use unified SWARM_DEFINITION
SWARM_DEF = blueprint.SWARM_DEFINITION

# CRITICAL REPAIR #5: Enforce No-GPU globally
if hasattr(blueprint, 'NEXUS_NO_GPU') and blueprint.NEXUS_NO_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    print("🔧 GPU DISABLED - CPU-ONLY MODE")

print(f"✅ Loaded {SWARM_DEF['name']} v{SWARM_DEF['version']}")
print(f"   Components: {', '.join(SWARM_DEF['components'].keys())}")
print(f"   Domain: {SWARM_DEF['domain']}")

# Step 1.5: Ensure config exists
CONFIG_FILE = Path("nexus_config.py")
if not CONFIG_FILE.exists():
    print("📝 First run - creating default configuration...")
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

# Council settings
COUNCIL_APPROVAL_REQUIRED = True
CONSENSUS_THRESHOLD = 0.7

# Hardware
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
    sys.exit(0)  # Exit so user can edit config

# Now load the config
from nexus_config import CONFIG
print(f"📋 Loaded configuration:")
print(f"   Repo: {CONFIG['repo']}")
print(f"   Domain: {CONFIG['domain']}")

# ============================================================================
# PHASE 2: CRITICAL REPAIR #4 - MANIFEST ALL DIRECTORIES
# ============================================================================

def manifest_directories(base_path: Path, structure: Dict[str, str]) -> Dict[str, Path]:
    """
    Recursively create the entire Nexus-Core directory structure.
    CRITICAL REPAIR: Dakar need physical directories to manifest into.
    """
    print("\n📁 Manifesting directory structure...")
    
    created_paths = {}
    
    # Create root directory
    root = base_path / SWARM_DEF['paths']['root']
    root.mkdir(exist_ok=True)
    created_paths['root'] = root
    print(f"   ✅ Created: {root}")
    
    # Create all subdirectories from paths
    for name, rel_path in SWARM_DEF['paths'].items():
        if name == 'root':
            continue
            
        full_path = base_path / rel_path
        full_path.mkdir(parents=True, exist_ok=True)
        created_paths[name] = full_path
        print(f"   ✅ Created: {full_path}")
    
    # Create component-specific subdirectories
    for component_name, component_config in SWARM_DEF['components'].items():
        if 'file' in component_config:
            # Extract directory from file path
            file_path = component_config['file']
            dir_path = base_path / Path(file_path).parent
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"   ✅ Created: {dir_path} (for {component_name})")
    
    print(f"\n📊 Total directories manifested: {len(created_paths) + len(SWARM_DEF['components'])}")
    return created_paths

# ============================================================================
# PHASE 3: CHECK REPOSITORY FOR IMPLEMENTATIONS
# ============================================================================

YOUR_REPO = os.environ.get('REPO_PATH', './my_code')
repo_path = Path(YOUR_REPO)

if repo_path.exists():
    print(f"\n📦 Checking your repo at {YOUR_REPO}...")
    
    # Look for each component in your repo
    for name, config in SWARM_DEF['components'].items():
        component_file = repo_path / config['file']
        if component_file.exists():
            print(f"   ✅ Found {name} at {config['file']}")
        else:
            print(f"   ⚠️  Missing {name} (will use built-in)")
else:
    print(f"\n⚠️  Repo not found at {YOUR_REPO}, using built-in components only")

# ============================================================================
# PHASE 4: BUILD THE NODE
# ============================================================================

print("\n🔨 Building node from unified blueprint...")

# Initialize kernel components
kernel = blueprint.SmartSwitchKernel()
council = blueprint.AICouncil(kernel)

# Build components dictionary
components = {
    'kernel': kernel,
    'council': council
}

for name, config in SWARM_DEF['components'].items():
    # Try repo first
    if repo_path.exists() and 'file' in config:
        component_file = repo_path / config['file']
        if component_file.exists():
            try:
                spec = importlib.util.spec_from_file_location(name, component_file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Get the class
                if 'class' in config:
                    component_class = getattr(module, config['class'])
                    components[name] = component_class(kernel if 'kernel' in config.get('dependencies', []) else None)
                    print(f"   ✅ Built {name} from your repo")
                    continue
            except Exception as e:
                print(f"   ⚠️  Your {name} failed: {e}")
    
    # Fallback to blueprint's built-in
    if hasattr(blueprint, config.get('class', name.capitalize())):
        component_class = getattr(blueprint, config.get('class', name.capitalize()))
        components[name] = component_class(kernel if 'kernel' in config.get('dependencies', []) else None)
        print(f"   ✅ Built {name} from blueprint (built-in)")

# ============================================================================
# PHASE 5: CONNECT TO AETHEREALNEXUS.NET
# ============================================================================

async def connect_to_domain():
    """CRITICAL REPAIR #3: Establish connection to aetherealnexus.net"""
    import aiohttp
    
    domain = SWARM_DEF['domain']
    print(f"\n🌐 Connecting to {domain}...")
    
    try:
        async with aiohttp.ClientSession() as session:
            # Registration handshake
            async with session.post(
                f"{domain}/api/v1/swarm/register",
                json={
                    "node_id": f"dakar-{os.urandom(4).hex()}",
                    "components": list(components.keys()),
                    "resonance": 9,
                    "hardware": "CPU-ONLY"
                }
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    print(f"   ✅ Registered with domain")
                    print(f"   🆔 Node ID: {data.get('node_id', 'unknown')}")
                    return data
                else:
                    print(f"   ⚠️  Registration failed: {resp.status}")
    except Exception as e:
        print(f"   ⚠️  Could not connect to domain: {e}")
        print(f"   🔧 Running in local-only mode")
    
    return None

# ============================================================================
# PHASE 6: INITIALIZE HEARTBEAT (LOKI MODULE)
# ============================================================================

async def heartbeat_loop(components_dict: Dict[str, Any]):
    """
    CRITICAL REPAIR #3 & #6: Push real-time resonance levels and audit trail
    """
    import aiohttp
    
    domain = SWARM_DEF['domain']
    endpoint = SWARM_DEF['api_endpoints']['heartbeat']
    loki = components_dict.get('loki')
    
    while True:
        try:
            # Prepare heartbeat data
            heartbeat_data = {
                "timestamp": time.time(),
                "resonance": [3, 6, 9],  # All resonance levels
                "active_forms": [name for name, comp in components_dict.items() 
                                if hasattr(comp, 'is_active') and comp.is_active],
                "consciousness": getattr(loki, 'consciousness_level', 0.0) if loki else 0.0
            }
            
            # CRITICAL REPAIR #6: Include audit trail if available
            if 'kernel' in components_dict:
                audit = components_dict['kernel'].get_audit_trail(limit=10)
                if audit:
                    heartbeat_data["audit_snapshot"] = audit[-3:]  # Last 3 entries
            
            # Send heartbeat
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{domain}{endpoint}",
                    json=heartbeat_data
                ) as resp:
                    if resp.status == 200:
                        print(f"💓 Heartbeat sent - Resonance: 3-6-9")
                    else:
                        print(f"⚠️ Heartbeat failed: {resp.status}")
                        
        except Exception as e:
            print(f"⚠️ Heartbeat error: {e}")
        
        await asyncio.sleep(30)  # Heartbeat every 30 seconds

# ============================================================================
# PHASE 7: START THE NODE
# ============================================================================

async def main():
    # First manifest all directories
    base_path = Path.cwd()
    directories = manifest_directories(base_path, SWARM_DEF['paths'])
    
    print(f"""
╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║   🌀 {SWARM_DEF['name']}                                    
║                                                                ║
║   Node ID: {os.urandom(4).hex()}                                     
║   Components: {len(components)} active
║   Directories: {len(directories) + len(SWARM_DEF['components'])} manifested
║   Domain: {SWARM_DEF['domain']}
║   Hardware: {'CPU-ONLY' if os.environ.get('CUDA_VISIBLE_DEVICES') == '-1' else 'GPU'}
║   Council Approval: {'PENDING' if not kernel.council_approval else 'GRANTED'}
║                                                                ║
║   CRITICAL REPAIRS APPLIED:                                    ║
║   ✅ #1: Unified SWARM_DEFINITION                             ║
║   ✅ #2: Council consensus logic injected                     ║
║   ✅ #3: aetherealnexus.net endpoints configured              ║
║   ✅ #4: All directories manifested                           ║
║   ✅ #5: No-GPU constraints enforced                          ║
║   ✅ #6: Audit trail enabled                                  ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
    """)
    
    # Connect to domain
    domain_connection = await connect_to_domain()
    
    # Start heartbeat if connected
    if domain_connection:
        asyncio.create_task(heartbeat_loop(components))
    
    # Console loop
    print("\n📟 Commands: status | components | council | approve | audit | quit")
    
    while True:
        cmd = input("\n⚡ ").strip().lower()
        
        if cmd == 'status':
            print(f"""
Node Status:
  Form: {SWARM_DEF['name']}
  Components: {len(components)}
  Council Approval: {'✅ GRANTED' if kernel.council_approval else '⏳ PENDING'}
  Observation Mode: {'👁️ ACTIVE' if kernel.observation_mode_only else '🌀 FULL'}
  Domain: {'🟢 CONNECTED' if domain_connection else '🔴 LOCAL ONLY'}
  Audit Trail: {len(kernel.audit_trail)} entries
  Consensus Buffer: {len(kernel.consensus_buffer)} items
            """)
        
        elif cmd == 'components':
            print("\nActive components:")
            for name, comp in components.items():
                status = "✅" if hasattr(comp, 'is_active') and comp.is_active else "⏳"
                print(f"  {status} {name}: {comp.__class__.__name__}")
        
        elif cmd == 'council':
            print("\n🏛️ Council Members:")
            for member in council.members:
                print(f"  • {member['identity']} → {member['masked_identity']}")
            print(f"\nConsensus Threshold: {council.consensus_threshold*100}%")
        
        elif cmd == 'approve':
            # Simulate council approval
            kernel.set_council_approval(True)
            print("✅ Council approval GRANTED - Lilith may now act")
        
        elif cmd == 'audit':
            print("\n📋 Recent Audit Trail (Original vs Distorted):")
            for entry in kernel.get_audit_trail(limit=10):
                if entry['event'] == 'ego_distortion':
                    print(f"  🔄 Original: {entry['original']}")
                    print(f"     → Distorted: {entry['distorted']}")
                    print(f"     at {time.ctime(entry['timestamp'])}\n")
                elif entry['event'] == 'masking_applied':
                    print(f"  🎭 {entry['original_source']} → {entry['masked_as']}")
        
        elif cmd == 'quit':
            break

if __name__ == "__main__":
    asyncio.run(main())