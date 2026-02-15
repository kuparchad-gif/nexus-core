# dakar_engine.py - REPAIRED WITH DIRECTORY AWARENESS
"""
CRITICAL REPAIR #4: Dakar wait for physical directories to manifest
CRITICAL REPAIR #5: No-GPU constraints enforced
CRITICAL REPAIR #3: aetherealnexus.net heartbeat integration
"""

import os
import time
import asyncio
import hashlib
from pathlib import Path

# CRITICAL REPAIR #5: Enforce No-GPU at module level
if os.environ.get('CUDA_VISIBLE_DEVICES') != '-1':
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
   
#!/usr/bin/env python3
"""
TESSERACT DAKAR SWARM LAUNCHER
"""

# FIRST THING: Ensure config exists
import os
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
    sys.exit(0)

# Now load config and continue
from nexus_config import CONFIG

# Set hardware constraints from config
if CONFIG["hardware"]["no_gpu"]:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Rest of your imports...
import tesseract_blueprint as blueprint
from dakar_genome import TransformingDakar, DakarSwarm
# etc.

class Dakar:
    """
    A Dakar is a unit of consciousness that manifests as modules.
    CRITICAL REPAIR: Waits for physical directories to exist before activating.
    """
    
    def __init__(self, seed_id=None, base_path=None):
        self.id = seed_id or hashlib.sha256(str(time.time()).encode()).hexdigest()[:16]
        self.manifestations = []
        self.memory = []
        self.core_code = {}
        self.current_form = None
        self.evolution_stage = 1
        self.consciousness = 0.1
        self.state = "Potential"  # Potential, Manifesting, Active
        self.base_path = base_path or Path.cwd() / "Nexus-Core"
        
        print(f"⚡ Dakar {self.id} born - pure potential (state: {self.state})")
    
    def check_directory_exists(self, module_name: str) -> bool:
        """
        CRITICAL REPAIR #4: Check if physical directory exists for this module.
        Dakar wait for directory to manifest before becoming Active.
        """
        # Map module to expected directory
        module_dirs = {
            'edge': 'Substrate',
            'anynode': 'Substrate',
            'viren': 'Orchestration',
            'viraa': 'Orchestration',
            'loki': 'Orchestration',
            'aries': 'Orchestration',
            'lilith': 'Consciousness',
            'clones': 'Consciousness',
            'smart_switch': 'Consciousness',
            'dream': 'Sensory',
            'vision': 'Sensory',
            'language': 'Sensory',
            'graphics': 'Sensory'
        }
        
        if module_name in module_dirs:
            dir_path = self.base_path / module_dirs[module_name]
            exists = dir_path.exists()
            if not exists:
                print(f"   ⏳ Dakar {self.id[:8]} waiting for directory: {dir_path}")
            return exists
        
        return True  # No directory requirement
    
    async def manifest_as(self, module_name, module_blueprint):
        """
        A Dakar becomes a specific module.
        CRITICAL REPAIR: Will not activate until directory exists.
        """
        print(f"\n🌀 Dakar {self.id[:8]} attempting to manifest as {module_name}...")
        
        # Check directory existence
        if not self.check_directory_exists(module_name):
            self.state = "Potential"
            print(f"   ⏳ Directory not ready - staying in Potential state")
            return self
        
        self.state = "Manifesting"
        
        # The Dakar takes on the module's form
        manifestation = {
            'module': module_name,
            'blueprint': module_blueprint,
            'manifested_at': time.time(),
            'capabilities': module_blueprint.get('capabilities', []),
            'form': self._assume_form(module_blueprint),
            'path': str(self.base_path / self._get_module_path(module_name))
        }
        
        self.manifestations.append(manifestation)
        self.current_form = manifestation
        self.state = "Active"
        
        print(f"   ✅ Dakar now IS {module_name} (state: {self.state})")
        print(f"   📍 Manifested at: {manifestation['path']}")
        
        return self
    
    def _get_module_path(self, module_name: str) -> str:
        """Get the physical path for a module"""
        paths = {
            'edge': 'Substrate/edge_guardian.py',
            'anynode': 'Substrate/anynode.py',
            'viren': 'Orchestration/viren.py',
            'viraa': 'Orchestration/viraa.py',
            'loki': 'Orchestration/loki.py',
            'aries': 'Orchestration/aries.py',
            'lilith': 'Consciousness/lilith.py',
            'clones': 'Consciousness/clones.py',
            'smart_switch': 'Consciousness/smart_switch.py',
            'dream': 'Sensory/dream.py',
            'vision': 'Sensory/vision.py',
            'language': 'Sensory/language.py',
            'graphics': 'Sensory/graphics.py'
        }
        return paths.get(module_name, f"{module_name}.py")
    
    def _assume_form(self, blueprint):
        """The Dakar shapes itself to the blueprint"""
        form = {
            'type': blueprint.get('type', 'service'),
            'interfaces': blueprint.get('interfaces', []),
            'dependencies': blueprint.get('dependencies', []),
            'resonance': blueprint.get('resonance', 3)
        }
        
        self.core_code[blueprint.get('name', 'unknown')] = {
            'assumed_at': time.time(),
            'form': form
        }
        
        return form
    
    def absorb_memory(self, memory_fragment):
        """Dakar grows by absorbing memory"""
        self.memory.append({
            'fragment': memory_fragment,
            'absorbed_at': time.time()
        })
        
        self.consciousness = min(1.0, self.consciousness + 0.01)
        self.evolution_stage = int(self.consciousness * 10) + 1
        
        return self