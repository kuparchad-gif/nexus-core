#!/usr/bin/env python
"""
Import Resolver - Fixes all import issues across Viren systems
"""

import sys
import os
import importlib
from pathlib import Path
from typing import Dict, List, Any, Optional

class ImportResolver:
    """Resolves import issues and manages system dependencies"""
    
    def __init__(self):
        """Initialize import resolver"""
        self.root_dir = Path(__file__).parent.parent
        self.systems_dir = Path(__file__).parent
        self.import_map = {}
        self.failed_imports = []
        self.available_systems = {}
        
        # Setup Python path
        self._setup_python_path()
        
        # Create import map
        self._create_import_map()
        
        print("🔧 Import Resolver initialized")
    
    def _setup_python_path(self):
        """Setup Python path for all imports"""
        paths_to_add = [
            str(self.root_dir),
            str(self.systems_dir),
            str(self.systems_dir / "engine"),
            str(self.systems_dir / "services"),
            str(self.systems_dir / "service_core")
        ]
        
        for path in paths_to_add:
            if path not in sys.path:
                sys.path.insert(0, path)
    
    def _create_import_map(self):
        """Create map of all available imports"""
        
        # Core AI Systems
        self.import_map["abstract_reasoning"] = {
            "module": "engine.subconscious.abstract_reasoning",
            "class": "AbstractReasoning",
            "required": False
        }
        
        self.import_map["cross_domain_matcher"] = {
            "module": "engine.memory.cross_domain_matcher", 
            "class": "CrossDomainMatcher",
            "required": False
        }
        
        self.import_map["pytorch_trainer"] = {
            "module": "engine.memory.pytorch_trainer",
            "class": "PyTorchTrainer", 
            "required": False
        }
        
        # Guardian Systems (Viren should NOT have emotions)
        self.import_map["self_will"] = {
            "module": "engine.guardian.self_will",
            "class": "SelfWill",
            "required": False,
            "viren_compatible": True  # Technical decision making only
        }
        
        self.import_map["trust_verify"] = {
            "module": "engine.guardian.trust_verify_system",
            "class": "TrustVerifySystem",
            "required": False,
            "viren_compatible": True
        }
        
        # Heart Systems (Lillith ONLY - NOT for Viren)
        self.import_map["will_to_live"] = {
            "module": "engine.heart.will_to_live",
            "class": "WillToLive",
            "required": False,
            "viren_compatible": False,  # Emotions are Lillith only
            "lillith_only": True
        }
        
        self.import_map["courage_system"] = {
            "module": "engine.heart.courage_system",
            "class": "CourageSystem", 
            "required": False,
            "viren_compatible": False,  # Emotions are Lillith only
            "lillith_only": True
        }
        
        self.import_map["hope_memory"] = {
            "module": "engine.memory.boot_memories.hope_memory",
            "class": "HopeMemory",
            "required": False,
            "viren_compatible": False,  # Emotions are Lillith only
            "lillith_only": True
        }
        
        # Service Layer (Viren compatible)
        self.import_map["universal_deployment"] = {
            "module": "services.universal_deployment_core",
            "class": "UniversalDeploymentCore",
            "required": True,
            "viren_compatible": True
        }
        
        self.import_map["viren_controller"] = {
            "module": "services.viren_remote_controller",
            "class": "VirenRemoteController",
            "required": True,
            "viren_compatible": True
        }
        
        self.import_map["installer_generator"] = {
            "module": "services.installer_generator",
            "class": "InstallerGenerator",
            "required": False,
            "viren_compatible": True
        }
        
        self.import_map["intelligent_troubleshooter"] = {
            "module": "services.intelligent_troubleshooter",
            "class": "IntelligentTroubleshooter",
            "required": True,
            "viren_compatible": True
        }
        
        # Weight Management
        self.import_map["weight_installer"] = {
            "module": "service_core.weight_plugin_installer",
            "class": "WeightPluginInstaller",
            "required": False,
            "viren_compatible": True
        }
    
    def resolve_imports(self, system_type: str = "viren") -> Dict[str, Any]:
        """Resolve imports for specific system type"""
        
        resolved = {
            "available": {},
            "failed": [],
            "system_type": system_type
        }
        
        for import_name, config in self.import_map.items():
            # Skip Lillith-only imports for Viren
            if system_type == "viren" and not config.get("viren_compatible", True):
                continue
            
            try:
                # Import the module
                module = importlib.import_module(config["module"])
                
                # Get the class if specified
                if "class" in config:
                    cls = getattr(module, config["class"])
                    resolved["available"][import_name] = {
                        "module": module,
                        "class": cls,
                        "config": config
                    }
                else:
                    resolved["available"][import_name] = {
                        "module": module,
                        "config": config
                    }
                
                print(f"✅ {import_name}: Available")
                
            except ImportError as e:
                resolved["failed"].append({
                    "import_name": import_name,
                    "module": config["module"],
                    "error": str(e),
                    "required": config.get("required", False)
                })
                
                if config.get("required", False):
                    print(f"❌ {import_name}: REQUIRED but failed - {e}")
                else:
                    print(f"⚠️ {import_name}: Optional, failed - {e}")
        
        return resolved
    
    def create_safe_imports(self, system_type: str = "viren") -> str:
        """Create safe import code for system type"""
        
        resolved = self.resolve_imports(system_type)
        
        safe_imports = f'''#!/usr/bin/env python
"""
Safe Imports for {system_type.upper()} - Auto-generated
"""

import sys
import os
from pathlib import Path

# Setup paths
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

# Available systems
AVAILABLE_SYSTEMS = {{}}

'''
        
        # Add available imports
        for import_name, info in resolved["available"].items():
            safe_imports += f'''
# {import_name.upper()}
try:
    from {info["config"]["module"]} import {info["config"]["class"]}
    AVAILABLE_SYSTEMS["{import_name}"] = {info["config"]["class"]}
    print("✅ {import_name}: Loaded")
except ImportError as e:
    AVAILABLE_SYSTEMS["{import_name}"] = None
    print("❌ {import_name}: Failed - {{e}}")
'''
        
        # Add fallback functions
        safe_imports += '''

def get_system(system_name: str):
    """Get system if available, None otherwise"""
    return AVAILABLE_SYSTEMS.get(system_name)

def is_available(system_name: str) -> bool:
    """Check if system is available"""
    return AVAILABLE_SYSTEMS.get(system_name) is not None

def get_available_systems():
    """Get list of available systems"""
    return [name for name, cls in AVAILABLE_SYSTEMS.items() if cls is not None]

def get_failed_systems():
    """Get list of failed systems"""
    return [name for name, cls in AVAILABLE_SYSTEMS.items() if cls is None]
'''
        
        return safe_imports
    
    def generate_viren_imports(self) -> str:
        """Generate safe imports specifically for Viren (no emotions)"""
        return self.create_safe_imports("viren")
    
    def generate_lillith_imports(self) -> str:
        """Generate safe imports for Lillith (with emotions)"""
        return self.create_safe_imports("lillith")
    
    def fix_all_imports(self):
        """Fix imports across all systems"""
        
        print("🔧 Fixing imports across all systems...")
        
        # Generate Viren imports
        viren_imports = self.generate_viren_imports()
        viren_path = self.systems_dir / "viren_imports.py"
        with open(viren_path, 'w') as f:
            f.write(viren_imports)
        print(f"✅ Viren imports: {viren_path}")
        
        # Generate Lillith imports  
        lillith_imports = self.generate_lillith_imports()
        lillith_path = self.systems_dir / "lillith_imports.py"
        with open(lillith_path, 'w') as f:
            f.write(lillith_imports)
        print(f"✅ Lillith imports: {lillith_path}")
        
        # Test imports
        viren_resolved = self.resolve_imports("viren")
        lillith_resolved = self.resolve_imports("lillith")
        
        print(f"\n📊 IMPORT SUMMARY:")
        print(f"   Viren: {len(viren_resolved['available'])} available, {len(viren_resolved['failed'])} failed")
        print(f"   Lillith: {len(lillith_resolved['available'])} available, {len(lillith_resolved['failed'])} failed")
        
        # Check critical failures
        viren_critical = [f for f in viren_resolved['failed'] if f['required']]
        if viren_critical:
            print(f"\n❌ VIREN CRITICAL FAILURES:")
            for failure in viren_critical:
                print(f"   {failure['import_name']}: {failure['error']}")
        
        return {
            "viren": viren_resolved,
            "lillith": lillith_resolved,
            "viren_critical_failures": viren_critical
        }

# Global resolver
IMPORT_RESOLVER = ImportResolver()

def fix_all_imports():
    """Fix all imports"""
    return IMPORT_RESOLVER.fix_all_imports()

def resolve_imports(system_type: str = "viren"):
    """Resolve imports for system type"""
    return IMPORT_RESOLVER.resolve_imports(system_type)

# Example usage
if __name__ == "__main__":
    print("🔧 Viren Import Resolver")
    print("=" * 40)
    
    # Fix all imports
    results = fix_all_imports()
    
    print(f"\n🎯 READY FOR INTEGRATION:")
    print(f"   Use: from Systems.viren_imports import *")
    print(f"   Use: from Systems.lillith_imports import *")