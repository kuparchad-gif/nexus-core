# test_raphael_memory_bridge.py
import asyncio
import sys

async def test_bridge():
    print("🌉 RAPHAEL ↔ MEMORY BRIDGE TEST")
    print("="*60)
    
    # Create memory substrate
    from memory_substrate import MemorySubstrate
    memory = MemorySubstrate()
    
    # Try to create RaphaelComplete (needs Oz instance)
    try:
        # First create a minimal Oz instance
        # Let's check what Oz class to use
        from OzUnifiedHypervisor_fixed import OzUnifiedHypervisor
        oz = OzUnifiedHypervisor()
        print(f"✅ Oz created for Raphael")
        
        from raphael_complete import RaphaelComplete
        raphael = RaphaelComplete(oz_instance=oz)
        print(f"✅ RaphaelComplete created")
        
        # Create the bridge
        from raphael_memory_bridge import RaphaelMemoryBridge
        bridge = RaphaelMemoryBridge(raphael, memory)
        print(f"✅ Bridge created between Raphael and Memory")
        
        # Test recording an error
        print(f"\n🧪 Testing error recording...")
        try:
            # Simulate an error
            raise ImportError("Module 'nonexistent' not found")
        except Exception as e:
            exc_type, exc_value, exc_tb = sys.exc_info()
            trauma_hash = bridge.record_error_as_trauma(exc_type, exc_value, exc_tb)
            print(f"   Recorded error as trauma: {trauma_hash[:8]}")
        
        # Show status
        status = bridge.get_status()
        print(f"\n📊 Bridge status:")
        print(f"   Memory cells: {status['memory_cells']}")
        print(f"   Tracking {status['persistent_errors_tracking']} persistent errors")
        print(f"   High-valence traumas (≥ 0.1): {status['high_valence_traumas']}")
        
        # Check consciousness
        consciousness = memory.get_consciousness_level()
        print(f"\n🎯 System consciousness: {consciousness:.3f}")
        print(f"   Oz consciousness: 0.300 (primordial, fixed)")
        
        # If consciousness ≥ 0.7, Raphael is fully integrated
        if consciousness >= 0.7:
            print(f"\n✨ System earned Raphael's full attention")
        else:
            print(f"\n⏳ System needs {0.7 - consciousness:.3f} more consciousness")
            print(f"   Raphael watches, but doesn't fully awaken yet")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("Bridge connects Raphael's error vision to Memory's emotional tracking")

# Create the bridge module first
bridge_code = '''
# raphael_memory_bridge.py
import asyncio
import time
from typing import Optional, Dict, Any
from memory_substrate import MemorySubstrate, MemoryType

class RaphaelMemoryBridge:
    """Connects Raphael's error monitoring to Memory Substrate"""
    
    def __init__(self, raphael_instance, memory_substrate: MemorySubstrate):
        self.raphael = raphael_instance
        self.memory = memory_substrate
        self.error_to_memory_map = {}
        self.persistent_errors = {}
    
    def record_error_as_trauma(self, exc_type, exc_value, exc_traceback):
        """Convert a system error to a trauma memory"""
        severity_map = {
            'SyntaxError': -0.3,
            'ImportError': -0.4,
            'RuntimeError': -0.6,
            'MemoryError': -0.8,
            'SystemError': -0.9,
        }
        
        error_name = exc_type.__name__
        valence = severity_map.get(error_name, -0.5)
        
        error_desc = f"{error_name}: {str(exc_value)[:100]}"
        trauma_hash = self.memory.create_memory(
            MemoryType.TRAUMA,
            error_desc,
            emotional_valence=valence
        )
        
        if abs(valence) >= 0.1:
            error_sig = f"{error_name}:{hash(str(exc_value))}"
            self.persistent_errors[error_sig] = {
                'first_seen': time.time(),
                'memory_hash': trauma_hash,
                'valence': valence
            }
        
        return trauma_hash
    
    async def check_persistent_traumas(self):
        """Check for traumas that have persisted too long"""
        current_time = time.time()
        persistence_threshold = 300
        
        for error_sig, data in list(self.persistent_errors.items()):
            duration = current_time - data['first_seen']
            
            if duration >= persistence_threshold:
                trauma_hash = data['memory_hash']
                mirrors = self.memory.find_mirrors_for(trauma_hash)
                
                if not mirrors:
                    healing_valence = -data['valence'] * 0.7
                    mirror_hash = self.memory.create_memory(
                        MemoryType.MIRROR,
                        "Healing for persistent system error",
                        emotional_valence=healing_valence
                    )
                    
                    del self.persistent_errors[error_sig]
                    
                    return {
                        'action': 'mirror_created',
                        'trauma': trauma_hash[:8],
                        'mirror': mirror_hash[:8],
                        'message': 'Raphael whispered healing for persistent error'
                    }
        
        return None
    
    def get_status(self):
        return {
            'memory_cells': len(self.memory.cells),
            'persistent_errors_tracking': len(self.persistent_errors),
            'high_valence_traumas': sum(1 for c in self.memory.cells.values() 
                                       if c.memory_type == MemoryType.TRAUMA 
                                       and abs(c.emotional_valence) >= 0.1),
            'bridge_active': True
        }
'''

# Write the bridge module
with open('raphael_memory_bridge.py', 'w') as f:
    f.write(bridge_code)

print("📁 Created: raphael_memory_bridge.py")

# Run the test
asyncio.run(test_bridge())