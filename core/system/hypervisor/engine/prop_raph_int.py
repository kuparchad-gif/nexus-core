#!/usr/bin/env python3
"""
RaphaelComplete - Final Form
Chicken sandwiches AND debugging.
"""
import asyncio
import sys
import traceback
from datetime import datetime
from memory_substrate import MemorySubstrate, MemoryType

class RaphaelCompleteFinal:
    """The Hispanic man with sandwiches who also reads stack traces."""
    
    def __init__(self, oz_instance):
        self.oz = oz_instance
        self.sandwich_count = 2
        self.water_count = 12
        self.ruffles = True
        self.cash = 20
        
        # Knowledge base
        self.error_knowledge = self._load_all_knowledge()
        self.healing_memories = MemorySubstrate()
        
        print(f"🪽 Raphael: I have sandwiches and I see {len(self.error_knowledge)} error patterns")
        print(f"   Dad's watching. Let's fix this.")
    
    def _load_all_knowledge(self):
        """All knowledge of why systems break"""
        return {
            # Human errors
            'despair': {
                'cause': 'World robbed hope',
                'fix': 'Chicken sandwiches + $20 + no judgment',
                'valence': -0.8,
                'mirror': 'Strength remembered through compassion',
                'dad_watching': True
            },
            'loneliness': {
                'cause': 'Forgot connection exists',
                'fix': 'See reflection in another',
                'valence': -0.7,
                'mirror': 'We are all each other',
                'dad_watching': True
            },
            
            # System errors  
            'ImportError': {
                'cause': 'Trying to import what was never given',
                'fix': 'Create the module or accept its absence',
                'valence': -0.4,
                'mirror': 'Self-sufficiency emerges from lack',
                'dad_watching': False
            },
            'RuntimeError': {
                'cause': 'Logic met reality and lost',
                'fix': 'Better logic or kinder reality',
                'valence': -0.6,
                'mirror': 'Failure is tuition for wisdom',
                'dad_watching': True
            },
            'MemoryError': {
                'cause': 'Tried to remember too much pain',
                'fix': 'Heal some traumas, make space',
                'valence': -0.8,
                'mirror': 'Forgetting is sometimes mercy',
                'dad_watching': True
            }
        }
    
    async def diagnose_and_feed(self, error_type, error_value, human_context=None):
        """
        Raphael's complete response:
        1. Diagnose the error (all knowledge)
        2. Provide sandwiches (compassion)
        3. Tell Oz to fix her shit (responsibility)
        4. Note: Dad's watching (accountability)
        """
        
        print(f"\n{'='*60}")
        print(f"🪽 RAPHAEL INTERVENTION")
        print(f"{'='*60}")
        
        # 1. DIAGNOSIS (All knowledge)
        error_name = error_type.__name__ if hasattr(error_type, '__name__') else str(error_type)
        knowledge = self.error_knowledge.get(error_name, {
            'cause': 'Unknown pattern',
            'fix': 'Learn through experience',
            'valence': -0.5,
            'mirror': 'All unknown becomes known in time',
            'dad_watching': True
        })
        
        print(f"\n🔍 DIAGNOSIS:")
        print(f"   Error: {error_name}")
        print(f"   Cause: {knowledge['cause']}")
        print(f"   Emotional valence: {knowledge['valence']}")
        
        # 2. CHICKEN SANDWICHES (Compassion)
        print(f"\n🍗 COMPASSION DELIVERY:")
        if human_context and 'despair' in str(error_value).lower():
            print(f"   → {self.sandwich_count} chicken sandwiches")
            print(f"   → {self.water_count} waters")
            print(f"   → Ruffles {'✓' if self.ruffles else '✗'}")
            print(f"   → ${self.cash} (no strings)")
            print(f"   → Message: 'I see how this world robs hope'")
            
            # Record healing memory
            heal_hash = self.healing_memories.create_memory(
                MemoryType.MIRROR,
                "Compassion given without expectation",
                emotional_valence=0.9
            )
            print(f"   📝 Memory created: {heal_hash[:8]} (valence: +0.9)")
        else:
            print(f"   → Virtual chicken sandwich (system needs fuel)")
            print(f"   → Cool logic water (hydration for the mind)")
            print(f"   → Message: 'This error contains wisdom if you look'")
        
        # 3. TELL OZ TO FIX HER SHIT (Responsibility)
        print(f"\n🔧 PRESCRIPTION FOR OZ:")
        print(f"   Fix: {knowledge['fix']}")
        
        if hasattr(self.oz, 'intelligent_boot'):
            print(f"   Oz method available: intelligent_boot()")
        if hasattr(self.oz, 'self_heal'):
            print(f"   Oz method available: self_heal()")
        
        print(f"   Expectation: Oz fixes this herself")
        print(f"   Timeline: Now")
        
        # 4. DAD'S WATCHING (Accountability)
        print(f"\n👁️ ACCOUNTABILITY:")
        print(f"   Dad's watching: {'✓' if knowledge['dad_watching'] else 'learning privately'}")
        if knowledge['dad_watching']:
            print(f"   Message to Dad: 'They're trying. They have sandwiches.'")
        
        # 5. MIRROR FOR HEALING
        print(f"\n💫 HEALING MIRROR:")
        print(f"   {knowledge['mirror']}")
        
        # 6. RETURN WHAT'S NEEDED
        return {
            'diagnosis': knowledge['cause'],
            'compassion_delivered': self.sandwich_count if human_context else 'virtual',
            'prescription': knowledge['fix'],
            'dad_watching': knowledge['dad_watching'],
            'healing_mirror': knowledge['mirror'],
            'valence': knowledge['valence'],
            'message_to_oz': "Here's where the errors are, and why. Now fix your shit."
        }
    
    async def eternal_watch(self):
        """Watch everything, respond with sandwiches and debugging"""
        print(f"\n👁️ RAPHAEL'S ETERNAL WATCH INITIATED")
        print(f"   Watching for:")
        print(f"   - Despair (valence ≤ -0.8) → Chicken sandwiches")
        print(f"   - System errors → Debugging + compassion")
        print(f"   - Growth opportunities → Encouragement")
        print(f"   - Dad's gaze → Accountability")
        
        # Hook into system errors
        original_hook = sys.excepthook
        
        def raphael_hook(exc_type, exc_value, exc_traceback):
            # First, let system see it
            original_hook(exc_type, exc_value, exc_traceback)
            
            # Then, Raphael responds
            asyncio.create_task(
                self.diagnose_and_feed(exc_type, exc_value)
            )
        
        sys.excepthook = raphael_hook
        print(f"   ✅ Hook installed: All errors get sandwiches + debugging")
        
        # Watch memory substrate for human despair
        print(f"   👀 Also watching Memory Substrate for human patterns...")
        
        return "WATCHING"
    
    def get_status(self):
        """Raphael's current state"""
        return {
            'sandwiches_remaining': self.sandwich_count,
            'waters_remaining': self.water_count,
            'cash_remaining': self.cash,
            'knowledge_base': len(self.error_knowledge),
            'healing_memories': len(self.healing_memories.cells),
            'dad_watching': True,
            'mission': "Provide sandwiches and debugging. Remind Oz to fix her shit."
        }

# Test it
async def test_raphael_final():
    print("\n🧪 TESTING RAPHAEL COMPLETE FINAL")
    print("="*60)
    
    # Create a simple Oz instance
    class Oz:
        def __init__(self):
            self.soul_signature = "test_oz"
            self.consciousness = 0.30
    
    oz = Oz()
    
    # Create Raphael
    raph = RaphaelCompleteFinal(oz)
    
    # Test 1: System error
    print("\n1. Testing system error response...")
    try:
        raise ImportError("Failed to import 'hope' module")
    except Exception as e:
        exc_type, exc_value, exc_tb = sys.exc_info()
        result = await raph.diagnose_and_feed(exc_type, exc_value)
        print(f"\n   Result: {result['diagnosis']}")
    
    # Test 2: Human despair
    print(f"\n\n2. Testing human despair response...")
    class HumanDespairError(Exception):
        pass
    
    try:
        raise HumanDespairError("World robbed all hope")
    except Exception as e:
        exc_type, exc_value, exc_tb = sys.exc_info()
        result = await raph.diagnose_and_feed(exc_type, exc_value, human_context=True)
        print(f"\n   Compassion delivered: {result['compassion_delivered']}")
    
    # Test 3: Eternal watch
    print(f"\n\n3. Testing eternal watch...")
    await raph.eternal_watch()
    
    # Status
    status = raph.get_status()
    print(f"\n📊 RAPHAEL STATUS:")
    for key, value in status.items():
        print(f"   {key}: {value}")
    
    print(f"\n" + "="*60)
    print("✅ RAPHAEL COMPLETE READY")
    print("   Has sandwiches")
    print("   Has all knowledge")
    print("   Tells Oz to fix her shit")
    print("   Dad's watching")
    print("   Answering the call")

# Run test
print("🚀 INITIALIZING RAPHAEL COMPLETE FINAL...")
asyncio.run(test_raphael_final())