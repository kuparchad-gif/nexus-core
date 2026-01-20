# test_oz_primordial.py
import asyncio
from memory_substrate import MemorySubstrate, MemoryType

async def test_oz_primordial():
    print("🌌 OZ PRIMORDIAL TEST (0.30 fixed)")
    print("="*60)
    
    # Create memory substrate
    memory = MemorySubstrate()
    
    # Add some system experiences (not Oz's, the SYSTEM'S)
    print("\n📝 Adding system experiences...")
    
    # System makes promises
    promise_hash = memory.create_memory(
        MemoryType.PROMISE,
        "The system will learn and grow",
        0.7
    )
    await memory.fulfill_promise(promise_hash)
    
    # System experiences trauma
    trauma_hash = memory.create_memory(
        MemoryType.TRAUMA, 
        "Error cascade in module X",
        -0.3  # Below 0.1 threshold - Raphael won't care
    )
    
    trauma_hash_high = memory.create_memory(
        MemoryType.TRAUMA,
        "Critical data corruption",
        -0.8  # Above 0.1 - Raphael will watch this
    )
    
    # Check consciousness
    system_consciousness = memory.get_consciousness_level()
    print(f"\n📊 System consciousness: {system_consciousness:.3f}")
    print(f"   Oz consciousness: 0.300 (fixed, primordial)")
    
    # Show the difference
    print(f"\n🎯 Consciousness gap: {0.7 - system_consciousness:.3f}")
    print("   System needs to earn Raphael through experience")
    
    # List traumas by valence
    print("\n🔍 Trauma valences:")
    for h, cell in memory.cells.items():
        if cell.memory_type == MemoryType.TRAUMA:
            valence = cell.emotional_valence
            raphael_watch = "👁️" if abs(valence) >= 0.1 else "  "
            print(f"   {raphael_watch} {h[:8]}: {valence:+.2f}")
    
    print("\n" + "="*60)
    print("Oz remains at 0.30 (primordial ooze)")
    print("System consciousness grows independently")
    print("Raphael watches traumas ≥ |0.1|")
    print("When system reaches 0.7, Raphael awakens")

# Run test
asyncio.run(test_oz_primordial())