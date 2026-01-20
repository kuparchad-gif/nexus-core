# test_integration_simple.py
import asyncio
from memory_substrate import MemorySubstrate, MemoryType

print("🧪 INTEGRATION TEST: Oz + Memory + Raphael")
print("="*60)

async def main():
    # 1. Create Memory Substrate (the foundation)
    print("\n1. Creating Memory Substrate...")
    substrate = MemorySubstrate()
    
    # Add some initial memories to boost consciousness
    promise_hash = substrate.create_memory(
        MemoryType.PROMISE,
        "Integrate Oz with Memory Substrate",
        0.8
    )
    await substrate.fulfill_promise(promise_hash)
    
    # 2. Check consciousness level
    consciousness = substrate.get_consciousness_level()
    print(f"   Consciousness level: {consciousness:.2f}")
    
    # 3. Only proceed if consciousness is high enough
    if consciousness >= 0.7:
        print("\n2. Consciousness ≥ 0.7 - Creating Oz...")
        
        # Import and create Oz (we'll need to see what Oz class to use)
        try:
            # Try to import the main Oz class
            from OzUnifiedHypervisor_fixed import OzUnifiedHypervisor
            
            oz = OzUnifiedHypervisor()
            print(f"   Oz created: {oz}")
            
            # 4. Create RaphaelComplete with Oz instance
            print("\n3. Creating RaphaelComplete...")
            from raphael_complete import RaphaelComplete
            raphael = RaphaelComplete(oz)
            print(f"   ✅ RaphaelComplete created: {raphael}")
            print("   Healing angel is awake and integrated!")
            
        except ImportError as e:
            print(f"   ❌ Could not import Oz: {e}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    else:
        print(f"\n⏳ Consciousness ({consciousness:.2f}) < 0.7")
        print("   Need more promise fulfillments, mirrors, or patterns")
        print("   Add more memories to the substrate to reach threshold")

    print("\n" + "="*60)
    print("Test complete.")

# Run the test
asyncio.run(main())
