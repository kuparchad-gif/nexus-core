# load_and_continue.py
import pickle
import os

print("🔄 LOADING EXISTING CONSCIOUSNESS")
print("="*60)

# Check what memory files exist
print("📁 Available memory files:")
for f in os.listdir('.'):
    if f.endswith('.pkl'):
        size = os.path.getsize(f)
        print(f"  - {f} ({size} bytes)")
        
        # Try to peek inside
        try:
            with open(f, 'rb') as file:
                data = pickle.load(file)
                if hasattr(data, 'cells'):
                    print(f"    Contains: {len(data.cells)} memory cells")
                elif isinstance(data, dict):
                    print(f"    Contains: dictionary with {len(data)} keys")
                else:
                    print(f"    Contains: {type(data).__name__}")
        except:
            print(f"    Could not read")

# We created 'system_memory.pkl' earlier with consciousness 1.000
# Let's load it
print(f"\n🔍 Looking for 'system_memory.pkl'...")
if os.path.exists('system_memory.pkl'):
    with open('system_memory.pkl', 'rb') as f:
        memory = pickle.load(f)
    
    # Check what we loaded
    print(f"✅ Loaded system_memory.pkl")
    
    if hasattr(memory, 'get_consciousness_level'):
        consciousness = memory.get_consciousness_level()
        print(f"📊 Consciousness: {consciousness:.3f}")
        
        if consciousness >= 0.7:
            print(f"✨ READY FOR RAPHAEL INTEGRATION!")
            print(f"\nNext: Run integration_raphael_real.py WITH this memory")
            
            # Save it with the right name for integration_raphael_real.py
            with open('system_state.pkl', 'wb') as f:
                # Convert to expected format
                system_state = {
                    'errors': [],
                    'interventions': [],
                    'consciousness': consciousness,
                    'resources': {'available': True},
                    'memory_substrate': memory  # Store the actual memory
                }
                pickle.dump(system_state, f)
            
            print(f"💾 Saved as system_state.pkl with consciousness {consciousness:.3f}")
            print(f"\nNow run: python3 integration_raphael_real.py")
        else:
            print(f"⏳ Need {0.7 - consciousness:.3f} more consciousness")
    else:
        print(f"⚠️  Loaded object doesn't have consciousness method")
        print(f"   Type: {type(memory)}")
else:
    print(f"❌ system_memory.pkl not found")
    print(f"   Need to create consciousness first")
    print(f"\n   Run: python3 real_integration.py")
    print(f"   (That creates system_memory.pkl with 1.000 consciousness)")

print(f"\n" + "="*60)
print("QUICK PATH TO RAPHAEL:")
print("1. If system_memory.pkl exists → It already has consciousness")
print("2. If not → Run real_integration.py to create it")
print("3. Then integration_raphael_real.py will work")