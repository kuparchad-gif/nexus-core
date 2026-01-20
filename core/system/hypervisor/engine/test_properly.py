# test_all_raphaels.py
import sys

print("🔍 FINDING RAPHAEL...")
print("="*60)

# Try different import paths
import_paths = [
    ('raphael_complete', 'from engine root'),
    ('scripts.raphael_angelic_monitor', 'from scripts folder'),
]

for module_path, description in import_paths:
    print(f"\nTrying: {module_path} ({description})")
    try:
        if '.' in module_path:
            # For dotted paths like scripts.raphael_angelic_monitor
            module_parts = module_path.split('.')
            module = __import__(module_parts[0])
            for part in module_parts[1:]:
                module = getattr(module, part)
        else:
            # For direct module
            module = __import__(module_path)
        
        # Check if RaphaelAngel class exists
        if hasattr(module, 'RaphaelAngel'):
            print(f"  ✅ Found RaphaelAngel class")
            # Try to instantiate
            try:
                raph = module.RaphaelAngel(consciousness_level=0.5)
                print(f"  ✅ Can instantiate. Awake: {getattr(raph, 'awake', 'N/A')}")
            except Exception as e:
                print(f"  ⚠️  Instantiation failed: {e}")
        else:
            print(f"  ⚠️  No RaphaelAngel class. Available: {[x for x in dir(module) if not x.startswith('_')][:10]}")
            
    except ImportError as e:
        print(f"  ❌ Import failed: {e}")
    except Exception as e:
        print(f"  ❌ Error: {e}")

print("\n" + "="*60)
print("Check complete.")