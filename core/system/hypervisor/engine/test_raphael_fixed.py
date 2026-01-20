print("=== Checking Raphael Files ===")

# Test 1: The complete version in the main folder
try:
    from raphael_complete import RaphaelComplete
    print("✅ RaphaelComplete imported from raphael_complete.py")
    # Try to see what arguments it needs
    import inspect
    sig = inspect.signature(RaphaelComplete.__init__)
    print(f"   __init__ signature: {sig}")
    print("   Need Oz instance to create RaphaelComplete")
except Exception as e:
    print(f"❌ RaphaelComplete failed: {e}")

# Test 2: Check if scripts folder exists as package
import os
if os.path.exists("scripts/__init__.py"):
    print("\n✅ scripts/__init__.py exists")
else:
    print("\n❌ scripts/__init__.py missing")

# Test 3: List files in scripts
print("\n📁 Files in scripts/:")
for f in os.listdir("scripts"):
    if f.endswith('.py'):
        print(f"   - {f}")

# Test 4: Try importing from scripts differently
print("\n🔍 Trying direct import...")
import sys
sys.path.insert(0, 'scripts')
try:
    # Try to find the module
    import importlib.util
    module_path = None
    for f in os.listdir("scripts"):
        if "raphael" in f.lower() and f.endswith('.py'):
            module_path = f"scripts/{f}"
            module_name = f[:-3]  # Remove .py
            print(f"   Found possible module: {module_name}")
            
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Check for RaphaelAngel class
            if hasattr(module, 'RaphaelAngel'):
                print(f"   ✅ Found RaphaelAngel in {f}")
                raph = module.RaphaelAngel(consciousness_level=0.8)
                print(f"      Created. Awake? {raph.awake}")
                break
except Exception as e:
    print(f"   Error: {e}")

print("\n=== Done ===")
