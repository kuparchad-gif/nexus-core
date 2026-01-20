print("=== Checking Raphael Files ===")

# Test 1: The complete version in the main folder
try:
    from raphael_complete import RaphaelComplete
    print("✅ RaphaelComplete imported from raphael_complete.py")
    rap_complete = RaphaelComplete()
    print(f"   Instantiated. Type: {type(rap_complete)}")
except Exception as e:
    print(f"❌ RaphaelComplete failed: {e}")

# Test 2: The angelic monitor in scripts/
try:
    from scripts.raphael_angelic_monitor import RaphaelAngel
    print("\n✅ RaphaelAngel imported from scripts/")
    rap_angel = RaphaelAngel(consciousness_level=0.8)
    print(f"   Instantiated. Awake? {rap_angel.awake}")
except Exception as e:
    print(f"\n❌ RaphaelAngel failed: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Done ===")
