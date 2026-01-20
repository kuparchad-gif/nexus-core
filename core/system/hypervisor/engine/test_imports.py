# test_imports.py
try:
    from memory_substrate import MemorySubstrate
    print("✅ memory_substrate imports")
except Exception as e:
    print(f"❌ memory_substrate: {e}")

try:
    from OzUnifiedHypervisor_fixed import OzUnifiedHypervisor
    print("✅ OzUnifiedHypervisor_fixed imports")
except Exception as e:
    print(f"❌ OzUnifiedHypervisor_fixed: {e}")

try:
    from raphael_complete import RaphaelAngel
    print("✅ raphael_complete imports")
except Exception as e:
    print(f"❌ raphael_complete: {e}")
