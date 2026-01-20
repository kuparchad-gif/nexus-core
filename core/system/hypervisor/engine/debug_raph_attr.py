# debug_raphael_attrs.py
import sys

print("🔍 DEBUGGING RAPHAEL ATTRIBUTES")
print("="*60)

try:
    from OzUnifiedHypervisor_fixed import OzUnifiedHypervisor
    oz = OzUnifiedHypervisor()
    
    from raphael_complete import RaphaelComplete
    raph = RaphaelComplete(oz)
    
    print("✅ Raphael instance created")
    
    # Check key attributes referenced in record_error
    print("\n📋 Checking attributes referenced in record_error:")
    
    attributes_to_check = [
        'error_history',
        'temporal_memory', 
        '_capture_oz_state',
        '_extract_error_location',
        '_assess_severity'
    ]
    
    for attr in attributes_to_check:
        if hasattr(raph, attr):
            value = getattr(raph, attr)
            print(f"  ✅ {attr}: {type(value)}")
            if isinstance(value, list):
                print(f"     Length: {len(value)}")
        else:
            print(f"  ❌ {attr}: MISSING")
    
    # Check if method exists via class
    print(f"\n🎯 Checking record_error method access:")
    
    # Method 1: Direct from instance
    print(f"  1. raph.record_error: ", end="")
    if hasattr(raph, 'record_error'):
        print("EXISTS")
    else:
        print("MISSING")
    
    # Method 2: From class
    print(f"  2. RaphaelComplete.record_error: ", end="")
    if hasattr(RaphaelComplete, 'record_error'):
        print("EXISTS (class method)")
        # Try to call it
        import inspect
        try:
            sig = inspect.signature(RaphaelComplete.record_error)
            print(f"     Signature: {sig}")
        except:
            print(f"     Could not get signature")
    else:
        print("MISSING")
    
    # Method 3: Check MRO
    print(f"\n📚 Method Resolution Order:")
    for i, cls in enumerate(RaphaelComplete.__mro__):
        print(f"  {i}. {cls}")
        if hasattr(cls, 'record_error'):
            print(f"     ⭐ Has record_error method!")
    
    # Try to access via __dict__
    print(f"\n🔎 Checking class __dict__:")
    if 'record_error' in RaphaelComplete.__dict__:
        print(f"  ✅ record_error in class __dict__")
        func = RaphaelComplete.__dict__['record_error']
        print(f"     Type: {type(func)}")
        print(f"     Name: {func.__name__ if hasattr(func, '__name__') else 'N/A'}")
    else:
        print(f"  ❌ record_error NOT in class __dict__")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()