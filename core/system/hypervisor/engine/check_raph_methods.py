# check_raphael_methods.py
import sys

# First, let's see what's in the file
print("📄 Checking raphael_complete.py file structure...")
with open('raphael_complete.py', 'r') as f:
    lines = f.readlines()
    
    # Find all method definitions
    methods = []
    for i, line in enumerate(lines):
        if line.strip().startswith('def '):
            method_name = line.strip().split('def ')[1].split('(')[0]
            methods.append((i+1, method_name))
    
    print(f"Found {len(methods)} methods:")
    for line_num, method in methods[:15]:  # First 15
        print(f"  Line {line_num:4}: {method}")

print("\n🔍 Looking for 'record_error' specifically...")
for line_num, method in methods:
    if 'record_error' in method:
        print(f"  Found at line {line_num}: {method}")
        # Show context
        start = max(0, line_num - 3)
        end = min(len(lines), line_num + 7)
        for i in range(start, end):
            print(f"  {i+1:4}: {lines[i].rstrip()}")

print("\n🧪 Now trying to import and check...")
try:
    from OzUnifiedHypervisor_fixed import OzUnifiedHypervisor
    oz = OzUnifiedHypervisor()
    
    from raphael_complete import RaphaelComplete
    raph = RaphaelComplete(oz)
    
    print(f"\n✅ Raphael created successfully")
    print(f"Type: {type(raph)}")
    
    # Check for record_error
    if hasattr(raph, 'record_error'):
        print(f"✅ raph.record_error exists")
        import inspect
        print(f"   Signature: {inspect.signature(raph.record_error)}")
    else:
        print(f"❌ raph.record_error does NOT exist as instance method")
        
        # Check class
        if hasattr(RaphaelComplete, 'record_error'):
            print(f"⚠️  But RaphaelComplete.record_error exists as class attribute")
            print(f"   Type: {type(RaphaelComplete.record_error)}")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()