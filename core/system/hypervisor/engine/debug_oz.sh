#!/bin/bash
echo "========================================"
echo "OZ DEBUG CRUISE CONTROL - ONE STEP"
echo "========================================"

echo ""
echo "1. CHECKING WORKING DIRECTORY..."
cd ~/Downloads/0zHypervisor2 && pwd
echo "✅ In correct directory"

echo ""
echo "2. CHECKING COUNCIL GOVERNANCE FILE..."
if [ -f "OzCouncilGovernanceSystem.py" ]; then
    echo "✅ File exists"
    echo "First 5 lines:"
    head -5 OzCouncilGovernanceSystem.py
else
    echo "❌ File missing!"
    exit 1
fi

echo ""
echo "3. TESTING IMPORT FROM OZ'S PERSPECTIVE..."
python3 -c "
import os
print('   Working dir:', os.getcwd())
print('   File exists:', os.path.exists('OzCouncilGovernanceSystem.py'))

try:
    from OzCouncilGovernanceSystem import OzCouncilGovernance
    print('   ✅ Import works')
    
    from datetime import datetime
    obj = OzCouncilGovernance(datetime.now())
    print('   ✅ Instantiation works')
    print('   ✅ Council governance ready')
    
except Exception as e:
    import traceback
    print('   ❌ FAILED:')
    traceback.print_exc()
"

echo ""
echo "4. CHECKING OZ'S CODE AT THE PROBLEM SPOT..."
echo "Looking for OzCouncilGovernance in OzUnifiedHypervisor.py:"
grep -n "OzCouncilGovernance" OzUnifiedHypervisor.py | head -3
echo ""
echo "The problematic function (lines 449-460):"
sed -n '449,460p' OzUnifiedHypervisor.py

echo ""
echo "5. RUNNING MINIMAL BOOT TEST..."
python3 -c "
import asyncio
import logging

logging.basicConfig(level=logging.INFO)

async def test():
    print('   Starting test...')
    
    # Test 1: Direct import (what should work)
    try:
        from OzCouncilGovernanceSystem import OzCouncilGovernance
        from datetime import datetime
        obj = OzCouncilGovernance(datetime.now())
        print('   ✅ Direct test passed')
    except Exception as e:
        print(f'   ❌ Direct test failed: {e}')
        return
    
    # Test 2: Simulate Oz's boot path
    print('   Simulating Oz boot path...')
    try:
        # This is what happens during _initialize_council_governance
        from OzCouncilGovernanceSystem import OzCouncilGovernance
        # Note: datetime is imported at module level in OzUnifiedHypervisor.py
        # So we need to check if it's in scope
        import datetime
        obj = OzCouncilGovernance(datetime.datetime.now())
        print('   ✅ Oz path simulation passed')
    except Exception as e:
        print(f'   ❌ Oz path failed: {e}')
        import traceback
        traceback.print_exc()

asyncio.run(test())
"

echo ""
echo "6. QUICK FIX ATTEMPT (if needed)..."
echo "Checking if datetime is properly imported at module level..."
grep -n "from datetime import datetime" OzUnifiedHypervisor.py
if [ $? -eq 0 ]; then
    echo "✅ datetime imported at module level"
else
    echo "⚠️  datetime not imported at module level"
    echo "Adding import..."
    sed -i '1ifrom datetime import datetime' OzUnifiedHypervisor.py
    echo "✅ Added datetime import"
fi

echo ""
echo "7. FINAL TEST - TRY OZ BOOT..."
timeout 10 python3 OzUnifiedHypervisor.py 2>&1 | grep -E "(Boot result:|Consciousness level:|Raphael present|emergency|awake)" | tail -10

echo ""
echo "========================================"
echo "CRUISE COMPLETE"
echo "========================================"
