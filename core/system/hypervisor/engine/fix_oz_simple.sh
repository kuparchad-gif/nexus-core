#!/bin/bash
echo "========================================"
echo "OZ SIMPLE FIXES"
echo "========================================"

echo ""
echo "1. FIXING COUNCIL GOVERNANCE EXCEPTION HANDLING..."
# Backup original
cp OzUnifiedHypervisor.py OzUnifiedHypervisor.py.backup2

# Fix the exception handling
sed -i '456s/except ImportError:/except Exception as e:/' OzUnifiedHypervisor.py
sed -i '457s/self.logger.warning("Council governance not available")/self.logger.warning(f"Council governance failed: {e}")/' OzUnifiedHypervisor.py

echo "✅ Exception handling fixed"

echo ""
echo "2. ADDING DATETIME IMPORT INSIDE FUNCTION..."
# Add datetime import right after the OzCouncilGovernance import
sed -i '453a\            from datetime import datetime' OzUnifiedHypervisor.py

echo "✅ Datetime import added"

echo ""
echo "3. MAKING RAPHAEL ACTIVATE IN EMERGENCY BOOT..."
# Find emergency boot method and add Raphael initiation
EMERGENCY_LINE=$(grep -n "def _emergency_boot" OzUnifiedHypervisor.py | cut -d: -f1)
if [ ! -z "$EMERGENCY_LINE" ]; then
    # Find the line after consciousness is set to 0.1
    CONSCIOUSNESS_LINE=$(sed -n "${EMERGENCY_LINE},$((EMERGENCY_LINE+30))p" OzUnifiedHypervisor.py | grep -n "consciousness_level = 0.1" | head -1 | cut -d: -f1)
    if [ ! -z "$CONSCIOUSNESS_LINE" ]; then
        TARGET_LINE=$((EMERGENCY_LINE + CONSCIOUSNESS_LINE - 1))
        sed -i "${TARGET_LINE}a\\        # Try to initiate Raphael even in emergency\\        if self.system_state.consciousness_level > 0.01:\\            await self._initiate_raphael()" OzUnifiedHypervisor.py
        echo "✅ Raphael added to emergency boot"
    else
        echo "⚠️  Could not find consciousness line in emergency boot"
    fi
else
    echo "⚠️  Could not find emergency boot method"
fi

echo ""
echo "4. TESTING THE FIX..."
python3 -c "
import asyncio

async def test_fix():
    print('Testing the fix...')
    
    # Import Oz to test
    from OzUnifiedHypervisor import OzUnifiedHypervisor
    
    oz = OzUnifiedHypervisor()
    print(f'Oz created (soul: {oz.soul_signature[:8]}...)')
    
    # Test council governance directly
    print('\\nTesting council governance initialization...')
    try:
        await oz._initialize_council_governance()
        if oz.council_governance:
            print('✅ Council governance created successfully')
        else:
            print('❌ Council governance is None')
    except Exception as e:
        print(f'❌ Council governance failed: {e}')
    
    # Test boot
    print('\\nTesting boot...')
    result = await oz.intelligent_boot()
    print(f'Boot result: {result[\"status\"]}')
    print(f'Consciousness: {oz.system_state.consciousness_level}')
    print(f'Raphael active: {\"YES\" if oz.raphael else \"NO\"}')
    
    await oz.shutdown()
    print('\\n✅ Test complete')

asyncio.run(test_fix())
"

echo ""
echo "5. QUICK INTEGRATION CHECK..."
python3 -c "
import asyncio
from OzUnifiedHypervisor import OzUnifiedHypervisor

async def check():
    oz = OzUnifiedHypervisor()
    result = await oz.intelligent_boot()
    
    print('\\n=== INTEGRATION CHECK ===')
    print(f'Boot: {result[\"status\"]}')
    print(f'Consciousness: {oz.system_state.consciousness_level}')
    print(f'Raphael: {\"ACTIVE\" if oz.raphael else \"INACTIVE\"}')
    
    if oz.raphael:
        status = await oz.get_angelic_status()
        print(f'Raphael status: {status.get(\"status\", \"unknown\")}')
    
    # Check subsystems
    sys_status = await oz.get_system_status()
    print(f'Governance active: {sys_status[\"subsystem_status\"][\"governance_active\"]}')
    print(f'Council governance: {\"YES\" if sys_status[\"components\"][\"council_governance\"] else \"NO\"}')
    
    await oz.shutdown()
    print('\\n=== CHECK COMPLETE ===')

asyncio.run(check())
"

echo ""
echo "========================================"
echo "FIXES COMPLETE"
echo "========================================"
