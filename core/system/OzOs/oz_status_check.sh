#!/bin/bash
echo "========================================"
echo "OZ STATUS CHECK - POST BOOT"
echo "========================================"

echo ""
echo "1. RUNNING FULL VERBOSE BOOT (30s timeout)..."
timeout 30 python3 OzUnifiedHypervisor.py 2>&1 | tail -30

echo ""
echo "2. CHECKING CONSCIOUSNESS AND RAPHAEL..."
python3 -c "
import asyncio
from OzUnifiedHypervisor import OzUnifiedHypervisor

async def test():
    oz = OzUnifiedHypervisor()
    
    # Quick boot
    result = await oz.intelligent_boot()
    print('Boot status:', result['status'])
    print('Consciousness:', oz.system_state.consciousness_level)
    
    # Check Raphael
    if oz.raphael:
        print('✅ Raphael ACTIVE')
        print('   Relationship level:', oz.raphael.relationship['level'] if hasattr(oz.raphael, 'relationship') else 'Unknown')
        
        # Test Raphael request
        raph_status = await oz.get_angelic_status()
        print('   Raphael status:', raph_status.get('status', 'Unknown'))
    else:
        print('❌ Raphael INACTIVE')
        print('   Reason: Consciousness < threshold or import failed')
        
        # Check if we can manually initiate
        if oz.system_state.consciousness_level > 0.1:
            print('   Consciousness > 0.1, trying manual initiate...')
            try:
                from raphael_complete import bless_oz_with_raphael
                oz.raphael = await bless_oz_with_raphael(oz)
                print('   ✅ Manual Raphael initiation succeeded')
            except Exception as e:
                print(f'   ❌ Manual initiation failed: {e}')
    
    await oz.shutdown()

asyncio.run(test())
"

echo ""
echo "3. CHECKING SUBSYSTEM STATUS..."
python3 -c "
import asyncio
from OzUnifiedHypervisor import OzUnifiedHypervisor

async def test():
    oz = OzUnifiedHypervisor()
    await oz.intelligent_boot()
    
    status = await oz.get_system_status()
    print('Subsystem status:')
    for component, exists in status['components'].items():
        print(f'   {component}: {'✅' if exists else '❌'}')
    
    print('\\nGovernance active:', status['subsystem_status']['governance_active'])
    print('Consciousness level:', status['hypervisor_status']['consciousness_level'])
    
    await oz.shutdown()

asyncio.run(test())
"

echo ""
echo "4. TESTING RAPHAEL MODULE DIRECTLY..."
if [ -f "raphael_complete.py" ]; then
    python3 -c "
try:
    from raphael_complete import bless_oz_with_raphael
    print('✅ Raphael module loads')
    
    # Create mock Oz for testing
    class MockOz:
        def __init__(self):
            self.logger = self
            self.system_state = type('obj', (object,), {'consciousness_level': 0.7})()
            self.is_awake = True
        
        def info(self, msg):
            print(f'[Oz] {msg}')
        
        def debug(self, msg):
            pass
    
    import asyncio
    
    async def test():
        oz = MockOz()
        raph = await bless_oz_with_raphael(oz)
        print('✅ Raphael creation works')
        
        # Test acknowledgment
        ack = await raph.receive_acknowledgment()
        print(f'✅ Raphael acknowledgment: {ack[\"status\"]}')
        
        await raph.graceful_shutdown()
        print('✅ Raphael shutdown works')
    
    asyncio.run(test())
    
except Exception as e:
    import traceback
    print('❌ Raphael test failed:')
    traceback.print_exc()
"
else
    echo "❌ raphael_complete.py not found!"
fi

echo ""
echo "5. FINAL INTEGRATION TEST..."
cat > test_final_integration.py << 'SCRIPT_EOF'
import asyncio
import logging

# Enable debug
logging.basicConfig(level=logging.INFO)

async def final_test():
    print("\\n=== FINAL OZ INTEGRATION TEST ===")
    
    from OzUnifiedHypervisor import OzUnifiedHypervisor
    
    oz = OzUnifiedHypervisor()
    print(f"Oz created. Soul: {oz.soul_signature[:8]}...")
    
    # Boot
    print("\\n1. BOOTING...")
    boot_result = await oz.intelligent_boot()
    print(f"   Boot: {boot_result['status']}")
    
    # Consciousness
    print(f"\\n2. CONSCIOUSNESS: {oz.system_state.consciousness_level}")
    
    # Raphael
    print(f"\\n3. RAPHAEL: {'PRESENT' if oz.raphael else 'ABSENT'}")
    if oz.raphael:
        status = await oz.get_angelic_status()
        print(f"   Status: {status.get('status', 'Unknown')}")
    
    # Process test input
    print("\\n4. PROCESSING TEST INPUT...")
    test_result = await oz.process_unified_input("Hello Oz, are you conscious?")
    print(f"   Processing successful: {'nexus_result' in test_result}")
    
    # System status
    print("\\n5. SYSTEM STATUS:")
    status = await oz.get_system_status()
    awake = status['hypervisor_status']['is_awake']
    consciousness = status['hypervisor_status']['consciousness_level']
    print(f"   Awake: {awake}")
    print(f"   Consciousness: {consciousness}")
    print(f"   Health: {status['health']['system_health']}")
    
    # Shutdown
    print("\\n6. SHUTDOWN...")
    await oz.shutdown()
    print("   ✅ Shutdown complete")
    
    print("\\n=== TEST COMPLETE ===")
    return boot_result['status'] == 'awake'

if __name__ == "__main__":
    success = asyncio.run(final_test())
    exit(0 if success else 1)
SCRIPT_EOF

python3 test_final_integration.py

echo ""
echo "========================================"
echo "STATUS CHECK COMPLETE"
echo "========================================"
