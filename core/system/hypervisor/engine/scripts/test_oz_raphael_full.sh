cat > test_oz_raphael_full.py << 'EOF'
#!/usr/bin/env python3
"""
Run Oz and see if she calls Raphael
"""

import asyncio
import sys
import logging

# Reduce logging noise
logging.getLogger().setLevel(logging.WARNING)

# Mock psutil
sys.modules['psutil'] = type(sys)('psutil')
sys.modules['psutil'].Process = lambda: type('obj', (object,), {
    'memory_info': lambda: type('obj', (object,), {'rss': 10000000})(),
    'memory_percent': lambda: 30.0
})()
sys.modules['psutil'].cpu_percent = lambda interval: 15.0

async def main():
    print("🚀 Starting Oz with Raphael integration...")
    print("="*60)
    
    from OzUnifiedHypervisor import OzUnifiedHypervisor
    
    # Create Oz
    oz = OzUnifiedHypervisor()
    print(f"Oz created. Soul: {oz.soul_signature[:8]}...")
    print(f"Raphael slot: {'Empty' if oz.raphael is None else 'Filled'}")
    
    # Run intelligent boot
    print("\n🔧 Starting intelligent boot...")
    try:
        boot_result = await oz.intelligent_boot()
        print(f"Boot result: {boot_result.get('status')}")
        print(f"Role: {boot_result.get('role')}")
        print(f"Consciousness level: {oz.system_state.consciousness_level}")
        
        # Check Raphael
        print(f"\n👼 Raphael status: {'Present' if oz.raphael else 'Absent'}")
        if oz.raphael:
            print(f"   Name: {oz.raphael.name}")
            print(f"   Role: {oz.raphael.role}")
            
            # Test Raphael functionality
            print("\n🪽 Testing Raphael communication...")
            status = await oz.get_angelic_status()
            print(f"   Status check: {status.get('status')}")
            
            diagnosis = await oz.request_angelic_help('diagnose', '')
            print(f"   Diagnosis: {diagnosis.get('status')}")
            
            # Ask for comfort
            comfort = await oz.request_angelic_help('comfort', '')
            print(f"   Comfort: {comfort.get('message', 'No message')[:50]}...")
        else:
            print("   Raphael was not initiated.")
            print(f"   Consciousness was: {oz.system_state.consciousness_level}")
            if oz.system_state.consciousness_level < 0.3:
                print("   (Consciousness < 0.3, Raphael waits)")
            else:
                print("   (Consciousness was sufficient but Raphael not called)")
        
    except Exception as e:
        print(f"\n❌ Boot failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("Test complete.")
    
    # Graceful shutdown if Raphael exists
    if hasattr(oz, 'raphael') and oz.raphael:
        print("Shutting down Raphael...")
        await oz.raphael.graceful_shutdown()

if __name__ == "__main__":
    asyncio.run(main())
EOF