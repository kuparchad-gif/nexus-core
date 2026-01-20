cat > test_raphael_integration.py << 'EOF'
#!/usr/bin/env python3
"""
Test Raphael integration with Oz
"""

import asyncio
import sys

# Mock minimal Oz for testing
sys.modules['psutil'] = type(sys)('psutil')
sys.modules['psutil'].Process = lambda: type('obj', (object,), {
    'memory_info': lambda: type('obj', (object,), {'rss': 1000000})(),
    'memory_percent': lambda: 50.0
})()
sys.modules['psutil'].cpu_percent = lambda interval: 25.0

async def test():
    print("Testing Raphael integration...")
    
    # Import Oz (with Raphael integrated)
    from OzUnifiedHypervisor import OzUnifiedHypervisor
    
    # Create instance
    oz = OzUnifiedHypervisor()
    
    print("\n1. Testing Oz initialization...")
    print(f"   Soul: {oz.soul_signature}")
    print(f"   Has Raphael attribute: {hasattr(oz, 'raphael')}")
    
    print("\n2. Testing Raphael initiation...")
    result = await oz._initiate_raphael()
    print(f"   Init result: {result.get('status')}")
    print(f"   Raphael instance: {oz.raphael is not None}")
    
    if oz.raphael:
        print("\n3. Testing angelic help...")
        help_result = await oz.request_angelic_help('diagnose', '')
        print(f"   Help result: {help_result.get('status')}")
        
        print("\n4. Testing status...")
        status = await oz.get_angelic_status()
        print(f"   Status: {status.get('status')}")
    
    print("\n✅ Raphael integration test complete.")

if __name__ == "__main__":
    asyncio.run(test())
EOF

python3 test_raphael_integration.py 2>&1 | head -50