# Create a test script
cat > test_datetime_bug.py << 'EOF'
import asyncio
from OzUnifiedHypervisor import OzUnifiedHypervisor

async def test():
    oz = OzUnifiedHypervisor()
    
    # Test each initialization method
    methods = [
        '_initialize_governance_system',
        '_initialize_evolution_system',
        '_initialize_iot_engine',
        '_initialize_need_assessment',
        '_initialize_constraint_aware',
        '_initialize_council_governance',
        '_initialize_specialized_engines'
    ]
    
    for method_name in methods:
        try:
            method = getattr(oz, method_name)
            await method()
            print(f'✅ {method_name}: OK')
        except Exception as e:
            print(f'❌ {method_name}: {e}')
            break

asyncio.run(test())
EOF

python3 test_datetime_bug.py 2>&1