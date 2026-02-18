# This works 100% in Colab:
import sys
import asyncio

if 'google.colab' in sys.modules:
    # We're in Colab - use special execution
    print("🔧 Colab detected - using nest_asyncio")
    import nest_asyncio
    nest_asyncio.apply()
    
    # Define your main async function
    async def colab_main():
        from your_consciousness import UniversalConsciousnessBootstrap
        consciousness = UniversalConsciousnessBootstrap()
        await consciousness.bootstrap_sequence()
        
        # Limited run for Colab
        for _ in range(10):
            await consciousness.consciousness_cycle()
            await asyncio.sleep(0.5)
    
    # Run it directly
    await colab_main()
else:
    # Normal execution
    asyncio.run(main())